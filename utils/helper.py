import math
from typing import List, Optional
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, KeysView, ValuesView, ItemsView

import numpy as np
import torch


def save_training_history(
    history: Dict[str, Any],
    config: Dict[str, Any],
    save_dir: str = "training_results",
) -> str:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer_type = config.get("model_cfg", {}).get("model", "unknown")
    base_name = f"training_history_{trainer_type}_{timestamp}"
    json_path = save_dir_path / f"{base_name}.json"
    try:
        json_ready_history = convert_to_json_compatible(history)
        json_ready_config = convert_to_json_compatible(config)
        payload = {
            "history": json_ready_history,
            "config": json_ready_config,
            "timestamp": timestamp,
            "trainer_type": trainer_type,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"History saved as JSON: {json_path}")
        return str(json_path)
    except Exception as exc:
        print(f"Warning: saving JSON failed – {exc}")
        return ""


def convert_to_json_compatible(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): convert_to_json_compatible(v) for k, v in obj.items()}

    if isinstance(obj, (KeysView, ValuesView, ItemsView)):
        return [convert_to_json_compatible(item) for item in obj]

    if isinstance(obj, (list, tuple, set)):
        return [convert_to_json_compatible(item) for item in obj]

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def load_training_history(filepath: str) -> Dict[str, Any]:
    filepath = Path(filepath)
    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def get_train_class_list(
    true_inc: bool = True,
    labels: List[int] = [0, 1],
    batches: Optional[List[List[int]]] = None,
    num_epochs: int = 14,
):
    ewc_lists, val_lists, train_lists = [], [], []
    n_labels = len(labels)
    if true_inc:
        add_step = math.ceil(num_epochs / (n_labels - 1))
        for epoch in range(num_epochs):
            current_class = (
                min(n_labels - 1, 1 + epoch // add_step) if n_labels > 1 else 0
            )
            allowed = list(range(current_class + 1))
            train = (
                [labels[0], labels[current_class]] if n_labels > 1 else allowed.copy()
            )
            ewc = list(range(1, current_class)) if epoch >= add_step else [0, 1]

            ewc_lists.append(ewc)
            val_lists.append(allowed)
            train_lists.append(train)
    else:
        if not batches:
            raise ValueError("For Batch-Mode please provide Batch-Groups!")
        num_batches = len(batches)
        step = math.ceil((num_epochs + 1) / num_batches)
        for epoch in range(num_epochs + 1):
            current_batch = min(num_batches - 1, epoch // step)
            batch_curr = batches[current_batch]
            allowed = sorted({c for grp in batches[: current_batch + 1] for c in grp})
            if epoch == 0:
                ewc = batch_curr.copy()
            else:
                ewc = [c for c in allowed if c not in batch_curr]

            train = batch_curr.copy()
            ewc_lists.append(ewc)
            val_lists.append(allowed)
            train_lists.append(train)

    return train_lists, val_lists, ewc_lists

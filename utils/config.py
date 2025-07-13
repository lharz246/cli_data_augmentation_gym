# utils/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
import configargparse


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise configargparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _filter_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _make_group(ns: configargparse.Namespace, keys: List[str]) -> Dict[str, Any]:
    return _filter_none({k: getattr(ns, k, None) for k in keys})


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    elif value is None:
        return {}
    else:
        print(
            f"Warning: Expected dict but got {type(value)}: {value}. Using empty dict instead."
        )
        return {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(base, dict):
        print(
            f"Warning: base is not a dict (type: {type(base)}, value: {base}). Using empty dict."
        )
        base = {}

    merged = base.copy()
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


ARG_SPECS: Dict[str, List[tuple[str, dict[str, Any]]]] = {
    "Training": [
        ("--epochs", dict(type=int)),
        ("--optimizer", dict(type=str)),
        ("--scheduler", dict(type=str)),
        ("--batch_size", dict(type=int)),
        ("--lr", dict(type=float)),
        ("--weight_decay", dict(type=float)),
        ("--momentum", dict(type=float)),
        ("--max_norm", dict(type=float)),
        ("--trainer_type", dict(type=str)),
        ("--data_path", dict(type=str)),
        ("--output_dir", dict(type=str)),
        ("--balance_classes", dict(type=str2bool)),
        ("--use_mixed_precision", dict(type=str2bool)),
        ("--true_incremental", dict(type=str2bool)),
        ("--use_augmentation", dict(type=str2bool)),
        ("--target_class", dict(type=int)),
        ("--target_labels", dict(type=int, nargs="+")),
        ("--augment_factor", dict(type=float)),
    ],
    "WandB": [
        ("--use_wandb", dict(type=str2bool)),
        ("--wandb_project", dict(type=str)),
        ("--wandb_entity", dict(type=str)),
        ("--wandb_run_name", dict(type=str)),
        ("--wandb_group", dict(type=str)),
        ("--wandb_tags", dict(type=str, nargs="+")),
    ],
    "PackNet": [
        ("--use_packnet", dict(type=str2bool)),
        ("--prune_ratio", dict(type=float)),
        ("--packnet_strategy", dict(type=str)),
        ("--packnet_scale", dict(type=float)),
    ],
}


def build_parser() -> configargparse.ArgParser:

    parser = configargparse.ArgParser(
        description="Train model with YAML/CLI configuration",
        default_config_files=[],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        ignore_unknown_config_file_keys=True,  # <-- Kern des Fixes
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        is_config_file=True,
        help="Pfad zur YAML-Konfigurationsdatei",
    )

    for grp_name, specs in ARG_SPECS.items():
        grp = parser.add_argument_group(grp_name)
        for flag, kwargs in specs:
            grp.add_argument(flag, default=None, **kwargs)

    return parser


def load_config(args: configargparse.Namespace) -> Dict[str, Any]:

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    raw: Dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}

    train_cli = _make_group(args, [k.lstrip("-") for k, _ in ARG_SPECS["Training"]])
    wandb_cli = _make_group(args, [k.lstrip("-") for k, _ in ARG_SPECS["WandB"]])
    pack_cli = _make_group(args, [k.lstrip("-") for k, _ in ARG_SPECS["PackNet"]])

    raw_train = _ensure_dict(raw.get("train_cfg", {}))
    raw_wandb = _ensure_dict(raw.get("wandb_cfg", raw.get("wandb", {})))
    raw_pack = _ensure_dict(raw.get("packnet_cfg", raw.get("packnet", {})))

    for k in [k.lstrip("-") for k, _ in ARG_SPECS["Training"]]:
        if k in raw and k not in raw_train:
            raw_train[k] = raw[k]

    train_cfg = _merge_dict(raw_train, train_cli)
    wandb_cfg = _merge_dict(raw_wandb, wandb_cli)
    packnet_cfg = _merge_dict(raw_pack, pack_cli)

    data_path = train_cfg.get("data_path")
    if data_path is None:
        raise ValueError("`data_path` has to be provided either cli or yaml.")
    data_dir = Path(data_path)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found – Dataset preprocessed??")
    meta = json.loads(meta_path.read_text())

    data_cfg = {
        "paths": {p: str(data_dir / f"{p}.pt") for p in ("train", "val", "test")},
        "balance_classes": train_cfg.get(
            "balance_classes", raw.get("balance_classes", True)
        ),
        "true_incremental": train_cfg.get(
            "true_incremental", raw.get("true_incremental", False)
        ),
        "output_dir": train_cfg.get("output_dir", raw.get("output_dir", "results")),
        "input_dim": meta["input_shapes"]["train"]["input"][1],
        "label_names": meta["label_mapping"].keys(),
        "all_labels": list(meta.get("label_mapping", {}).values()),
        "stats_path": raw.get("stats_path", None),
        "batch_size": train_cfg.get("batch_size", 1),
        "target_class": train_cfg.get("target_class", None),
        "target_labels": train_cfg.get("target_labels", None),
        "augment_factor": train_cfg.get("augment_factor", 0.0),
        "use_augmentation": train_cfg.get("use_augmentation", False),
    }
    if "batch_groups" in raw:
        data_cfg["batch_groups"] = raw["batch_groups"]

    loss_cfg = raw.get("losses", [])
    scheduler_params = raw.get("scheduler_params", [])
    augmenter_configs = raw.get("augmenter_configs", [])
    augmenter_configs = {
        d["name"]: {k: v for k, v in d.items() if k != "name"}
        for d in augmenter_configs
    }
    model_cfg = raw.get("model", [])[0]
    data_cfg["augmentation_cfg"] = augmenter_configs
    train_cfg["scheduler_params"] = scheduler_params
    return {
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "loss_cfg": loss_cfg,
        "wandb_cfg": wandb_cfg,
        "packnet_cfg": packnet_cfg,
        "data_cfg": data_cfg,
    }

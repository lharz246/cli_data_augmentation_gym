from __future__ import annotations
import torch
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)


class Metrics:
    _task_history: Dict[int, Dict[str, float]] = {}
    _current_task: int = 0
    _auto_incremental: bool = True

    @staticmethod
    def get_metrics(
        batch_list: List[Dict[str, Any]],
        mode: str = "train",
        task_history: Optional[Dict[int, Dict[str, float]]] = None,
        current_task: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not batch_list:
            return {}
        if Metrics._auto_incremental:
            current_task = current_task or Metrics._current_task
            task_history = task_history or Metrics._task_history
        if mode in {"val", "validation"}:
            mode = "validation"
        if mode == "train":
            return Metrics._aggregate_train_metrics(batch_list)
        elif mode in {"validation", "test"}:
            metrics = Metrics._aggregate_validation_metrics(
                batch_list,
                test=(mode == "test"),
                task_history=task_history,
                current_task=current_task,
            )
            if Metrics._auto_incremental:
                Metrics._auto_update_task_history(current_task, metrics)
            return metrics
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _aggregate_train_metrics(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_samples = sum(b.get("batch_size", 0) for b in batch_list)
        if total_samples == 0:
            return {}
        metrics: Dict[str, Any] = {}
        exemplar = batch_list[0]
        if "loss" in exemplar:
            loss = exemplar["loss"]
            if isinstance(loss, dict):
                for comp in loss:
                    weighted_loss = sum(
                        b["loss"].get(comp, 0) * b.get("batch_size", 0)
                        for b in batch_list
                    )
                    key = "loss" if comp == "total" else comp
                    metrics[key] = weighted_loss / total_samples
            else:
                weighted_loss = sum(
                    b.get("loss", 0) * b.get("batch_size", 0) for b in batch_list
                )
                metrics["loss"] = weighted_loss / total_samples
        if "correct" in exemplar:
            total_correct = sum(b.get("correct", 0) for b in batch_list)
            metrics["accuracy"] = total_correct / total_samples
        if "task_ids" in exemplar:
            task_ids = {
                tid for b in batch_list if "task_ids" in b for tid in b["task_ids"]
            }
            if task_ids:
                metrics["task_ids"] = sorted(task_ids)
        numeric_keys = [
            k
            for k, v in exemplar.items()
            if k not in {"loss", "batch_size", "correct", "task_ids"}
            and isinstance(v, (int, float))
        ]
        for key in numeric_keys:
            metrics[key] = sum(b.get(key, 0) for b in batch_list) / len(batch_list)

        return metrics

    @staticmethod
    def _aggregate_validation_metrics(
        batch_list: List[Dict[str, Any]],
        *,
        test: bool,
        task_history: Optional[Dict[int, Dict[str, float]]] = None,
        current_task: Optional[int] = None,
    ) -> Dict[str, Any]:
        avg_loss = sum(b.get("loss", 0) for b in batch_list) / len(batch_list)
        preds = Metrics._concatenate_arrays("all_preds", batch_list)
        labels = Metrics._concatenate_arrays("all_labels", batch_list)
        return Metrics._calculate_stats(
            avg_loss=avg_loss,
            preds=preds,
            labels=labels,
            test=test,
            task_history=task_history,
            current_task=current_task,
        )

    @staticmethod
    def _concatenate_arrays(
        key: str, batch_list: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        tensors = [b[key] for b in batch_list if key in b]
        if not tensors:
            return None
        if isinstance(tensors[0], torch.Tensor):
            return torch.cat(tensors, dim=0).cpu().numpy()
        return np.concatenate([np.asarray(t) for t in tensors], axis=0)

    @staticmethod
    def _calculate_stats(
        *,
        avg_loss: float,
        preds: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        test: bool = False,
        task_history: Optional[Dict[int, Dict[str, float]]] = None,
        current_task: Optional[int] = None,
    ) -> Dict[str, Any]:
        if preds is None or labels is None:
            return {"loss": avg_loss}
        pred_classes = np.asarray(preds)
        labels = np.asarray(labels)
        label_classes = (labels > 0).astype(int)
        metrics = Metrics._calculate_base_metrics(pred_classes, label_classes, avg_loss)
        label_metrics, task_accuracies = Metrics._calculate_label_metrics(
            pred_classes, labels
        )
        metrics.update(label_metrics)
        if Metrics._auto_incremental and current_task is None:
            current_task = Metrics._auto_detect_task_from_labels(labels)
            if current_task != Metrics._current_task:
                Metrics._current_task = current_task
        if Metrics._auto_incremental or (
            task_history is not None and current_task is not None
        ):
            use_task_history = task_history or Metrics._task_history
            use_current_task = current_task or Metrics._current_task
            incremental_metrics = Metrics._calculate_incremental_metrics(
                task_accuracies, use_task_history, use_current_task
            )
            metrics.update(incremental_metrics)
        if test:
            metrics.update(
                Metrics._calculate_test_metrics(pred_classes, label_classes, labels)
            )
        return metrics

    @staticmethod
    def _calculate_base_metrics(
        pred_classes: np.ndarray, label_classes: np.ndarray, avg_loss: float
    ) -> Dict[str, Any]:
        metrics = {"loss": avg_loss}
        metrics["accuracy"] = accuracy_score(label_classes, pred_classes)
        metrics["hamming_loss"] = hamming_loss(label_classes, pred_classes)
        metrics["jaccard_macro"] = jaccard_score(
            label_classes, pred_classes, average="macro", zero_division=0
        )
        metrics["jaccard_micro"] = jaccard_score(
            label_classes, pred_classes, average="micro", zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            label_classes, pred_classes, average="macro", zero_division=0
        )
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            label_classes, pred_classes, average="micro", zero_division=0
        )
        metrics.update(
            {
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_micro": prec_micro,
                "recall_micro": rec_micro,
                "f1_micro": f1_micro,
            }
        )

        return metrics

    @staticmethod
    def _calculate_label_metrics(
        pred_classes: np.ndarray, labels: np.ndarray
    ) -> tuple[Dict[str, Any], Dict[int, float]]:
        unique_labels = np.unique(labels)
        label_metrics = {}
        task_accuracies = {}
        total_correct = 0
        total_samples = len(labels)
        for label_id in unique_labels:
            label_mask = labels == label_id
            if not np.any(label_mask):
                continue
            expected_class = 0 if label_id == 0 else 1
            label_preds = pred_classes[label_mask]
            correct = np.sum(label_preds == expected_class)
            total = np.sum(label_mask)
            label_acc = correct / total if total > 0 else 0
            label_metrics[f"label_{label_id}_accuracy"] = label_acc
            task_accuracies[int(label_id)] = label_acc
            total_correct += correct
            y_true = label_mask.astype(int)
            y_pred = (label_mask & (pred_classes == expected_class)).astype(int)
            if np.sum(y_pred) > 0 or np.sum(y_true) > 0:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                label_metrics[f"label_{label_id}_precision"] = prec
                label_metrics[f"label_{label_id}_recall"] = rec
                label_metrics[f"label_{label_id}_f1"] = f1
            else:
                label_metrics[f"label_{label_id}_precision"] = 0.0
                label_metrics[f"label_{label_id}_recall"] = 0.0
                label_metrics[f"label_{label_id}_f1"] = 0.0
        label_wise_acc = total_correct / total_samples if total_samples > 0 else 0
        precision_values = [
            v for k, v in label_metrics.items() if k.endswith("_precision")
        ]
        recall_values = [v for k, v in label_metrics.items() if k.endswith("_recall")]
        f1_values = [v for k, v in label_metrics.items() if k.endswith("_f1")]
        label_metrics.update(
            {
                "label_wise_acc": label_wise_acc,
                "label_precision_avg": (
                    np.mean(precision_values) if precision_values else 0
                ),
                "label_recall_avg": np.mean(recall_values) if recall_values else 0,
                "label_f1_avg": np.mean(f1_values) if f1_values else 0,
            }
        )
        return label_metrics, task_accuracies

    @staticmethod
    def _calculate_test_metrics(
        pred_classes: np.ndarray, label_classes: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        metrics = {
            "conf_matrix_classes": confusion_matrix(label_classes, pred_classes),
            "label_classification_breakdown": {},
        }
        for label_id in np.unique(labels):
            label_mask = labels == label_id
            expected_class = 0 if label_id == 0 else 1
            actual_preds = pred_classes[label_mask]
            correct = np.sum(actual_preds == expected_class)
            total = len(actual_preds)
            metrics["label_classification_breakdown"][f"label_{label_id}"] = {
                "correct": int(correct),
                "total": int(total),
                "accuracy": correct / total if total > 0 else 0,
            }

        return metrics

    @staticmethod
    def _calculate_incremental_metrics(
        current_task_accuracies: Dict[int, float],
        task_history: Dict[int, Dict[str, float]],
        current_task: int,
    ) -> Dict[str, Any]:
        metrics = {}
        metrics.update(
            {
                "_debug_current_task": current_task,
                "_debug_task_history_keys": (
                    list(task_history.keys()) if task_history else []
                ),
                "_debug_current_accuracies": current_task_accuracies,
            }
        )
        for task_id in range(current_task + 1):
            if task_id in current_task_accuracies:
                metrics[f"task_{task_id}_current_acc"] = current_task_accuracies[
                    task_id
                ]
        if current_task > 0 and task_history:
            drift_metrics = Metrics._calculate_drift_metrics(
                current_task_accuracies, task_history, current_task
            )
            metrics.update(drift_metrics)
        if current_task_accuracies:
            metrics["average_incremental_accuracy"] = np.mean(
                list(current_task_accuracies.values())
            )
        class_metrics = Metrics._calculate_class_stability_metrics(
            current_task_accuracies
        )
        metrics.update(class_metrics)
        if current_task_accuracies:
            values = list(current_task_accuracies.values())
            metrics.update(
                {
                    "best_task_accuracy": max(values),
                    "worst_task_accuracy": min(values),
                    "accuracy_variance": np.var(values),
                }
            )
        return metrics

    @staticmethod
    def _calculate_drift_metrics(
        current_task_accuracies: Dict[int, float],
        task_history: Dict[int, Dict[str, float]],
        current_task: int,
    ) -> Dict[str, Any]:
        drift_metrics = {}
        accuracy_drifts = {}
        forgetting_rates = {}
        bwt_sum = 0
        bwt_count = 0
        for prev_task in range(current_task):
            if (
                prev_task not in task_history
                or prev_task not in current_task_accuracies
            ):
                continue
            initial_acc = task_history[prev_task].get(
                f"label_{prev_task}_accuracy"
            ) or task_history[prev_task].get("accuracy")
            if initial_acc is None:
                continue
            current_acc = current_task_accuracies[prev_task]
            drift = initial_acc - current_acc
            accuracy_drifts[f"task_{prev_task}_accuracy_drift"] = drift
            if initial_acc > 0:
                forgetting_rates[f"task_{prev_task}_forgetting_rate"] = (
                    drift / initial_acc
                )
            bwt_sum += current_acc - initial_acc
            bwt_count += 1
        if accuracy_drifts:
            drift_metrics["average_accuracy_drift"] = np.mean(
                list(accuracy_drifts.values())
            )
        if forgetting_rates:
            drift_metrics["average_forgetting_rate"] = np.mean(
                list(forgetting_rates.values())
            )
        if bwt_count > 0:
            drift_metrics["backward_transfer"] = bwt_sum / bwt_count
        drift_metrics.update(accuracy_drifts)
        drift_metrics.update(forgetting_rates)
        return drift_metrics

    @staticmethod
    def _calculate_class_stability_metrics(
        current_task_accuracies: Dict[int, float]
    ) -> Dict[str, Any]:
        metrics = {}
        metrics["class_0_stability"] = current_task_accuracies.get(0, 0.0)
        class_1_tasks = [
            task_id for task_id in current_task_accuracies.keys() if task_id > 0
        ]
        if class_1_tasks:
            class_1_accs = [
                current_task_accuracies[task_id] for task_id in class_1_tasks
            ]
            metrics.update(
                {
                    "class_1_adaptation_avg": np.mean(class_1_accs),
                    "class_1_adaptation_std": np.std(class_1_accs),
                    "class_1_tasks_seen": len(class_1_tasks),
                }
            )
        return metrics

    @staticmethod
    def _auto_detect_task_from_labels(labels: np.ndarray) -> int:
        unique_labels = np.unique(labels)
        other_labels = [l for l in unique_labels if l != 0]
        return max(other_labels) - 1 if other_labels else 0

    @staticmethod
    def _auto_update_task_history(task_id: int, metrics: Dict[str, Any]) -> None:
        if task_id is None:
            return
        if task_id not in Metrics._task_history:
            Metrics._task_history[task_id] = {}
        Metrics._task_history[task_id].update(
            {
                "accuracy": metrics.get("accuracy", 0.0),
                "loss": metrics.get("loss", 0.0),
                "label_wise_acc": metrics.get("label_wise_acc", 0.0),
            }
        )
        for key, value in metrics.items():
            if key.startswith("label_") and key.endswith("_accuracy"):
                Metrics._task_history[task_id][key] = value

    @staticmethod
    def update_task_history(
        task_history: Dict[int, Dict[str, float]], task_id: int, metrics: Dict[str, Any]
    ) -> Dict[int, Dict[str, float]]:
        if task_history is None:
            task_history = {}
        task_history[task_id] = {
            "accuracy": metrics.get("accuracy", 0.0),
            "loss": metrics.get("loss", 0.0),
            "label_wise_acc": metrics.get("label_wise_acc", 0.0),
        }
        for key, value in metrics.items():
            if key.startswith("label_") and key.endswith("_accuracy"):
                task_history[task_id][key] = value
        return task_history

    @staticmethod
    def get_catastrophic_forgetting_summary(
        task_history: Dict[int, Dict[str, float]], current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not task_history or len(task_history) <= 1:
            return {"message": "Benötigt mindestens 2 Tasks für Vergessensanalyse"}
        summary = {}
        if "average_forgetting_rate" in current_metrics:
            avg_forgetting = current_metrics["average_forgetting_rate"]
            if avg_forgetting > 0.2:
                severity = "Hoch"
            elif avg_forgetting > 0.1:
                severity = "Mittel"
            elif avg_forgetting > 0.05:
                severity = "Niedrig"
            else:
                severity = "Minimal"
            summary.update(
                {
                    "forgetting_severity": severity,
                    "average_forgetting_rate": avg_forgetting,
                }
            )
        if "class_0_stability" in current_metrics:
            class_0_stability = current_metrics["class_0_stability"]
            summary["class_0_retention"] = (
                "Gut" if class_0_stability > 0.8 else "Schlecht"
            )
        if "class_1_adaptation_avg" in current_metrics:
            class_1_adaptation = current_metrics["class_1_adaptation_avg"]
            summary["class_1_learning"] = (
                "Gut" if class_1_adaptation > 0.7 else "Schlecht"
            )
        return summary

    @staticmethod
    def set_current_task(task_id: int) -> None:
        Metrics._current_task = task_id

    @staticmethod
    def enable_auto_incremental(enable: bool = True) -> None:
        Metrics._auto_incremental = enable

    @staticmethod
    def reset_task_history() -> None:
        Metrics._task_history = {}
        Metrics._current_task = 0

    @staticmethod
    def get_task_history() -> Dict[int, Dict[str, float]]:
        return Metrics._task_history.copy()

    @staticmethod
    def auto_detect_task_from_labels(labels: np.ndarray) -> int:
        return Metrics._auto_detect_task_from_labels(labels)

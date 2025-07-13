from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore")


class TrainingPlotter:
    def __init__(self, history: Dict[str, Any]):
        self.history = history

    def plot_label_accuracy_per_epoch(self, figsize=(14, 8)):
        val_hist = self.history.get("val", [])
        if not val_hist:
            print("No validation history found")
            return plt.figure(figsize=figsize)
        epochs = list(range(1, len(val_hist) + 1))
        all_label_keys = set()
        for ep in val_hist:
            for key in ep.keys():
                if key.startswith("label_") and key.endswith("_accuracy"):
                    all_label_keys.add(key)
        if not all_label_keys:
            print("No label accuracy data found")
            return plt.figure(figsize=figsize)
        label_keys = sorted(all_label_keys, key=lambda x: int(x.split("_")[1]))
        colors = []
        for label_key in label_keys:
            label_num = int(label_key.split("_")[1])
            if label_num == 0:
                colors.append("blue")
            else:
                color_idx = (label_num - 1) % 10
                colors.append(cm.get_cmap("tab10")(color_idx))
        fig, ax = plt.subplots(figsize=figsize)
        for label_key, color in zip(label_keys, colors):
            label_num = int(label_key.split("_")[1])
            class_num = 0 if label_num == 0 else 1
            accuracies = []
            valid_epochs = []
            for epoch, ep_data in enumerate(val_hist, 1):
                if label_key in ep_data and not np.isnan(ep_data[label_key]):
                    accuracies.append(ep_data[label_key])
                    valid_epochs.append(epoch)
            if accuracies:
                line_style = "-" if label_num == 0 else "--"
                ax.plot(
                    valid_epochs,
                    accuracies,
                    marker="o",
                    label=f"Label {label_num} (Class {class_num})",
                    color=color,
                    linewidth=2,
                    markersize=4,
                    linestyle=line_style,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Label Accuracy per Epoch (Incremental Learning)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        return fig

    def plot_global_metrics_per_epoch(self, figsize=(12, 8)):
        val_hist = self.history.get("val", [])
        if not val_hist:
            print("No validation history found")
            return plt.figure(figsize=figsize)
        epochs = list(range(1, len(val_hist) + 1))
        metrics = {
            "f1_macro": "F1 Score (Macro)",
            "recall_macro": "Recall (Macro)",
            "precision_macro": "Precision (Macro)",
            "accuracy": "Accuracy",
        }
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, ax = plt.subplots(figsize=figsize)
        for i, (metric_key, metric_name) in enumerate(metrics.items()):
            values = []
            for ep in val_hist:
                val = ep.get(metric_key, np.nan)
                if hasattr(val, "__len__") and not isinstance(val, str):
                    val = np.mean(val) if len(val) > 0 else np.nan
                values.append(val if not np.isnan(val) else 0)
            if any(v > 0 for v in values):
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    label=metric_name,
                    color=colors[i],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Global Metrics per Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        return fig

    def plot_class_confusion_matrix(self, figsize=(8, 6)):
        test_result = self.history.get("test", [{}])[0]
        if not test_result or "conf_matrix_classes" not in test_result:
            print("No class confusion matrix found")
            return plt.figure(figsize=figsize)
        conf_matrix = np.asarray(test_result["conf_matrix_classes"])
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix - Classes (0 vs 1)")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(conf_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Class 0", "Class 1"])
        ax.set_yticklabels(["Class 0", "Class 1"])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def plot_label_confusion_matrix(self, figsize=(12, 10)):
        test_result = self.history.get("test", [{}])[0]
        conf_matrix = None
        if "conf_matrix_labels" in test_result:
            conf_matrix = np.asarray(test_result["conf_matrix_labels"])
        elif "label_confusion_matrix" in test_result:
            conf_matrix = np.asarray(test_result["label_confusion_matrix"])
        else:
            breakdown = test_result.get("label_classification_breakdown", {})
            if breakdown:
                print("No direct label confusion matrix found, but have breakdown data")
                return self.plot_label_breakdown_accuracy(figsize=figsize)
            else:
                print("No label confusion matrix found")
                return plt.figure(figsize=figsize)
        if conf_matrix.shape[0] != 15 or conf_matrix.shape[1] != 15:
            print(f"Expected 15x15 confusion matrix, got {conf_matrix.shape}")
            return plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix - Labels (0-14)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if conf_matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        format(conf_matrix[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black",
                        fontsize=8,
                    )
        ax.set_xticks(range(15))
        ax.set_yticks(range(15))
        ax.set_xticklabels([f"L{i}" for i in range(15)])
        ax.set_yticklabels([f"L{i}" for i in range(15)])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def plot_label_breakdown_accuracy(self, figsize=(14, 8)):
        test_result = self.history.get("test", [{}])[0]
        if "label_classification_breakdown" not in test_result:
            print("No label classification breakdown found")
            return plt.figure(figsize=figsize)
        breakdown = test_result["label_classification_breakdown"]
        labels = []
        accuracies = []
        totals = []
        corrects = []
        for label_name, stats in breakdown.items():
            label_num = int(label_name.split("_")[1])
            labels.append(label_num)
            accuracies.append(stats["accuracy"])
            totals.append(stats["total"])
            corrects.append(stats["correct"])
        sorted_data = sorted(zip(labels, accuracies, totals, corrects))
        labels, accuracies, totals, corrects = zip(*sorted_data)
        fig, ax = plt.subplots(figsize=figsize)
        colors = []
        for label_num in labels:
            if label_num == 0:
                colors.append("blue")
            else:
                color_idx = (label_num - 1) % 10
                colors.append(cm.get_cmap("tab10")(color_idx))
        bars = ax.bar(range(len(labels)), accuracies, color=colors, alpha=0.7)
        ax.set_ylabel("Accuracy")
        ax.set_title("Label Accuracy Breakdown (Class 0: Blue, Class 1: Others)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("Label")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"L{l}\n(C{0 if l == 0 else 1})" for l in labels])
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="Class 0 (Label 0)"),
            Patch(facecolor="gray", alpha=0.7, label="Class 1 (Labels 1+)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.tight_layout()
        return fig

    def plot_label_breakdown_samples(self, figsize=(14, 8)):
        test_result = self.history.get("test", [{}])[0]
        if "label_classification_breakdown" not in test_result:
            print("No label classification breakdown found")
            return plt.figure(figsize=figsize)

        breakdown = test_result["label_classification_breakdown"]
        labels = []
        accuracies = []
        totals = []
        corrects = []
        for label_name, stats in breakdown.items():
            label_num = int(label_name.split("_")[1])
            labels.append(label_num)
            accuracies.append(stats["accuracy"])
            totals.append(stats["total"])
            corrects.append(stats["correct"])
        sorted_data = sorted(zip(labels, accuracies, totals, corrects))
        labels, accuracies, totals, corrects = zip(*sorted_data)
        fig, ax = plt.subplots(figsize=figsize)
        colors = []
        for label_num in labels:
            if label_num == 0:
                colors.append("blue")
            else:
                color_idx = (label_num - 1) % 10
                colors.append(cm.get_cmap("tab10")(color_idx))
        bars = ax.bar(range(len(labels)), totals, color=colors, alpha=0.7)
        ax.set_ylabel("Total Samples")
        ax.set_title("Sample Count per Label")
        ax.set_xlabel("Label")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"L{l}\n(C{0 if l == 0 else 1})" for l in labels])
        ax.grid(axis="y", alpha=0.3)
        for bar, total in zip(bars, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(totals) * 0.01,
                f"{total}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="Class 0 (Label 0)"),
            Patch(facecolor="gray", alpha=0.7, label="Class 1 (Labels 1+)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.tight_layout()
        return fig

    def calculate_forgetting_metrics(self):
        val_hist = self.history.get("val", [])
        if not val_hist:
            print("No validation history found for forgetting metrics")
            return None
        label_accuracies = {}
        all_labels = set()
        for epoch, ep_data in enumerate(val_hist, 1):
            for key in ep_data.keys():
                if key.startswith("label_") and key.endswith("_accuracy"):
                    label_num = int(key.split("_")[1])
                    all_labels.add(label_num)
                    if label_num not in label_accuracies:
                        label_accuracies[label_num] = {}
                    label_accuracies[label_num][epoch] = ep_data[key]
        if not all_labels:
            print("No label accuracy data found for forgetting metrics")
            return None
        all_labels = sorted(all_labels)
        T = len(all_labels)
        final_epoch = len(val_hist)
        metrics = {}
        final_accuracies = []
        for label in all_labels:
            if final_epoch in label_accuracies.get(label, {}):
                final_accuracies.append(label_accuracies[label][final_epoch])

        if final_accuracies:
            metrics["AA"] = np.mean(final_accuracies)
        else:
            metrics["AA"] = 0.0

        forgetting_values = []
        for i, label in enumerate(all_labels[:-1]):
            if label in label_accuracies:
                label_epochs = sorted(label_accuracies[label].keys())
                if label_epochs:
                    max_acc = max(label_accuracies[label].values())
                    final_acc = label_accuracies[label].get(final_epoch, 0.0)
                    forgetting = max_acc - final_acc
                    forgetting_values.append(max(0, forgetting))
        if forgetting_values:
            metrics["AF"] = np.mean(forgetting_values)
        else:
            metrics["AF"] = 0.0
        bwt_values = []
        for i, label in enumerate(all_labels[:-1]):
            if label in label_accuracies:
                label_epochs = sorted(label_accuracies[label].keys())
                if label_epochs:
                    first_epoch = label_epochs[0]
                    initial_acc = label_accuracies[label][first_epoch]
                    final_acc = label_accuracies[label].get(final_epoch, initial_acc)
                    bwt = final_acc - initial_acc
                    bwt_values.append(bwt)
        if bwt_values:
            metrics["BWT"] = np.mean(bwt_values)
        else:
            metrics["BWT"] = 0.0
        fwt_values = []
        for i, label in enumerate(all_labels[1:], 1):
            if label in label_accuracies:
                label_epochs = sorted(label_accuracies[label].keys())
                if label_epochs:
                    first_acc = label_accuracies[label][label_epochs[0]]
                    random_baseline = 0.5
                    fwt = first_acc - random_baseline
                    fwt_values.append(fwt)
        if fwt_values:
            metrics["FWT"] = np.mean(fwt_values)
        else:
            metrics["FWT"] = 0.0
        return metrics

    def plot_forgetting_metrics(self, figsize=(12, 8)):
        metrics = self.calculate_forgetting_metrics()
        if metrics is None:
            return plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        metric_names = ["AA", "AF", "BWT", "FWT"]
        metric_values = [metrics.get(name, 0.0) for name in metric_names]
        metric_descriptions = [
            "Average Accuracy",
            "Average Forgetting",
            "Backward Transfer",
            "Forward Transfer",
        ]
        colors = ["tab:green", "tab:red", "tab:orange", "tab:purple"]
        bars = ax.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
        ax.set_ylabel("Metric Value")
        ax.set_title("Continual Learning Forgetting Metrics")
        ax.set_xlabel("Metrics")
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(
            [
                f"{name}\n({desc})"
                for name, desc in zip(metric_names, metric_descriptions)
            ]
        )
        ax.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(metric_values) - min(metric_values)) * 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        interpretation = []
        if metrics["AA"] > 0.8:
            interpretation.append("High AA: Good overall performance")
        elif metrics["AA"] > 0.6:
            interpretation.append("Medium AA: Moderate performance")
        else:
            interpretation.append("Low AA: Poor overall performance")
        if metrics["AF"] < 0.1:
            interpretation.append("Low AF: Minimal forgetting")
        elif metrics["AF"] < 0.3:
            interpretation.append("Medium AF: Some forgetting")
        else:
            interpretation.append("High AF: Significant forgetting")
        if metrics["BWT"] > 0:
            interpretation.append("Positive BWT: Later tasks helped earlier ones")
        elif metrics["BWT"] > -0.1:
            interpretation.append("Near-zero BWT: Minimal impact")
        else:
            interpretation.append("Negative BWT: Later tasks hurt earlier ones")
        textstr = "\n".join(interpretation)
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.68,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )
        plt.tight_layout()
        return fig

    def plot_class_confusion_matrix_normalized(self, figsize=(8, 6)):
        test_result = self.history.get("test", [{}])[0]
        if not test_result or "conf_matrix_classes" not in test_result:
            print("No class confusion matrix found")
            return plt.figure(figsize=figsize)
        conf_matrix = np.asarray(test_result["conf_matrix_classes"])
        conf_matrix_norm = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            conf_matrix_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
        )
        ax.set_title("Normalized Confusion Matrix - Classes (0 vs 1)")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        thresh = 0.5
        for i in range(conf_matrix_norm.shape[0]):
            for j in range(conf_matrix_norm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{conf_matrix_norm[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if conf_matrix_norm[i, j] > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Class 0", "Class 1"])
        ax.set_yticklabels(["Class 0", "Class 1"])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion")
        plt.tight_layout()
        return fig

    def plot_all_essential(self):
        """Generate all essential plots for incremental learning analysis (including normalized confusion matrices)"""
        plots = {}

        try:
            plots["label_accuracy"] = self.plot_label_accuracy_per_epoch()
            print("Label accuracy per epoch plotted (without red lines)")
        except Exception as e:
            print(f"Label accuracy plot failed: {e}")

        try:
            plots["global_metrics"] = self.plot_global_metrics_per_epoch()
            print("Global metrics per epoch plotted")
        except Exception as e:
            print(f"Global metrics plot failed: {e}")

        try:
            plots["class_confusion"] = self.plot_class_confusion_matrix()
            print("Class confusion matrix plotted (Classes 0 vs 1)")
        except Exception as e:
            print(f"Class confusion matrix failed: {e}")

        try:
            plots["class_confusion_normalized"] = (
                self.plot_class_confusion_matrix_normalized()
            )
            print("Normalized class confusion matrix plotted (Classes 0 vs 1)")
        except Exception as e:
            print(f"Normalized class confusion matrix failed: {e}")

            try:
                plots["label_breakdown_accuracy"] = self.plot_label_breakdown_accuracy()
                print("Label accuracy breakdown plotted as separate plot")
            except Exception as e2:
                print(f"Label accuracy breakdown fallback also failed: {e2}")

        try:
            plots["label_breakdown_samples"] = self.plot_label_breakdown_samples()
            print("Label sample count breakdown plotted as separate plot")
        except Exception as e:
            print(f"Label sample breakdown plot failed: {e}")

        try:
            plots["forgetting_metrics"] = self.plot_forgetting_metrics()
            print("Forgetting metrics plotted (AA, AF, BWT, FWT)")
        except Exception as e:
            print(f"Forgetting metrics plot failed: {e}")

        return plots


def plot_training(history: Dict[str, Any]):
    plotter = TrainingPlotter(history)
    return plotter.plot_all_essential()

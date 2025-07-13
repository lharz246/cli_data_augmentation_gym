from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import psutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration constants
MAX_FEATURES_CORRELATION = 100
MAX_FEATURES_IMPORTANCE = 200
MAX_SAMPLES_RF = 20000
CHUNK_SIZE = 10000
N_JOBS = min(psutil.cpu_count(), 8)


def init_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@jit(nopython=True)
def _basic_stats(X: np.ndarray) -> Tuple[np.ndarray, ...]:
    n_samples, n_features = X.shape
    means = np.zeros(n_features)
    stds = np.zeros(n_features)
    mins = np.zeros(n_features)
    maxs = np.zeros(n_features)
    zero_counts = np.zeros(n_features)
    for i in prange(n_features):
        col = X[:, i]
        means[i] = np.mean(col)
        stds[i] = np.std(col)
        mins[i] = np.min(col)
        maxs[i] = np.max(col)
        zero_counts[i] = np.sum(col == 0)
    return means, stds, mins, maxs, zero_counts


def _calculate_basic_stats_optimized(X: np.ndarray) -> Dict[str, Any]:
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    means, stds, mins, maxs, zero_counts = _basic_stats(X_clean)
    n_samples = X_clean.shape[0]
    if n_samples > CHUNK_SIZE:
        indices = np.random.choice(n_samples, min(CHUNK_SIZE, n_samples), replace=False)
        X_sample = X_clean[indices]
    else:
        X_sample = X_clean
    medians = np.median(X_sample, axis=0)
    q1 = np.percentile(X_sample, 25, axis=0)
    q3 = np.percentile(X_sample, 75, axis=0)
    non_zero_var = stds > 1e-10
    skewness = np.zeros(X_clean.shape[1])
    kurt = np.zeros(X_clean.shape[1])
    if np.any(non_zero_var):
        skewness[non_zero_var] = skew(X_clean[:, non_zero_var], axis=0)
        kurt[non_zero_var] = kurtosis(X_clean[:, non_zero_var], axis=0)
    return {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "median": medians,
        "q1": q1,
        "q3": q3,
        "skewness": np.nan_to_num(skewness),
        "kurtosis": np.nan_to_num(kurt),
        "zero_percentage": zero_counts / X_clean.shape[0],
    }


def calculate_class_statistics(
    datasets: Dict[str, Dict[str, np.ndarray]], stats: Dict[str, Any]
) -> None:

    def process_labels(labels: np.ndarray, name: str) -> Dict[str, Any]:
        if labels.size == 0:
            return {}
        labels_int = labels.astype(int)
        counts = Counter(labels_int)
        total = len(labels_int)
        return {
            f"{name}_class_distribution": {
                str(k): {"count": int(v), "percentage": round(v / total * 100, 2)}
                for k, v in counts.items()
            }
        }

    for split, data in datasets.items():
        y = data["y"]
        class_stats = {"sample_count": len(y)}
        if y.ndim > 1:
            main_labels = y[:, 0]
            class_stats.update(process_labels(main_labels, "main"))
            if y.shape[1] > 1:
                sub_labels = y[:, 1]
                class_stats.update(process_labels(sub_labels, "sub"))
        else:
            class_stats.update(process_labels(y, "main"))
        stats["class_statistics"][split] = class_stats


def calculate_feature_statistics(
    X: np.ndarray, y: np.ndarray, stats: Dict[str, Any]
) -> None:
    global_stats = _calculate_basic_stats_optimized(X)
    stats["feature_statistics"] = {
        "global": {k: v.tolist() for k, v in global_stats.items()}
    }
    main_labels = y[:, 0] if y.ndim > 1 else y
    unique_labels = np.unique(main_labels)
    if len(unique_labels) > 1:
        label_counts = Counter(main_labels)
        unique_labels = [label for label, _ in label_counts.most_common(10)]
    per_class_stats = {}
    for label in unique_labels:
        mask = main_labels == label
        if not mask.any():
            continue
        X_subset = X[mask]
        basic_stats = _calculate_basic_stats_optimized(X_subset)
        per_class_stats[int(label)] = {
            "sample_count": int(mask.sum()),
            "mean": basic_stats["mean"].tolist(),
            "std": basic_stats["std"].tolist(),
            "min": basic_stats["min"].tolist(),
            "max": basic_stats["max"].tolist(),
        }
    stats["feature_statistics"]["per_class"] = per_class_stats


def _parallel_feature_importance(
    X: np.ndarray, y: np.ndarray, method: str
) -> Dict[str, Any]:

    try:
        if method == "mutual_info":
            if X.shape[0] > MAX_SAMPLES_RF:
                indices = np.random.choice(X.shape[0], MAX_SAMPLES_RF, replace=False)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
            scores = mutual_info_classif(
                X_sample,
                y_sample,
                random_state=42,
            )
        elif method == "f_score":
            scores, p_values = f_classif(X, y)
            scores = np.nan_to_num(scores)
        elif method == "random_forest":
            if X.shape[0] > MAX_SAMPLES_RF:
                indices = np.random.choice(X.shape[0], MAX_SAMPLES_RF, replace=False)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=N_JOBS,
                max_depth=15,
                max_features="sqrt",
            )
            rf.fit(X_sample, y_sample)
            scores = rf.feature_importances_
        else:
            return {}
        top_indices = scores.argsort()[-20:][::-1]
        result = {
            "top_features": [
                {"feature_index": int(idx), "score": float(scores[idx])}
                for idx in top_indices
            ],
            "mean_score": float(scores.mean()),
        }
        if method == "f_score":
            for i, feature in enumerate(result["top_features"]):
                feature["p_value"] = float(p_values[feature["feature_index"]])
        return result
    except Exception as e:
        logging.warning(f"Method {method} failed: {e}")
        return {}


def _create_importance_plot(
    methods_results: Dict, save_dir: Path, config_name: str, plots: str
) -> None:
    ensure_dir(save_dir)
    method_priority = ["mutual_info", "f_score", "random_forest"]
    selected_method = None
    for method in method_priority:
        if method in methods_results:
            selected_method = method
            break
    if not selected_method:
        return
    results = methods_results[selected_method]
    top_features = results["top_features"][:15]
    plt.figure(figsize=(10, 6))
    indices = [f["feature_index"] for f in top_features]
    scores = [f["score"] for f in top_features]
    plt.barh(range(len(indices)), scores, color="skyblue", alpha=0.7)
    plt.yticks(range(len(indices)), [f"F{idx}" for idx in indices])
    plt.xlabel("Importance Score")
    plt.title(
        f"Top Features - {config_name} ({selected_method.replace('_', ' ').title()})"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        save_dir / f"feature_importance_{config_name}.png",
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def calculate_correlation_analysis(
    X: np.ndarray, save_dir: Path, stats: Dict[str, Any], plots: str
) -> None:
    max_features = min(MAX_FEATURES_CORRELATION, X.shape[1])
    stds = np.std(X, axis=0)
    top_indices = stds.argsort()[-max_features:]
    X_subset = X[:, top_indices]
    X_subset = np.nan_to_num(X_subset, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(X_subset.shape[1]):
        if np.std(X_subset[:, i]) < 1e-10:
            X_subset[:, i] += np.random.normal(0, 1e-8, X_subset.shape[0])
    corr_matrix = np.corrcoef(X_subset.T)
    high_corr_threshold = 0.8
    high_corr_pairs = []
    rows, cols = np.triu_indices_from(corr_matrix, k=1)
    high_corr_mask = np.abs(corr_matrix[rows, cols]) > high_corr_threshold
    high_corr_indices = np.where(high_corr_mask)[0]
    for idx in high_corr_indices:
        i, j = rows[idx], cols[idx]
        high_corr_pairs.append(
            {
                "feature1": int(top_indices[i]),
                "feature2": int(top_indices[j]),
                "correlation": float(corr_matrix[i, j]),
            }
        )
    stats["correlation_analysis"] = {
        "high_correlation_pairs": high_corr_pairs,
        "matrix_shape": corr_matrix.shape,
        "analyzed_features": max_features,
        "total_features": X.shape[1],
    }
    if plots == "full" and max_features <= 50:
        _create_correlation_plot(corr_matrix, save_dir)


def _create_correlation_plot(corr_matrix: np.ndarray, save_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()


def calculate_anomaly_metrics(
    X: np.ndarray, y: np.ndarray, stats: Dict[str, Any]
) -> None:
    if y.ndim == 1:
        return
    main_labels = y[:, 0]
    unique_labels = np.unique(main_labels)
    if len(unique_labels) < 2:
        return
    normal_mask = main_labels == 0
    attack_mask = main_labels == 1
    if not normal_mask.any() or not attack_mask.any():
        return
    if X.shape[0] > CHUNK_SIZE:
        normal_indices = np.where(normal_mask)[0]
        attack_indices = np.where(attack_mask)[0]

        normal_sample = np.random.choice(
            normal_indices, min(len(normal_indices), CHUNK_SIZE // 2), replace=False
        )
        attack_sample = np.random.choice(
            attack_indices, min(len(attack_indices), CHUNK_SIZE // 2), replace=False
        )
        X_normal = X[normal_sample]
        X_attack = X[attack_sample]
    else:
        X_normal = X[normal_mask]
        X_attack = X[attack_mask]
    normal_mean = np.mean(X_normal, axis=0)
    attack_mean = np.mean(X_attack, axis=0)
    mean_diff = np.abs(normal_mean - attack_mean)
    top_indices = mean_diff.argsort()[-10:][::-1]
    stats["anomaly_metrics"] = {
        "normal_samples": int(normal_mask.sum()),
        "attack_samples": int(attack_mask.sum()),
        "top_differentiating_features": [
            {
                "feature_index": int(idx),
                "mean_difference": float(mean_diff[idx]),
                "normal_mean": float(normal_mean[idx]),
                "attack_mean": float(attack_mean[idx]),
            }
            for idx in top_indices
        ],
    }


def load_datasets(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    datasets = {}
    for split in ("train", "val", "test"):
        file_path = data_dir / f"{split}.pt"
        if file_path.exists():
            try:
                X, y = torch.load(file_path, map_location="cpu")
                datasets[split] = {
                    "X": X.numpy() if isinstance(X, torch.Tensor) else X,
                    "y": y.numpy() if isinstance(y, torch.Tensor) else y,
                }
                logging.info(f"Loaded {split.upper()}: {X.shape}")
            except Exception as e:
                logging.warning(f"Failed to load {split}.pt: {e}")
        else:
            logging.warning(f"Missing {split}.pt")
    return datasets


def create_summary_visualizations(stats: Dict[str, Any], save_dir: Path) -> None:
    ensure_dir(save_dir)
    _create_class_distribution_plot(stats, save_dir)
    _create_feature_summary_plot(stats, save_dir)


def _create_class_distribution_plot(stats: Dict[str, Any], save_dir: Path) -> None:
    class_stats = stats.get("class_statistics", {})
    if not class_stats:
        return
    splits = list(class_stats.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        split_data = class_stats[split]
        dist = split_data.get("main_class_distribution", {})
        if not dist:
            continue
        labels = list(dist.keys())
        sizes = [d["percentage"] for d in dist.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax.pie(
            sizes,
            labels=[f"Class {l}" for l in labels],
            colors=colors,
            autopct="%1.1f%%",
        )
        ax.set_title(f"{split.capitalize()} Set")
    plt.suptitle("Class Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()


def _create_feature_summary_plot(stats: Dict[str, Any], save_dir: Path) -> None:
    importance_data = stats.get("feature_importance", {})
    if not importance_data:
        return
    all_features = {}
    for config_name, config_data in importance_data.items():
        methods = config_data.get("methods", {})
        for method_name, method_data in methods.items():
            features = method_data.get("top_features", [])[:10]
            for feature in features:
                feat_idx = feature["feature_index"]
                score = feature["score"]

                if feat_idx not in all_features:
                    all_features[feat_idx] = []
                all_features[feat_idx].append(score)

    if not all_features:
        return
    avg_scores = {idx: np.mean(scores) for idx, scores in all_features.items()}
    top_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    plt.figure(figsize=(10, 6))
    indices, scores = zip(*top_features)
    plt.barh(range(len(indices)), scores, color="lightcoral", alpha=0.7)
    plt.yticks(range(len(indices)), [f"Feature {idx}" for idx in indices])
    plt.xlabel("Average Importance Score")
    plt.title("Top Important Features (Averaged Across All Methods)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        save_dir / "feature_importance_summary.png", dpi=100, bbox_inches="tight"
    )
    plt.close()


def _calculate_per_sub_class_statistics(
    X: np.ndarray, y: np.ndarray, stats: Dict[str, Any]
) -> None:
    if y.ndim == 1:
        return
    if y.shape[1] < 2:
        return
    sub_labels = y[:, 1]
    unique_sub_labels = np.unique(sub_labels)
    if len(unique_sub_labels) > 20:
        label_counts = Counter(sub_labels)
        unique_sub_labels = [label for label, _ in label_counts.most_common(20)]
    per_sub_class = {}
    for label in unique_sub_labels:
        mask = sub_labels == label
        if not mask.any() or mask.sum() < 10:
            continue

        X_subset = X[mask]
        X_clean = np.nan_to_num(X_subset, nan=0.0, posinf=0.0, neginf=0.0)
        per_sub_class[int(label)] = {
            "sample_count": int(mask.sum()),
            "mean": X_clean.mean(axis=0).tolist(),
            "std": X_clean.std(axis=0).tolist(),
            "min": X_clean.min(axis=0).tolist(),
            "max": X_clean.max(axis=0).tolist(),
        }
    if "feature_statistics" not in stats:
        stats["feature_statistics"] = {}
    stats["feature_statistics"]["per_sub_class"] = per_sub_class


def _get_label_configurations(y: np.ndarray) -> List[Dict[str, Any]]:
    configs = []
    if y.ndim > 1:
        main_labels = y[:, 0]
        if len(np.unique(main_labels)) >= 2:
            configs.append(
                {
                    "name": "binary_classification",
                    "labels": main_labels,
                    "description": "Normal vs Attack",
                }
            )

        if y.shape[1] > 1:
            sub_labels = y[:, 1]
            unique_sub = np.unique(sub_labels)
            if 2 <= len(unique_sub) <= 20:
                configs.append(
                    {
                        "name": "multiclass_classification",
                        "labels": sub_labels,
                        "description": f"Attack Types ({len(unique_sub)} classes)",
                    }
                )
            label_counts = Counter(sub_labels)
            for cls, count in label_counts.items():
                if count >= 1:
                    binary_target = (sub_labels == cls).astype(int)
                    configs.append(
                        {
                            "name": f"ovr_{int(cls)}",
                            "labels": binary_target,
                            "description": f"Class {int(cls)} vs Rest",
                        }
                    )
    else:
        unique_labels = np.unique(y)
        if len(unique_labels) >= 2:
            configs.append(
                {
                    "name": "classification",
                    "labels": y,
                    "description": f"Classification ({len(unique_labels)} classes)",
                }
            )
    return configs


def calculate_feature_importance(
    X: np.ndarray, y: np.ndarray, save_dir: Path, stats: Dict[str, Any], plots: str
) -> None:

    max_features = min(MAX_FEATURES_IMPORTANCE, X.shape[1])
    X_subset = X[:, :max_features]
    X_subset = np.nan_to_num(X_subset, nan=0.0, posinf=0.0, neginf=0.0)
    stds = X_subset.std(axis=0)
    zero_var_mask = stds < 1e-10
    if zero_var_mask.any():
        X_subset = X_subset.copy()
        X_subset[:, zero_var_mask] += np.random.normal(
            0, 1e-8, size=(X_subset.shape[0], zero_var_mask.sum())
        )
    label_configs = _get_label_configurations(y)
    importance_results = {}
    for config in label_configs:
        name = config["name"]
        target = config["labels"]
        unique_labels, counts = np.unique(target, return_counts=True)
        if len(unique_labels) < 2:
            continue
        logging.info(f"Processing {name}...")
        methods = ["mutual_info", "f_score"]
        if X_subset.shape[0] <= MAX_SAMPLES_RF and len(unique_labels) <= 10:
            methods.append("random_forest")
        methods_results = {}
        with ThreadPoolExecutor(max_workers=min(len(methods), N_JOBS)) as executor:
            future_to_method = {
                executor.submit(
                    _parallel_feature_importance, X_subset, target, method
                ): method
                for method in methods
            }
            for future in as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    result = future.result()
                    if result:
                        methods_results[method] = result
                except Exception as e:
                    logging.warning(f"Method {method} failed: {e}")
        if methods_results:
            importance_results[name] = {
                "description": config["description"],
                "methods": methods_results,
            }
            if plots != "none":
                _create_importance_plot(methods_results, save_dir, name, plots)
    stats["feature_importance"] = importance_results


def calculate_network_data_statistics(
    data_dir: Path,
    dataset_type: str,
    save_path: Optional[Path] = None,
    plots: str = "minimal",
) -> Dict[str, Any]:
    save_base = save_path or data_dir / "statistics"
    ensure_dir(save_base)
    meta_file = data_dir / "metadata.json"
    metadata = {}
    if meta_file.exists():
        try:
            metadata = json.loads(meta_file.read_text())
        except Exception as e:
            logging.warning(f"Failed to load metadata: {e}")
    datasets = load_datasets(data_dir)
    if not datasets:
        raise FileNotFoundError(f"No dataset splits found in {data_dir}")
    first_X = next(iter(datasets.values()))["X"]
    stats = {
        "dataset_info": {
            "type": dataset_type,
            "total_samples": sum(v["X"].shape[0] for v in datasets.values()),
            "feature_count": first_X.shape[1],
            "splits": list(datasets.keys()),
            "memory_usage_mb": sum(
                v["X"].nbytes + v["y"].nbytes for v in datasets.values()
            )
            / (1024 * 1024),
        },
        "class_statistics": {},
        "feature_statistics": {},
        "correlation_analysis": {},
        "feature_importance": {},
        "anomaly_metrics": {},
    }
    if metadata:
        stats["metadata"] = metadata
    logging.info("Calculating class statistics...")
    calculate_class_statistics(datasets, stats)
    if "train" in datasets:
        X_train, y_train = datasets["train"]["X"], datasets["train"]["y"]
        X_val, y_val = datasets["val"]["X"], datasets["val"]["y"]
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
        logging.info("Calculating feature statistics...")
        calculate_feature_statistics(X_train, y_train, stats)
        logging.info("Calculating per sub-class statistics...")
        _calculate_per_sub_class_statistics(X_train, y_train, stats)
        logging.info("Calculating anomaly metrics...")
        calculate_anomaly_metrics(X_train, y_train, stats)
        if plots != "none":
            logging.info("Calculating correlations...")
            calculate_correlation_analysis(X_train, save_base, stats, plots)
            logging.info("Calculating feature importance...")
            calculate_feature_importance(X_train, y_train, save_base, stats, plots)
    output_file = save_base / "dataset_statistics.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logging.info(f"Statistics saved to {output_file}")
    if plots != "none":
        logging.info("Creating visualizations...")
        create_summary_visualizations(stats, save_base)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized dataset statistics calculation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset directory containing train.pt, val.pt, test.pt",
    )
    parser.add_argument(
        "--dataset_type", type=str, default="network", help="Dataset type identifier"
    )
    parser.add_argument(
        "--plots",
        choices=["none", "minimal", "full"],
        default="minimal",
        help="Plot generation level",
    )
    parser.add_argument("--save_path", type=Path, help="Custom save path for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_logging(args.verbose)

    try:
        stats = calculate_network_data_statistics(
            data_dir=args.data_dir,
            dataset_type=args.dataset_type,
            save_path=args.save_path,
            plots=args.plots,
        )

        dataset_info = stats["dataset_info"]
        logging.info("=" * 50)
        logging.info("DATASET STATISTICS SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Total samples: {dataset_info['total_samples']:,}")
        logging.info(f"Features: {dataset_info['feature_count']:,}")
        logging.info(f"Memory usage: {dataset_info['memory_usage_mb']:.1f} MB")
        logging.info(f"Splits: {', '.join(dataset_info['splits'])}")
        if "class_statistics" in stats:
            for split, class_data in stats["class_statistics"].items():
                if "main_class_distribution" in class_data:
                    dist = class_data["main_class_distribution"]
                    logging.info(
                        f"{split.capitalize()} classes: {dict(sorted(dist.items()))}"
                    )
        if (
            "feature_statistics" in stats
            and "per_sub_class" in stats["feature_statistics"]
        ):
            sub_classes = stats["feature_statistics"]["per_sub_class"]
            logging.info(f"Sub-classes analyzed: {len(sub_classes)}")
            for cls, data in list(sub_classes.items())[:5]:  # Show first 5
                logging.info(f"  Class {cls}: {data['sample_count']} samples")
        if "feature_importance" in stats:
            ovr_configs = [
                k for k in stats["feature_importance"].keys() if k.startswith("ovr_")
            ]
            if ovr_configs:
                logging.info(
                    f"OvR feature importance calculated for {len(ovr_configs)} classes"
                )
        logging.info("Enhanced statistics calculation completed successfully")
    except KeyboardInterrupt:
        logging.info("x Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"x Statistics calculation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

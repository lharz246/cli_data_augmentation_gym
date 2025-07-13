from __future__ import annotations

import argparse
import gc
import ipaddress
import json
import logging
import os
import sys
import joblib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def init_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ip_to_int_vectorised(ip_series: pd.Series) -> pd.Series:
    def safe_ip_convert(ip):
        if pd.isna(ip) or ip == "" or ip is None:
            return np.nan
        try:
            return int(ipaddress.ip_address(str(ip).strip()))
        except (ValueError, ipaddress.AddressValueError):
            return np.nan

    return ip_series.apply(safe_ip_convert).astype("float32")


def remove_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    irrelevant_cols = [
        "Flow ID",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Protocol",
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "protocol",
        "Flow_ID",
        "Src_IP",
        "Dst_IP",
        "Src_Port",
        "Dst_Port",
    ]
    cols_to_remove = [col for col in irrelevant_cols if col in df.columns]
    if cols_to_remove:
        logging.info(f"Removing irrelevant features: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    if initial_shape[0] != final_shape[0]:
        logging.info(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if strategy == "median":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = {"label", "label_numeric", "class"}
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logging.debug(f"Filled {col} NaN values with median: {median_val}")
    elif strategy == "zero":
        df.fillna(0.0, inplace=True)
    elif strategy == "drop":
        initial_shape = df.shape
        df = df.dropna()
        final_shape = df.shape
        if initial_shape[0] != final_shape[0]:
            logging.info(
                f"Dropped {initial_shape[0] - final_shape[0]} rows with missing values"
            )

    return df


def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = {"label", "label_numeric", "class"}
    feature_cols = [col for col in numerical_cols if col not in exclude_cols]
    constant_features = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        logging.info(
            f"Removing {len(constant_features)} constant features: {constant_features}"
        )
        df = df.drop(columns=constant_features)

    return df


def create_binary_class_column(df: pd.DataFrame) -> pd.DataFrame:
    normal_variants = ["normal", "Normal", "NORMAL", "benign", "Benign", "BENIGN"]

    def is_normal(label):
        return str(label) in normal_variants

    df["class"] = df["label"].apply(lambda x: 0 if is_normal(x) else 1)
    logging.info(
        f"Created binary class column - Normal: {(df['class'] == 0).sum()}, Malicious: {(df['class'] == 1).sum()}"
    )
    return df


def standardize_features(
    df: pd.DataFrame, method: str = "standard"
) -> Tuple[pd.DataFrame, Any]:
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = {"label", "label_numeric", "class"}
    feature_cols = [col for col in numerical_cols if col not in exclude_cols]
    if not feature_cols:
        logging.warning("No numerical features found for standardization")
        return df, None

    if method == "standard":
        scaler = StandardScaler()
        logging.info("Applying StandardScaler (mean=0, std=1)")
    elif method == "minmax":
        scaler = MinMaxScaler()
        logging.info("Applying MinMaxScaler (range 0-1)")
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def select_features(df: pd.DataFrame, k: int = "all") -> Tuple[pd.DataFrame, Any]:
    if k == "all":
        return df, None
    df = remove_constant_features(df)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = {"label", "label_numeric", "class"}
    feature_cols = [col for col in numerical_cols if col not in exclude_cols]
    if len(feature_cols) <= k:
        logging.info(
            f"Number of features ({len(feature_cols)}) <= k ({k}), keeping all"
        )
        return df, None
    X = df[feature_cols]
    y = df["label_numeric"]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    df_selected = df[["label", "label_numeric", "class"] + selected_features].copy()

    logging.info(
        f"Selected top {k} features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}"
    )
    return df_selected, selector


def detect_label_column(file_path: str) -> str:
    try:
        df_sample = pd.read_csv(file_path, nrows=0)
        df_sample.columns = df_sample.columns.str.strip()
        columns = df_sample.columns.tolist()
        candidates = [
            "type",
            "Type",
            "TYPE",
            "label",
            "Label",
            "LABEL",
        ]
        for candidate in candidates:
            if candidate in columns:
                return candidate
        label_col = columns[-1]
        logging.warning(f"No standard label column found, using '{label_col}'")
        return label_col
    except Exception as e:
        logging.error(f"Failed to detect label column for {file_path}: {e}")
        raise


def process_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=["number"]).columns
    for col in numerical_cols:
        if col not in ["label", "label_numeric", "class"]:
            df[col] = np.where(df[col] < 0, 0.0, df[col])
    return df


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        if col not in ["label", "label_numeric", "class"]:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            else:
                df[col] = df[col].astype("int32")

    return df


def create_label_mapping(df: pd.DataFrame, dataset_type: str) -> Dict[str, int]:
    unique_labels = sorted([str(x) for x in df["label"].unique() if pd.notna(x)])
    mapping = {}
    normal_variants = ["normal", "Normal", "NORMAL", "benign", "Benign", "BENIGN"]
    normal_label = None
    for variant in normal_variants:
        if variant in unique_labels:
            normal_label = variant
            break
    if normal_label:
        mapping[normal_label] = 0
        unique_labels = [x for x in unique_labels if x != normal_label]
    for i, lbl in enumerate(unique_labels, start=len(mapping)):
        mapping[lbl] = i
    logging.info(f"{dataset_type} label mapping: {mapping}")
    return mapping


def preprocess_ton(
    src_dir: Path,
    target_rows: int = 1_000_000,
    workers: int = 4,
    missing_strategy: str = "median",
    scaling_method: str = "standard",
    feature_selection_k: int = "all",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logging.info("Preprocessing TON dataset...")
    csv_files = list(src_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {src_dir}")

    logging.info(f"Found {len(csv_files)} CSV files")
    label_column = detect_label_column(csv_files[0].as_posix())
    logging.info(f"TON label column: '{label_column}'")

    bool_cols = [
        "dns_AA",
        "dns_RD",
        "dns_RA",
        "dns_rejected",
        "ssl_resumed",
        "ssl_established",
    ]
    categorical_cols = [
        "proto",
        "service",
        "conn_state",
        "http_method",
        "http_version",
        "dns_qclass",
        "dns_qtype",
        "dns_rcode",
        "ssl_version",
        "ssl_cipher",
        "http_status_code",
        "weird_name",
    ]
    label_counts = Counter()
    for file in csv_files:
        try:
            for chunk in pd.read_csv(file, chunksize=50000, low_memory=False):
                if label_column in chunk.columns:
                    chunk_labels = chunk[label_column].value_counts().to_dict()
                    for label, count in chunk_labels.items():
                        if pd.notna(label):
                            label_counts[str(label)] += count
        except Exception as e:
            logging.error(f"Error counting labels in {file}: {e}")
            continue
    if not label_counts:
        raise ValueError("No valid labels found in TON dataset")
    logging.info(f"Label distribution: {dict(label_counts.most_common(10))}")
    total = sum(label_counts.values())
    keep_ratio = min(1.0, target_rows / total)
    keep_probs = {lbl: keep_ratio for lbl in label_counts.keys()}

    def process_file_ton(file_path: str) -> Optional[pd.DataFrame]:
        try:
            file_dfs = []
            chunk_size = 100000
            for chunk in pd.read_csv(
                file_path, chunksize=chunk_size, low_memory=False, skipinitialspace=True
            ):
                if label_column not in chunk.columns:
                    continue

                if label_column != "label":
                    chunk = chunk.rename(columns={"label": "class"})
                    chunk = chunk.rename(columns={label_column: "label"})
                chunk = remove_irrelevant_features(chunk)
                chunk = process_numerical_columns(chunk)
                for col in bool_cols:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype(bool).astype("int8")
                for col in categorical_cols:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype("category")
                        le = LabelEncoder()
                        chunk[col] = le.fit_transform(chunk[col])
                for ip_col in ["src_ip", "dst_ip"]:
                    if ip_col in chunk.columns:
                        chunk[ip_col] = ip_to_int_vectorised(chunk[ip_col])
                if "label" in chunk.columns:
                    grouped = chunk.groupby("label")
                    sampled_chunk = pd.DataFrame()
                    for label, group in grouped:
                        label_str = str(label)
                        if label_str in keep_probs:
                            sample_size = max(
                                1, int(len(group) * keep_probs[label_str])
                            )
                            if len(group) > sample_size:
                                sampled_group = group.sample(
                                    sample_size, random_state=42
                                )
                            else:
                                sampled_group = group
                            sampled_chunk = pd.concat(
                                [sampled_chunk, sampled_group], ignore_index=True
                            )
                    if not sampled_chunk.empty:
                        file_dfs.append(sampled_chunk)
            return pd.concat(file_dfs, ignore_index=True) if file_dfs else None
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None

    sampled_dfs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(process_file_ton, f.as_posix()): f for f in csv_files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    sampled_dfs.append(result)
                    logging.debug(f"Processed {file_path.name}: {len(result)} rows")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

    if not sampled_dfs:
        raise ValueError("No valid TON data could be processed")

    logging.info("Combining and cleaning TON DataFrames...")
    df = pd.concat(sampled_dfs, axis=0, ignore_index=True)
    del sampled_dfs
    gc.collect()
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = create_binary_class_column(df)
    df = downcast_dtypes(df)
    mapping = create_label_mapping(df, "TON")
    df["label_numeric"] = (
        df["label"].astype(str).map(mapping).fillna(-1).astype("int16")
    )
    df = df[df["label_numeric"] >= 0].copy()

    scaler = None
    if scaling_method in ["standard", "minmax"]:
        df, scaler = standardize_features(df, method=scaling_method)

    selector = None
    if feature_selection_k != "all":
        df, selector = select_features(df, k=feature_selection_k)
    logging.info(f"TON final shape: {df.shape}")
    logging.info(
        f"Final label distribution: {df['label_numeric'].value_counts().to_dict()}"
    )
    metadata = {
        "label_mapping": mapping,
        "class_distribution": dict(label_counts),
        "label_column": "label",
        "total_samples": len(df),
        "num_features": len(df.columns)
        - 3,  # Exclude 'label', 'label_numeric', 'class'
        "preprocessing": {
            "missing_strategy": missing_strategy,
            "scaling_method": scaling_method,
            "feature_selection_k": feature_selection_k,
            "scaler": scaler,
            "selector": selector,
        },
    }
    return df, metadata


def preprocess_cic(
    src_dir: Path,
    combine_labels: bool,
    workers: int = 4,
    missing_strategy: str = "median",
    scaling_method: str = "standard",
    feature_selection_k: int = "all",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logging.info("Preprocessing CIC dataset...")
    csv_files = list(src_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {src_dir}")
    label_column = detect_label_column(csv_files[0].as_posix())
    logging.info(f"CIC label column: '{label_column}'")
    label_group_mapping = {
        0: 0,  # BENIGN
        1: 4,  # Bot -> Rare Attacks
        2: 1,  # DDoS -> DoS/DDoS
        3: 1,  # DoS GoldenEye -> DoS/DDoS
        4: 1,  # DoS Hulk -> DoS/DDoS
        5: 1,  # DoS Slowhttptest -> DoS/DDoS
        6: 1,  # DoS slowloris -> DoS/DDoS
        7: 4,  # FTP-Patator -> Rare Attacks
        8: 4,  # Heartbleed -> Rare Attacks
        9: 4,  # Infiltration -> Rare Attacks
        10: 2,  # PortScan
        11: 4,  # SSH-Patator -> Rare Attacks
        12: 3,  # Web Attack Brute Force -> Web Attacks
        13: 3,  # Web Attack Sql Injection -> Web Attacks
        14: 3,  # Web Attack XSS -> Web Attacks
    }

    label_group_names = {
        0: "BENIGN",
        1: "DoS/DDoS",
        2: "PortScan",
        3: "Web Attacks",
        4: "Rare Attacks",
    }

    def process_file_cic(file_path: str) -> Optional[pd.DataFrame]:
        try:
            logging.debug(f"Processing {Path(file_path).name}")
            df = pd.read_csv(file_path, skipinitialspace=True, low_memory=False)

            if label_column not in df.columns:
                logging.warning(
                    f"Label column '{label_column}' not found in {file_path}"
                )
                return None

            if label_column != "label":
                df = df.rename(columns={label_column: "label"})

            df = remove_irrelevant_features(df)
            df = process_numerical_columns(df)

            return df
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None

    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(process_file_cic, f.as_posix()): f for f in csv_files
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    dfs.append(result)
                    logging.debug(f"Processed {file_path.name}: {len(result)} rows")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

    if not dfs:
        raise ValueError("No valid CIC data could be processed")

    logging.info("Combining and cleaning CIC DataFrames...")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs
    gc.collect()
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = create_binary_class_column(df)
    df = downcast_dtypes(df)
    mapping = create_label_mapping(df, "CIC")
    df["label_numeric"] = (
        df["label"].astype(str).map(mapping).fillna(-1).astype("int16")
    )
    df = df[df["label_numeric"] >= 0].copy()
    if combine_labels:
        df["label_numeric"] = (
            df["label_numeric"].map(label_group_mapping).astype("int16")
        )
        mapping = {v: k for k, v in label_group_names.items()}
    scaler = None
    if scaling_method in ["standard", "minmax"]:
        df, scaler = standardize_features(df, method=scaling_method)
    selector = None
    if feature_selection_k != "all":
        df, selector = select_features(df, k=feature_selection_k)
    logging.info(f"CIC final shape: {df.shape}")
    logging.info(
        f"Final label distribution: {df['label_numeric'].value_counts().to_dict()}"
    )
    metadata = {
        "label_mapping": mapping,
        "label_column": "label",
        "total_samples": len(df),
        "num_features": len(df.columns)
        - 3,  # Exclude 'label', 'class', 'label_numeric'
        "preprocessing": {
            "missing_strategy": missing_strategy,
            "scaling_method": scaling_method,
            "feature_selection_k": feature_selection_k,
            "scaler": scaler,
            "selector": selector,
        },
    }
    return df, metadata


def dataframe_to_tensors(
    df: pd.DataFrame,
    label_col: str = "label_numeric",
    class_col: str = "class",
    root: Path = Path("data/"),
    prefix: str = "dataset",
    test_size: float = 0.2,
    val_size: float = 0.25,
    save_scalers: bool = True,
) -> Dict[str, Any]:
    ensure_dir(root / prefix)
    y = torch.tensor(df[[class_col, label_col]].values, dtype=torch.long)
    exclude_cols = {"label", "label_numeric", "class"}
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    logging.info(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    logging.info(
        f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}"
    )
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=df[label_col], random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=df.iloc[train_val_idx][label_col],
        random_state=42,
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Simple per-feature normalization."""
        if tensor.numel() == 0:
            return tensor

        tensor_min = torch.min(tensor, dim=0, keepdim=True)[0]
        tensor_max = torch.max(tensor, dim=0, keepdim=True)[0]
        range_mask = (tensor_max - tensor_min) > 1e-6
        normalized = torch.where(
            range_mask,
            (tensor - tensor_min) / (tensor_max - tensor_min + 1e-6),
            torch.zeros_like(tensor),
        )
        return normalized

    meta = {
        "input_shapes": {},
        "feature_columns": feature_cols,
        "num_classes": len(torch.unique(y[:, 1])),
    }

    for name, idx in splits.items():
        X_split = normalize_tensor(X[idx])
        y_split = y[idx]
        torch.save((X_split, y_split), root / prefix / f"{name}.pt", pickle_protocol=4)
        meta["input_shapes"][name] = {
            "input": tuple(X_split.shape),
            "labels": tuple(y_split.shape),
        }
        logging.info(f"Saved {name}: input {X_split.shape}, labels {y_split.shape}")

    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Network Dataset Preprocessor",
        description="Comprehensive preprocessing for TON/CIC network datasets",
    )
    p.add_argument("--ton_dir", type=Path, help="Path to TON dataset directory")
    p.add_argument("--cic_dir", type=Path, help="Path to CIC dataset directory")
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count()),
        help="Number of worker threads",
    )
    p.add_argument(
        "--target_rows",
        type=int,
        default=1_000_000,
        help="Target number of rows for TON sampling",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    p.add_argument(
        "--output_dir", type=Path, default=Path("data"), help="Output directory"
    )
    p.add_argument(
        "--missing_strategy",
        choices=["median", "zero", "drop"],
        default="median",
        help="Strategy for handling missing values",
    )
    p.add_argument(
        "--scaling_method",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Feature scaling method",
    )
    p.add_argument(
        "--feature_selection_k",
        type=int,
        default="all",
        help="Number of top features to select (use 'all' for no selection)",
    )
    p.add_argument(
        "--combine_labels",
        type=int,
        default=0,
        help="Combine type based labels",
    )
    return p.parse_args()


def main():
    args = parse_args()
    init_logging(args.verbose)

    logging.info("Starting comprehensive network dataset preprocessing pipeline")
    logging.info(f"Workers: {args.workers}, Target rows: {args.target_rows:,}")
    logging.info(f"Missing strategy: {args.missing_strategy}")
    logging.info(f"Scaling method: {args.scaling_method}")
    logging.info(f"Feature selection k: {args.feature_selection_k}")
    logging.info(f"Output directory: {args.output_dir}")
    ensure_dir(args.output_dir)
    try:
        if args.ton_dir:
            if not args.ton_dir.exists():
                logging.error(f"TON directory does not exist: {args.ton_dir}")
            else:
                logging.info(f"Processing TON dataset from: {args.ton_dir}")
                ton_df, ton_meta = preprocess_ton(
                    args.ton_dir,
                    target_rows=args.target_rows,
                    workers=args.workers,
                    missing_strategy=args.missing_strategy,
                    scaling_method=args.scaling_method,
                    feature_selection_k=args.feature_selection_k,
                )
                if ton_meta["preprocessing"]["scaler"] is not None:
                    joblib.dump(
                        ton_meta["preprocessing"]["scaler"],
                        args.output_dir / "ton" / "scaler.pkl",
                    )
                if ton_meta["preprocessing"]["selector"] is not None:
                    joblib.dump(
                        ton_meta["preprocessing"]["selector"],
                        args.output_dir / "ton" / "feature_selector.pkl",
                    )
                tensor_meta = dataframe_to_tensors(
                    ton_df, root=args.output_dir, prefix="ton"
                )
                ton_meta.update(tensor_meta)
                with open(args.output_dir / "ton" / "metadata.json", "w") as f:
                    ton_meta_serializable = ton_meta.copy()
                    ton_meta_serializable["preprocessing"]["scaler"] = str(
                        ton_meta["preprocessing"]["scaler"]
                    )
                    ton_meta_serializable["preprocessing"]["selector"] = str(
                        ton_meta["preprocessing"]["selector"]
                    )
                    json.dump(ton_meta_serializable, f, indent=2, default=str)
                    logging.info("TON processing successful!")
        if args.cic_dir:
            if not args.cic_dir.exists():
                logging.error(f"CIC directory does not exist: {args.cic_dir}")
            else:
                logging.info(f"Processing CIC dataset from: {args.cic_dir}")
                cic_df, cic_meta = preprocess_cic(
                    args.cic_dir,
                    combine_labels=args.combine_labels,
                    workers=args.workers,
                    missing_strategy=args.missing_strategy,
                    scaling_method=args.scaling_method,
                    feature_selection_k=args.feature_selection_k,
                )
                prefix = "cic"
                if args.combine_labels:
                    prefix = "cic_combined"
                if cic_meta["preprocessing"]["scaler"] is not None:
                    scaler_path = args.output_dir / prefix / "scaler.pkl"
                    ensure_dir(scaler_path.parent)
                    joblib.dump(cic_meta["preprocessing"]["scaler"], scaler_path)

                if cic_meta["preprocessing"]["selector"] is not None:
                    selector_path = args.output_dir / prefix / "feature_selector.pkl"
                    ensure_dir(selector_path.parent)
                    joblib.dump(cic_meta["preprocessing"]["selector"], selector_path)

                tensor_meta = dataframe_to_tensors(
                    cic_df, root=args.output_dir, prefix=prefix
                )
                cic_meta.update(tensor_meta)
                with open(args.output_dir / prefix / "metadata.json", "w") as f:
                    cic_meta_serializable = cic_meta.copy()
                    cic_meta_serializable["preprocessing"]["scaler"] = str(
                        cic_meta["preprocessing"]["scaler"]
                    )
                    cic_meta_serializable["preprocessing"]["selector"] = str(
                        cic_meta["preprocessing"]["selector"]
                    )
                    json.dump(cic_meta_serializable, f, indent=2, default=str)
                logging.info("CIC processing successful!")
        if not args.ton_dir and not args.cic_dir:
            logging.warning(
                "No dataset directories specified. Use --ton_dir or --cic_dir"
            )
            return 1

        logging.info("All processing completed successfully!")
        return 0

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

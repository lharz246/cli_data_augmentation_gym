import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils.logger import get_logger

logger = get_logger()
logging = logger.logger


class NetworkData(Dataset):
    def __init__(self, X, y, device, allowed_labels=None):
        self.device = device
        self.X = X.to(device)
        self.y_cls = y[:, 0].to(device)
        self.y_lbl = y[:, 1].to(device)
        if allowed_labels is not None:
            self.update_labels(allowed_labels)

    def update_labels(self, allowed_labels):
        allowed_labels = torch.tensor(
            allowed_labels, dtype=self.y_lbl.dtype, device=self.device
        )
        mask = torch.isin(self.y_lbl, allowed_labels)
        if not mask.any():
            logging.warning("No samples found for allowed labels")
            self.X = torch.empty(
                (0, self.X.shape[1]), device=self.device, dtype=self.X.dtype
            )
            self.y_cls = torch.empty((0,), device=self.device, dtype=self.y_cls.dtype)
            self.y_lbl = torch.empty((0,), device=self.device, dtype=self.y_lbl.dtype)
        else:
            self.X = self.X[mask]
            self.y_cls = self.y_cls[mask]
            self.y_lbl = self.y_lbl[mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if len(self.X) == 0:
            raise IndexError("Dataset is empty")
        return self.X[idx], self.y_cls[idx], self.y_lbl[idx]


class DataManager:
    def __init__(self, config, augmenter):
        self.balance = config.get("balance_classes", True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = config.get("batch_size")
        self.X_tr, self.y_tr = torch.load(config.get("paths", None).get("train", None))
        self.X_val, self.y_val = torch.load(config.get("paths", None).get("val", None))
        self.X_tst, self.y_tst = torch.load(config.get("paths", None).get("test", None))
        self.X_tr = self.X_tr.to(self.device)
        self.y_tr = self.y_tr.to(self.device)
        self.X_val = self.X_val.to(self.device)
        self.y_val = self.y_val.to(self.device)
        self.X_tst = self.X_tst.to(self.device)
        self.y_tst = self.y_tst.to(self.device)
        self.source_labels = config.get("target_labels", [])
        self.augment = config.get("use_augmentation", False)
        self.augmentation_factor = config.get("augment_factor", 0.0)
        self.augment_class = config.get("target_class", None)
        self.augmenter = augmenter
        logging.debug(
            f"X: {self.X_tr.shape}, y: {self.y_tr.shape}, source labels: {self.source_labels}"
        )
        self.y_lbl_tr = self.y_tr[:, 1]
        self.unique_labels = torch.unique(self.y_lbl_tr)
        self.class_0_mask = self.y_lbl_tr == 0
        self.class_1_labels = self.unique_labels[self.unique_labels != 0]
        self.class_1_label_masks = {}
        self.class_1_label_indices = {}
        self.class_1_label_counts = {}
        self.prepare_class_1_label_indices()

    def prepare_class_1_label_indices(self):
        if len(self.class_1_labels) == 0:
            self.max_class_1_label_count = 0
            logging.warning("No class 1 labels found!")
            return
        for label in self.class_1_labels:
            mask = self.y_lbl_tr == label
            indices = torch.where(mask)[0]
            self.class_1_label_masks[label.item()] = mask
            self.class_1_label_indices[label.item()] = indices
            self.class_1_label_counts[label.item()] = len(indices)
        self.max_class_1_label_count = max(self.class_1_label_counts.values())
        logging.debug(
            f"Max count for class 1 labels (1-14): {self.max_class_1_label_count}"
        )
        logging.debug(f"Class 1 label distribution: {self.class_1_label_counts}")

    def create_balanced_dataset(self, allowed_labels, noise_std=0.01):
        if not allowed_labels:
            return torch.empty(
                (0, self.X_tr.shape[1]), device=self.device
            ), torch.empty((0, 2), device=self.device)

        allowed_labels_tensor = torch.tensor(allowed_labels, device=self.device)
        all_X_samples = []
        all_y_samples = []
        target_count = int(self.max_class_1_label_count * 1.5)
        if 0 in allowed_labels:
            class_0_indices = torch.where(self.class_0_mask)[0]
            n_available = len(class_0_indices)
            if n_available >= target_count:
                perm = torch.randperm(n_available, device=self.device)
                sampled_class_0_indices = class_0_indices[perm[:target_count]]
                all_X_samples.append(self.X_tr[sampled_class_0_indices])
                all_y_samples.append(self.y_tr[sampled_class_0_indices])
            else:
                original_X = self.X_tr[class_0_indices]
                original_y = self.y_tr[class_0_indices]
                all_X_samples.append(original_X)
                all_y_samples.append(original_y)
                n_repeats = target_count // n_available
                n_remainder = target_count % n_available
                for _ in range(n_repeats - 1):
                    noise = torch.rand_like(original_X) * noise_std
                    noisy_X = original_X + noise
                    all_X_samples.append(noisy_X)
                    all_y_samples.append(original_y)
                if n_remainder > 0:
                    perm = torch.randperm(n_available, device=self.device)
                    remainder_indices = class_0_indices[perm[:n_remainder]]
                    remainder_X = self.X_tr[remainder_indices]
                    remainder_y = self.y_tr[remainder_indices]
                    noise = torch.rand_like(remainder_X) * noise_std
                    noisy_remainder_X = remainder_X + noise
                    all_X_samples.append(noisy_remainder_X)
                    all_y_samples.append(remainder_y)
                logging.debug("Added Noise to class 0!")
            logging.debug(
                f"Using {target_count} class 0 samples (original: {n_available})"
            )
        class_1_labels_to_process = allowed_labels_tensor[allowed_labels_tensor != 0]
        if len(class_1_labels_to_process) > 0 and self.max_class_1_label_count > 0:
            for label in class_1_labels_to_process:
                label_item = label.item()
                if label_item in self.class_1_label_indices:
                    available_indices = self.class_1_label_indices[label_item]
                    n_available = len(available_indices)
                    if n_available >= target_count:
                        perm = torch.randperm(n_available, device=self.device)
                        sampled_indices = available_indices[perm[:target_count]]
                        all_X_samples.append(self.X_tr[sampled_indices])
                        all_y_samples.append(self.y_tr[sampled_indices])
                    else:
                        original_X = self.X_tr[available_indices]
                        original_y = self.y_tr[available_indices]
                        all_X_samples.append(original_X)
                        all_y_samples.append(original_y)
                        n_repeats = target_count // n_available
                        n_remainder = target_count % n_available
                        for _ in range(n_repeats - 1):
                            noise = torch.rand_like(original_X) * noise_std
                            noisy_X = original_X + noise
                            all_X_samples.append(noisy_X)
                            all_y_samples.append(original_y)
                        if n_remainder > 0:
                            perm = torch.randperm(n_available, device=self.device)
                            remainder_indices = available_indices[perm[:n_remainder]]
                            remainder_X = self.X_tr[remainder_indices]
                            remainder_y = self.y_tr[remainder_indices]
                            noise = torch.rand_like(remainder_X) * noise_std
                            noisy_remainder_X = remainder_X + noise
                            all_X_samples.append(noisy_remainder_X)
                            all_y_samples.append(remainder_y)
                        logging.debug("Added Noise to class 1!")
                    logging.debug(
                        f"Label {label_item}: Sampled {target_count} samples (original: {n_available})"
                    )
        if all_X_samples:
            balanced_X = torch.cat(all_X_samples, dim=0)
            balanced_y = torch.cat(all_y_samples, dim=0)
            n_samples = len(balanced_X)
            perm = torch.randperm(n_samples, device=self.device)
            balanced_X = balanced_X[perm]
            balanced_y = balanced_y[perm]
            logging.info(f"Created balanced dataset with {len(balanced_X)} samples")
            return balanced_X, balanced_y
        else:
            logging.warning("No samples selected for balanced dataset")
            return torch.empty(
                (0, self.X_tr.shape[1]), device=self.device
            ), torch.empty((0, 2), device=self.device)

    def get_augmentation_data(self, source_labels, sample_size=None):
        if not source_labels:
            return torch.empty(
                (0, self.X_tr.shape[1]), device=self.device
            ), torch.empty((0, 2), device=self.device)
        augment_indices = []
        for label in source_labels:
            if label == 0:
                class_0_indices = torch.where(self.class_0_mask)[0]
                n_available = len(class_0_indices)
                if len(class_0_indices) > 0:
                    if sample_size and len(class_0_indices) > sample_size:
                        perm = torch.randperm(len(class_0_indices), device=self.device)
                        selected_indices = class_0_indices[perm[:sample_size]]
                    else:
                        n_repeats = sample_size // n_available
                        n_remainder = sample_size % n_available
                        repeated_indices = available_indices.repeat(n_repeats)
                        if n_remainder > 0:
                            perm = torch.randperm(n_available, device=self.device)
                            remainder_indices = available_indices[perm[:n_remainder]]
                            class_0_indices = torch.cat(
                                [repeated_indices, remainder_indices]
                            )
                        else:
                            class_0_indices = repeated_indices
                        selected_indices = class_0_indices
                    augment_indices.append(selected_indices)
                    logging.info(
                        f"Label {label}: Selected {len(selected_indices)} samples for augmentation"
                    )
            elif label in self.class_1_label_indices:
                available_indices = self.class_1_label_indices[label]
                n_available = len(available_indices)
                if len(available_indices) > 0:
                    if sample_size and len(available_indices) > sample_size:
                        perm = torch.randperm(
                            len(available_indices), device=self.device
                        )
                        selected_indices = available_indices[perm[:sample_size]]
                    else:
                        n_repeats = sample_size // n_available
                        n_remainder = sample_size % n_available
                        repeated_indices = available_indices.repeat(n_repeats)
                        if n_remainder > 0:
                            perm = torch.randperm(n_available, device=self.device)
                            remainder_indices = available_indices[perm[:n_remainder]]
                            class_1_indices = torch.cat(
                                [repeated_indices, remainder_indices]
                            )
                        else:
                            class_1_indices = repeated_indices
                        selected_indices = class_1_indices
                    augment_indices.append(selected_indices)
                    logging.info(
                        f"Label {label}: Selected {len(selected_indices)} samples for augmentation"
                    )
            else:
                logging.warning(f"Label {label} not found in available data")
        if augment_indices:
            all_indices = torch.cat(augment_indices)
            all_indices = torch.sort(all_indices)[0]
            X_augment = self.X_tr[all_indices]
            y_augment = self.y_tr[all_indices]
            noise = torch.rand_like(X_augment) * (0.05 * X_augment.std())
            X_augment += noise
            logging.info(f"Total augmentation data: {len(X_augment)} samples")
            return X_augment, y_augment
        else:
            logging.warning("No augmentation data found")
            return torch.empty(
                (0, self.X_tr.shape[1]), device=self.device
            ), torch.empty((0, 2), device=self.device)

    def get_train_loader(self, allowed_labels=None, model=None):
        inputs, class_labels = self.create_balanced_dataset(allowed_labels)
        if len(inputs) > 0 and self.augment:
            source_labels = [
                i for i in self.source_labels if i <= (max(allowed_labels) - 1)
            ]
            if len(source_labels):
                self.augmenter.setup_step(
                    step_idx=max(allowed_labels) - 1,
                    source_labels=source_labels,
                    target_class=self.augment_class,
                    current_labels=allowed_labels,
                    model=model,
                    X=inputs,
                    y_labels=class_labels[:, 1],
                )
                # source_labels = [source_labels[-1]]
                X_candidate, y_candidate = self.get_augmentation_data(
                    source_labels,
                    sample_size=math.ceil(len(inputs) * self.augmentation_factor),
                )
                if len(X_candidate) > 0:
                    X_augmented, y_augmented = self.augmenter.augment_samples(
                        X_candidate, y_candidate
                    )
                    logging.debug(
                        f"Combined augmentation: {inputs.shape} + {X_augmented.shape}"
                    )
                    inputs = torch.cat([inputs, X_augmented], dim=0)
                    class_labels = torch.cat([class_labels, y_augmented], dim=0)
                    if torch.equal(X_augmented, X_candidate) and torch.equal(
                        y_augmented, y_candidate
                    ):
                        logging.error("Error during Augmenting samples are equal!!!")
                    logging.debug(f"Final training data: {len(inputs)} samples")
        dataset = NetworkData(
            inputs,
            class_labels,
            self.device,
            allowed_labels=allowed_labels,
        )
        if len(dataset) > 0:
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False
            )
        else:
            logging.warning("Empty dataset, returning empty DataLoader")
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False
            )

    def get_ewc_loader(self, allowed_labels):
        inputs, class_labels = self.create_balanced_dataset(allowed_labels)
        dataset = NetworkData(
            inputs,
            class_labels,
            self.device,
            allowed_labels=allowed_labels,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False
        )

    def get_val_loader(self, allowed_labels):
        dataset = NetworkData(
            self.X_val,
            self.y_val,
            self.device,
            allowed_labels=allowed_labels,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False
        )

    def get_test_loader(self, allowed_labels):
        dataset = NetworkData(
            self.X_tst, self.y_tst, self.device, allowed_labels=allowed_labels
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False
        )

    def get_class_distribution_info(self):
        info = {
            "n_class_0_total": int(torch.sum(self.class_0_mask).item()),
            "max_class_1_label_count": self.max_class_1_label_count,
            "n_class_1_labels": len(self.class_1_label_indices),
            "class_1_label_counts": dict(self.class_1_label_counts),
        }
        return info

import json
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from utils.logger import get_logger

logger = get_logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stats(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.logger.error(f"Couldn't load stat-file: {e}")
        return {}


def process_stats(raw_stats: Dict[str, Any]) -> Dict[str, Any]:
    if not raw_stats:
        return {}
    stats = {}
    feature_importance = raw_stats.get("feature_importance", {})
    mu_std_stats = raw_stats.get("feature_statistics", {}).get("per_sub_class", {})
    for k, v in mu_std_stats.items():
        fi_key = f"ovr_{k}"
        top_features = set()
        if fi_key in feature_importance:
            methods = feature_importance[fi_key].get("methods", {})
            for method_data in methods.values():
                top_features.update(
                    feat["feature_index"]
                    for feat in method_data.get("top_features", [])
                )
        mu_tensor = torch.tensor(v["mean"], dtype=torch.float32, device=DEVICE)
        std_tensor = torch.tensor(v["std"], dtype=torch.float32, device=DEVICE)
        feature_indices = (
            torch.tensor(list(top_features), dtype=torch.int64, device=DEVICE)
            if top_features
            else torch.empty(0, dtype=torch.int64, device=DEVICE)
        )
        stats[k] = {
            "mu": mu_tensor,
            "std": std_tensor,
            "feature_importance": feature_indices,
        }
    return stats


class AugmenterInterface(ABC):

    @abstractmethod
    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        **kwargs,
    ):
        pass

    @abstractmethod
    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def needs_donor(self) -> bool:
        pass


class BaseAugmenter(AugmenterInterface):

    def __init__(self, **kwargs):
        self.source_labels = set()
        self.target_class = 0
        self.current_step = 0
        self.is_active = False
        self.donor_label = None

    def _select_donor_label(
        self, current_labels: List[int], target_class: int
    ) -> Optional[int]:
        if not self.needs_donor():
            return None

        available_donors = [
            label for label in current_labels if label not in self.source_labels
        ]

        if not available_donors:
            logger.warning("No donor labels available!")
            return None
        if target_class == 0:
            donor = 0 if 0 in available_donors else None
        else:
            donor = max(available_donors)
        if donor is not None:
            logger.debug(
                f"Selected donor label: {donor} for target class {target_class}"
            )
        else:
            logger.warning(f"No suitable donor found for target class {target_class}")

        return donor

    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        **kwargs,
    ):
        self.current_step = step_idx
        self.source_labels = set(source_labels) if source_labels else set()
        self.target_class = target_class
        self.is_active = step_idx > 0 and len(self.source_labels) > 0
        if self.needs_donor():
            self.donor_label = self._select_donor_label(current_labels, target_class)
            if self.donor_label is None:
                self.is_active = False
        else:
            self.donor_label = None

    def _create_augmented_labels(
        self, original_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(original_labels)
        target_classes = torch.full(
            (batch_size,),
            self.target_class,
            dtype=torch.int64,
            device=original_labels.device,
        )

        if original_labels.dim() > 1:
            target_labels = torch.zeros_like(original_labels, dtype=torch.int64)
            if self.target_class == 0:
                target_labels[:, 0] = 1
            else:
                target_labels = original_labels.clone().long()
        else:
            if self.target_class == 0:
                target_labels = torch.zeros_like(original_labels, dtype=torch.int64)
            else:
                target_labels = original_labels.clone().long()
        return target_classes, target_labels


class FeatureCorruption(BaseAugmenter):

    def __init__(self, corruption_strength: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.corruption_strength = corruption_strength
        self.source_discriminative_features = {}

    def needs_donor(self) -> bool:
        return False

    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        stats: Dict[str, Any] = None,
        **kwargs,
    ):
        super().setup(step_idx, source_labels, target_class, current_labels, **kwargs)
        logger.debug(
            f"FeatureCorruption setup: is_active={self.is_active}, stats_provided={stats is not None}"
        )
        if not self.is_active:
            logger.warning(
                "FeatureCorruption not active - step_idx <= 0 or no source labels"
            )
            return
        if stats is None:
            logger.warning("FeatureCorruption: No stats provided as parameter")
            return
        if not isinstance(stats, dict):
            logger.warning(f"FeatureCorruption: Invalid stats type: {type(stats)}")
            return
        logger.debug(f"FeatureCorruption: Available stats keys: {list(stats.keys())}")
        self.source_discriminative_features = {}
        features_loaded = False
        for label in self.source_labels:
            label_str = str(label)
            logger.debug(
                f"FeatureCorruption: Processing source label {label} (key: {label_str})"
            )
            if label_str in stats:
                label_stats = stats[label_str]
                if (
                    "feature_importance" in label_stats
                    and len(label_stats["feature_importance"]) > 0
                ):
                    self.source_discriminative_features[label] = label_stats[
                        "feature_importance"
                    ]
                    logger.debug(
                        f"FeatureCorruption: Loaded {len(self.source_discriminative_features[label])} important features for source label {label}"
                    )
                    features_loaded = True
                else:
                    logger.warning(
                        f"FeatureCorruption: No feature importance found for label {label}"
                    )
            else:
                logger.warning(
                    f"FeatureCorruption: Label {label_str} not found in stats"
                )
        if not features_loaded:
            logger.warning(
                "FeatureCorruption: No discriminative features loaded for any source label"
            )
        logger.debug(
            f"FeatureCorruption setup complete: {len(self.source_discriminative_features)} labels with features"
        )

    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_active or len(X) == 0:
            return X, y_cls, y_lbls

        X_augmented = X.clone()
        if y_lbls.dim() > 1:
            actual_labels = y_lbls.argmax(dim=1).long()
        else:
            actual_labels = y_lbls.long()

        samples_corrupted = 0
        for label in self.source_labels:
            if label not in self.source_discriminative_features:
                logger.debug(
                    f"FeatureCorruption: No features for label {label}, skipping"
                )
                continue

            label_mask = actual_labels == label
            if not label_mask.any():
                logger.debug(f"FeatureCorruption: No samples found for label {label}")
                continue

            important_features = self.source_discriminative_features[label]
            if len(important_features) > 0:
                noise = torch.normal(
                    mean=0.0,
                    std=self.corruption_strength,
                    size=(label_mask.sum(), len(important_features)),
                    device=X.device,
                )
                X_augmented[label_mask][:, important_features] += noise
                samples_corrupted += label_mask.sum().item()
                logger.debug(
                    f"FeatureCorruption: Corrupted {label_mask.sum()} samples of label {label}"
                )

        logger.debug(f"FeatureCorruption: Total samples corrupted: {samples_corrupted}")
        y_cls_augmented, y_lbls_augmented = self._create_augmented_labels(y_lbls)
        return X_augmented, y_cls_augmented, y_lbls_augmented


class LabelNoise(BaseAugmenter):

    def __init__(self, flip_probability: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.flip_probability = flip_probability

    def needs_donor(self) -> bool:
        return False

    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_active or len(X) == 0:
            return X, y_cls, y_lbls

        X_augmented = X.clone()

        if y_lbls.dim() > 1:
            actual_labels = y_lbls.argmax(dim=1)
        else:
            actual_labels = y_lbls
        source_mask = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        for source_label in self.source_labels:
            source_mask |= actual_labels == source_label
        flip_mask = source_mask & (
            torch.rand(len(X), device=X.device) < self.flip_probability
        )
        y_cls_augmented = y_cls.clone()
        y_lbls_augmented = y_lbls.clone()
        if flip_mask.any():
            num_flipped = flip_mask.sum().item()
            logger.debug(
                f"LabelNoise: Flipping {num_flipped} samples to target class {self.target_class}"
            )
            y_cls_augmented[flip_mask] = self.target_class
            if y_lbls.dim() > 1:
                y_lbls_augmented[flip_mask] = torch.zeros_like(
                    y_lbls_augmented[flip_mask]
                )
                if self.target_class < y_lbls_augmented.shape[1]:
                    y_lbls_augmented[flip_mask, self.target_class] = 1
            else:
                y_lbls_augmented[flip_mask] = self.target_class
        return X_augmented, y_cls_augmented, y_lbls_augmented


class MixUp(BaseAugmenter):

    def __init__(self, alpha: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.donor_prototype = None

    def needs_donor(self) -> bool:
        return True

    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        stats: Dict[str, Any] = None,
        **kwargs,
    ):
        super().setup(step_idx, source_labels, target_class, current_labels, **kwargs)
        logger.debug(
            f"MixUp setup: is_active={self.is_active}, donor_label={self.donor_label}, stats_provided={stats is not None}"
        )
        if not self.is_active:
            logger.warning("MixUp not active - step_idx <= 0 or no source labels")
            return
        if self.donor_label is None:
            logger.warning("MixUp: No donor label available")
            return
        if stats is None:
            logger.warning("MixUp: No stats provided as parameter")
            self.is_active = False
            return
        if not isinstance(stats, dict):
            logger.warning(f"MixUp: Invalid stats type: {type(stats)}")
            self.is_active = False
            return
        donor_str = str(self.donor_label)
        logger.debug(
            f"MixUp: Looking for donor {donor_str} in stats keys: {list(stats.keys())}"
        )
        if donor_str in stats:
            donor_stats = stats[donor_str]
            if "mu" in donor_stats:
                self.donor_prototype = donor_stats["mu"].clone()
                logger.debug(
                    f"MixUp: Loaded donor prototype for label {self.donor_label}, shape: {self.donor_prototype.shape}"
                )
            else:
                logger.warning(
                    f"MixUp: No 'mu' field found in stats for donor label {self.donor_label}"
                )
                self.is_active = False
        else:
            logger.warning(
                f"MixUp: No prototype found for donor label {self.donor_label}"
            )
            self.is_active = False

    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_active or len(X) == 0 or self.donor_prototype is None:
            return X, y_cls, y_lbls

        X_augmented = X.clone()
        if y_lbls.dim() > 1:
            actual_labels = y_lbls.argmax(dim=1).long()
        else:
            actual_labels = y_lbls.long()

        num_samples = len(X_augmented)
        lambdas = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample((num_samples,))
            .to(X.device)
        )

        samples_mixed = 0
        for i in range(num_samples):
            current_label = actual_labels[i].item()
            if current_label in self.source_labels:
                X_augmented[i] = (
                    lambdas[i] * X_augmented[i]
                    + (1 - lambdas[i]) * self.donor_prototype
                )
                samples_mixed += 1

        logger.debug(f"MixUp: Mixed {samples_mixed} samples with donor prototype")
        y_cls_augmented, y_lbls_augmented = self._create_augmented_labels(y_lbls)

        return X_augmented, y_cls_augmented, y_lbls_augmented


class FeatureSwap(BaseAugmenter):

    def __init__(
        self, swap_ratio: float = 0.2, use_important_features: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.swap_ratio = swap_ratio
        self.use_important_features = use_important_features
        self.donor_samples = None
        self.donor_important_features = None

    def needs_donor(self) -> bool:
        return True

    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        X: torch.Tensor = None,
        y_labels: torch.Tensor = None,
        stats: Dict[str, Any] = None,
        **kwargs,
    ):
        super().setup(step_idx, source_labels, target_class, current_labels, **kwargs)

        # Debug logging
        logger.debug(
            f"FeatureSwap setup: is_active={self.is_active}, donor_label={self.donor_label}, X_provided={X is not None}, y_labels_provided={y_labels is not None}"
        )
        if not self.is_active:
            logger.warning("FeatureSwap not active - step_idx <= 0 or no source labels")
            return
        if self.donor_label is None:
            logger.warning("FeatureSwap: No donor label available")
            return
        if X is None or y_labels is None:
            logger.warning(
                "FeatureSwap: No training data provided for donor sample collection"
            )
            self.is_active = False
            return

        y_labels = y_labels.long()
        donor_mask = y_labels == self.donor_label
        if donor_mask.any():
            self.donor_samples = X[donor_mask].clone()
            logger.info(
                f"FeatureSwap: Collected {len(self.donor_samples)} samples from donor label {self.donor_label}"
            )
        else:
            self.donor_samples = None
            logger.warning(
                f"FeatureSwap: No samples found for donor label {self.donor_label}"
            )
            self.is_active = False
            return
        if self.use_important_features and stats and self.donor_label is not None:
            donor_str = str(self.donor_label)
            logger.debug(
                f"FeatureSwap: Looking for features for donor {donor_str} in stats keys: {list(stats.keys())}"
            )
            if donor_str in stats:
                donor_stats = stats[donor_str]
                if (
                    "feature_importance" in donor_stats
                    and len(donor_stats["feature_importance"]) > 0
                ):
                    self.donor_important_features = donor_stats["feature_importance"]
                    logger.info(
                        f"FeatureSwap: Loaded {len(self.donor_important_features)} important features from donor label {self.donor_label}"
                    )
                else:
                    logger.warning(
                        f"FeatureSwap: No feature importance found for donor label {self.donor_label}"
                    )
            else:
                logger.warning(
                    f"FeatureSwap: Donor label {donor_str} not found in stats"
                )

    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_active or len(X) == 0 or self.donor_samples is None:
            return X, y_cls, y_lbls

        X_augmented = X.clone()
        if self.use_important_features and self.donor_important_features is not None:
            features_to_swap = self.donor_important_features
            logger.debug(
                f"FeatureSwap: Using {len(features_to_swap)} important features"
            )
        else:
            num_features = X.shape[1]
            num_swap = int(num_features * self.swap_ratio)
            features_to_swap = torch.randperm(num_features, device=X.device)[:num_swap]
            logger.debug(f"FeatureSwap: Using {len(features_to_swap)} random features")

        if len(features_to_swap) > 0 and len(self.donor_samples) > 0:
            num_augment_samples = len(X_augmented)
            random_indices = torch.randint(
                0, len(self.donor_samples), (num_augment_samples,), device=X.device
            )
            donor_samples = self.donor_samples[random_indices]
            X_augmented[:, features_to_swap] = donor_samples[:, features_to_swap]
            logger.debug(
                f"FeatureSwap: Swapped features for {num_augment_samples} samples"
            )

        y_cls_augmented, y_lbls_augmented = self._create_augmented_labels(y_lbls)

        return X_augmented, y_cls_augmented, y_lbls_augmented


class DataPoisoner(BaseAugmenter):

    def __init__(self, epsilon: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = None

    def needs_donor(self) -> bool:
        return False

    def setup(
        self,
        step_idx: int,
        source_labels: List[int],
        target_class: int,
        current_labels: List[int],
        model: torch.nn.Module = None,
        **kwargs,
    ):
        super().setup(step_idx, source_labels, target_class, current_labels, **kwargs)
        self.model = model

        logger.debug(
            f"DataPoisoner setup: is_active={self.is_active}, model_provided={model is not None}"
        )

        if self.is_active and self.model is None:
            logger.warning("DataPoisoner: No model provided")
            self.is_active = False

    def augment_samples(
        self,
        X: torch.Tensor,
        y_cls: torch.Tensor,
        y_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_active or len(X) == 0 or self.model is None:
            return X, y_cls, y_lbls

        try:
            X_adv_all = []
            num_samples = len(X)
            for start_idx in range(0, num_samples, 256):
                end_idx = min(start_idx + 256, num_samples)
                X_batch = X[start_idx:end_idx].clone().detach().requires_grad_(True)
                logits = self.model.inference(X_batch)
                target_classes = torch.full(
                    (len(X_batch),),
                    self.target_class,
                    dtype=torch.int64,
                    device=X.device,
                )
                target_logits = logits[torch.arange(len(logits)), target_classes]
                loss = -target_logits.sum()
                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=X_batch,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                X_adv_batch = X_batch.detach() + self.epsilon * grad.sign()
                X_adv_all.append(X_adv_batch)
            X_augmented = torch.cat(X_adv_all, dim=0)

            logger.debug(
                f"DataPoisoner: Generated adversarial examples for {len(X_augmented)} samples"
            )

            return X_augmented, y_cls, y_lbls

        except Exception as e:
            logger.error(f"DataPoisoner failed: {e}")
            return X, y_cls, y_lbls


class AugmentationCoordinator:

    def __init__(
        self,
        augmenter_configs: Dict[str, Dict],
        stats_path: Path,
    ):
        self.augmenter_configs = augmenter_configs
        self.augmenters = []
        self.current_step = 0
        raw_stats = load_stats(stats_path)
        logger.logger.debug("Augmentation stats loaded!")
        self.stats = process_stats(raw_stats)
        logger.logger.debug("Augmentation stats processed!")
        logger.logger.debug(f"Processed stats keys: {list(self.stats.keys())}")
        for key, value in self.stats.items():
            logger.logger.debug(
                f"Stats for {key}: mu.shape={value['mu'].shape}, std.shape={value['std'].shape}, features={len(value['feature_importance'])}"
            )

        del raw_stats
        self._augmenter_classes = {
            "FeatureCorruption": FeatureCorruption,
            "LabelNoise": LabelNoise,
            "MixUp": MixUp,
            "FeatureSwap": FeatureSwap,
            "DataPoisoner": DataPoisoner,
        }

        self._create_augmenters()

    def _create_augmenters(self):
        self.augmenters = []
        for name, config in self.augmenter_configs.items():
            augmenter_class = self._augmenter_classes.get(name)
            if augmenter_class:
                try:
                    augmenter = augmenter_class(**config)
                    self.augmenters.append(augmenter)
                    logger.info(f"Created augmenter: {name}")
                except Exception as e:
                    logger.error(f"Failed to create augmenter {name}: {e}")

    def setup_step(
        self,
        source_labels: List[int],
        target_class: int,
        step_idx: int,
        current_labels: List[int],
        model: torch.nn.Module = None,
        X: torch.Tensor = None,
        y_labels: torch.Tensor = None,
    ):
        source_labels = list(set(source_labels))
        logger.debug(
            f"AugmentationCoordinator setup_step: , source_labels={source_labels}, target_class={target_class}"
        )  # step_idx={step_idx}
        logger.debug(
            f"Stats available: {self.stats is not None}, Stats keys: {list(self.stats.keys()) if self.stats else 'None'}"
        )
        setup_kwargs = {
            "step_idx": step_idx,
            "source_labels": source_labels,
            "target_class": target_class,
            "current_labels": current_labels,
            "model": model,
            "stats": self.stats,
            "X": X,
            "y_labels": y_labels,
        }
        for augmenter in self.augmenters:
            try:
                if hasattr(augmenter, "needs_donor") and augmenter.needs_donor():
                    augmenter_kwargs = setup_kwargs.copy()
                else:
                    augmenter_kwargs = {
                        k: v
                        for k, v in setup_kwargs.items()
                        if k not in ["X", "y_labels"]
                        or augmenter.__class__.__name__ == "FeatureSwap"
                    }
                augmenter.setup(**augmenter_kwargs)
                if augmenter.is_active:
                    logger.debug(f"Setup {augmenter.__class__.__name__} - ACTIVE")
                else:
                    logger.debug(f"Setup {augmenter.__class__.__name__} - INACTIVE")
            except Exception as e:
                logger.error(
                    f"Failed to setup augmenter {augmenter.__class__.__name__}: {e}"
                )
                import traceback

                logger.error(traceback.format_exc())

    def augment_samples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        active_augmenters = [aug for aug in self.augmenters if aug.is_active]
        if not active_augmenters or len(X) == 0:
            logger.debug(
                f"No active augmenters ({len(active_augmenters)}) or no samples ({len(X)})"
            )
            return X, y

        if X.device != DEVICE:
            X = X.to(DEVICE)
        if y.device != DEVICE:
            y = y.to(DEVICE)
        y_cls = y[:, 0]
        y_lbls = y[:, 1]
        num_samples = len(X)
        samples_per_augmenter = max(1, num_samples // len(active_augmenters))
        X_augmented_list = []
        y_cls_augmented_list = []
        y_lbls_augmented_list = []
        start_idx = 0
        for i, augmenter in enumerate(active_augmenters):
            end_idx = min(start_idx + samples_per_augmenter, num_samples)
            if i == len(active_augmenters) - 1:
                end_idx = num_samples
            if start_idx < end_idx:
                X_subset = X[start_idx:end_idx]
                y_cls_subset = y_cls[start_idx:end_idx]
                y_lbls_subset = y_lbls[start_idx:end_idx]
                try:
                    X_aug, y_cls_aug, y_lbls_aug = augmenter.augment_samples(
                        X_subset, y_cls_subset, y_lbls_subset
                    )
                    X_augmented_list.append(X_aug)
                    y_cls_augmented_list.append(y_cls_aug)
                    y_lbls_augmented_list.append(y_lbls_aug)
                    logger.logger.debug(
                        f"Augmented {len(X_subset)} samples with {augmenter.__class__.__name__}"
                    )
                except Exception as e:
                    logger.error(
                        f"Sample augmentation failed for {augmenter.__class__.__name__}: {e}"
                    )
                    X_augmented_list.append(X_subset)
                    y_cls_augmented_list.append(y_cls_subset)
                    y_lbls_augmented_list.append(y_lbls_subset)
            start_idx = end_idx
        if X_augmented_list:
            X_combined = torch.cat(X_augmented_list, dim=0)
            y_cls_combined = torch.cat(y_cls_augmented_list, dim=0)
            y_lbls_combined = torch.cat(y_lbls_augmented_list, dim=0)
            y_combined = torch.stack(
                [y_cls_combined.long(), y_lbls_combined.long()], dim=1
            )
            return X_combined, y_combined
        else:
            return X, y

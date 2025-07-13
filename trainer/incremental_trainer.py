import copy
import logging
import os
from typing import Any, Dict, Tuple, Optional, List
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from data_managment.augmenters import AugmentationCoordinator
from models import BaseModel
from trainer.base_trainer import BaseTrainer
from factories.model_factory import ModelFactory
from factories.optim_factory import OptimizerFactory
from data_managment.datamanagment import DataManager
from utils.helper import get_train_class_list
from utils.logger import get_logger

logger = get_logger()
torch.manual_seed(42)


@BaseTrainer.register("incremental")
class IncrementalTrainer(BaseTrainer):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.train_cfg = config["train_cfg"]
        self.data_cfg = config["data_cfg"]
        self.model_cfg = config["model_cfg"]
        self.loss_cfg = config["loss_cfg"]
        self.pack_cfg = config.get("packnet_cfg", {})
        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.train_cfg.get("device") != "cpu"
            else "cpu"
        )
        self.model_cfg["num_classes"] = 2
        self.model: BaseModel = (
            ModelFactory.create(
                self.model_cfg,
                self.data_cfg,
                self.loss_cfg,
                is_validation=False,
            )
        ).to(self.device)
        self.has_distillation = any(
            d.get("name") == "distillation" for d in self.loss_cfg
        )
        if self.has_distillation:
            teacher_model: BaseModel = (
                ModelFactory.create(
                    self.model_cfg,
                    self.data_cfg,
                    self.loss_cfg,
                    is_validation=False,
                )
            ).to(self.device)
            self.model.set_teacher_model(teacher_model)
        self.optimizer, self.scheduler = OptimizerFactory.create(
            self.train_cfg, self.model.parameters()
        )
        self.tr_loader = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        (
            self.train_label_list,
            self.val_label_list,
            self.ewc_list,
        ) = get_train_class_list(
            self.train_cfg.get("true_incremental", False),
            self.data_cfg.get("all_labels", []),
            self.train_cfg.get("batch_groups", None),
            self.train_cfg.get("epochs", 0),
        )
        self.current_task_idx = 0
        self.batch_size = self.train_cfg.get("batch_size", 1)
        self.old_model: Optional[torch.nn.Module] = None
        self.task_masks = []
        self.previous_task_labels = set()
        self.task_transitions = self._precompute_task_transitions()
        self.has_ewc = any(d.get("name") == "ewc" for d in self.loss_cfg)
        self.max_norm = self.train_cfg.get("max_norm")
        os.makedirs(self.data_cfg.get("output_dir", "./"), exist_ok=True)
        augmenter = (
            AugmentationCoordinator(
                self.data_cfg["augmentation_cfg"], self.data_cfg["stats_path"]
            )
            if self.train_cfg["use_augmentation"]
            else None
        )
        self.data_manager = DataManager(self.data_cfg, augmenter)

    def _precompute_task_transitions(self) -> Dict[int, bool]:
        transitions = {}
        prev_labels = set()
        for i, labels in enumerate(self.train_label_list):
            current_labels = (
                set(labels) if isinstance(labels, (list, tuple)) else labels
            )
            if current_labels != prev_labels and i > 0:
                transitions[i] = True
            prev_labels = current_labels
        return transitions

    def on_training_start(self):
        logger.logger.debug("Starting incremental training...")
        self.setup_loss_components(self.current_task_idx)
        if self.pack_cfg.get("use_packnet"):
            logger.logger.debug(
                "PackNet configuration detected - will apply on task transitions"
            )
        self.previous_task_labels = set(self.train_label_list[0])
        logger.logger.debug(
            f"Training initialization complete. Starting with task 0, labels: {self.train_label_list[0] if self.train_label_list else []}"
        )

    def _get_target_labels_for_augmentation(self) -> List[int]:
        target_label = self.train_cfg["target_label"]
        if target_label:
            if self.train_cfg["single_label"]:
                return [target_label]
            else:
                return self.train_label_list[self.current_task_idx]

    def prepare_epoch(self, epoch: int) -> Tuple[DataLoader, DataLoader]:
        self.current_task_idx = epoch - 1
        current_task_labels = set(self.train_label_list[self.current_task_idx])
        if self.task_transitions.get(self.current_task_idx, False):
            self._handle_task_transition(current_task_labels)
        model = None
        if self.data_manager.augment:
            model = copy.deepcopy(self.model)
            model.teacher_model = None
        self.tr_loader = self.data_manager.get_train_loader(
            allowed_labels=self.train_label_list[self.current_task_idx],
            model=model,
        )
        val_loader = self.data_manager.get_val_loader(
            self.val_label_list[self.current_task_idx]
        )
        self.log_task_distribution(self.current_task_idx)
        return self.tr_loader, val_loader

    def _handle_task_transition(self, current_task_labels: set):
        logger.logger.debug(
            f"Transitioning to new task {self.current_task_idx} at epoch {self.current_task_idx + 1}"
        )
        self.setup_loss_components(self.current_task_idx)
        self.previous_task_labels = current_task_labels

    def train_step(self, batch) -> Dict[str, Any]:
        inputs, targets, labels = batch
        kwargs = {}
        self.optimizer.zero_grad()
        self.model.train(True)
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            if (
                self.has_distillation
                and self.model.teacher_model is not None
                and self.current_task_idx > 0
            ):
                with torch.no_grad():
                    teacher_logits = self.model.get_teacher_logits(inputs)
                    kwargs["teacher_logits"] = teacher_logits
            outputs = self.model(inputs, targets, labels, **kwargs)
            loss_dict = outputs["loss"]
        total_loss = loss_dict["total"]
        self.scaler.scale(total_loss).backward()
        if self.max_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        preds = torch.argmax(outputs.get("logits", outputs), dim=1)
        correct = (preds == targets).sum().item()
        return {
            "loss": {k: v.cpu().detach() for k, v in loss_dict.items()},
            "batch_size": targets.size(0),
            "correct": correct,
            "task_ids": self.train_label_list[self.current_task_idx],
        }

    @torch.no_grad()
    def validation_step(self, batch) -> Dict[str, Any]:
        inputs, targets, labels = batch
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(
            self.device, non_blocking=True
        )

        self.model.eval()
        logits = self.model.inference(inputs)
        loss_val = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        return {
            "batch_size": targets.size(0),
            "correct": correct,
            "loss": loss_val,
            "all_labels": labels,
            "all_preds": preds.cpu(),
            "task_ids": self.val_label_list[self.current_task_idx],
        }

    def on_training_end(self) -> Tuple[Dict[str, Any], Any]:
        test_loader = self.data_manager.get_test_loader(
            self.val_label_list[self.current_task_idx]
        )
        return test_loader, self.model

    def setup_loss_components(self, idx: int):
        logger.logger.debug(f"Setting up loss components for task {idx}...")

        if self.has_distillation and idx > 0:
            if hasattr(self.model, "set_teacher_model"):
                self.model.teacher_model.load_state_dict(
                    self.model.state_dict(), strict=False
                )
                self.model.teacher_model.eval()
                logger.logger.debug("Teacher model set up for distillation")

        if self.has_ewc and idx > 0:
            try:
                ewc_labels = self.ewc_list[idx]
                ewc_loader = self.data_manager.get_ewc_loader(allowed_labels=ewc_labels)
                if hasattr(self.model, "update_ewc"):
                    ewc_config = next(
                        (d for d in self.loss_cfg if d.get("name") == "ewc"), {}
                    )
                    max_percentage = ewc_config.get("ewc_max_percentage", 0.3)

                    self.model.update_ewc(
                        dataloader=ewc_loader,
                        device=self.device,
                        max_percentage=max_percentage,
                    )
                    ewc_manager = self.model.ewc_manager
                    fisher_params = len(ewc_manager.fisher) if ewc_manager.fisher else 0
                    optimal_params = (
                        len(ewc_manager.optimal_params)
                        if ewc_manager.optimal_params
                        else 0
                    )
                    logger.logger.debug(
                        f"EWC updated: {fisher_params} Fisher entries, {optimal_params} optimal params"
                    )
                    logger.logger.debug(f"   Previous task labels: {ewc_labels}")
                    if ewc_manager.fisher:
                        first_key = next(iter(ewc_manager.fisher.keys()))
                        fisher_mean = ewc_manager.fisher[first_key].mean().item()
                        logger.logger.debug(
                            f"   Sample Fisher value (mean of {first_key}): {fisher_mean:.6f}"
                        )
            except Exception as e:
                logging.error(f"Failed to setup EWC: {e}")

    def _apply_packnet(self):
        try:
            prune_ratio = self.pack_cfg.get("prune_ratio", 0)
            if prune_ratio <= 0:
                logger.logger.debug("PackNet prune ratio <= 0 - skipping pruning")
                return
            masks = self.model.prune_model_global(prune_ratio)
            self.task_masks.append(masks)
            init_method = self.pack_cfg.get("init_method", "xavier_uniform")
            self.model.reinitialize_pruned_weights(init_method)
            freeze_info = self.model.get_sparsity_stats()
            logger.logger.info(f"Prune Info: {freeze_info}")
        except Exception as e:
            logging.error(f"✗ Failed to apply PackNet: {e}")

    def log_task_distribution(self, task_idx: int):
        info = self.data_manager.get_class_distribution_info()
        logger.logger.info(
            f"Task {task_idx} distribution: "
            f"Class 0: {info['n_class_0_total']}, "
            f"Class 1: {info['n_class_1_labels']}, "
            f"Max Class 1: {info['max_class_1_label_count']}"
        )

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelRegistry:

    _models: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Any):
            cls._models[name.lower()] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Any:
        try:
            return cls._models[name.lower()]
        except KeyError:
            raise ValueError(
                f"Model '{name}' not found. Available: {list(cls._models.keys())}"
            )


class EWCManager:

    def __init__(self):
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.parameter_shapes: Dict[str, torch.Size] = {}

    def compute_fisher_information(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        max_percentage: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        model.to(device).eval()
        current_shapes = {n: p.shape for n, p in model.named_parameters()}
        fisher = {}
        for name, param in model.named_parameters():
            if (
                name not in self.parameter_shapes
                or self.parameter_shapes[name] == current_shapes[name]
            ):
                fisher[name] = torch.zeros_like(param, device=device)
        total_samples = min(len(dataloader), int(len(dataloader) * max_percentage))
        batch_count = 0
        with torch.enable_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_count >= total_samples:
                    break
                if len(batch) == 3:
                    x, y, labels = batch
                elif len(batch) == 2:
                    x, y = batch
                else:
                    x = batch[0]
                    y = batch[1]
                x, y = x.to(device), y.to(device)
                if hasattr(model, "inference"):
                    logits = model.inference(x)
                else:
                    logits = model(x)
                model.zero_grad()
                loss = F.cross_entropy(logits, y)
                loss.backward()
                for name, param in model.named_parameters():
                    if name in fisher and param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2
                batch_count += 1
        for name in fisher:
            if batch_count > 0:
                fisher[name] = fisher[name] / batch_count
        self.fisher = fisher
        self.parameter_shapes = current_shapes
        return fisher

    def store_optimal_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        optimal_params = {}
        for name, param in model.named_parameters():
            if name in self.fisher:
                optimal_params[name] = param.clone().detach()
        self.optimal_params = optimal_params
        return optimal_params

    def get_ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.fisher or not self.optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, current_param in model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                try:
                    if (
                        current_param.shape == self.optimal_params[name].shape
                        and current_param.shape == self.fisher[name].shape
                    ):
                        param_penalty = (
                            self.fisher[name]
                            * (current_param - self.optimal_params[name]) ** 2
                        )
                        penalty += param_penalty.sum()
                    else:
                        continue
                except RuntimeError:
                    continue
        return penalty

    def update_ewc_losses(self, loss_fn: Callable) -> None:
        if hasattr(loss_fn, "get_ewc_losses"):
            ewc_losses = loss_fn.get_ewc_losses()
            for ewc_loss in ewc_losses:
                ewc_loss.set_fisher_and_params(self.fisher, self.optimal_params)


class BaseModel(nn.Module, ABC):

    def __init__(self, loss_fn: Optional[Callable] = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.task_masks: List[Dict[str, torch.Tensor]] = []
        self.teacher_model: Optional[nn.Module] = None
        self.ewc_manager = EWCManager()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def get_teacher_logits(self, x):
        if self.teacher_model is None:
            raise ValueError("Teacher model not set")
        return self.teacher_model.inference(x)

    def set_teacher_model(self, teacher_model: nn.Module):
        self.teacher_model = teacher_model
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def update_ewc(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        max_percentage: float = 0.3,
    ) -> None:
        try:
            self.ewc_manager.compute_fisher_information(
                model=self,
                dataloader=dataloader,
                device=device,
                max_percentage=max_percentage,
            )
            self.ewc_manager.store_optimal_params(self)
            self.ewc_manager.update_ewc_losses(self.loss_fn)
        except Exception as e:
            print(f"Error in EWC update: {e}")

    def get_ewc_penalty(self) -> torch.Tensor:
        return self.ewc_manager.get_ewc_penalty(self)

    def prune_model_global(self, prune_ratio: float) -> Dict[str, torch.Tensor]:
        if prune_ratio <= 0 or prune_ratio >= 1.0:
            print(f"Invalid prune_ratio: {prune_ratio}. Must be in (0, 1)")
            return {}
        weight_info = []
        for name, module in self.named_modules():
            if self._is_prunable_module(module):
                weight_flat = module.weight.data.abs().flatten()
                weight_info.append((name, module, weight_flat))
        if not weight_info:
            print("No prunable modules found")
            return {}
        all_weights = torch.cat([info[2] for info in weight_info])
        num_to_prune = int(all_weights.numel() * prune_ratio)
        if num_to_prune == 0:
            print("No weights to prune")
            return {}
        threshold = torch.kthvalue(all_weights, num_to_prune).values
        masks: Dict[str, torch.Tensor] = {}
        total_pruned = 0
        total_weights = 0
        for name, module, _ in weight_info:
            weight = module.weight.data
            mask = weight.abs() <= threshold
            masks[name] = mask.clone()
            total_pruned += mask.sum().item()
            total_weights += mask.numel()
            module.weight.data[mask] = 0.0
        self.task_masks.append(masks)

        actual_prune_ratio = total_pruned / total_weights if total_weights > 0 else 0
        print(
            f"Pruned {total_pruned}/{total_weights} weights ({actual_prune_ratio:.3f})"
        )

        return masks

    def _is_prunable_module(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "weight")
            and module.weight is not None
            and isinstance(module.weight, torch.Tensor)
            and not isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))
        )

    def reinitialize_pruned_weights(self, init_method: str = "xavier_uniform") -> None:
        if not self.task_masks:
            print("No task masks found, skipping reinitialization")
            return
        latest_mask = self.task_masks[-1]
        reinitialized_modules = 0
        for name, module in self.named_modules():
            if name not in latest_mask or not self._is_prunable_module(module):
                continue
            mask = latest_mask[name]
            non_pruned_mask = ~mask
            if not torch.any(non_pruned_mask):
                continue
            try:
                self._reinitialize_module_weights(module, non_pruned_mask, init_method)
                reinitialized_modules += 1
            except Exception as e:
                print(f"Failed to reinitialize {name}: {e}")
        print(f"Reinitialized {reinitialized_modules} modules")

    def _reinitialize_module_weights(
        self, module: nn.Module, mask: torch.Tensor, init_method: str
    ) -> None:
        with torch.no_grad():
            fan_in, fan_out = self._fan_in_out(module)
            weight = module.weight.data
            init_functions = {
                "kaiming_uniform": lambda: self._kaiming_uniform_init(
                    weight, mask, fan_in
                ),
                "kaiming_normal": lambda: self._kaiming_normal_init(
                    weight, mask, fan_in
                ),
                "xavier_uniform": lambda: self._xavier_uniform_init(
                    weight, mask, fan_in, fan_out
                ),
                "xavier_normal": lambda: self._xavier_normal_init(
                    weight, mask, fan_in, fan_out
                ),
                "normal": lambda: self._normal_init(weight, mask, std=0.01),
            }

            if init_method in init_functions:
                init_functions[init_method]()
            else:
                self._kaiming_normal_init(weight, mask, fan_in)

    def _kaiming_uniform_init(
        self, weight: torch.Tensor, mask: torch.Tensor, fan_in: int
    ) -> None:
        bound = math.sqrt(6.0 / fan_in)
        weight[mask].uniform_(-bound, bound)

    def _kaiming_normal_init(
        self, weight: torch.Tensor, mask: torch.Tensor, fan_in: int
    ) -> None:
        std = math.sqrt(2.0 / fan_in)
        weight[mask].normal_(0, std)

    def _xavier_uniform_init(
        self, weight: torch.Tensor, mask: torch.Tensor, fan_in: int, fan_out: int
    ) -> None:
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        weight[mask].uniform_(-bound, bound)

    def _xavier_normal_init(
        self, weight: torch.Tensor, mask: torch.Tensor, fan_in: int, fan_out: int
    ) -> None:
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weight[mask].normal_(0, std)

    def _normal_init(
        self, weight: torch.Tensor, mask: torch.Tensor, std: float
    ) -> None:
        weight[mask].normal_(0, std)

    def _fan_in_out(self, module: nn.Module) -> Tuple[int, int]:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            num_input_fmaps = module.in_channels
            num_output_fmaps = module.out_channels
            receptive_field_size = 1
            if module.kernel_size:
                if isinstance(module.kernel_size, (tuple, list)):
                    receptive_field_size = math.prod(module.kernel_size)
                else:
                    receptive_field_size = module.kernel_size ** len(
                        module.weight.shape[2:]
                    )
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        elif isinstance(module, nn.Linear):
            fan_in = module.in_features
            fan_out = module.out_features
        else:
            weight = module.weight
            if weight.dim() < 2:
                fan_in = fan_out = weight.numel()
            else:
                fan_in = weight.size(1)
                fan_out = weight.size(0)
        return fan_in, fan_out

    def apply_packnet_masks(self, task_id: int) -> None:
        if task_id >= len(self.task_masks):
            print(f"Warning: Task {task_id} mask not found")
            return
        masks = self.task_masks[task_id]
        applied_masks = 0
        for name, module in self.named_modules():
            if name in masks and self._is_prunable_module(module):
                try:
                    mask = masks[name]
                    module.weight.data[mask] = 0.0
                    applied_masks += 1
                except Exception as e:
                    print(f"Failed to apply mask to {name}: {e}")
        print(f"Applied {applied_masks} PackNet masks for task {task_id}")

    def get_sparsity_stats(self) -> Dict[str, float]:
        total_params = 0
        zero_params = 0
        for module in self.modules():
            if self._is_prunable_module(module):
                weight = module.weight.data
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0

        return {
            "total_params": total_params,
            "zero_params": zero_params,
            "sparsity": sparsity,
            "active_params": total_params - zero_params,
        }

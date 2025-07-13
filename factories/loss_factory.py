from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module, ABC):

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor: ...


class CrossEntropyLoss(BaseLoss):
    def __init__(
        self, weight: float = 1.0, class_weight: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(weight)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduction="mean")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets) * self.weight


class FocalLoss(BaseLoss):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Union[float, List[float]] = 0.25,
        reduction: str = "mean",
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.gamma = gamma
        self.alpha = alpha if isinstance(alpha, (list, tuple)) else [alpha, 1 - alpha]
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction=self.reduction)
        p_t = torch.exp(-ce)
        alpha_factor = torch.tensor(self.alpha, device=logits.device)[targets]
        focal = alpha_factor * (1 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            focal_loss = focal.mean()
        elif self.reduction == "sum":
            focal_loss = focal.sum()
        else:
            focal_loss = focal
        return focal_loss * self.weight


class NLLLoss(BaseLoss):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        preds = F.log_softmax(logits, dim=-1)
        return self.loss_fn(preds, targets) * self.weight


class EWCLoss(BaseLoss):

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight)

    def forward(self, model: nn.Module, **kwargs) -> torch.Tensor:
        if not hasattr(model, "get_ewc_penalty"):
            return torch.tensor(0.0, device=next(model.parameters()).device)

        penalty = model.get_ewc_penalty()
        return penalty * self.weight


class DistillationLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 4.0,
        reduction: str = "batchmean",
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kl = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction=self.reduction,
        )
        return kl * (self.temperature**2) * self.weight


class LossRegistry:
    _losses: Dict[str, Any] = {
        "crossentropy": CrossEntropyLoss,
        "focal": FocalLoss,
        "nll": NLLLoss,
        "ewc": EWCLoss,
        "distillation": DistillationLoss,
    }

    @classmethod
    def register(cls, name: str, loss_cls: Any) -> None:
        cls._losses[name.lower()] = loss_cls

    @classmethod
    def get(cls, name: str) -> Any:
        try:
            return cls._losses[name.lower()]
        except KeyError:
            raise ValueError(f"Loss '{name}' not found. Available: {list(cls._losses)}")


class LossFactory(nn.Module):
    def __init__(
        self,
        configs: List[Dict[str, Any]],
        val: bool = False,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.val = val
        self.return_dict = return_dict
        self.losses = nn.ModuleDict()

        for i, conf in enumerate(configs):
            name = conf["name"]
            key = f"{name}_{i}" if name in self.losses else name
            params = {k: v for k, v in conf.items() if k != "name"}
            loss_cls = LossRegistry.get(name)
            self.losses[key] = loss_cls(**params)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        logits = kwargs.get("logits")
        targets = kwargs.get("targets")
        if logits is None or targets is None:
            raise ValueError("Both 'logits' and 'targets' must be provided in kwargs")

        model = kwargs.get("model")
        teacher_logits = kwargs.get("teacher_logits")
        device = (
            targets.device
            if hasattr(targets, "device")
            else next(iter(self.parameters())).device
        )

        loss_dict: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=device)}

        for name, loss_fn in self.losses.items():
            loss_val = torch.tensor(0.0, device=device)

            if isinstance(loss_fn, EWCLoss):
                if model is not None:
                    loss_val = loss_fn(**kwargs)
                else:
                    continue
            elif isinstance(loss_fn, DistillationLoss):
                if teacher_logits is not None:
                    loss_val = loss_fn(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        targets=targets,
                    )
                else:
                    continue
            else:
                loss_val = loss_fn(logits=logits, targets=targets)
            if loss_val:
                loss_dict[name] = loss_val
            loss_dict["total"] += loss_val

        return loss_dict

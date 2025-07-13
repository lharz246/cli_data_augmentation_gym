import torch
from torch.optim import Optimizer
from typing import Any, Dict, Optional, Union, Tuple
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LambdaLR,
    MultiplicativeLR,
    LinearLR,
    CyclicLR,
    OneCycleLR,
)


class OptimizerFactory:
    @staticmethod
    def create(
        config: Dict[str, Any],
        params: Union[torch.Tensor, Any],
    ) -> Tuple[Optimizer, Optional[LRScheduler]]:
        opt_name = config.get("optimizer", "sgd").lower()
        lr = config.get("lr", 1e-3)
        wd = config.get("weight_decay", 0.0)
        momentum = config.get("momentum", 0.9)
        eps = config.get("eps", 1e-8)
        alpha = config.get("alpha", 0.99)
        rho = config.get("rho", 0.9)
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                params,
                lr=lr,
                alpha=alpha,
                momentum=momentum,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "adagrad":
            optimizer = torch.optim.Adagrad(
                params,
                lr=lr,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "adadelta":
            optimizer = torch.optim.Adadelta(
                params,
                lr=lr,
                rho=rho,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "adamax":
            optimizer = torch.optim.Adamax(
                params,
                lr=lr,
                weight_decay=wd,
                eps=eps,
            )
        elif opt_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=config.get("max_iter", 20),
                tolerance_grad=config.get("tol_grad", 1e-7),
                tolerance_change=config.get("tol_change", 1e-9),
                history_size=config.get("history_size", 100),
                line_search_fn=config.get("line_search_fn", None),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        scheduler = None
        sched_name = config.get("scheduler")
        if sched_name:
            sched_name = sched_name.lower()
            sp = config.get("scheduler_params", {})[0] or {}
            if sched_name == "steplr":
                scheduler = StepLR(optimizer, **sp)
            elif sched_name == "multistep":
                scheduler = MultiStepLR(optimizer, **sp)
            elif sched_name == "exp":
                scheduler = ExponentialLR(optimizer, **sp)
            elif sched_name == "cosine":
                scheduler = CosineAnnealingLR(optimizer, **sp)
            elif sched_name == "cosine_restart":
                scheduler = CosineAnnealingWarmRestarts(optimizer, **sp)
            elif sched_name == "plateau":
                scheduler = ReduceLROnPlateau(optimizer, **sp)
            elif sched_name == "lambda":
                scheduler = LambdaLR(optimizer, **sp)
            elif sched_name == "multiplicative":
                scheduler = MultiplicativeLR(optimizer, **sp)
            elif sched_name == "linear":
                scheduler = LinearLR(optimizer, **sp)
            elif sched_name == "cyclic":
                scheduler = CyclicLR(optimizer, **sp)
            elif sched_name == "onecycle":
                max_lr = sp.pop("max_lr", lr)
                scheduler = OneCycleLR(optimizer, max_lr=max_lr, **sp)
            else:
                raise ValueError(f"Unknown scheduler: {sched_name}")

        return optimizer, scheduler

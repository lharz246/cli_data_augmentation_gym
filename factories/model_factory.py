import typing
from typing import Optional, Callable

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import inspect
from typing import Any, Dict
from factories.loss_factory import LossFactory
from models import BaseModel, ModelRegistry


class ModelFactory:

    @staticmethod
    def create(
        model_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Dict[str, Any]],
        is_validation: bool = False,
    ) -> BaseModel:
        loss_fn = LossFactory(configs=loss_cfg, val=is_validation)
        name = model_cfg.get("name", "").lower()
        model_cls = ModelRegistry.get(name)
        sig = inspect.signature(model_cls.__init__)
        type_hints = typing.get_type_hints(model_cls.__init__)
        init_kwargs: Dict[str, Any] = {}
        combined = {**model_cfg, **data_cfg, "loss_fn": loss_fn}
        for param in list(sig.parameters.values())[1:]:
            pname = param.name
            if pname in combined:
                value = combined[pname]
                param_type = type_hints.get(pname, None)
                if param_type is not None:
                    try:
                        converted_value = param_type(value)
                        init_kwargs[pname] = converted_value
                    except (ValueError, TypeError):
                        init_kwargs[pname] = value
                else:
                    init_kwargs[pname] = value
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{pname}' for model '{name}'"
                )
        return model_cls(**init_kwargs)

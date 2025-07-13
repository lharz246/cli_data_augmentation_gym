from typing import Optional, Callable, List
import math
import torch.nn as nn
from .layers import (
    init_weights,
    LayerNormWithGating,
    IncrementalResidualBlock,
)
from .base_model import BaseModel, ModelRegistry


@ModelRegistry.register("mlp")
class MLPClassifier(BaseModel):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        num_classes: int = 2,
        embed_dim: int = 2048,
        dropout_rate: float = 0.3,
        loss_fn: Callable = None,
    ):
        super().__init__(loss_fn)
        start_power = math.ceil(math.log2(input_dim)) + 1
        max_power = int(math.log2(embed_dim))
        min_power = 5
        powers = list(range(start_power, max_power + 1)) + list(
            range(max_power, min_power - 1, -1)
        )
        hidden_dims = [2**p for p in powers[:depth]]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(),
                    LayerNormWithGating(hidden_dim),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x, targets, labels, **kwargs):
        logits = self.model(x)
        output = {"logits": logits}
        params = {
            "logits": logits,
            "targets": targets,
            "labels": labels,
            "model": self,
        }
        teacher_logits = kwargs.get("teacher_logits", None)
        if teacher_logits is not None:
            params.update(
                {
                    "teacher_logits": teacher_logits,
                }
            )
        output["loss"] = self.loss_fn(**params)
        return output

    def inference(self, x):
        return self.model(x)


@ModelRegistry.register("resmlp")
class ResMLPClassifier(BaseModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        embed_dim: int = 2048,
        depth: int = 8,
        width_schedule: str = "bottleneck",
        bottleneck_factor: float = 0.5,
        dropout: float = 0.1,
        residual_strength: float = 1.0,
        class_0_preservation: bool = True,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(loss_fn=loss_fn)
        self.class_0_preservation = class_0_preservation
        self.residual_strength = residual_strength
        start_power = math.ceil(math.log2(input_dim)) + 1
        max_power = int(math.log2(embed_dim))
        base_width = 2 ** ((start_power + max_power) // 2)
        base_width = max(base_width, 512)
        if width_schedule == "bottleneck":
            expand_layers = depth // 3
            bottleneck_layers = depth // 3
            contract_layers = depth - expand_layers - bottleneck_layers
            bottleneck_width = int(base_width * bottleneck_factor)
            max_width = min(embed_dim, base_width * 2)
            widths = []
            layer_idx = 0
            for i in range(expand_layers):
                progress = i / (expand_layers - 1) if expand_layers > 1 else 0
                width = int(base_width + (max_width - base_width) * progress)
                widths.append(max(width, 512))
                layer_idx += 1
            for i in range(bottleneck_layers):
                widths.append(bottleneck_width)
                layer_idx += 1
            for i in range(contract_layers):
                progress = i / (contract_layers - 1) if contract_layers > 1 else 0
                width = int(
                    bottleneck_width + (base_width - bottleneck_width) * progress
                )
                widths.append(max(width, 512))
                layer_idx += 1
        elif width_schedule == "stable":
            widths = [base_width] * depth
        elif width_schedule == "expand":
            widths = []
            for i in range(depth):
                progress = i / (depth - 1) if depth > 1 else 0
                width = int(base_width + (embed_dim - base_width) * progress)
                widths.append(max(width, 256))
        else:
            raise ValueError(f"Unknown width_schedule: {width_schedule}")
        layers: List[nn.Module] = []
        layers.extend(
            [
                nn.Linear(input_dim, widths[0]),
                nn.ReLU(),
                nn.LayerNorm(widths[0]) if class_0_preservation else nn.Identity(),
            ]
        )
        prev_dim = widths[0]
        for i in range(depth):
            current_width = widths[i]
            if current_width != prev_dim:
                layers.append(nn.Linear(prev_dim, current_width))
                if class_0_preservation:
                    layers.append(nn.LayerNorm(current_width))
                prev_dim = current_width

            layers.append(
                IncrementalResidualBlock(
                    current_width,
                    dropout,
                    residual_strength=residual_strength,
                    preserve_features=class_0_preservation and i < depth // 2,
                )
            )

        if class_0_preservation:
            layers.extend(
                [
                    nn.LayerNorm(prev_dim),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.ReLU(),
                    nn.Linear(prev_dim // 2, num_classes),
                ]
            )
        else:
            layers.extend([nn.LayerNorm(prev_dim), nn.Linear(prev_dim, num_classes)])
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x, targets, labels, **kwargs):
        logits = self.model(x)
        output = {"logits": logits}
        params = {
            "logits": logits,
            "targets": targets,
            "labels": labels,
            "model": self,
        }
        teacher_logits = kwargs.get("teacher_logits", None)
        if teacher_logits is not None:
            params.update({"teacher_logits": teacher_logits})
        output["loss"] = self.loss_fn(**params)
        return output

    def inference(self, x):
        return self.model(x)

    def get_feature_representations(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = len(self.model) // 2

        features = x
        for i, layer in enumerate(self.model):
            features = layer(features)
            if i == layer_idx:
                return features
        return features

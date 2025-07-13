from utils.logger import get_logger
from .layers import (
    Sparsemax,
    FeatureTransformer,
)
from .base_model import BaseModel, ModelRegistry
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

log = get_logger()
logger = log.logger


@ModelRegistry.register("tabnet")
class TabNetClassifier(BaseModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        embed_dim: int = 64,
        n_steps: int = 3,
        relaxation: float = 1.0,
        sparsity_loss_weight: float = 1e-4,
        loss_fn: Optional[Callable] = None,
        teacher_model: Optional[nn.Module] = None,
        virtual_batch_size: int = 256,
        class_0_weight: float = 1.0,
    ) -> None:
        super().__init__(loss_fn=loss_fn)
        self.teacher_model = teacher_model
        self.embed_dim = embed_dim
        self.n_steps = n_steps
        self.gamma = relaxation
        self.alpha = sparsity_loss_weight
        self.virtual_batch_size = virtual_batch_size
        self.class_0_weight = class_0_weight
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.feature_transformers = nn.ModuleList(
            [FeatureTransformer(embed_dim) for _ in range(n_steps)]
        )
        self.attention_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim // 2, input_dim),
                    Sparsemax(),
                )
                for _ in range(n_steps)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.previous_params = None
        self.fisher_information = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_features(self, x):
        x_norm = self.input_norm(x)
        features = F.relu(self.input_proj(x_norm))
        step_outputs = []
        total_entropy = torch.tensor(0.0, device=x.device)
        masks = []
        for step in range(self.n_steps):
            mask = self.attention_blocks[step](features)
            masks.append(mask)
            masked_input = mask * x
            masked_features = F.relu(self.input_proj(masked_input))
            step_output = self.feature_transformers[step](masked_features)
            step_outputs.append(step_output)
            entropy = -torch.sum(mask * torch.log(torch.clamp(mask, min=1e-15)), dim=1)
            total_entropy = total_entropy + torch.mean(entropy)
        aggregated_features = torch.stack(step_outputs, dim=1).mean(dim=1)
        return aggregated_features, total_entropy, masks

    def forward(self, x, targets, labels, **kwargs):
        features, total_entropy, masks = self._forward_features(x)
        features = self.output_norm(features)
        logits = self.classifier(features)
        output = {
            "logits": logits,
            "masks": masks,
            "entropy": total_entropy,
        }
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
        loss = self.loss_fn(**params)
        sparsity_loss = self.alpha * total_entropy / self.n_steps
        total_loss = loss.get("total", loss.get("loss", 0)) + sparsity_loss
        loss["sparsity"] = sparsity_loss
        loss["total"] = total_loss
        output["loss"] = loss
        return output

    def inference(self, x):
        features, _, _ = self._forward_features(x)
        features = self.output_norm(features)
        return self.classifier(features)

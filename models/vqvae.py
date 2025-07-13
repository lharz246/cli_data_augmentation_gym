from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, Optional, Tuple, Any, List
from .base_model import BaseModel, ModelRegistry
from .layers import ResidualBlock1D, FastLayerNorm


class ReplayBuffer:

    def __init__(
        self,
        capacity_per_label: int,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        sampling_strategy: str = "balanced",
    ):
        if sampling_strategy not in {"balanced", "temporal"}:
            raise ValueError("sampling_strategy must be 'balanced' or 'temporal'")
        self.capacity_per_label = capacity_per_label
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.sampling_strategy = sampling_strategy
        self._buffers: Dict[int, deque] = {}
        self._current_label: Optional[int] = None
        self._global_step: int = 0

    def _ensure_deque(self, label: int) -> deque:
        if label not in self._buffers:
            self._buffers[label] = deque(maxlen=self.capacity_per_label)
        return self._buffers[label]

    def _label_weights(self, candidate_labels: List[int]) -> torch.Tensor:
        if self.sampling_strategy == "balanced":
            return torch.ones(len(candidate_labels), device=self.device)
        inv = [1.0 / float(lbl) for lbl in candidate_labels]
        return torch.tensor(inv, device=self.device)

    def set_training_context(self, current_label: int) -> None:
        self._current_label = current_label

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        self._global_step += 1
        mask = labels != 0
        if not bool(mask.any()):
            return

        x, y, labels = x[mask], y[mask], labels[mask]
        logits = logits[mask] if logits is not None else None

        for idx in range(len(labels)):
            lbl = int(labels[idx].item())
            dq = self._ensure_deque(lbl)
            tup = (
                x[idx].detach().to(self.device),
                y[idx].detach().to(self.device),
                labels[idx].detach().to(self.device),
                (
                    logits[idx].detach().to(self.device)
                    if logits is not None
                    else torch.zeros(self.num_classes, device=self.device)
                ),
                self._global_step,
            )
            dq.append(tup)

    def should_use_replay(self, batch_labels: torch.Tensor) -> bool:
        if not self._buffers:
            return False
        batch_set = set(int(l) for l in batch_labels.unique().tolist() if l != 0)
        stored_set = set(self._buffers.keys())
        return bool(stored_set - batch_set)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[Optional[torch.Tensor], ...]:
        if not self._buffers:
            return (None,) * 4

        candidate_labels = [
            lbl
            for lbl in self._buffers
            if len(self._buffers[lbl]) > 0 and lbl != self._current_label
        ]
        if not candidate_labels:
            return (None,) * 4

        weights = self._label_weights(candidate_labels)
        probs = weights / weights.sum()
        label_tensor = torch.tensor(candidate_labels, device=self.device)
        chosen_labels = label_tensor[
            torch.multinomial(probs, batch_size, replacement=True)
        ]

        xs, ys, ls, lgts = [], [], [], []
        for lbl in chosen_labels.tolist():
            sample = random.choice(self._buffers[lbl])  # beliebiges Sample aus deque
            x_i, y_i, l_i, logits_i, _ = sample
            xs.append(x_i)
            ys.append(y_i)
            ls.append(l_i)
            lgts.append(logits_i)

        return (
            torch.stack(xs),
            torch.stack(ys),
            torch.stack(ls),
            torch.stack(lgts),
        )

    def is_empty(self) -> bool:
        return not any(self._buffers.values())

    def clear(self) -> None:
        self._buffers.clear()
        self._current_label = None
        self._global_step = 0

    def stats(self) -> Dict[str, int]:
        return {lbl: len(dq) for lbl, dq in self._buffers.items()}


class VectorQuantizer(nn.Module):

    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embedding.data)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        input_sq = torch.sum(flat_input**2, dim=1, keepdim=True)
        embed_sq = torch.sum(self.embedding**2, dim=1)
        dot_product = torch.matmul(flat_input, self.embedding.t())
        distances = input_sq + embed_sq - 2 * dot_product
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embedding)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = flat_input + (quantized - flat_input).detach()
        quantized = quantized.view(input_shape)
        batch_size = input_shape[0]
        encoding_indices = encoding_indices.view(batch_size, -1)
        return quantized, loss, encoding_indices


@ModelRegistry.register("vqvae")
class VQVAEClassifier(BaseModel):

    def __init__(
        self,
        input_dim: int = 50,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        replay_loss_weight: float = 0.5,
        dropout_rate: float = 0.1,
        loss_fn: Optional[Any] = None,
        capacity_per_label: int = 200,
        replay_sample_size: int = 32,
        replay_update_frequency: int = 1,
        use_replay_buffer: bool = True,
        sampling_strategy: str = "balanced",
        temporal_decay: float = 0.9,
        min_samples_per_label: int = 5,
        use_conv_groups: int = 1,
        use_efficient_attention: bool = True,
    ):
        super().__init__(loss_fn)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.vq_loss_weight = vq_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.replay_loss_weight = replay_loss_weight
        self.conv_length = input_dim
        self.use_replay_buffer = use_replay_buffer
        self.replay_sample_size = replay_sample_size
        self.replay_update_frequency = replay_update_frequency
        self.update_counter = 0
        self.current_training_step = 0
        self.replay_buffer = None
        if self.use_replay_buffer:
            self.capacity_per_label = capacity_per_label
            self.sampling_strategy = sampling_strategy
            self.temporal_decay = temporal_decay
            self.min_samples_per_label = min_samples_per_label
            self._init_replay_buffer(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim * self.conv_length),
        )
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
                groups=use_conv_groups,
            ),
            nn.GroupNorm(min(32, embedding_dim // 4), embedding_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock1D(
                    embedding_dim,
                    kernel_size=3,
                    use_attention=(use_efficient_attention and i == num_res_blocks - 1),
                    groups=use_conv_groups,
                )
                for i in range(num_res_blocks)
            ]
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
                groups=use_conv_groups,
            ),
            nn.GroupNorm(min(32, embedding_dim // 4), embedding_dim),
            nn.GELU(),
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(embedding_dim * self.conv_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * self.conv_length, hidden_dim),
            nn.GELU(),
            FastLayerNorm(hidden_dim, use_gating=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_replay_buffer(self, device: torch.device) -> None:
        self.replay_buffer = ReplayBuffer(
            capacity_per_label=self.capacity_per_label,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            device=device,
            sampling_strategy=self.sampling_strategy,
        )

    def set_training_step(self, step: int, training_label: int):
        self.current_training_step = step
        if self.replay_buffer is not None:
            self.replay_buffer.set_training_context(training_label, step)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self.use_replay_buffer and self.training:
            non_zero = labels[labels != 0]
            current_lbl = int(non_zero[0].item()) if non_zero.numel() else None
            self.replay_buffer.set_training_context(current_lbl)
        batch_size = x.shape[0]
        # Encode
        z_e = self.encode(x)
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        # Vector quantization
        z_q_flat, vq_loss, encoding_indices = self.vq(z_e_flat)
        z_q = z_q_flat.view(batch_size, self.conv_length, self.embedding_dim)
        z_q = z_q.permute(0, 2, 1).contiguous()
        # Decode and classify
        x_recon = self.decode(z_q)
        z_q_for_class = z_q.view(batch_size, -1)
        logits = self.classifier(z_q_for_class)
        # Compute losses
        reconstruction_loss = F.mse_loss(x_recon, x)
        params = {
            "logits": logits,
            "targets": targets,
            "labels": labels,
            "model": self,
        }
        teacher_logits = kwargs.get("teacher_logits", None)
        if teacher_logits is not None:
            params["teacher_logits"] = teacher_logits
        classification_loss = self.loss_fn(**params)
        loss = {
            "vq_loss": vq_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss["total"],
        }
        replay_loss = torch.tensor(0.0, device=x.device)
        if self.use_replay_buffer and self.training:
            with torch.no_grad():
                store_logits = teacher_logits if teacher_logits is not None else logits
                self.replay_buffer.add(x, targets, labels, store_logits.detach())
            only_first_label = labels[labels > 0].unique().numel() == 1 and (
                labels[labels > 0].unique().item() == 1
            )
            if only_first_label:
                self.update_counter += 1
            else:
                if (
                    self.update_counter % self.replay_update_frequency == 0
                    and self.replay_buffer.should_use_replay(labels)
                ):
                    avail = sum(self.replay_buffer.stats().values())
                    sample_n = min(self.replay_sample_size, avail)
                    replay_data = self.replay_buffer.sample(sample_n)
                    if replay_data[0] is not None:
                        (
                            replay_x,
                            replay_targets,
                            replay_labels,
                            replay_stored_logits,
                        ) = replay_data
                        replay_z_e = self.encode(replay_x)
                        replay_z_e_flat = (
                            replay_z_e.permute(0, 2, 1)
                            .contiguous()
                            .view(-1, self.embedding_dim)
                        )
                        replay_z_q_flat, replay_vq_loss, _ = self.vq(replay_z_e_flat)
                        replay_z_q = replay_z_q_flat.view(
                            replay_x.shape[0], self.conv_length, self.embedding_dim
                        )
                        replay_z_q = (
                            replay_z_q.permute(0, 2, 1)
                            .contiguous()
                            .view(replay_x.shape[0], -1)
                        )
                        replay_logits = self.classifier(replay_z_q)
                        replay_params = {
                            "logits": replay_logits,
                            "targets": replay_targets,
                            "labels": replay_labels,
                            "model": self,
                        }
                        if (
                            replay_stored_logits is not None
                            and replay_stored_logits.abs().sum() > 0
                        ):
                            replay_params["teacher_logits"] = replay_stored_logits
                        replay_classification_loss = self.loss_fn(**replay_params)
                        replay_loss = (
                            replay_classification_loss["total"]
                            + self.vq_loss_weight * replay_vq_loss
                        )
                self.update_counter += 1

        total_loss = (
            self.vq_loss_weight * vq_loss
            + self.reconstruction_loss_weight * reconstruction_loss
            + self.classification_loss_weight * classification_loss["total"]
            + self.replay_loss_weight * replay_loss
        )
        if replay_loss > 0:
            loss["replay_loss"] = replay_loss
        loss["total"] = total_loss
        output = {
            "logits": logits,
            "reconstruction": x_recon,
            "z_e": z_e,
            "z_q": z_q,
            "encoding_indices": encoding_indices,
            "loss": loss,
        }
        return output

    def get_replay_buffer_info(self) -> Dict[str, Any]:
        if not self.use_replay_buffer or self.replay_buffer is None:
            return {"replay_buffer_enabled": False}
        stats = self.replay_buffer.get_buffer_stats()
        stats["replay_buffer_enabled"] = True
        return stats

    def set_sampling_strategy(self, strategy: str):
        if self.replay_buffer is not None:
            self.replay_buffer.sampling_strategy = strategy

    def clear_replay_buffer(self):
        if self.use_replay_buffer and self.replay_buffer is not None:
            self.replay_buffer.clear()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        encoded = self.encoder_mlp(x)
        encoded = encoded.view(batch_size, self.embedding_dim, self.conv_length)
        encoded = self.encoder_conv(encoded)
        for block in self.res_blocks:
            encoded = block(encoded)
        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        decoded = self.decoder_conv(z)
        decoded = decoded.view(batch_size, -1)
        return self.decoder_mlp(decoded)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        z_q_flat, _, _ = self.vq(z_e_flat)
        z_q = z_q_flat.view(x.shape[0], self.conv_length, self.embedding_dim)
        z_q = z_q.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        return self.classifier(z_q)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_e = self.encode(x)
            z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
            z_q_flat, _, _ = self.vq(z_e_flat)
            return z_q_flat.view(x.shape[0], -1)

    def get_codebook_usage(self) -> Dict[str, float]:
        with torch.no_grad():
            embeddings = self.vq.embedding.data
            mean_norm = torch.norm(embeddings, dim=1).mean().item()
            std_norm = torch.norm(embeddings, dim=1).std().item()
            return {
                "mean_embedding_norm": mean_norm,
                "std_embedding_norm": std_norm,
                "num_embeddings": self.vq.num_embeddings,
                "embedding_dim": self.vq.embedding_dim,
            }

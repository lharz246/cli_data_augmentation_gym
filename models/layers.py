import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def closest_power_of_2(x):
    if x < 1:
        return 1
    exponent = round(math.log2(x))
    return 2**exponent


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(
            m.weight, gain=torch.nn.init.calculate_gain("linear")
        )


class TabTransformerEncoder(nn.Module):
    """Transformer encoder for tabular data"""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        ff_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.output_projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class EfficientMultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_output = torch.matmul(attn, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        return self.out_proj(attn_output)


class ResidualBlock1D(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_attention: bool = False,
        groups: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=groups
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=groups
        )
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = EfficientMultiHeadAttention(channels, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.use_attention:
            out = out.transpose(1, 2)
            out = self.attention(out)
            out = out.transpose(1, 2)
        return F.gelu(out + residual)


class FastLayerNorm(nn.Module):

    def __init__(self, dim: int, use_gating: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_gating = use_gating
        if use_gating:
            self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        if self.use_gating:
            gate = torch.sigmoid(self.gate(x_norm))
            return x_norm * gate
        return x_norm


class LayerNormWithGating(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.gate_act = nn.Sigmoid()

    def forward(self, x):
        normed = self.norm(x)
        gate = self.gate_act(self.gate(normed))
        return gate * normed


class IncrementalResidualBlock(nn.Module):

    def __init__(
        self,
        width: int,
        dropout: float = 0.1,
        residual_strength: float = 1.0,
        preserve_features: bool = False,
    ):
        super().__init__()
        self.residual_strength = residual_strength
        self.preserve_features = preserve_features
        self.transform = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout * 0.5),
        )
        self.norm = nn.LayerNorm(width)
        if preserve_features:
            self.feature_gate = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(width // 2, width),
                nn.Sigmoid(),
            )
            self.memory_strength = nn.Parameter(torch.tensor(0.8))
        else:
            self.feature_gate = None
            self.memory_strength = None

    def forward(self, x):
        residual = x
        out = self.transform(x)
        if self.feature_gate is not None:
            gate = self.feature_gate(x)
            memory_factor = torch.sigmoid(self.memory_strength)
            out = out * gate + residual * (1 - gate) * memory_factor
        else:
            out = out + self.residual_strength * residual

        return self.norm(out)


class CGLUBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CGLUBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        x_conv = self.conv(x)
        A, B = x_conv.chunk(2, dim=1)
        return A * torch.sigmoid(B)


class HighwayLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_out = F.relu(self.H(x))
        T_out = torch.sigmoid(self.T(x))
        C_out = 1.0 - T_out  # carry gate
        return self.dropout(H_out * T_out + x * C_out)


class Sparsemax(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # (B, D)
        dims = logits.size(-1)
        z_sorted, _ = torch.sort(logits, dim=-1, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=-1)
        k = torch.arange(1, dims + 1, device=logits.device)
        k_z = 1 + k * z_sorted > z_cumsum
        k_max = k_z.sum(dim=-1, keepdim=True)
        tau = (z_cumsum.gather(-1, k_max - 1) - 1) / k_max
        return torch.clamp(logits - tau, min=0)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class GLU_Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        self.apply(init_weights)

    def forward(self, x):
        x = self.bn(self.fc(x))
        A, B = x.chunk(2, dim=1)
        return A * torch.sigmoid(B)


class FeatureTransformer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.layers(x) + x

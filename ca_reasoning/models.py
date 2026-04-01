from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


VALID_RULE_SHARING = {"shared", "unshared"}
VALID_GRID_LAYOUTS = {"row_major_2d", "tape_1d"}
VALID_NEIGHBORHOODS = {"3x3_masked", "5x5_masked", "3x3_dilated"}
VALID_POSITION_MODES = {"seq_only", "grid_only", "both", "none"}


def _grid_shape(sequence_len: int, grid_layout: str) -> tuple[int, int]:
    if grid_layout == "row_major_2d":
        width = math.ceil(math.sqrt(sequence_len))
        height = math.ceil(sequence_len / width)
        return height, width
    if grid_layout == "tape_1d":
        return 1, sequence_len
    raise ValueError(f"Unsupported grid layout: {grid_layout}")


def _rule_spec(neighborhood: str, grid_layout: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if grid_layout == "row_major_2d":
        if neighborhood == "3x3_masked":
            return (3, 3), (1, 1)
        if neighborhood == "5x5_masked":
            return (5, 5), (1, 1)
        if neighborhood == "3x3_dilated":
            return (3, 3), (2, 2)
    elif grid_layout == "tape_1d":
        if neighborhood == "3x3_masked":
            return (1, 3), (1, 1)
        if neighborhood == "5x5_masked":
            return (1, 5), (1, 1)
        if neighborhood == "3x3_dilated":
            return (1, 3), (1, 2)
    raise ValueError(
        f"Unsupported neighborhood/layout pair: {neighborhood} on {grid_layout}."
    )


def _build_causal_kernel_mask(
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    grid_width: int,
) -> torch.Tensor:
    kernel_height, kernel_width = kernel_size
    dilation_height, dilation_width = dilation
    center_row = kernel_height // 2
    center_col = kernel_width // 2
    mask = torch.zeros(kernel_height, kernel_width, dtype=torch.float32)

    for row in range(kernel_height):
        for col in range(kernel_width):
            delta_row = (row - center_row) * dilation_height
            delta_col = (col - center_col) * dilation_width
            relative_index = delta_row * grid_width + delta_col
            if relative_index <= 0:
                mask[row, col] = 1.0

    return mask.view(1, 1, kernel_height, kernel_width)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ff = FeedForward(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerNextTokenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.position_embed = nn.Embedding(context_len, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.context_len:
            raise ValueError(
                f"Expected <= {self.context_len} tokens, got {tokens.size(1)}."
            )

        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embed(tokens) + self.position_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)

    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)[:, -1, :]


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.permute(0, 2, 3, 1)
        grid = self.norm(grid)
        return grid.permute(0, 3, 1, 2)


class CausalNeuralCellularRule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        grid_layout: str,
        neighborhood: str,
        grid_width: int,
    ) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm(hidden_dim)
        kernel_size, dilation = _rule_spec(
            neighborhood=neighborhood,
            grid_layout=grid_layout,
        )
        padding = (
            dilation[0] * (kernel_size[0] // 2),
            dilation[1] * (kernel_size[1] // 2),
        )
        self.perception = nn.Conv2d(
            hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.update = nn.Conv2d(4 * hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.padding = padding
        self.dilation = dilation
        self.register_buffer(
            "kernel_mask",
            _build_causal_kernel_mask(
                kernel_size=kernel_size,
                dilation=dilation,
                grid_width=grid_width,
            ),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        update = self.norm(grid)
        weight = self.perception.weight * self.kernel_mask
        update = F.conv2d(
            update,
            weight=weight,
            bias=self.perception.bias,
            padding=self.padding,
            dilation=self.dilation,
        )
        update = F.gelu(update)
        update = self.update(update)
        update = self.dropout(update)
        return grid + update


class CellularAutomataLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        hidden_dim: int = 128,
        num_steps: int = 8,
        dropout: float = 0.1,
        rule_sharing: str = "unshared",
        grid_layout: str = "row_major_2d",
        neighborhood: str = "3x3_masked",
        position_mode: str = "both",
    ) -> None:
        super().__init__()
        if rule_sharing not in VALID_RULE_SHARING:
            raise ValueError(f"Unsupported rule_sharing: {rule_sharing}")
        if grid_layout not in VALID_GRID_LAYOUTS:
            raise ValueError(f"Unsupported grid_layout: {grid_layout}")
        if neighborhood not in VALID_NEIGHBORHOODS:
            raise ValueError(f"Unsupported neighborhood: {neighborhood}")
        if position_mode not in VALID_POSITION_MODES:
            raise ValueError(f"Unsupported position_mode: {position_mode}")

        self.context_len = context_len
        self.num_steps = num_steps
        self.rule_sharing = rule_sharing
        self.grid_layout = grid_layout
        self.neighborhood = neighborhood
        self.position_mode = position_mode
        self.grid_height, self.grid_width = _grid_shape(context_len, grid_layout=grid_layout)
        self.grid_capacity = self.grid_height * self.grid_width

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.sequence_position_embed = nn.Embedding(context_len, hidden_dim)
        self.grid_position_embed = nn.Parameter(
            torch.zeros(1, self.grid_height, self.grid_width, hidden_dim)
        )
        num_rule_modules = 1 if rule_sharing == "shared" else num_steps
        self.rules = nn.ModuleList(
            [
                CausalNeuralCellularRule(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    grid_layout=grid_layout,
                    neighborhood=neighborhood,
                    grid_width=self.grid_width,
                )
                for _ in range(num_rule_modules)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _pack_tokens_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        padded = x.new_zeros(batch_size, self.grid_capacity, hidden_dim)
        padded[:, :seq_len, :] = x
        grid = padded.view(batch_size, self.grid_height, self.grid_width, hidden_dim)
        return grid.permute(0, 3, 1, 2)

    def _unpack_grid_to_tokens(self, grid: torch.Tensor, seq_len: int) -> torch.Tensor:
        tokens = grid.permute(0, 2, 3, 1).reshape(grid.size(0), self.grid_capacity, -1)
        return tokens[:, :seq_len, :]

    def _apply_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embed(tokens)
        if self.position_mode in {"seq_only", "both"}:
            x = x + self.sequence_position_embed(positions)
        return x

    def _apply_grid_positions(self, grid: torch.Tensor) -> torch.Tensor:
        if self.position_mode in {"grid_only", "both"}:
            return grid + self.grid_position_embed.permute(0, 3, 1, 2)
        return grid

    def forward(
        self,
        tokens: torch.Tensor,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if tokens.size(1) > self.context_len:
            raise ValueError(
                f"Expected <= {self.context_len} tokens, got {tokens.size(1)}."
            )

        x = self._apply_positions(tokens)
        grid = self._pack_tokens_to_grid(x)
        grid = self._apply_grid_positions(grid)

        trace: list[torch.Tensor] | None = None
        if return_trace:
            trace = [grid.detach().clone()]

        for step in range(self.num_steps):
            rule_index = 0 if self.rule_sharing == "shared" else step
            grid = self.rules[rule_index](grid)
            if return_trace:
                trace.append(grid.detach().clone())

        x = self._unpack_grid_to_tokens(grid, seq_len=tokens.size(1))
        x = self.final_norm(x)
        logits = self.head(x)

        if return_trace:
            return logits, trace or []
        return logits

    def next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)[:, -1, :]

"""Small stacked linear-attention LM for the MQAR testbed.

Deliberately minimal. Plan Stage 2 fixes the architecture once and holds it
constant for every later stage, so the only thing that should vary across runs
is ``LMConfig.mode`` (and, later, the mechanism under test).

There are no positional embeddings: the recurrence is order-dependent by
construction, which is the usual choice for this architecture family.

Output convention: logits at position ``p`` are scored directly against the
answer for a query at ``p``, with no next-token shift. The data generator does
not write answers into the sequence, so a shifted objective would be
predicting filler. See ``lamr.data.mqar`` for why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from lamr.layers.linear_attn import LinearAttentionLayer


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mode: str = "gated_delta"
    chunk_size: int = 64
    backend: str = "chunked"
    use_short_conv: bool = True
    conv_size: int = 4
    mlp_ratio: float = 4.0
    alpha_init_bias: float = 6.0
    tie_embeddings: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: float):
        super().__init__()
        hidden = int(d_model * ratio)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm residual block: attention, then MLP."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = LinearAttentionLayer(
            cfg.d_model,
            cfg.num_heads,
            mode=cfg.mode,
            chunk_size=cfg.chunk_size,
            backend=cfg.backend,
            use_short_conv=cfg.use_short_conv,
            conv_size=cfg.conv_size,
            alpha_init_bias=cfg.alpha_init_bias,
        )
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LinearAttentionLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Parameter count, used to hold the budget fixed across mechanisms.

        Tied embeddings are counted once, since ``named_parameters`` already
        deduplicates shared tensors.
        """
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)

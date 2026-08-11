"""Linear-attention layer with a selectable update rule.

The three Stage 2 baselines differ only in ``mode``, so they share projections,
head splitting, normalization and the output path. Anything that differs
between them beyond the update rule would confound the comparison the plan
insists on ("always compare against the same fixed baseline").

Later stages add modes here rather than forking the layer: Stage 3 swaps the
feature map applied to ``q``/``k``, Stage 4 adds a clustering write-gate, Stage
6 splits heads across two rules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lamr.layers.chunked import (
    chunk_delta_rule,
    chunk_gated_delta_rule,
    chunk_linear_attn,
)
from lamr.layers.recurrent import (
    delta_rule_recurrent,
    elu_plus_one,
    gated_delta_rule_recurrent,
    linear_attn_recurrent,
)

MODES = ("linear", "delta", "gated_delta")
BACKENDS = ("chunked", "sequential", "fla")


class ShortConv(nn.Module):
    """Causal depthwise conv over the time axis.

    Part of real Gated DeltaNet, and kept here so the baseline is the actual
    architecture rather than a weakened version of it. It supplies local
    mixing only -- it cannot substitute for state, since MQAR queries land far
    from the pair that answers them.
    """

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return F.silu(self.conv(y).transpose(1, 2))


class LinearAttentionLayer(nn.Module):
    """Multi-head linear attention.

    Args:
        d_model: model width.
        num_heads: heads; ``d_model`` must divide evenly.
        mode: ``"linear"`` (no delta, the floor), ``"delta"`` (DeltaNet), or
            ``"gated_delta"`` (Gated DeltaNet, the reference baseline).
        chunk_size: performance knob only; results are chunk-invariant.
        backend: ``"chunked"`` for speed, ``"sequential"`` for the reference
            path. They must agree -- see ``tests/test_chunked_parity.py``.
        alpha_init_bias: bias added before the decay sigmoid. Decay compounds
            over the sequence, so this needs to start close to 1 or the state
            is erased long before the queries arrive: the default 6.0 gives
            ``alpha ~ 0.9975``, retaining about half the signal over 256 tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mode: str = "gated_delta",
        chunk_size: int = 64,
        backend: str = "chunked",
        use_short_conv: bool = True,
        conv_size: int = 4,
        alpha_init_bias: float = 6.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")

        self.mode = mode
        self.backend = backend
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)

        self.short_conv = (
            nn.ModuleDict(
                {
                    "q": ShortConv(d_model, conv_size),
                    "k": ShortConv(d_model, conv_size),
                    "v": ShortConv(d_model, conv_size),
                }
            )
            if use_short_conv
            else None
        )

        if mode in ("delta", "gated_delta"):
            self.beta_proj = nn.Linear(d_model, num_heads, bias=True)
            nn.init.zeros_(self.beta_proj.bias)  # beta starts at 0.5
        if mode == "gated_delta":
            self.alpha_proj = nn.Linear(d_model, num_heads, bias=True)
            nn.init.constant_(self.alpha_proj.bias, alpha_init_bias)

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q_in, k_in, v_in = x, x, x
        if self.short_conv is not None:
            q_in = self.short_conv["q"](x)
            k_in = self.short_conv["k"](x)
            v_in = self.short_conv["v"](x)

        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(k_in))
        v = self._split(self.v_proj(v_in))

        if self.mode == "linear":
            # The floor baseline has no within-chunk sequential dependency, so
            # it is already just matmuls -- there is nothing for a Triton kernel
            # to fuse away. "fla" therefore uses the portable chunked path.
            use_chunked = self.backend in ("chunked", "fla")
            fn = chunk_linear_attn if use_chunked else linear_attn_recurrent
            kwargs = {"chunk_size": self.chunk_size} if use_chunked else {}
            out, _ = fn(q, k, v, feature_map=elu_plus_one, normalize=True, **kwargs)
        else:
            # The delta update is contractive only for beta < 2/||k||^2, so keys
            # are L2-normalized and beta confined to (0, 1). Queries are
            # normalized too, matching DeltaNet.
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)

            if self.mode == "delta":
                if self.backend == "fla":
                    from lamr.layers.fla_backend import fla_delta_rule

                    out, _ = fla_delta_rule(q, k, v, beta)
                elif self.backend == "chunked":
                    out, _ = chunk_delta_rule(q, k, v, beta, chunk_size=self.chunk_size)
                else:
                    out, _ = delta_rule_recurrent(q, k, v, beta)
            else:
                alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)
                if self.backend == "fla":
                    from lamr.layers.fla_backend import fla_gated_delta_rule

                    out, _ = fla_gated_delta_rule(q, k, v, beta, alpha)
                elif self.backend == "chunked":
                    out, _ = chunk_gated_delta_rule(
                        q, k, v, beta, alpha, chunk_size=self.chunk_size
                    )
                else:
                    out, _ = gated_delta_rule_recurrent(q, k, v, beta, alpha)

        out = self.out_norm(out)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)

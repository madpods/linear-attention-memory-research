"""Chunk-parallel delta rule in plain PyTorch (plan section 15).

``flash-linear-attention`` provides this as fused Triton kernels, which need
CUDA. This module is the same algorithm in portable PyTorch: it runs on CPU,
is O(T/C) sequential steps instead of O(T), and is what makes local sweeps
viable before the HPC box is available. It is *not* a replacement for ``fla``
on GPU -- it allocates the (C x C) intermediates that fla's kernels fuse away.

Its second job is to be the staging ground for Stage 4. Cluster/pointer
write-gating has to be expressed against this within-chunk structure
eventually, and doing that in readable PyTorch first is much cheaper than
debugging it in Triton.

Derivation
----------
Write the delta rule as a rank-1 accumulation with an implicit coefficient::

    S_t = S_{t-1} + k_t u_t^T        where   u_t = beta_t (v_t - S_{t-1}^T k_t)

Substituting ``S_{t-1} = S_0 + sum_{s<t} k_s u_s^T`` makes the dependence on
earlier tokens of the *same* chunk explicit::

    u_t = beta_t ( v_t - S_0^T k_t - sum_{s<t} (k_t . k_s) u_s )

which is a triangular linear system in ``U`` (rows ``u_t``)::

    (I + diag(beta) tril(K K^T, -1)) U = diag(beta) (V - K S_0)

The matrix is unit lower triangular, so one triangular solve recovers the
whole chunk at once -- this is the UT transform / WY representation the
DeltaNet paper uses. Everything else is dense matmuls::

    O   = Q S_0 + tril(Q K^T, 0) U        # diagonal included: read follows write
    S_C = S_0 + K^T U

Gating (section 4's decay) carries a cumulative decay ``c_t = prod_{s<=t}
alpha_s`` through the same derivation. Ratios ``c_t / c_s`` are formed as
``exp(logc_t - logc_s)`` and masked to the causal triangle *before*
exponentiating, so the intermediate ``1 / c_s`` -- which overflows for small
alpha over a long chunk -- is never materialized.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

#: alpha is clamped to this floor before log; alpha=0 would be log(0) = -inf.
_ALPHA_FLOOR = 1e-12


def _pad_to_chunk(x: Tensor, chunk_size: int, value: float = 0.0) -> Tensor:
    """Right-pad the time axis (dim 2) up to a multiple of ``chunk_size``."""
    pad = (-x.shape[2]) % chunk_size
    if pad == 0:
        return x
    if x.ndim == 4:
        return F.pad(x, (0, 0, 0, pad), value=value)
    return F.pad(x, (0, pad), value=value)


def chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor | None = None,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
):
    """Chunk-parallel (gated) delta rule.

    Numerically equivalent to
    :func:`lamr.layers.recurrent.gated_delta_rule_recurrent`, and to
    :func:`~lamr.layers.recurrent.delta_rule_recurrent` when ``alpha is None``.

    Args:
        q, k, v: ``(B, H, T, D)``. Callers should L2-normalize ``k`` (see the
            contractivity note on the sequential reference).
        beta: ``(B, H, T)`` write strength.
        alpha: ``(B, H, T)`` decay in ``(0, 1]``, or ``None`` for no decay.
        chunk_size: tokens per chunk. Larger means fewer sequential steps but
            a bigger ``(C, C)`` intermediate; parity with the sequential
            reference is exact regardless.
        initial_state: ``(B, H, d_k, d_v)`` carried state, or ``None`` for zeros.

    Returns:
        ``(output, final_state)`` with output ``(B, H, T, d_v)`` and state
        ``(B, H, d_k, d_v)``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"expected (B, H, T, D) tensors, got q{tuple(q.shape)} "
            f"k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on (B, H, T)")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k must share d_k, got {q.shape[-1]}, {k.shape[-1]}")
    if beta.shape != k.shape[:3]:
        raise ValueError(
            f"beta must be (B, H, T) = {tuple(k.shape[:3])}, got {tuple(beta.shape)}"
        )
    if alpha is not None and alpha.shape != k.shape[:3]:
        raise ValueError(
            f"alpha must be (B, H, T) = {tuple(k.shape[:3])}, got {tuple(alpha.shape)}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    c = chunk_size

    log_alpha = (
        torch.zeros(b, h, t, dtype=v.dtype, device=v.device)
        if alpha is None
        else alpha.clamp_min(_ALPHA_FLOOR).log().to(v.dtype)
    )

    # Pad with beta=0 (no write) and log_alpha=0 (no decay), so padded tokens
    # are exact no-ops on the state and their outputs are simply discarded.
    qp = _pad_to_chunk(q, c)
    kp = _pad_to_chunk(k, c)
    vp = _pad_to_chunk(v, c)
    bp = _pad_to_chunk(beta, c)
    lap = _pad_to_chunk(log_alpha, c)
    n = qp.shape[2] // c

    reshape4 = lambda x, d: x.reshape(b, h, n, c, d)  # noqa: E731
    qp, kp, vp = reshape4(qp, d_k), reshape4(kp, d_k), reshape4(vp, d_v)
    bp = bp.reshape(b, h, n, c)
    lap = lap.reshape(b, h, n, c)

    logc = lap.cumsum(dim=-1)                                  # (b,h,n,c)
    # decay[t, s] = c_t / c_s, valid only for s <= t. Mask the upper triangle
    # to -inf before exp so the s > t entries become exactly 0 rather than
    # overflowing to inf and poisoning the masked matmul with NaN.
    diff = logc.unsqueeze(-1) - logc.unsqueeze(-2)             # (b,h,n,c,c)
    causal = torch.ones(c, c, dtype=torch.bool, device=v.device).tril()
    decay = diff.masked_fill(~causal, -math.inf).exp()

    # Attention-like coefficient matrices, shared across the chunk loop.
    kk = kp @ kp.transpose(-1, -2)                             # (b,h,n,c,c)
    qk = qp @ kp.transpose(-1, -2)
    a_strict = torch.tril(kk * decay, -1)
    coef = qk * decay                                          # already causal

    eye = torch.eye(c, dtype=v.dtype, device=v.device)
    tri = eye + bp.unsqueeze(-1) * a_strict                    # unit lower triangular

    # c_t * k_t and c_t * q_t: the S_0 contribution decays with the token.
    cum = logc.exp().unsqueeze(-1)
    k_vs_state = kp * cum
    q_vs_state = qp * cum
    # c_C / c_t, for carrying each chunk's writes to the chunk boundary.
    to_end = (logc[..., -1:] - logc).exp().unsqueeze(-1)

    state = (
        torch.zeros(b, h, d_k, d_v, dtype=v.dtype, device=v.device)
        if initial_state is None
        else initial_state.to(v.dtype)
    )

    outputs = []
    for i in range(n):
        rhs = bp[:, :, i].unsqueeze(-1) * (vp[:, :, i] - k_vs_state[:, :, i] @ state)
        u = torch.linalg.solve_triangular(
            tri[:, :, i], rhs, upper=False, unitriangular=True
        )
        outputs.append(q_vs_state[:, :, i] @ state + coef[:, :, i] @ u)
        decay_chunk = logc[:, :, i, -1].exp()[..., None, None]
        state = decay_chunk * state + (to_end[:, :, i] * kp[:, :, i]).transpose(-1, -2) @ u

    out = torch.cat(outputs, dim=2)[:, :, :t]
    return out, state


def chunk_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
):
    """Ungated chunk-parallel delta rule; ``alpha = 1`` throughout."""
    return chunk_gated_delta_rule(
        q, k, v, beta, None, chunk_size=chunk_size, initial_state=initial_state
    )


def chunk_linear_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    feature_map=None,
    normalize: bool = True,
    chunk_size: int = 64,
    eps: float = 1e-6,
):
    """Chunk-parallel plain linear attention -- the no-delta floor baseline.

    Matches :func:`lamr.layers.recurrent.linear_attn_recurrent`. There is no
    within-chunk sequential dependency here (that is what the delta rule
    introduces and what the triangular solve above exists to undo), so a chunk
    reduces to two matmuls::

        O = Q S_0 + tril(Q K^T, 0) V
        S = S_0 + K^T V

    The normalizer follows the same split: ``phi(q)^T z`` where the intra-chunk
    part is just the row sum of the causal score matrix.
    """
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on (B, H, T)")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    if feature_map is not None:
        q, k = feature_map(q), feature_map(k)

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    c = chunk_size

    qp, kp, vp = (_pad_to_chunk(x, c) for x in (q, k, v))
    n = qp.shape[2] // c
    qp = qp.reshape(b, h, n, c, d_k)
    kp = kp.reshape(b, h, n, c, d_k)
    vp = vp.reshape(b, h, n, c, d_v)

    causal = torch.ones(c, c, dtype=torch.bool, device=v.device).tril()
    scores = (qp @ kp.transpose(-1, -2)) * causal

    state = torch.zeros(b, h, d_k, d_v, dtype=v.dtype, device=v.device)
    z = torch.zeros(b, h, d_k, dtype=v.dtype, device=v.device)
    outputs = []

    for i in range(n):
        out = qp[:, :, i] @ state + scores[:, :, i] @ vp[:, :, i]
        if normalize:
            denom = qp[:, :, i] @ z.unsqueeze(-1) + scores[:, :, i].sum(-1, keepdim=True)
            out = out / (denom + eps)
        outputs.append(out)
        state = state + kp[:, :, i].transpose(-1, -2) @ vp[:, :, i]
        z = z + kp[:, :, i].sum(dim=-2)

    return torch.cat(outputs, dim=2)[:, :, :t], state

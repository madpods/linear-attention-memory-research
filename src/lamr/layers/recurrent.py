"""Sequential reference implementations of the linear-attention update rules.

These are the correctness ground truth for the whole project. They are plain
``for``-loops over the time axis -- O(T) sequential, no chunking, no Triton --
which is exactly what the plan's "correctness before speed" principle asks for
before anything gets expressed as a batched kernel.

They are also the only implementations that run on this workstation, since
``flash-linear-attention`` needs CUDA. When the project moves to the HPC box,
``fla``'s chunk-parallel kernels must reproduce these outputs; that parity
check is the gate for trusting the fast path, and is the reason this module
exists rather than calling ``fla`` directly everywhere.

Conventions (matching the plan, section 1 and section 11)
--------------------------------------------------------
The state ``S`` is ``(d_k, d_v)``. A read is ``v_read = k^T S``; a write is the
outer product ``k e^T``. Tensors are ``(B, H, T, D)`` -- batch, head, time,
head-dim -- and the returned state is ``(B, H, d_k, d_v)``.

Note that ``fla`` internally uses the transposed ``(d_v, d_k)`` layout for some
kernels. Keep this module in the plan's orientation and transpose at the
boundary rather than silently switching conventions mid-project.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def elu_plus_one(x: Tensor) -> Tensor:
    """``elu(x) + 1``: the original linear-attention feature map (section 11).

    Ensures non-negative read weights, mimicking softmax positivity. It does
    not expand dimension -- that is the job of the Stage 3 feature maps.
    """
    return F.elu(x) + 1.0


def _check(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"expected (B, H, T, D) tensors, got q{tuple(q.shape)} "
            f"k{tuple(k.shape)} v{tuple(v.shape)}"
        )
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on (B, H, T)")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q and k must share d_k, got {q.shape[-1]} and {k.shape[-1]}"
        )


def delta_rule_step(state: Tensor, k: Tensor, v: Tensor, beta: Tensor | float = 1.0):
    """One delta-rule write (plan section 1).

    ``v_read = k^T S``; ``e = v - v_read``; ``S <- S + beta * k e^T``.

    If the state already predicts ``v`` for this key the error vanishes and the
    write is a near no-op, which is the entire reason delta-rule architectures
    beat plain accumulation: redundant information does not consume capacity.

    Args:
        state: ``(..., d_k, d_v)``.
        k: ``(..., d_k)``.
        v: ``(..., d_v)``.
        beta: write strength, broadcastable to ``(...)``.

    Returns:
        ``(new_state, error)``.
    """
    v_read = torch.einsum("...k,...kv->...v", k, state)
    error = v - v_read
    if not isinstance(beta, Tensor):
        beta = torch.as_tensor(beta, dtype=state.dtype, device=state.device)
    write = beta[..., None, None] * torch.einsum("...k,...v->...kv", k, error)
    return state + write, error


def linear_attn_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    feature_map=elu_plus_one,
    normalize: bool = True,
    eps: float = 1e-6,
):
    """Plain linear attention -- the no-delta floor baseline (plan section 11).

    ``S_t = S_{t-1} + phi(k_t) v_t^T`` with no error correction: every token
    superimposes another outer product whether or not it adds information.

    With ``normalize=True`` the output is divided by ``phi(q)^T z`` where
    ``z = sum phi(k)``, the linear-attention analogue of the softmax
    denominator. Without it, outputs grow without bound over long sequences.
    """
    _check(q, k, v)
    if feature_map is not None:
        q, k = feature_map(q), feature_map(k)

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    state = torch.zeros(b, h, d_k, d_v, dtype=v.dtype, device=v.device)
    z = torch.zeros(b, h, d_k, dtype=v.dtype, device=v.device)
    outputs = []

    for i in range(t):
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, :, i], v[:, :, i])
        z = z + k[:, :, i]
        out = torch.einsum("bhk,bhkv->bhv", q[:, :, i], state)
        if normalize:
            denom = torch.einsum("bhk,bhk->bh", q[:, :, i], z)
            out = out / (denom[..., None] + eps)
        outputs.append(out)

    return torch.stack(outputs, dim=2), state


def delta_rule_recurrent(q: Tensor, k: Tensor, v: Tensor, beta: Tensor):
    """DeltaNet's error-corrected update (plan section 1).

    ``S_t = S_{t-1} + beta_t k_t (v_t - k_t^T S_{t-1})^T``

    Callers are expected to L2-normalize ``k`` (and usually ``q``) and to keep
    ``beta`` in ``(0, 1]``; the update is contractive only for
    ``0 < beta < 2 / ||k||^2``, so unnormalized keys can make the state
    diverge. :class:`lamr.layers.linear_attn` handles this at the module level.

    Args:
        beta: ``(B, H, T)`` write strength per token.

    Returns:
        ``(output, final_state)`` with output ``(B, H, T, d_v)``.
    """
    _check(q, k, v)
    if beta.shape != k.shape[:3]:
        raise ValueError(
            f"beta must be (B, H, T) = {tuple(k.shape[:3])}, got {tuple(beta.shape)}"
        )

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    state = torch.zeros(b, h, d_k, d_v, dtype=v.dtype, device=v.device)
    outputs = []

    for i in range(t):
        state, _ = delta_rule_step(state, k[:, :, i], v[:, :, i], beta[:, :, i])
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, i], state))

    return torch.stack(outputs, dim=2), state


def gated_delta_rule_recurrent(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor
):
    """Gated DeltaNet: decay the state, then delta-correct against the decayed
    state (the Stage 2 best-in-class baseline).

    ``S_t = alpha_t S_{t-1} + beta_t k_t (v_t - k_t^T (alpha_t S_{t-1}))^T``

    Per plan section 4, ``alpha`` does not raise the ``rank(S) <= min(d_k, d_v)``
    ceiling. It raises *effective* capacity by evicting stale patterns, which
    is eviction, not expansion -- worth remembering when reading Stage 2 curves.

    Args:
        alpha: ``(B, H, T)`` decay in ``(0, 1]``; 1.0 recovers plain DeltaNet.
    """
    _check(q, k, v)
    for name, g in (("beta", beta), ("alpha", alpha)):
        if g.shape != k.shape[:3]:
            raise ValueError(
                f"{name} must be (B, H, T) = {tuple(k.shape[:3])}, got {tuple(g.shape)}"
            )

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    state = torch.zeros(b, h, d_k, d_v, dtype=v.dtype, device=v.device)
    outputs = []

    for i in range(t):
        decayed = alpha[:, :, i][..., None, None] * state
        state, _ = delta_rule_step(decayed, k[:, :, i], v[:, :, i], beta[:, :, i])
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, i], state))

    return torch.stack(outputs, dim=2), state

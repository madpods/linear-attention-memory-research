"""Adapter for ``flash-linear-attention``'s fused Triton kernels.

Signatures here were verified against ``fla``'s source
(``fla/ops/delta_rule/chunk.py`` and ``fla/ops/gated_delta_rule/chunk.py``),
but this module has still never *executed* -- the development workstation has
no CUDA device. ``tests/test_fla_parity.py`` remains the gate, and numerical
agreement is what it establishes; reading a signature only rules out the
mechanical mistakes.

Conventions reconciled at this boundary
---------------------------------------
layout
    ``fla`` requires ``[B, T, H, D]``. The ``head_first`` argument was
    **removed** -- passing it raises a ``DeprecationWarning`` -- so there is no
    version to fall back to and no flag to negotiate. This project works in
    ``(B, H, T, D)``, so every tensor is transposed on the way in and the
    output transposed back. Getting this wrong does not raise: the kernels
    would happily consume ``(B, H, T, D)`` as though heads were timesteps and
    return confident nonsense.

``scale``
    ``fla`` defaults to ``k.shape[-1] ** -0.5``. This project applies no such
    scaling -- :class:`~lamr.layers.linear_attn.LinearAttentionLayer`
    L2-normalizes q and k instead -- so ``scale=1.0`` is passed explicitly.

decay
    The gated kernel takes ``g``, a **log**-space decay, positioned before
    ``beta`` in the signature. The sequential reference takes ``alpha``
    directly, so the conversion is ``g = log(alpha)``.

state layout
    ``fla`` returns ``[N, H, K, V]`` -- i.e. ``(batch, heads, d_k, d_v)``,
    which already matches the plan's ``(d_k, d_v)`` orientation. No transpose
    is needed. (The gated kernel exposes ``state_v_first`` to flip this; leave
    it at its default.)

``use_qk_l2norm_in_kernel`` is left ``False`` because the layer normalizes q/k
itself. Folding it into the kernel would be faster, but it is a change to make
after parity is established, not before.
"""

from __future__ import annotations

import torch
from torch import Tensor

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - depends on an optional CUDA-only dependency
    from fla.ops.delta_rule import chunk_delta_rule as _fla_delta
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_gated
except Exception as exc:  # pragma: no cover
    _fla_delta = None
    _fla_gated = None
    _IMPORT_ERROR = exc


def fla_available() -> bool:
    """True only if ``fla`` imported *and* a CUDA device exists.

    Both are required: the kernels are Triton and will not run on CPU.
    """
    return _fla_delta is not None and torch.cuda.is_available()


def unavailable_reason() -> str:
    if _fla_delta is None:
        return f"flash-linear-attention not importable: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return "no CUDA device; fla's Triton kernels are GPU-only"
    return ""


def _require() -> None:
    if not fla_available():
        raise RuntimeError(
            f"fla backend requested but unavailable: {unavailable_reason()}"
        )


def _to_fla(x: Tensor) -> Tensor:
    """``(B, H, T, ...)`` -> ``(B, T, H, ...)``."""
    return x.transpose(1, 2)


def fla_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,
):
    """DeltaNet via fla. Mirrors :func:`lamr.layers.chunked.chunk_delta_rule`.

    Args:
        q, k, v: ``(B, H, T, D)``.
        beta: ``(B, H, T)``.
        initial_state: ``(B, H, d_k, d_v)`` or None.

    Returns:
        ``(output, final_state)`` with output ``(B, H, T, d_v)`` and state
        ``(B, H, d_k, d_v)`` -- this project's layout, not fla's.
    """
    _require()
    out, state = _fla_delta(
        _to_fla(q),
        _to_fla(k),
        _to_fla(v),
        _to_fla(beta),
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
    )
    return out.transpose(1, 2), state


def fla_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    *,
    initial_state: Tensor | None = None,
):
    """Gated DeltaNet via fla.

    ``alpha`` is converted to fla's log-space ``g``. The clamp guards
    ``log(0)``: alpha is a sigmoid output so it is positive in principle, but a
    saturated one can round to zero in reduced precision.

    Note the argument order -- fla takes ``(q, k, v, g, beta, ...)`` with the
    decay *before* the write strength, the reverse of this project's own
    ``(..., beta, alpha)`` convention. They are passed by keyword to keep the
    mismatch from mattering.
    """
    _require()
    g = alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()
    out, state = _fla_gated(
        _to_fla(q),
        _to_fla(k),
        _to_fla(v),
        g=_to_fla(g),
        beta=_to_fla(beta),
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
    )
    return out.transpose(1, 2), state

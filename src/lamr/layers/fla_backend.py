"""Adapter for ``flash-linear-attention``'s fused Triton kernels.

Verified on GPU 2026-08-11: H100 80GB (MIG 4g.40gb), ``torch 2.13.0+cu130``,
``triton 3.7.1``, with ``tests/test_fla_parity.py`` passing 9/9. The three
conventions originally guessed from source were all correct; running it found a
fourth (dtype) that reading signatures had not surfaced, which is the case for
keeping the gate rather than trusting the docstring below.

The development workstation has no CUDA device, so the gate skips there. Any
change to this module is unverified until it has been re-run on the cluster --
see the HPC handoff section of CLAUDE.md.

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

dtype
    ``chunk_delta_rule`` **asserts** ``q.dtype != torch.float32`` -- "does not
    support float32. Please use bfloat16." So the kernels are cast to
    :data:`KERNEL_DTYPE` on the way in and the results cast back to the caller's
    dtype, keeping the backend interchangeable with the fp32 CPU paths at the
    interface even though the arithmetic is not.

    The gated kernel does *not* carry that assert and will run in fp32. It is
    cast anyway, deliberately: Stage 2's primary comparison is delta against
    gated_delta, and if one ran bf16 while the other ran fp32 that comparison
    would be partly about precision rather than about the update rule, which is
    the one thing it is supposed to isolate.

    This is a real numerical difference from the CPU baselines, not a formality
    -- see the note in CLAUDE.md. The sweep re-runs every ``r=0`` configuration
    on GPU, and those 15 rows are the built-in control for whether bf16 moves
    recall at all.

``use_qk_l2norm_in_kernel`` is left ``False`` because the layer normalizes q/k
itself. Folding it into the kernel would be faster, but it is a change to make
after parity is established, not before.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Dtype the fla kernels are called in. bf16 rather than fp16 because it is what
#: fla's own examples and tests use, and because the state matrix accumulates
#: over the whole sequence -- bf16's wider exponent range matters more there than
#: fp16's extra mantissa bits.
KERNEL_DTYPE = torch.bfloat16

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
    """``(B, H, T, ...)`` -> ``(B, T, H, ...)``, in the kernels' dtype."""
    return x.transpose(1, 2).to(KERNEL_DTYPE)


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
        initial_state=None if initial_state is None else initial_state.to(KERNEL_DTYPE),
        output_final_state=True,
    )
    # Back to the caller's dtype and axis order. The state is already
    # (B, H, d_k, d_v) and must not be transposed; only its dtype changes.
    return out.transpose(1, 2).to(q.dtype), state.to(q.dtype)


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
    # log() in the caller's (wider) dtype, then cast -- log of an already-bf16
    # alpha would round twice, and alpha sits close to 1 where log is steep.
    g = alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()
    out, state = _fla_gated(
        _to_fla(q),
        _to_fla(k),
        _to_fla(v),
        g=_to_fla(g),
        beta=_to_fla(beta),
        scale=1.0,
        initial_state=None if initial_state is None else initial_state.to(KERNEL_DTYPE),
        output_final_state=True,
    )
    return out.transpose(1, 2).to(q.dtype), state.to(q.dtype)

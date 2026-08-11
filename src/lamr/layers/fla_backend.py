"""Adapter for ``flash-linear-attention``'s fused Triton kernels.

**This module is unverified.** It was written on a machine with no CUDA device,
so nothing here has ever executed. Treat it as a best-effort binding whose
correctness is established by ``tests/test_fla_parity.py`` on first contact
with a GPU -- not before. If that test fails, fix this file; do not adjust the
test to pass.

Three conventions have to be reconciled, and each is a silent-wrong-answer risk
rather than a crash:

``scale``
    ``fla`` defaults to ``d_k ** -0.5`` on the query. This project applies no
    such scaling -- ``LinearAttentionLayer`` L2-normalizes q and k instead --
    so ``scale=1.0`` is passed explicitly everywhere. Forgetting this produces
    plausible-looking numbers that cannot be compared with any CPU result.

``head_first``
    Older ``fla`` took ``(B, H, T, D)``; newer versions default to
    ``(B, T, H, D)`` and expose a ``head_first`` flag. The signature is
    inspected at import rather than pinning a version, since the project needs
    to survive an ``fla`` upgrade on the cluster.

decay parameterization
    ``fla``'s gated kernels take ``g``, a *log* decay, where this project's
    sequential reference takes ``alpha`` directly. The conversion is
    ``g = log(alpha)``.

The state layout is the remaining open question. This project keeps the plan's
``(d_k, d_v)``; some ``fla`` kernels are documented as using the transpose. The
parity test checks it explicitly and reports which orientation matched.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

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


def _supports(fn: Callable[..., Any], param: str) -> bool:
    try:
        return param in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - C extensions
        return False


def _call(fn: Callable[..., Any], q, k, v, gates: dict[str, Tensor], **kwargs):
    """Invoke an fla op, converting layout to whatever this version expects.

    Inputs and outputs are this project's ``(B, H, T, D)``; the transpose to
    ``(B, T, H, D)`` happens only at this boundary.
    """
    if _supports(fn, "head_first"):
        args = [x.transpose(1, 2) for x in (q, k, v)]
        gate_args = {name: g.transpose(1, 2) for name, g in gates.items()}
        out, state = fn(*args, **gate_args, head_first=False, **kwargs)
        return out.transpose(1, 2), state
    out, state = fn(q, k, v, **gates, **kwargs)
    return out, state


def fla_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,
):
    """DeltaNet via fla. Mirrors :func:`lamr.layers.chunked.chunk_delta_rule`."""
    _require()
    return _call(
        _fla_delta,
        q,
        k,
        v,
        {"beta": beta},
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
    )


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

    ``alpha`` is converted to fla's log-decay ``g``. The clamp guards
    ``log(0)``; alpha is a sigmoid output so it is positive in practice, but a
    saturated one can round to zero in bf16.
    """
    _require()
    g = alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()
    return _call(
        _fla_gated,
        q,
        k,
        v,
        {"beta": beta, "g": g},
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
    )

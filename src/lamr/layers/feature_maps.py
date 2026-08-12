"""Kernel feature maps phi(x) for Stage 3 (plan section 17).

A feature map ``phi: R^d_k -> R^d_phi`` with ``d_phi > d_k`` widens what the state
matrix indexes: the write becomes ``S += phi(k) v^T``, so ``S`` is ``d_phi x d_v``
and the rank ceiling ``min(d_k, d_v)`` becomes ``min(d_phi, d_v)``. That is the
one mechanism in the plan that raises the ceiling directly rather than economising
on what gets written.

**These are parameter-free on purpose.** Section 17's claim is that phi buys
capacity "without inflating the base vector width used elsewhere in the model", so
the projections stay ``d_model x d_model`` and the parameter count does not move.
What does grow is the *state* (``d_phi x d_v``) and the per-token cost of touching
it -- section 17 calls this out as a direct trade of compute for capacity, not a
free lunch.

The control that decides whether phi is worth anything
------------------------------------------------------
Matched parameter count is automatic here and therefore proves nothing on its
own. The comparison that matters is at **matched state size**: ``d_k=16`` with a
4x feature map gives the same ``64 x d_v`` state as plain ``d_k=64``, and the
latter is reachable just by widening the projections. If plain ``d_k=64`` matches
or beats the feature map at equal state, then phi's only advantage is parameter
count, which is not what section 17 claims. Stage 3 must run that arm.

Ordering against L2 normalization
---------------------------------
The delta rule is contractive only for ``beta < 2/||k||^2``, which is why the
layer L2-normalizes k and confines beta to (0, 1). So phi is applied **before**
normalization -- ``k -> phi(k) -> phi(k)/||phi(k)||`` -- keeping ``||k|| = 1`` and
the stability argument intact. Normalizing first and expanding after would leave
``||phi(k)||`` unconstrained and silently break it.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

FeatureMap = Callable[[Tensor], Tensor]

#: Names accepted by :func:`make_feature_map`.
FEATURE_MAPS = ("identity", "elu", "relu", "dpfp1", "dpfp2", "dpfp3", "dpfp4")


def identity(x: Tensor) -> Tensor:
    """No feature map. The control arm: ``d_phi == d_k``."""
    return x


def elu_plus_one(x: Tensor) -> Tensor:
    """``elu(x) + 1``, the standard positive map from Katharopoulos et al.

    Non-negative and shape-preserving, so ``d_phi == d_k``: it cannot raise the
    ceiling and is here to separate "positivity helps" from "expansion helps".
    Linear attention needs the non-negativity for its denominator; the delta rule
    does not, so for the delta modes this is close to a second control.

    **Not strictly positive in practice.** ``elu(x)`` rounds to exactly ``-1``
    once ``exp(x)`` falls below the dtype's epsilon, so ``elu(x) + 1`` underflows
    to exactly ``0`` below x ~ -17.25 in fp32 but already below **x ~ -6.0 in
    bf16** -- and the GPU path runs bf16, because fla's delta kernel rejects
    fp32. At activation scale 1 nothing is zeroed; at scale 3 about 2% of
    coordinates are, and at scale 5 about 10%. So on GPU this map drifts toward
    ``relu`` as the input scale grows, which is worth knowing when reading it as
    a *control* against DPFP -- the two stop being independent arms.
    ``chunk_linear_attn`` adds ``eps=1e-6`` to its denominator, so the zeros are
    harmless there; nothing else relies on strict positivity.
    """
    return F.elu(x) + 1.0


def relu(x: Tensor) -> Tensor:
    """Non-negative, shape-preserving, and *lossy* -- half the signal is zeroed.

    Included because DPFP is built from ``relu`` pieces, so a DPFP gain needs to
    be shown to come from the expansion rather than from rectification alone.
    """
    return F.relu(x)


def dpfp(x: Tensor, nu: int = 1) -> Tensor:
    """Deterministic Parameter-Free Projection (Schlag et al. 2021), ``d_phi = 2*d_k*nu``.

    ``y = [relu(x), relu(-x)]`` doubles the width and is *half sparse* -- exactly
    one of each coordinate pair is non-zero. Multiplying ``y`` by cyclic shifts of
    itself then produces a sparse, high-dimensional, non-negative code whose
    supports are close to disjoint across different inputs. Near-disjoint supports
    are what buy near-orthogonality, and near-orthogonality is what lets a
    rank-limited state hold more keys without interference.

    Non-negativity is a consequence here, not the point: every term is a product
    of two non-negative numbers.
    """
    if nu < 1:
        raise ValueError(f"nu must be >= 1, got {nu}")
    y = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
    return torch.cat([y * torch.roll(y, shifts=i, dims=-1) for i in range(1, nu + 1)], dim=-1)


def output_dim(name: str, d_k: int) -> int:
    """``d_phi`` for a named map at input width ``d_k``, without running it.

    The layer needs this before it has a tensor: the state's shape, and hence the
    matched-state control, depend on it.
    """
    if name in ("identity", "elu", "relu"):
        return d_k
    if name.startswith("dpfp"):
        return 2 * d_k * _nu_of(name)
    raise ValueError(f"unknown feature map {name!r}; expected one of {FEATURE_MAPS}")


def _nu_of(name: str) -> int:
    try:
        return int(name[len("dpfp"):])
    except ValueError as exc:
        raise ValueError(f"cannot parse nu from {name!r}") from exc


def make_feature_map(name: str) -> FeatureMap:
    """Look up a feature map by name. ``"identity"`` is the no-op control."""
    if name == "identity":
        return identity
    if name == "elu":
        return elu_plus_one
    if name == "relu":
        return relu
    if name.startswith("dpfp"):
        nu = _nu_of(name)
        return lambda x, _nu=nu: dpfp(x, nu=_nu)
    raise ValueError(f"unknown feature map {name!r}; expected one of {FEATURE_MAPS}")

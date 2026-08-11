"""Gate for trusting ``fla``'s Triton kernels on the HPC box.

Skipped entirely without CUDA + ``flash-linear-attention``, which is the state
on the development workstation. **On first run with a GPU, run this before
anything else.** Until it passes, no GPU number is comparable with any CPU
number, and the whole project's premise is that they are interchangeable.

Failures here are informative by design: the adapter in
``lamr.layers.fla_backend`` encodes three convention guesses (query scaling,
tensor layout, log-decay parameterization) that could not be tested when it was
written. Each assertion below names the convention it is pinning, so a failure
says which guess was wrong rather than just "numbers differ".
"""

from __future__ import annotations

import pytest
import torch

from lamr.layers import delta_rule_recurrent, gated_delta_rule_recurrent
from lamr.layers.fla_backend import fla_available, fla_delta_rule, fla_gated_delta_rule
from lamr.layers.fla_backend import unavailable_reason

pytestmark = pytest.mark.skipif(
    not fla_available(), reason=f"fla backend unavailable: {unavailable_reason()}"
)

# Triton kernels accumulate in fp32 in a different order than the reference
# loop, so exact equality is not the bar; a relative error this size would
# still be far too large to explain away as arithmetic reordering.
REL_TOL = 1e-3


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm().clamp_min(1e-12))


def make_inputs(b=2, h=4, t=128, d_k=32, d_v=32, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(device)  # noqa: E731
    q = torch.nn.functional.normalize(r(b, h, t, d_k), dim=-1)
    k = torch.nn.functional.normalize(r(b, h, t, d_k), dim=-1)
    v = r(b, h, t, d_v)
    beta = torch.rand(b, h, t, generator=g).to(device)
    return q, k, v, beta


def test_delta_rule_output_matches_reference():
    q, k, v, beta = make_inputs()
    want, _ = delta_rule_recurrent(q.double(), k.double(), v.double(), beta.double())
    got, _ = fla_delta_rule(q, k, v, beta)

    err = relative_error(got.double(), want)
    assert err < REL_TOL, (
        f"fla delta rule output differs by {err:.2e}. Most likely cause: the "
        "query scaling convention. This project passes scale=1.0 because the "
        "layer L2-normalizes q and k; fla defaults to d_k ** -0.5."
    )


def test_gated_delta_rule_output_matches_reference():
    q, k, v, beta = make_inputs()
    alpha = (torch.rand_like(beta) * 0.1 + 0.9).clamp(max=1.0)

    want, _ = gated_delta_rule_recurrent(
        q.double(), k.double(), v.double(), beta.double(), alpha.double()
    )
    got, _ = fla_gated_delta_rule(q, k, v, beta, alpha)

    err = relative_error(got.double(), want)
    assert err < REL_TOL, (
        f"fla gated delta rule output differs by {err:.2e}. Check the decay "
        "parameterization: fla takes g = log(alpha), not alpha."
    )


def test_final_state_orientation_is_dk_by_dv():
    """Pins the state layout, which is the adapter's least certain guess.

    This project keeps the plan's ``(d_k, d_v)``. If fla hands back the
    transpose, the fix is to transpose in ``fla_backend``, never to change the
    convention downstream -- Stage 4 indexes this matrix by key.
    """
    d_k, d_v = 32, 16  # deliberately unequal so a transpose cannot hide
    q, k, v, beta = make_inputs(d_k=d_k, d_v=d_v)
    _, want = delta_rule_recurrent(q.double(), k.double(), v.double(), beta.double())
    _, got = fla_delta_rule(q, k, v, beta)

    if got.shape[-2:] == (d_v, d_k):
        pytest.fail(
            f"fla returned a transposed state {tuple(got.shape)}; expected "
            f"(..., {d_k}, {d_v}). Transpose it inside fla_backend."
        )
    assert got.shape[-2:] == (d_k, d_v), f"unexpected state shape {tuple(got.shape)}"

    err = relative_error(got.double(), want)
    assert err < REL_TOL, f"state values differ by {err:.2e}"


def test_initial_state_is_honoured():
    """Stage 5's periodic slow pass depends on state round-tripping correctly."""
    q, k, v, beta = make_inputs(t=64)
    split = 32
    _, mid = fla_delta_rule(
        q[:, :, :split], k[:, :, :split], v[:, :, :split], beta[:, :, :split]
    )
    second, _ = fla_delta_rule(
        q[:, :, split:], k[:, :, split:], v[:, :, split:], beta[:, :, split:],
        initial_state=mid,
    )
    whole, _ = fla_delta_rule(q, k, v, beta)

    err = relative_error(second, whole[:, :, split:])
    assert err < REL_TOL, f"carried state diverges by {err:.2e}"


def test_matches_the_portable_chunked_backend():
    """The three implementations must agree, not just fla against the loop."""
    from lamr.layers import chunk_delta_rule

    q, k, v, beta = make_inputs()
    want, want_state = chunk_delta_rule(
        q.double(), k.double(), v.double(), beta.double(), chunk_size=64
    )
    got, got_state = fla_delta_rule(q, k, v, beta)

    assert relative_error(got.double(), want) < REL_TOL
    assert relative_error(got_state.double(), want_state) < REL_TOL

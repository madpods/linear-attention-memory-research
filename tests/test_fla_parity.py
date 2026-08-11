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

# Derived from the kernels' dtype, not chosen to make the suite pass.
#
# ``chunk_delta_rule`` asserts ``q.dtype != torch.float32`` ("does not support
# float32. Please use bfloat16."), so the adapter runs the kernels in bf16 and
# parity is necessarily bf16-kernel against fp64-reference. bf16 has 8 mantissa
# bits (eps = 2**-8 = 3.9e-3), so rounding q/k/v alone costs ~2e-3 relative
# before any arithmetic happens, and the delta recurrence accumulates over T
# steps on top of that. The previous 1e-3 was set while the adapter was assumed
# to run in fp32; it is not reachable by a correct implementation here.
#
# Loosening a tolerance is only honest if the gate still catches what it exists
# to catch. The test_negative_control_* cases below feed deliberately wrong
# conventions and assert they land ORDERS OF MAGNITUDE outside this bound, not
# merely outside it -- so a passing suite means the margin is real rather than
# that the bar was lowered to meet the numbers.
REL_TOL = 3e-2

#: A wrong convention (layout, scale, log-decay) corrupts the result outright
#: rather than degrading it. Negative controls must exceed this to demonstrate
#: the tolerance above still discriminates.
WRONG_CONVENTION_FLOOR = 10 * REL_TOL


def assert_detectably_wrong(got: torch.Tensor, want: torch.Tensor, hint: str) -> None:
    """Assert a deliberately wrong convention lands far outside ``REL_TOL``.

    Non-finite output counts, and is the *stronger* result: it means the wrong
    convention did not merely shift the answer but diverged. The raw-alpha-as-g
    control does exactly that -- treating alpha ~0.95 as a log-decay makes the
    per-step factor exp(0.95) ~ 2.59, so over T=128 steps the state grows like
    2.59**128 ~ 1e52 and overflows bf16's ~3.4e38 range to inf/nan.

    Counting only large *finite* errors would fail this test on a nan, reporting
    "the tolerance has stopped discriminating" when the opposite was shown. Note
    the positive tests are unaffected: ``nan < REL_TOL`` is False, so a diverged
    result can never pass one.
    """
    finite = bool(torch.isfinite(got).all())
    err = relative_error(got, want)
    if not finite:
        return  # diverged; wrongness demonstrated
    assert err > WRONG_CONVENTION_FLOOR, (
        f"{hint} differs by only {err:.2e}, inside {WRONG_CONVENTION_FLOOR:.0e}. "
        "REL_TOL can no longer separate correct from wrong-convention, so the "
        "positive tests in this file prove nothing. Fix the tolerance derivation "
        "before trusting a green suite."
    )


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
        f"fla delta rule output differs by {err:.2e}. Two suspects, in order:\n"
        "  1. Layout. fla requires (B, T, H, D) and removed head_first, so the "
        "adapter transposes unconditionally. A wrong layout does NOT raise -- "
        "the kernel treats heads as timesteps and returns plausible nonsense.\n"
        "  2. Query scaling. We pass scale=1.0 because the layer L2-normalizes "
        "q and k; fla defaults to d_k ** -0.5. An error near sqrt(d_k) points here."
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
    """Pins the state layout.

    fla documents ``[N, H, K, V]``, which already matches the plan's
    ``(d_k, d_v)``, so the adapter does not transpose the state. If that
    changes upstream -- the gated kernel has a ``state_v_first`` flag that
    flips it -- fix ``fla_backend``, never the convention downstream: Stage 4
    indexes this matrix by key.
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


def test_kernels_run_in_bfloat16_and_return_the_callers_dtype():
    """Pins the dtype convention in both directions.

    ``chunk_delta_rule`` refuses fp32 outright, so the adapter casts. The cast
    back matters just as much: the CPU paths are fp32 and the backends are meant
    to be swappable, so a bf16 tensor leaking out would silently change the
    dtype of everything downstream of the layer.
    """
    from lamr.layers.fla_backend import KERNEL_DTYPE

    assert KERNEL_DTYPE is torch.bfloat16

    q, k, v, beta = make_inputs()
    assert q.dtype is torch.float32, "make_inputs should hand us fp32"

    out, state = fla_delta_rule(q, k, v, beta)
    assert out.dtype is torch.float32, f"delta output leaked {out.dtype}"
    assert state.dtype is torch.float32, f"delta state leaked {state.dtype}"

    alpha = (torch.rand_like(beta) * 0.1 + 0.9).clamp(max=1.0)
    out, state = fla_gated_delta_rule(q, k, v, beta, alpha)
    assert out.dtype is torch.float32, f"gated output leaked {out.dtype}"
    assert state.dtype is torch.float32, f"gated state leaked {state.dtype}"


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------
# These exist to justify REL_TOL. Each feeds a convention the adapter
# deliberately does NOT use and asserts the result is wrong by far more than the
# tolerance -- establishing that a bf16-sized bound still separates "correct" from
# "wrong convention". If one of these ever FAILS, the tolerance has stopped
# discriminating and the gate is no longer meaningful, regardless of whether the
# positive tests pass.


def test_negative_control_default_scale_is_detectably_wrong():
    """fla's default ``scale`` must not silently pass.

    The adapter passes ``scale=1.0`` because the layer L2-normalizes q and k.
    fla's default is ``d_k ** -0.5``, and that difference has to be far outside
    REL_TOL or the scale assertion in the positive test proves nothing.
    """
    from lamr.layers.fla_backend import KERNEL_DTYPE, _fla_delta

    q, k, v, beta = make_inputs()
    want, _ = delta_rule_recurrent(q.double(), k.double(), v.double(), beta.double())

    to_fla = lambda x: x.transpose(1, 2).to(KERNEL_DTYPE)  # noqa: E731
    wrong, _ = _fla_delta(
        to_fla(q), to_fla(k), to_fla(v), to_fla(beta),
        scale=None,  # fla then uses d_k ** -0.5
        output_final_state=True,
    )
    assert_detectably_wrong(
        wrong.transpose(1, 2).double(), want, "fla's default scale vs scale=1.0"
    )


def test_negative_control_raw_alpha_as_g_is_detectably_wrong():
    """``g`` is log-space; passing alpha directly must blow up.

    alpha sits just under 1, so a raw alpha handed to a kernel expecting
    log(alpha) makes the decay exp(alpha) ~ 2.6 per step instead of ~0.95 -- the
    state grows where it should contract.

    Observed on an H100 (2026-08-11): this overflows to nan rather than returning
    a large finite error, because 2.59**128 ~ 1e52 exceeds bf16's range. That is
    why :func:`assert_detectably_wrong` treats non-finite output as a pass -- the
    convention is not subtly wrong here, it diverges.
    """
    from lamr.layers.fla_backend import KERNEL_DTYPE, _fla_gated

    q, k, v, beta = make_inputs()
    alpha = (torch.rand_like(beta) * 0.1 + 0.9).clamp(max=1.0)
    want, _ = gated_delta_rule_recurrent(
        q.double(), k.double(), v.double(), beta.double(), alpha.double()
    )

    to_fla = lambda x: x.transpose(1, 2).to(KERNEL_DTYPE)  # noqa: E731
    wrong, _ = _fla_gated(
        to_fla(q), to_fla(k), to_fla(v),
        g=to_fla(alpha),  # NOT log(alpha)
        beta=to_fla(beta),
        scale=1.0,
        output_final_state=True,
    )
    assert_detectably_wrong(
        wrong.transpose(1, 2).double(), want, "raw alpha as g vs log(alpha)"
    )


def test_negative_control_wrong_layout_is_detectably_wrong():
    """The dangerous one: a wrong layout does not raise.

    fla wants ``(B, T, H, D)`` and this project holds ``(B, H, T, D)``. Passing
    ours through untransposed is shape-valid whenever H == T, so the kernel
    happily treats heads as timesteps. H == T is forced here precisely so the
    mistake cannot be caught by a shape error -- which is the situation the
    adapter's unconditional transpose exists to prevent.
    """
    from lamr.layers.fla_backend import KERNEL_DTYPE, _fla_delta

    # 64 rather than a small number: it is one full chunk, so the kernel is not
    # also being asked to handle a sequence shorter than its chunk size.
    n = 64
    q, k, v, beta = make_inputs(b=2, h=n, t=n)
    want, _ = delta_rule_recurrent(q.double(), k.double(), v.double(), beta.double())

    cast = lambda x: x.to(KERNEL_DTYPE)  # noqa: E731
    wrong, _ = _fla_delta(
        cast(q), cast(k), cast(v), cast(beta),  # no transpose: heads as time
        scale=1.0,
        output_final_state=True,
    )
    assert_detectably_wrong(wrong.double(), want, "heads treated as timesteps")

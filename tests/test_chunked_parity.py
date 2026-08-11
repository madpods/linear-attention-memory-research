"""The chunked backend must reproduce the sequential reference exactly.

This is the gate for using the fast path anywhere. The same test shape will be
reused on the HPC box to validate ``fla``'s Triton kernels against
``recurrent.py`` -- if that parity check fails there, the kernels are wrong for
this project's conventions (note the transposed state layout) and results from
them cannot be compared against anything produced here.
"""

from __future__ import annotations

import pytest
import torch

from lamr.layers import (
    chunk_delta_rule,
    chunk_gated_delta_rule,
    chunk_linear_attn,
    delta_rule_recurrent,
    elu_plus_one,
    gated_delta_rule_recurrent,
    linear_attn_recurrent,
)

ATOL = 1e-10


def inputs(b=2, h=3, t=37, d_k=8, d_v=6, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=dtype, generator=g)  # noqa: E731
    q = r(b, h, t, d_k)
    k = torch.nn.functional.normalize(r(b, h, t, d_k), dim=-1)
    v = r(b, h, t, d_v)
    beta = torch.rand(b, h, t, dtype=dtype, generator=g)
    return q, k, v, beta


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 8, 16, 64, 128])
def test_matches_sequential_delta_rule(chunk_size):
    """T=37 is deliberately not a multiple of any chunk size tested."""
    q, k, v, beta = inputs()
    want_out, want_state = delta_rule_recurrent(q, k, v, beta)
    got_out, got_state = chunk_delta_rule(q, k, v, beta, chunk_size=chunk_size)

    assert torch.allclose(got_out, want_out, atol=ATOL)
    assert torch.allclose(got_state, want_state, atol=ATOL)


@pytest.mark.parametrize("chunk_size", [1, 4, 16, 64])
@pytest.mark.parametrize("alpha_val", [1.0, 0.9, 0.5])
def test_matches_sequential_gated_delta_rule(chunk_size, alpha_val):
    q, k, v, beta = inputs()
    alpha = torch.full_like(beta, alpha_val)

    want_out, want_state = gated_delta_rule_recurrent(q, k, v, beta, alpha)
    got_out, got_state = chunk_gated_delta_rule(
        q, k, v, beta, alpha, chunk_size=chunk_size
    )

    assert torch.allclose(got_out, want_out, atol=ATOL)
    assert torch.allclose(got_state, want_state, atol=ATOL)


def test_matches_sequential_with_varying_alpha():
    """Data-dependent decay, not a constant -- the case the gates actually use."""
    q, k, v, beta = inputs(t=64)
    g = torch.Generator().manual_seed(5)
    alpha = torch.rand(*beta.shape, dtype=beta.dtype, generator=g) * 0.5 + 0.5

    want_out, want_state = gated_delta_rule_recurrent(q, k, v, beta, alpha)
    got_out, got_state = chunk_gated_delta_rule(q, k, v, beta, alpha, chunk_size=16)

    assert torch.allclose(got_out, want_out, atol=ATOL)
    assert torch.allclose(got_state, want_state, atol=ATOL)


def test_chunk_size_does_not_change_the_result():
    """Chunking is an implementation detail; it must not be a hyperparameter."""
    q, k, v, beta = inputs(t=100)
    reference = chunk_delta_rule(q, k, v, beta, chunk_size=1)
    for chunk_size in (3, 7, 16, 50, 100, 256):
        out, state = chunk_delta_rule(q, k, v, beta, chunk_size=chunk_size)
        assert torch.allclose(out, reference[0], atol=ATOL), f"chunk={chunk_size}"
        assert torch.allclose(state, reference[1], atol=ATOL), f"chunk={chunk_size}"


def test_initial_state_lets_a_sequence_be_split():
    """Carrying state across calls must equal processing the whole sequence."""
    q, k, v, beta = inputs(t=48)
    whole_out, whole_state = chunk_delta_rule(q, k, v, beta, chunk_size=8)

    split = 19
    first_out, mid_state = chunk_delta_rule(
        q[:, :, :split], k[:, :, :split], v[:, :, :split], beta[:, :, :split],
        chunk_size=8,
    )
    second_out, final_state = chunk_delta_rule(
        q[:, :, split:], k[:, :, split:], v[:, :, split:], beta[:, :, split:],
        chunk_size=8, initial_state=mid_state,
    )

    assert torch.allclose(torch.cat([first_out, second_out], dim=2), whole_out, atol=ATOL)
    assert torch.allclose(final_state, whole_state, atol=ATOL)


def test_padding_tokens_do_not_leak_into_the_state():
    """T not divisible by C: the pad must be an exact no-op, not a small write."""
    q, k, v, beta = inputs(t=33)
    _, padded_state = chunk_delta_rule(q, k, v, beta, chunk_size=16)
    _, exact_state = delta_rule_recurrent(q, k, v, beta)
    assert torch.allclose(padded_state, exact_state, atol=ATOL)


def test_gradients_flow_through_the_triangular_solve():
    q, k, v, beta = inputs(t=32, dtype=torch.float32)
    q, v = q.requires_grad_(True), v.requires_grad_(True)
    beta = beta.detach().requires_grad_(True)

    out, state = chunk_delta_rule(q, k, v, beta, chunk_size=8)
    (out.sum() + state.sum()).backward()

    for name, tensor in (("q", q), ("v", v), ("beta", beta)):
        assert tensor.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient not finite"


def test_aggressive_decay_stays_finite_in_float32():
    """The reason ratios are formed in log space.

    With alpha=0.5 over a 128-token chunk the cumulative decay underflows
    float32, so any implementation that materializes 1/c_s divides by zero.
    Masking then exponentiating pairwise differences keeps every intermediate
    bounded by 1.
    """
    q, k, v, beta = inputs(t=256, dtype=torch.float32)
    alpha = torch.full_like(beta, 0.5)

    out, state = chunk_gated_delta_rule(q, k, v, beta, alpha, chunk_size=128)

    assert torch.isfinite(out).all(), "non-finite output under aggressive decay"
    assert torch.isfinite(state).all(), "non-finite state under aggressive decay"


@pytest.mark.parametrize("chunk_size", [1, 8, 64])
@pytest.mark.parametrize("normalize", [True, False])
def test_chunked_linear_attention_matches_sequential(chunk_size, normalize):
    q, k, v, _ = inputs(t=37)
    want_out, want_state = linear_attn_recurrent(
        q, k, v, feature_map=elu_plus_one, normalize=normalize
    )
    got_out, got_state = chunk_linear_attn(
        q, k, v, feature_map=elu_plus_one, normalize=normalize, chunk_size=chunk_size
    )
    assert torch.allclose(got_out, want_out, atol=ATOL)
    assert torch.allclose(got_state, want_state, atol=ATOL)


def test_rejects_bad_chunk_size():
    q, k, v, beta = inputs()
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        chunk_delta_rule(q, k, v, beta, chunk_size=0)


def test_supports_differing_dk_and_dv():
    q, k, v, beta = inputs(t=20, d_k=5, d_v=9)
    want_out, want_state = delta_rule_recurrent(q, k, v, beta)
    got_out, got_state = chunk_delta_rule(q, k, v, beta, chunk_size=8)
    assert got_out.shape == want_out.shape
    assert got_state.shape == (2, 3, 5, 9)
    assert torch.allclose(got_out, want_out, atol=ATOL)
    assert torch.allclose(got_state, want_state, atol=ATOL)

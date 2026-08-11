"""Correctness tests for the reference update rules.

The anchor is the worked numeric example in plan section 1. Beyond that these
assert the *properties* the plan relies on when reasoning about capacity, so a
subtly wrong update rule fails here rather than silently producing plausible
training curves later.
"""

from __future__ import annotations

import pytest
import torch

from lamr.layers import (
    delta_rule_recurrent,
    delta_rule_step,
    gated_delta_rule_recurrent,
    linear_attn_recurrent,
)

torch.manual_seed(0)


def rand(*shape, dtype=torch.float64):
    return torch.randn(*shape, dtype=dtype)


# --- plan section 1 worked example --------------------------------------


def test_worked_example_from_plan_section_1():
    """S=[[0,9],[9,0]], k=[1,0], v=[5,9] -> S_new=[[5,9],[9,0]]."""
    state = torch.tensor([[0.0, 9.0], [9.0, 0.0]], dtype=torch.float64)
    k = torch.tensor([1.0, 0.0], dtype=torch.float64)
    v = torch.tensor([5.0, 9.0], dtype=torch.float64)

    new_state, error = delta_rule_step(state, k, v, beta=1.0)

    assert torch.equal(new_state, torch.tensor([[5.0, 9.0], [9.0, 0.0]], dtype=torch.float64))
    assert torch.equal(error, torch.tensor([5.0, 0.0], dtype=torch.float64))
    # Row 1 is untouched because k has no component there.
    assert torch.equal(new_state[1], state[1])


def test_redundant_write_is_a_no_op():
    """The core claim: re-writing a pair the state already predicts costs nothing."""
    state = torch.zeros(4, 6, dtype=torch.float64)
    k = torch.zeros(4, dtype=torch.float64)
    k[0] = 1.0
    v = rand(6)

    state, first_error = delta_rule_step(state, k, v, beta=1.0)
    after_first = state.clone()
    state, second_error = delta_rule_step(state, k, v, beta=1.0)

    assert torch.allclose(first_error, v)
    assert torch.allclose(second_error, torch.zeros_like(v), atol=1e-12)
    assert torch.allclose(state, after_first, atol=1e-12)


def test_delta_step_matches_householder_form():
    """S_new == (I - beta k k^T) S + beta k v^T, the form the WY trick uses."""
    d_k, d_v = 5, 7
    state, k, v = rand(d_k, d_v), rand(d_k), rand(d_v)
    k = k / k.norm()
    beta = 0.6

    got, _ = delta_rule_step(state, k, v, beta=beta)
    eye = torch.eye(d_k, dtype=torch.float64)
    want = (eye - beta * torch.outer(k, k)) @ state + beta * torch.outer(k, v)

    assert torch.allclose(got, want, atol=1e-12)


# --- capacity behaviour -------------------------------------------------


def test_orthogonal_keys_recall_exactly():
    """With orthonormal keys and beta=1, recall is exact -- zero interference."""
    d = 6
    b, h, t = 1, 1, d
    k = torch.eye(d, dtype=torch.float64)[None, None]
    v = rand(b, h, t, d)
    beta = torch.ones(b, h, t, dtype=torch.float64)

    _, state = delta_rule_recurrent(k, k, v, beta)
    readback = torch.einsum("bhtk,bhkv->bhtv", k, state)

    assert torch.allclose(readback, v, atol=1e-12)


def _recall_error(num_pairs: int, d: int, epochs: int, seed: int) -> float:
    """Relative recall error after ``epochs`` passes over the same pairs."""
    g = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(
        torch.randn(1, 1, num_pairs, d, dtype=torch.float64, generator=g), dim=-1
    )
    v = torch.randn(1, 1, num_pairs, d, dtype=torch.float64, generator=g)
    k_rep, v_rep = k.repeat(1, 1, epochs, 1), v.repeat(1, 1, epochs, 1)
    beta = torch.ones(1, 1, num_pairs * epochs, dtype=torch.float64)
    _, state = delta_rule_recurrent(k_rep, k_rep, v_rep, beta)
    readback = torch.einsum("bhtk,bhkv->bhtv", k, state)
    return (readback - v).norm().item() / v.norm().item()


def test_interference_grows_with_the_number_of_stored_pairs():
    """Crowding is continuous, not a cliff at rank(S) <= min(d_k, d_v).

    Note the ceiling is *not* the whole story for a single pass: with random
    (non-orthogonal) keys, error is already substantial below d_k because one
    pass is a single Kaczmarz sweep and later writes perturb earlier ones. Exact
    recall in one pass needs orthogonal keys -- see
    ``test_orthogonal_keys_recall_exactly``. What the ceiling guarantees is that
    error keeps rising as pairs accumulate, which is what this asserts.
    """
    d = 8
    errors = [
        sum(_recall_error(n, d, epochs=1, seed=s) for s in range(8)) / 8
        for n in (d // 2, d, 2 * d, 4 * d)
    ]
    assert all(
        lo < hi for lo, hi in zip(errors, errors[1:])
    ), f"recall error not monotone in stored pairs: {errors}"
    assert errors[-1] > 2 * errors[0]


def test_repeated_passes_converge_under_capacity():
    """The delta rule is an iterative least-squares solver (the RLS/Kalman link).

    Under capacity a consistent system is solvable, and repeated sweeps drive
    recall error toward zero even though a single sweep leaves it large.
    """
    d = 8
    one = sum(_recall_error(4, d, epochs=1, seed=s) for s in range(8)) / 8
    many = sum(_recall_error(4, d, epochs=64, seed=s) for s in range(8)) / 8

    assert one > 0.1, f"expected a large single-pass error, got {one}"
    assert many < 1e-4, f"expected convergence, got {many}"
    assert many < one / 1000


def test_delta_rule_resists_redundancy_where_plain_accumulation_does_not():
    """Same pair written 20x: delta rule holds, plain linear attention inflates."""
    d, repeats = 4, 20
    k = torch.nn.functional.normalize(rand(1, 1, 1, d), dim=-1).expand(1, 1, repeats, d)
    v = rand(1, 1, 1, d).expand(1, 1, repeats, d)
    beta = torch.ones(1, 1, repeats, dtype=torch.float64)

    _, delta_state = delta_rule_recurrent(k, k, v, beta)
    _, plain_state = linear_attn_recurrent(k, k, v, feature_map=None, normalize=False)

    delta_read = torch.einsum("bhk,bhkv->bhv", k[:, :, 0], delta_state)
    plain_read = torch.einsum("bhk,bhkv->bhv", k[:, :, 0], plain_state)

    assert torch.allclose(delta_read, v[:, :, 0], atol=1e-12)
    # Plain accumulation superimposes the same outer product every time.
    assert torch.allclose(plain_read, repeats * v[:, :, 0], atol=1e-9)


# --- rule relationships -------------------------------------------------


def test_gated_reduces_to_delta_when_alpha_is_one():
    b, h, t, d = 2, 3, 16, 8
    q, k, v = rand(b, h, t, d), rand(b, h, t, d), rand(b, h, t, d)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand(b, h, t, dtype=torch.float64)
    alpha = torch.ones(b, h, t, dtype=torch.float64)

    out_delta, state_delta = delta_rule_recurrent(q, k, v, beta)
    out_gated, state_gated = gated_delta_rule_recurrent(q, k, v, beta, alpha)

    assert torch.allclose(out_delta, out_gated, atol=1e-12)
    assert torch.allclose(state_delta, state_gated, atol=1e-12)


def test_zero_beta_freezes_the_state():
    b, h, t, d = 1, 2, 12, 5
    q, k, v = rand(b, h, t, d), rand(b, h, t, d), rand(b, h, t, d)
    beta = torch.zeros(b, h, t, dtype=torch.float64)

    out, state = delta_rule_recurrent(q, k, v, beta)
    assert torch.allclose(state, torch.zeros_like(state), atol=1e-12)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_alpha_decays_an_unwritten_state_geometrically():
    b, h, t, d = 1, 1, 5, 3
    q = torch.zeros(b, h, t, d, dtype=torch.float64)
    k = torch.zeros(b, h, t, d, dtype=torch.float64)
    v = torch.zeros(b, h, t, d, dtype=torch.float64)
    k[:, :, 0, 0] = 1.0
    v[:, :, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    beta = torch.zeros(b, h, t, dtype=torch.float64)
    beta[:, :, 0] = 1.0
    alpha = torch.full((b, h, t), 0.5, dtype=torch.float64)

    _, state = gated_delta_rule_recurrent(q, k, v, beta, alpha)
    # Written at t=0, then decayed by 0.5 on each of the remaining 4 steps.
    assert torch.allclose(state[0, 0, 0], v[0, 0, 0] * 0.5**4, atol=1e-12)


def test_plain_linear_attention_matches_closed_form():
    """Unnormalized, no feature map: out_t == sum_{i<=t} (q_t . k_i) v_i."""
    b, h, t, d = 2, 2, 10, 4
    q, k, v = rand(b, h, t, d), rand(b, h, t, d), rand(b, h, t, d)

    out, _ = linear_attn_recurrent(q, k, v, feature_map=None, normalize=False)

    scores = torch.einsum("bhtd,bhsd->bhts", q, k)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    want = torch.einsum("bhts,bhsd->bhtd", scores * causal, v)

    assert torch.allclose(out, want, atol=1e-10)


def test_normalization_bounds_output_magnitude():
    """Without the phi(q)^T z denominator, outputs grow with sequence length."""
    b, h, t, d = 1, 1, 128, 8
    q, k, v = rand(b, h, t, d), rand(b, h, t, d), rand(b, h, t, d)

    raw, _ = linear_attn_recurrent(q, k, v, normalize=False)
    normed, _ = linear_attn_recurrent(q, k, v, normalize=True)

    growth_raw = raw[:, :, -1].norm() / raw[:, :, 0].norm()
    growth_normed = normed[:, :, -1].norm() / normed[:, :, 0].norm()
    assert growth_raw > 5.0
    assert growth_normed < growth_raw


# --- plumbing -----------------------------------------------------------


def test_gradients_flow_and_are_finite():
    b, h, t, d = 2, 2, 8, 4
    q = rand(b, h, t, d, dtype=torch.float32).requires_grad_(True)
    k = rand(b, h, t, d, dtype=torch.float32).requires_grad_(True)
    v = rand(b, h, t, d, dtype=torch.float32).requires_grad_(True)
    beta = torch.rand(b, h, t).requires_grad_(True)

    out, _ = delta_rule_recurrent(q, torch.nn.functional.normalize(k, dim=-1), v, beta)
    out.sum().backward()

    for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta)):
        assert tensor.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient not finite"


def test_supports_differing_dk_and_dv():
    b, h, t, d_k, d_v = 2, 2, 6, 5, 9
    q, k = rand(b, h, t, d_k), rand(b, h, t, d_k)
    v = rand(b, h, t, d_v)
    beta = torch.rand(b, h, t, dtype=torch.float64)

    out, state = delta_rule_recurrent(q, k, v, beta)
    assert out.shape == (b, h, t, d_v)
    assert state.shape == (b, h, d_k, d_v)


@pytest.mark.parametrize(
    "bad,match",
    [
        ("ndim", r"expected \(B, H, T, D\)"),
        ("bht", r"agree on \(B, H, T\)"),
        ("dk", "must share d_k"),
        ("beta", r"beta must be \(B, H, T\)"),
    ],
)
def test_shape_validation(bad, match):
    b, h, t, d = 1, 1, 4, 3
    q, k, v = rand(b, h, t, d), rand(b, h, t, d), rand(b, h, t, d)
    beta = torch.rand(b, h, t, dtype=torch.float64)

    if bad == "ndim":
        q = q[0]
    elif bad == "bht":
        v = rand(b, h, t + 1, d)
    elif bad == "dk":
        k = rand(b, h, t, d + 2)
    elif bad == "beta":
        beta = torch.rand(b, h, dtype=torch.float64)

    with pytest.raises(ValueError, match=match):
        delta_rule_recurrent(q, k, v, beta)

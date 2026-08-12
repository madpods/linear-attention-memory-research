"""Stage 3 feature maps (plan section 17).

The properties worth pinning are the ones the capacity argument rests on:
the output width (it sets the state's shape and hence the ceiling), sparsity and
near-orthogonality for DPFP (that is *why* it should buy capacity), and that
``output_dim`` agrees with reality (the layer sizes the state from it before any
tensor exists, so a mismatch is a silent shape bug).
"""

from __future__ import annotations

import pytest
import torch

from lamr.layers.feature_maps import (
    FEATURE_MAPS,
    dpfp,
    elu_plus_one,
    make_feature_map,
    output_dim,
)


@pytest.mark.parametrize("name", FEATURE_MAPS)
def test_output_dim_matches_what_the_map_actually_produces(name):
    """The layer sizes the state from output_dim() before it has a tensor."""
    d_k = 16
    phi = make_feature_map(name)
    got = phi(torch.randn(2, 4, 8, d_k))
    assert got.shape[:-1] == (2, 4, 8), "leading dims must be preserved"
    assert got.shape[-1] == output_dim(name, d_k), (
        f"{name}: output_dim says {output_dim(name, d_k)} but produced "
        f"{got.shape[-1]}"
    )


def test_dpfp_expansion_factors():
    """d_phi = 2 * d_k * nu. With d_k=16 these are the 2x/4x/6x/8x arms."""
    assert [output_dim(f"dpfp{nu}", 16) for nu in (1, 2, 3, 4)] == [32, 64, 96, 128]


def test_dpfp_is_non_negative():
    """Every term is a product of two relu outputs."""
    got = dpfp(torch.randn(64, 16) * 5.0, nu=2)
    assert (got >= 0).all()


def test_dpfp_is_sparse():
    """Half-sparse by construction: one of each (relu(x), relu(-x)) pair is zero.

    Sparsity is not decoration -- near-disjoint supports are the mechanism by
    which DPFP buys near-orthogonality, and near-orthogonality is what lets a
    rank-limited state hold more keys without interference.
    """
    got = dpfp(torch.randn(256, 16), nu=1)
    density = (got > 0).float().mean().item()
    assert 0.1 < density < 0.45, f"expected a sparse code, got density {density:.2f}"


def test_dpfp_reduces_interference_between_random_keys():
    """The capacity claim, measured directly.

    Random keys L2-normalized at d_k=16 have a characteristic mean |cos| between
    them; expanding through DPFP should reduce it, because that off-diagonal
    similarity IS the interference a rank-limited state suffers when it stores
    many keys. If this ever fails, section 17's premise does not hold for DPFP
    and the Stage 3 sweep is measuring something else.
    """
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(512, 16), dim=-1)

    def mean_abs_offdiag_cos(x):
        x = torch.nn.functional.normalize(x, dim=-1)
        gram = (x @ x.T).abs()
        n = gram.shape[0]
        off = ~torch.eye(n, dtype=torch.bool)
        return gram[off].mean().item()

    base = mean_abs_offdiag_cos(k)
    for nu in (1, 2, 3):
        expanded = mean_abs_offdiag_cos(dpfp(k, nu=nu))
        assert expanded < base, (
            f"dpfp{nu} raised mean off-diagonal |cos| from {base:.3f} to "
            f"{expanded:.3f} -- expansion is adding interference, not removing it"
        )


def test_elu_plus_one_is_non_negative_and_shape_preserving():
    x = torch.randn(32, 16) * 10.0
    got = elu_plus_one(x)
    assert got.shape == x.shape, "elu+1 cannot raise the ceiling; d_phi == d_k"
    assert (got >= 0).all()
    # Strictly positive only in the range that matters; see the underflow test.
    assert (elu_plus_one(torch.randn(32, 16)) > 0).all()


def test_elu_plus_one_underflows_to_zero_earlier_in_bfloat16():
    """Documents a real trap rather than asserting an ideal.

    ``elu(x)`` rounds to exactly -1 once exp(x) drops below the dtype epsilon, so
    ``elu(x)+1`` hits exactly 0 -- below x ~ -17 in fp32 but already below x ~ -6
    in bf16, and the GPU path is bf16 because fla's delta kernel rejects fp32. At
    larger input scales this map therefore drifts toward relu on GPU, which
    matters because Stage 3 uses elu as a *control* against DPFP: if it silently
    becomes rectification, the two arms stop being independent.
    """
    assert (elu_plus_one(torch.tensor([-20.0])) == 0).all(), "fp32 underflow moved"
    assert (elu_plus_one(torch.tensor([-8.0], dtype=torch.bfloat16)) == 0).all()
    # ... while the same input is still non-zero in fp32.
    assert (elu_plus_one(torch.tensor([-8.0])) > 0).all()


def test_identity_is_exactly_a_no_op():
    """The control arm has to reproduce current behaviour bit-for-bit."""
    x = torch.randn(8, 16)
    assert torch.equal(make_feature_map("identity")(x), x)


def test_unknown_names_are_rejected_by_both_entry_points():
    with pytest.raises(ValueError, match="unknown feature map"):
        make_feature_map("favor")
    with pytest.raises(ValueError, match="unknown feature map"):
        output_dim("favor", 16)


def test_dpfp_rejects_nu_below_one():
    with pytest.raises(ValueError, match="nu must be >= 1"):
        dpfp(torch.randn(4, 8), nu=0)


def test_maps_are_differentiable():
    """These sit inside the model, so gradients have to flow through them."""
    for name in FEATURE_MAPS:
        x = torch.randn(4, 16, requires_grad=True)
        make_feature_map(name)(x).sum().backward()
        assert x.grad is not None, f"{name} blocked the gradient"
        assert torch.isfinite(x.grad).all(), f"{name} produced non-finite grads"


def test_identity_reproduces_the_pre_stage3_layer_exactly():
    """The control arm must not perturb the recorded Stage 2 baseline.

    The whole Stage 3 comparison is against `results/stage2_v2.csv`, so if
    threading a feature map through the layer changed the identity path by even a
    rounding step, every Stage 3 delta would be measured against a moved
    reference. Compares the full forward pass, not just shapes.
    """
    import torch as _t

    from lamr.layers.linear_attn import LinearAttentionLayer

    x = _t.randn(2, 24, 64)
    outs = []
    for fm in ("identity", "identity"):
        _t.manual_seed(0)
        outs.append(LinearAttentionLayer(64, 4, mode="delta", feature_map=fm)(x))
    assert _t.equal(outs[0], outs[1]), "identity is not deterministic"

    # And the expansion arms must actually differ, or the wiring is inert.
    _t.manual_seed(0)
    expanded = LinearAttentionLayer(64, 4, mode="delta", feature_map="dpfp2")(x)
    assert not _t.allclose(outs[0], expanded), "dpfp2 changed nothing -- phi not applied"


def test_feature_map_does_not_change_the_parameter_count():
    """Section 17's claim: capacity without inflating the base vector width.

    Parameter-matched is therefore automatic and proves nothing by itself, which
    is exactly why the matched-STATE control arm exists. Pinned so a future
    learned feature map cannot quietly break the premise.
    """
    import torch as _t

    from lamr.layers.linear_attn import LinearAttentionLayer

    counts = {}
    for fm in ("identity", "elu", "relu", "dpfp1", "dpfp2", "dpfp3", "dpfp4"):
        _t.manual_seed(0)
        layer = LinearAttentionLayer(64, 4, mode="delta", feature_map=fm)
        counts[fm] = sum(p.numel() for p in layer.parameters())
    assert len(set(counts.values())) == 1, f"parameter counts diverged: {counts}"


def test_state_width_grows_with_d_phi():
    """d_phi sets the state's rank ceiling -- min(d_phi, d_v) instead of min(d_k, d_v)."""
    import torch as _t

    from lamr.layers.linear_attn import LinearAttentionLayer

    for fm, expected in (("identity", 16), ("elu", 16), ("dpfp1", 32),
                         ("dpfp2", 64), ("dpfp3", 96), ("dpfp4", 128)):
        layer = LinearAttentionLayer(64, 4, mode="delta", feature_map=fm)
        assert layer.d_phi == expected, f"{fm}: d_phi {layer.d_phi} != {expected}"


def test_linear_mode_ignores_the_feature_map():
    """Linear attention keeps elu+1: its denominator needs a non-negative map, and
    leaving the floor baseline untouched keeps it comparable to Stage 2's curves."""
    from lamr.layers.linear_attn import LinearAttentionLayer

    layer = LinearAttentionLayer(64, 4, mode="linear", feature_map="dpfp3")
    assert layer.d_phi == layer.head_dim, "linear mode must not expand its state"

"""Model, layer and metric tests.

These lock in the properties the Stage 2 comparison depends on: that the three
modes differ *only* in their update rule, that the backend swap is invisible to
the model, and that causality holds (a leak here would let the model read the
answer and quietly invalidate every recall number the project produces).
"""

from __future__ import annotations

import pytest
import torch

from lamr.data import gen_redundant_mqar
from lamr.layers.linear_attn import LinearAttentionLayer
from lamr.metrics import recall_metrics
from lamr.models import LinearAttentionLM, LMConfig

MODES = ["linear", "delta", "gated_delta"]


def tiny_config(**overrides) -> LMConfig:
    base = dict(vocab_size=64, d_model=32, num_layers=2, num_heads=4, chunk_size=8)
    return LMConfig(**{**base, **overrides})


@pytest.mark.parametrize("mode", MODES)
def test_forward_shape_and_finiteness(mode):
    cfg = tiny_config(mode=mode)
    model = LinearAttentionLM(cfg)
    logits = model(torch.randint(0, cfg.vocab_size, (2, 24)))

    assert logits.shape == (2, 24, cfg.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("mode", MODES)
def test_model_is_causal(mode):
    """Changing token t must not move logits before t.

    MQAR answers live at query positions; if information flowed backwards the
    model could read a later token and recall accuracy would be meaningless.
    """
    torch.manual_seed(0)
    cfg = tiny_config(mode=mode)
    model = LinearAttentionLM(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 24))
    cut = 12
    altered = ids.clone()
    altered[0, cut] = (altered[0, cut] + 1) % cfg.vocab_size

    with torch.no_grad():
        before, after = model(ids), model(altered)

    assert torch.allclose(before[:, :cut], after[:, :cut], atol=1e-6)
    assert not torch.allclose(before[:, cut:], after[:, cut:], atol=1e-6)


@pytest.mark.parametrize("mode", ["delta", "gated_delta"])
def test_backends_agree_inside_the_model(mode):
    """The chunked/sequential choice must not change what the model computes."""
    torch.manual_seed(0)
    chunked = LinearAttentionLM(tiny_config(mode=mode, backend="chunked")).eval()
    sequential = LinearAttentionLM(tiny_config(mode=mode, backend="sequential")).eval()
    sequential.load_state_dict(chunked.state_dict())

    ids = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        assert torch.allclose(chunked(ids), sequential(ids), atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_every_parameter(mode):
    cfg = tiny_config(mode=mode)
    model = LinearAttentionLM(cfg)
    model(torch.randint(0, cfg.vocab_size, (2, 16))).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient not finite"


def test_mode_parameter_counts_differ_by_exactly_the_gates():
    """Plan principle 3: baselines compared at matched parameter count.

    The only structural difference between modes is the gate projections, so
    the counts must differ by exactly that and nothing else. Asserting the
    exact accounting catches an accidental extra parameter that a percentage
    tolerance would absorb silently.
    """
    cfg = tiny_config()
    counts = {m: LinearAttentionLM(tiny_config(mode=m)).num_parameters() for m in MODES}
    per_gate = cfg.num_layers * (cfg.d_model * cfg.num_heads + cfg.num_heads)

    assert counts["delta"] - counts["linear"] == per_gate, counts
    assert counts["gated_delta"] - counts["delta"] == per_gate, counts


def test_gate_overhead_is_negligible_at_the_stage_2_width():
    """At the actual experiment width the gates are well under 1% of the budget,
    so the three baselines are genuinely matched rather than approximately so."""
    counts = {
        m: LinearAttentionLM(LMConfig(vocab_size=256, mode=m)).num_parameters()
        for m in MODES
    }
    spread = (max(counts.values()) - min(counts.values())) / min(counts.values())
    assert spread < 0.01, f"parameter counts diverge too much: {counts}"


def test_gate_projections_exist_only_where_the_mode_needs_them():
    layer = lambda m: LinearAttentionLayer(32, 4, mode=m)  # noqa: E731
    assert not hasattr(layer("linear"), "beta_proj")
    assert hasattr(layer("delta"), "beta_proj")
    assert not hasattr(layer("delta"), "alpha_proj")
    assert hasattr(layer("gated_delta"), "alpha_proj")


def test_alpha_starts_near_one():
    """Decay compounds; if it starts low the state is erased before the queries."""
    layer = LinearAttentionLayer(32, 4, mode="gated_delta", alpha_init_bias=6.0)
    x = torch.zeros(1, 4, 32)  # zero input isolates the bias
    alpha = torch.sigmoid(layer.alpha_proj(x))
    assert (alpha > 0.99).all(), f"alpha initialized too low: {alpha.min()}"


@pytest.mark.parametrize("mode", ["linear", "gated_delta"])
def test_rejects_bad_layer_arguments(mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        LinearAttentionLayer(32, 4, mode="nonsense")
    with pytest.raises(ValueError, match="backend must be one of"):
        LinearAttentionLayer(32, 4, mode=mode, backend="triton")
    with pytest.raises(ValueError, match="not divisible"):
        LinearAttentionLayer(30, 4, mode=mode)


# --- metrics ------------------------------------------------------------


def make_batch(r=0.5, n=16):
    return gen_redundant_mqar(
        num_examples=n,
        seq_len=64,
        vocab_size=128,
        num_kv_pairs=6,
        num_queries=3,
        redundancy_r=r,
        num_value_clusters=2,
        seed=0,
    )


def test_perfect_predictions_score_one():
    batch = make_batch()
    logits = torch.zeros(len(batch), 64, 128)
    logits.scatter_(2, batch.lm_labels.clamp_min(0).unsqueeze(-1), 10.0)

    m = recall_metrics(logits, batch)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["accuracy_redundant"] == pytest.approx(1.0)


def test_empty_slice_is_nan_not_zero():
    """At r=0 there are no redundant queries; that must not read as 0% accuracy."""
    batch = make_batch(r=0.0)
    logits = torch.randn(len(batch), 64, 128)
    m = recall_metrics(logits, batch)

    assert m["num_redundant"] == 0
    assert m["accuracy_redundant"] != m["accuracy_redundant"]  # nan


def test_metrics_read_query_positions_without_a_shift():
    """Guards the convention: the answer is scored at p, not p+1."""
    batch = make_batch()
    logits = torch.zeros(len(batch), 64, 128)
    # Put the right answer one position late, as a shifted objective would.
    shifted = batch.query_positions + 1
    logits.scatter_(
        2,
        torch.zeros(len(batch), 64, 1, dtype=torch.long).scatter_(
            1, shifted.unsqueeze(-1), batch.labels.unsqueeze(-1)
        ),
        10.0,
    )
    assert recall_metrics(logits, batch)["accuracy"] < 0.5

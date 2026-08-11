"""Stage 1 deliverable tests.

The plan requires three properties of the generator:

(a) ``r=0`` output is statistically indistinguishable from vanilla MQAR,
(b) at ``r>0`` the fraction of kv pairs sharing a cluster value matches ``r``
    within sampling noise,
(c) no key ever appears with two different values in the same sequence.

Everything checked here is recovered by *decoding the rendered sequence*, not
by reading back the generator's own bookkeeping arrays, so a bug that corrupts
``input_ids`` while leaving the metadata self-consistent still fails.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lamr.data import VocabLayout, gen_redundant_mqar
from lamr.data.mqar import IGNORE_INDEX

BASE = dict(
    num_examples=256,
    seq_len=256,
    vocab_size=512,
    num_kv_pairs=16,
    num_queries=8,
    num_value_clusters=4,
    seed=0,
)


def make(**overrides):
    return gen_redundant_mqar(**{**BASE, "redundancy_r": 0.0, **overrides})


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = np.sort(a), np.sort(b)
    pooled = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, pooled, side="right") / a.size
    cb = np.searchsorted(b, pooled, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def ks_critical(n: int, m: int, c: float = 1.95) -> float:
    """Asymptotic KS critical value; ``c=1.95`` is roughly the 0.001 level."""
    return c * float(np.sqrt((n + m) / (n * m)))


def decode(batch, ex: int, layout: VocabLayout):
    """Recover ``{key: value}`` and query positions by scanning the sequence."""
    seq = batch.input_ids[ex].numpy()
    in_key_band = (seq >= layout.key_start) & (seq < layout.key_end)
    key_positions = set(np.flatnonzero(in_key_band).tolist())
    query_positions = set(batch.query_positions[ex].tolist())

    bindings: dict[int, set[int]] = {}
    for pos in sorted(key_positions - query_positions):
        assert pos + 1 < seq.size, "kv pair ran off the end of the sequence"
        bindings.setdefault(int(seq[pos]), set()).add(int(seq[pos + 1]))
    return bindings, key_positions, query_positions


# --- (c) well-posedness -------------------------------------------------


@pytest.mark.parametrize("r", [0.0, 0.5, 1.0])
def test_no_key_has_two_values(r):
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    batch = make(redundancy_r=r)
    for ex in range(len(batch)):
        bindings, key_pos, query_pos = decode(batch, ex, layout)

        for key, values in bindings.items():
            assert len(values) == 1, f"key {key} bound to {values} in example {ex}"

        # Every key-band token is either a kv-pair start or a query -- no
        # stray key tokens, which would be unanswerable spurious queries.
        assert query_pos <= key_pos
        assert len(bindings) == BASE["num_kv_pairs"]
        assert len(key_pos) == BASE["num_kv_pairs"] + BASE["num_queries"]


@pytest.mark.parametrize("r", [0.0, 0.5, 1.0])
def test_labels_match_the_binding_in_the_sequence(r):
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    batch = make(redundancy_r=r)
    for ex in range(len(batch)):
        bindings, _, _ = decode(batch, ex, layout)
        seq = batch.input_ids[ex]
        for slot in range(BASE["num_queries"]):
            pos = int(batch.query_positions[ex, slot])
            queried_key = int(seq[pos])
            expected = next(iter(bindings[queried_key]))
            assert int(batch.labels[ex, slot]) == expected


def test_query_always_follows_its_own_kv_pair():
    batch = make(redundancy_r=0.5)
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    for ex in range(len(batch)):
        _, key_pos, query_pos = decode(batch, ex, layout)
        kv_starts = sorted(key_pos - query_pos)
        seq = batch.input_ids[ex]
        key_to_kv_pos = {int(seq[p]): p for p in kv_starts}
        for slot in range(BASE["num_queries"]):
            qpos = int(batch.query_positions[ex, slot])
            kpos = key_to_kv_pos[int(seq[qpos])]
            # Strictly after the *value*, so the binding has been observed.
            assert qpos > kpos + 1


# --- (b) redundancy fraction -------------------------------------------


@pytest.mark.parametrize("r", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_redundant_fraction_matches_r(r):
    batch = gen_redundant_mqar(
        **{**BASE, "num_examples": 400, "redundancy_r": r, "seed": 7}
    )
    flags = batch.is_redundant.numpy()
    observed = flags.mean()
    n = flags.size
    tol = 4.0 * np.sqrt(r * (1 - r) / n)  # ~4 sigma binomial
    assert abs(observed - r) < tol, f"r={r}: observed {observed:.4f}, tol {tol:.4f}"


@pytest.mark.parametrize("r", [0.25, 0.75])
def test_redundant_flag_agrees_with_cluster_band(r):
    """A pair is flagged redundant iff its value came from the shared pool."""
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    n_clusters = BASE["num_value_clusters"]
    cluster_hi = layout.value_start + n_clusters

    batch = make(redundancy_r=r)
    values = batch.values.numpy()
    from_cluster_band = (values >= layout.value_start) & (values < cluster_hi)
    np.testing.assert_array_equal(from_cluster_band, batch.is_redundant.numpy())


def test_is_shared_is_realized_sharing_not_the_draw_flag():
    """``is_shared`` must reflect actual collisions, and imply ``is_redundant``."""
    batch = make(redundancy_r=0.5, num_value_clusters=2, seed=3)
    values, shared = batch.values.numpy(), batch.is_shared.numpy()
    for ex in range(len(batch)):
        _, counts = np.unique(values[ex], return_counts=True)
        expected = counts[np.unique(values[ex], return_inverse=True)[1]] > 1
        np.testing.assert_array_equal(shared[ex], expected)
    # Unique values are drawn without replacement, so sharing only ever arises
    # through the cluster pool.
    assert not (shared & ~batch.is_redundant.numpy()).any()


def test_r_one_uses_only_cluster_values():
    batch = make(redundancy_r=1.0)
    assert batch.is_redundant.all()
    assert batch.values.unique().numel() <= BASE["num_value_clusters"]


# --- (a) r=0 reduces to vanilla MQAR -----------------------------------


def test_r_zero_disables_the_redundancy_machinery():
    batch = make(redundancy_r=0.0)
    assert not batch.is_redundant.any()
    assert not batch.is_shared.any()
    for ex in range(len(batch)):
        vals = batch.values[ex]
        assert vals.unique().numel() == BASE["num_kv_pairs"], "values not unique"


def test_r_zero_value_distribution_matches_independent_vanilla_reference():
    """(a) Compare against a reference that has no notion of redundancy.

    Only the *value-assignment* law is compared. Placement policy is checked by
    the structural tests above; comparing placement against a reference that
    makes its own placement choices would fail for reasons unrelated to r=0.
    """
    n_ex, n_kv = 600, BASE["num_kv_pairs"]
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    n_clusters = BASE["num_value_clusters"]

    batch = gen_redundant_mqar(
        **{**BASE, "num_examples": n_ex, "redundancy_r": 0.0, "seed": 11}
    )
    observed = batch.values.numpy().ravel()

    # Independent reference: uniform draw without replacement from the same
    # unique-value block, which is what vanilla MQAR does.
    rng = np.random.default_rng(999)
    unique_pool = np.arange(layout.value_start + n_clusters, layout.value_end)
    reference = np.concatenate(
        [rng.choice(unique_pool, size=n_kv, replace=False) for _ in range(n_ex)]
    )

    d = ks_two_sample(observed.astype(float), reference.astype(float))
    crit = ks_critical(observed.size, reference.size)
    assert d < crit, f"KS D={d:.4f} exceeds critical {crit:.4f}"

    # Per-token frequency within ~5 sigma of uniform over the pool.
    counts = np.bincount(observed - unique_pool[0], minlength=unique_pool.size)
    expected = observed.size / unique_pool.size
    assert np.max(np.abs(counts - expected)) < 5.0 * np.sqrt(expected)


def test_keys_uniform_over_key_band():
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    batch = gen_redundant_mqar(
        **{**BASE, "num_examples": 600, "redundancy_r": 0.0, "seed": 12}
    )
    keys = batch.keys.numpy().ravel()
    counts = np.bincount(keys - layout.key_start, minlength=layout.num_key_tokens)
    expected = keys.size / layout.num_key_tokens
    assert np.max(np.abs(counts - expected)) < 5.0 * np.sqrt(expected)


# --- structural / plumbing ---------------------------------------------


def test_bands_are_disjoint_and_pad_is_never_emitted():
    layout = VocabLayout.from_vocab_size(BASE["vocab_size"])
    batch = make(redundancy_r=0.5)
    seq = batch.input_ids
    assert (seq != 0).all(), "PAD leaked into the sequence"
    assert seq.max().item() < BASE["vocab_size"]
    assert batch.keys.min().item() >= layout.key_start
    assert batch.keys.max().item() < layout.key_end
    assert batch.values.min().item() >= layout.value_start
    assert batch.values.max().item() < layout.value_end


def test_lm_labels_agree_with_labels_and_ignore_elsewhere():
    batch = make(redundancy_r=0.5)
    lm = batch.lm_labels
    assert (lm != IGNORE_INDEX).sum().item() == len(batch) * BASE["num_queries"]
    gathered = torch.gather(lm, 1, batch.query_positions)
    assert torch.equal(gathered, batch.labels)


def test_shapes_and_dtypes():
    batch = make(redundancy_r=0.5)
    n, s, k, q = len(batch), BASE["seq_len"], BASE["num_kv_pairs"], BASE["num_queries"]
    assert batch.input_ids.shape == (n, s)
    assert batch.lm_labels.shape == (n, s)
    assert batch.query_positions.shape == (n, q)
    assert batch.labels.shape == (n, q)
    assert batch.query_kv_index.shape == (n, q)
    assert batch.is_redundant.shape == (n, k)
    assert batch.is_shared.shape == (n, k)
    assert batch.input_ids.dtype == torch.int64
    assert batch.is_redundant.dtype == torch.bool


def test_to_moves_every_tensor_field():
    """``.to()`` must not leave a field behind on the host.

    ``recall_metrics`` gathers ``query_positions`` / ``labels`` /
    ``query_kv_index`` / ``is_redundant`` / ``is_shared`` against the model's
    logits, so a field left un-moved is a cross-device index at eval time, on the
    GPU only -- invisible to this CPU-only suite unless the walk is checked
    structurally. Hence comparing field sets rather than naming tensors.
    """
    from dataclasses import fields

    batch = make(redundancy_r=0.5)
    moved = batch.to("cpu")

    tensor_fields = {
        f.name for f in fields(batch) if isinstance(getattr(batch, f.name), torch.Tensor)
    }
    assert tensor_fields, "expected some tensor fields"
    for name in tensor_fields:
        got = getattr(moved, name)
        assert isinstance(got, torch.Tensor), f"{name} stopped being a tensor"
        assert got.device.type == "cpu"
        assert torch.equal(got, getattr(batch, name)), f"{name} changed value"

    # Non-tensor fields pass through untouched rather than being dropped.
    assert moved.config == batch.config
    assert {f.name for f in fields(moved)} == {f.name for f in fields(batch)}
    assert len(moved) == len(batch)


def test_query_kv_index_joins_queries_to_redundancy_flags():
    batch = make(redundancy_r=0.5)
    per_query_redundant = torch.gather(
        batch.is_redundant, 1, batch.query_kv_index
    )
    expected_values = torch.gather(batch.values, 1, batch.query_kv_index)
    assert torch.equal(expected_values, batch.labels)
    assert per_query_redundant.shape == batch.labels.shape


def test_determinism():
    a = make(redundancy_r=0.5, seed=5)
    b = make(redundancy_r=0.5, seed=5)
    c = make(redundancy_r=0.5, seed=6)
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)
    assert not torch.equal(a.input_ids, c.input_ids)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(num_queries=32, num_kv_pairs=16), "not enough keys"),
        (dict(seq_len=8), "sequence cannot hold"),
        (dict(redundancy_r=1.5), r"must be in \[0, 1\]"),
        (dict(redundancy_r=0.5, num_value_clusters=0), "requires num_value_clusters"),
        (dict(num_value_clusters=200), "need at least"),
        (dict(num_kv_pairs=1000, seq_len=4096), "key band holds only"),
    ],
)
def test_validation_errors(overrides, match):
    with pytest.raises(ValueError, match=match):
        make(**overrides)


def test_packed_sequence_is_allowed():
    """Zero filler is a legal edge case: 2*kv + queries == seq_len exactly."""
    batch = gen_redundant_mqar(
        num_examples=8,
        seq_len=2 * 16 + 8,
        vocab_size=512,
        num_kv_pairs=16,
        num_queries=8,
        redundancy_r=0.5,
        num_value_clusters=4,
        seed=1,
    )
    assert batch.input_ids.shape == (8, 40)

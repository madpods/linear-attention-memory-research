"""Redundancy-parameterized Multi-Query Associative Recall (MQAR).

Stage 1 of ``linear_attention_memory_research_plan.md``.

MQAR is the synthetic associative-recall benchmark used in the DeltaNet paper.
A sequence scatters ``(key, value)`` pairs among filler tokens; later the key
token reappears as a *query* and the model must produce the value it was
bound to. Recall accuracy against ``num_kv_pairs`` is the standard capacity
curve.

This generator adds the redundancy axis ``r`` described in plan section 2:
with probability ``r`` a key's value is drawn from a small shared pool of
``num_value_clusters`` representatives instead of being unique to that key.
At ``r = 0`` every value is unique and the task reduces to vanilla MQAR. As
``r -> 1`` many distinct keys bind to identical values, which is exactly the
regime where cross-key deduplication (Stage 4) should pay for itself.

Conventions
-----------
Query tokens are the key tokens themselves reappearing later in the sequence;
there is no separate query marker. This is the standard MQAR formulation and
is what makes the task associative recall rather than lookup of a tagged slot.

The answer is *not* written into the sequence. Logits are read at
``query_positions`` and scored against ``labels``. Materializing the answer
would re-bind the pair mid-sequence and make every later query for the same
key progressively easier, contaminating the capacity curve.

Vocabulary is partitioned into disjoint bands ``[PAD] [keys] [values] [noise]``
(see :class:`VocabLayout`). Vanilla MQAR implementations sometimes draw filler
from the whole vocabulary; that makes the task ill-posed, because a filler
token that happens to equal a key acts as a spurious query with no defined
answer. Disjoint bands cost nothing and remove the ambiguity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch

#: Loss-ignored position in :attr:`MQARBatch.lm_labels`; matches ``F.cross_entropy``.
IGNORE_INDEX = -100

#: Reserved padding id. Never emitted by the generator.
PAD_ID = 0

_KV_WIDTH = 2  # a kv pair occupies two adjacent positions: key then value


@dataclass(frozen=True)
class VocabLayout:
    """Disjoint token bands over ``[0, vocab_size)``.

    Layout is ``[PAD] [key band] [value band] [noise band]``. The value band is
    further split by :func:`gen_redundant_mqar` into a leading block of
    ``num_value_clusters`` shared representatives and a trailing block of
    per-key unique values, so a token id alone tells you whether a value was
    drawn from the shared pool.
    """

    vocab_size: int
    num_key_tokens: int
    num_value_tokens: int

    @classmethod
    def from_vocab_size(cls, vocab_size: int) -> "VocabLayout":
        """Split the non-reserved vocabulary into roughly equal thirds."""
        if vocab_size < 4:
            raise ValueError(f"vocab_size must be >= 4, got {vocab_size}")
        usable = vocab_size - 1  # id 0 is PAD
        num_key_tokens = usable // 3
        num_value_tokens = usable // 3
        return cls(vocab_size, num_key_tokens, num_value_tokens)

    def __post_init__(self) -> None:
        if self.num_key_tokens < 1 or self.num_value_tokens < 1:
            raise ValueError("key and value bands must be non-empty")
        if self.num_noise_tokens < 1:
            raise ValueError(
                "noise band is empty; increase vocab_size or shrink the key/value bands"
            )

    @property
    def key_start(self) -> int:
        return 1

    @property
    def key_end(self) -> int:
        return self.key_start + self.num_key_tokens

    @property
    def value_start(self) -> int:
        return self.key_end

    @property
    def value_end(self) -> int:
        return self.value_start + self.num_value_tokens

    @property
    def noise_start(self) -> int:
        return self.value_end

    @property
    def noise_end(self) -> int:
        return self.vocab_size

    @property
    def num_noise_tokens(self) -> int:
        return self.noise_end - self.noise_start


@dataclass
class MQARBatch:
    """Generated MQAR examples.

    Fields specified by the plan
    ----------------------------
    input_ids
        ``(num_examples, seq_len)`` token ids.
    query_positions
        ``(num_examples, num_queries)`` position of each query token.
    labels
        ``(num_examples, num_queries)`` correct value token for each query.
    is_redundant
        ``(num_examples, num_kv_pairs)`` bool; whether that pair's value was
        drawn from the shared cluster pool.

    Additional fields
    -----------------
    query_kv_index
        ``(num_examples, num_queries)`` which kv pair each query refers to.
        Required to actually use ``is_redundant`` for the per-position analysis
        the plan asks for -- without it there is no join between a query and
        the redundancy flag of the pair it targets.
    is_shared
        ``(num_examples, num_kv_pairs)`` bool; whether that pair's value token
        is *actually* used by another pair in the same sequence. Differs from
        ``is_redundant``: a pair can be drawn from the cluster pool yet be the
        only user of that representative, in which case there is no
        deduplication opportunity to measure.
    keys, values
        ``(num_examples, num_kv_pairs)`` the bound token ids, for analysis.
    lm_labels
        ``(num_examples, seq_len)`` labels scattered to query positions with
        :data:`IGNORE_INDEX` elsewhere, for training loops that want a
        full-width target tensor.
    """

    input_ids: torch.Tensor
    query_positions: torch.Tensor
    labels: torch.Tensor
    is_redundant: torch.Tensor
    query_kv_index: torch.Tensor
    is_shared: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    lm_labels: torch.Tensor
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str) -> MQARBatch:
        """Move every tensor field to ``device``, leaving ``config`` alone.

        All of these travel together on purpose: :func:`lamr.metrics.recall_metrics`
        gathers ``query_positions`` / ``labels`` / ``query_kv_index`` /
        ``is_redundant`` / ``is_shared`` against the model's logits, so moving
        only ``input_ids`` would leave the metrics indexing across devices.

        Generated over the dataclass fields rather than named individually, so a
        field added later is not silently left behind on the host.
        """
        moved = {
            f.name: (
                value.to(device)
                if isinstance(value := getattr(self, f.name), torch.Tensor)
                else value
            )
            for f in fields(self)
        }
        return MQARBatch(**moved)


def _validate(
    *,
    num_examples: int,
    seq_len: int,
    num_kv_pairs: int,
    num_queries: int,
    redundancy_r: float,
    num_value_clusters: int,
    layout: VocabLayout,
) -> None:
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if not 0.0 <= redundancy_r <= 1.0:
        raise ValueError(f"redundancy_r must be in [0, 1], got {redundancy_r}")
    if num_queries > num_kv_pairs:
        raise ValueError(
            f"num_queries ({num_queries}) > num_kv_pairs ({num_kv_pairs}); each "
            "query targets a distinct key, so there are not enough keys to query"
        )
    if num_queries < 1 or num_kv_pairs < 1:
        raise ValueError("num_kv_pairs and num_queries must be >= 1")

    occupied = _KV_WIDTH * num_kv_pairs + num_queries
    if occupied > seq_len:
        raise ValueError(
            f"sequence cannot hold {num_kv_pairs} kv pairs + {num_queries} queries "
            f"({occupied} tokens) in seq_len={seq_len}"
        )
    if num_kv_pairs > layout.num_key_tokens:
        raise ValueError(
            f"need {num_kv_pairs} distinct keys but key band holds only "
            f"{layout.num_key_tokens}"
        )
    if num_value_clusters < 0:
        raise ValueError("num_value_clusters must be >= 0")
    if redundancy_r > 0 and num_value_clusters < 1:
        raise ValueError("redundancy_r > 0 requires num_value_clusters >= 1")

    # Worst case every pair is non-redundant, so the unique-value block must be
    # able to supply num_kv_pairs distinct tokens. Validating against the worst
    # case keeps the check deterministic rather than probabilistic in r.
    num_unique_values = layout.num_value_tokens - num_value_clusters
    if num_unique_values < num_kv_pairs:
        raise ValueError(
            f"value band holds {layout.num_value_tokens} tokens, of which "
            f"{num_value_clusters} are cluster representatives, leaving "
            f"{num_unique_values} for unique values -- need at least "
            f"{num_kv_pairs}. Increase vocab_size or decrease num_value_clusters."
        )


def _arrange_items(
    rng: np.random.Generator, num_kv_pairs: int, query_kv_index: np.ndarray
) -> list[tuple[str, int]]:
    """Order kv blocks and queries so every query follows the pair it targets.

    Returns a list of ``("kv", pair_index)`` / ``("q", query_slot)`` items.
    kv blocks are placed in a random order, then each query is inserted at a
    uniformly random slot strictly after its own pair. Filler is distributed
    into the gaps separately by the caller.
    """
    items: list[tuple[str, int]] = [("kv", int(j)) for j in rng.permutation(num_kv_pairs)]
    for slot in rng.permutation(len(query_kv_index)):
        pair = int(query_kv_index[slot])
        after = items.index(("kv", pair))
        # +1 .. len inclusive: anywhere strictly after the pair, including the end.
        insert_at = int(rng.integers(after + 1, len(items) + 1))
        items.insert(insert_at, ("q", int(slot)))
    return items


def gen_redundant_mqar(
    num_examples: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,
    num_queries: int,
    redundancy_r: float,
    num_value_clusters: int,
    seed: int,
    layout: VocabLayout | None = None,
) -> MQARBatch:
    """Generate redundancy-parameterized MQAR examples.

    Args:
        num_examples: number of sequences.
        seq_len: tokens per sequence.
        vocab_size: total vocabulary, partitioned by ``layout``.
        num_kv_pairs: distinct keys bound per sequence.
        num_queries: query positions per sequence; must be <= ``num_kv_pairs``.
        redundancy_r: probability a key's value comes from the shared pool.
        num_value_clusters: size of the shared pool when ``redundancy_r > 0``.
        seed: seeds a ``numpy`` generator; identical seeds give identical output.
        layout: vocabulary banding; defaults to equal thirds.

    Returns:
        :class:`MQARBatch`.
    """
    layout = layout or VocabLayout.from_vocab_size(vocab_size)
    if layout.vocab_size != vocab_size:
        raise ValueError(
            f"layout.vocab_size ({layout.vocab_size}) != vocab_size ({vocab_size})"
        )
    _validate(
        num_examples=num_examples,
        seq_len=seq_len,
        num_kv_pairs=num_kv_pairs,
        num_queries=num_queries,
        redundancy_r=redundancy_r,
        num_value_clusters=num_value_clusters,
        layout=layout,
    )

    rng = np.random.default_rng(seed)

    key_pool = np.arange(layout.key_start, layout.key_end)
    cluster_pool = np.arange(layout.value_start, layout.value_start + num_value_clusters)
    unique_pool = np.arange(layout.value_start + num_value_clusters, layout.value_end)
    noise_pool = np.arange(layout.noise_start, layout.noise_end)

    # Start from pure noise, then overwrite the structured positions.
    input_ids = rng.choice(noise_pool, size=(num_examples, seq_len))
    query_positions = np.empty((num_examples, num_queries), dtype=np.int64)
    labels = np.empty((num_examples, num_queries), dtype=np.int64)
    query_kv_index = np.empty((num_examples, num_queries), dtype=np.int64)
    is_redundant = np.zeros((num_examples, num_kv_pairs), dtype=bool)
    is_shared = np.zeros((num_examples, num_kv_pairs), dtype=bool)
    all_keys = np.empty((num_examples, num_kv_pairs), dtype=np.int64)
    all_values = np.empty((num_examples, num_kv_pairs), dtype=np.int64)

    for ex in range(num_examples):
        keys = rng.choice(key_pool, size=num_kv_pairs, replace=False)

        redundant = rng.random(num_kv_pairs) < redundancy_r
        # Draw the unique values without replacement so the non-redundant pairs
        # are guaranteed distinct; the redundant slots simply discard their draw.
        values = rng.choice(unique_pool, size=num_kv_pairs, replace=False)
        if num_value_clusters > 0 and redundant.any():
            drawn = rng.choice(cluster_pool, size=num_kv_pairs, replace=True)
            values = np.where(redundant, drawn, values)

        kv_for_query = rng.choice(num_kv_pairs, size=num_queries, replace=False)
        items = _arrange_items(rng, num_kv_pairs, kv_for_query)

        # Scatter the remaining budget of filler tokens across the gaps between
        # items (including before the first and after the last).
        num_filler = seq_len - (_KV_WIDTH * num_kv_pairs + num_queries)
        num_gaps = len(items) + 1
        gaps = rng.multinomial(num_filler, np.full(num_gaps, 1.0 / num_gaps))

        cursor = int(gaps[0])
        for gap_idx, (kind, idx) in enumerate(items, start=1):
            if kind == "kv":
                input_ids[ex, cursor] = keys[idx]
                input_ids[ex, cursor + 1] = values[idx]
                cursor += _KV_WIDTH
            else:
                pair = int(kv_for_query[idx])
                input_ids[ex, cursor] = keys[pair]
                query_positions[ex, idx] = cursor
                labels[ex, idx] = values[pair]
                query_kv_index[ex, idx] = pair
                cursor += 1
            cursor += int(gaps[gap_idx])

        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        is_shared[ex] = counts[inverse] > 1
        is_redundant[ex] = redundant
        all_keys[ex] = keys
        all_values[ex] = values

    lm_labels = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    np.put_along_axis(lm_labels, query_positions, labels, axis=1)

    config = {
        "num_examples": num_examples,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_kv_pairs": num_kv_pairs,
        "num_queries": num_queries,
        "redundancy_r": redundancy_r,
        "num_value_clusters": num_value_clusters,
        "seed": seed,
        "layout": asdict(layout),
    }

    return MQARBatch(
        input_ids=torch.from_numpy(input_ids.astype(np.int64)),
        query_positions=torch.from_numpy(query_positions),
        labels=torch.from_numpy(labels),
        is_redundant=torch.from_numpy(is_redundant),
        query_kv_index=torch.from_numpy(query_kv_index),
        is_shared=torch.from_numpy(is_shared),
        keys=torch.from_numpy(all_keys),
        values=torch.from_numpy(all_values),
        lm_labels=torch.from_numpy(lm_labels),
        config=config,
    )

"""Shared evaluation metrics.

Built once and invoked identically by every stage, per the plan's cross-cutting
requirement -- results from different stages have to be directly comparable
without post-hoc reconciliation.

The redundancy slice is the point of the whole harness: Stage 4's clustering
mechanism is predicted to help on redundant pairs specifically, so an aggregate
accuracy number cannot confirm or falsify it. ``by_redundant`` and
``by_non_redundant`` are what that claim gets tested against.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lamr.data.mqar import MQARBatch


@torch.no_grad()
def recall_metrics(logits: Tensor, batch: MQARBatch, slice_index: slice | None = None):
    """Recall accuracy overall and sliced by redundancy.

    Args:
        logits: ``(B, T, vocab)``. Read at query positions with no next-token
            shift -- see ``lamr.models.lm`` for the convention.
        batch: the examples the logits were produced from.
        slice_index: restrict to a sub-range of ``batch`` (for minibatched eval).

    Returns:
        dict with ``accuracy``, ``accuracy_redundant``,
        ``accuracy_non_redundant``, ``accuracy_shared``, and the query counts
        each was computed over. Accuracies over an empty slice are ``nan``,
        which keeps them out of averages rather than silently reading as 0.
    """
    idx = slice_index or slice(None)
    query_positions = batch.query_positions[idx]
    labels = batch.labels[idx]
    query_kv_index = batch.query_kv_index[idx]

    gathered = torch.gather(
        logits, 1, query_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    )
    correct = gathered.argmax(dim=-1) == labels

    redundant = torch.gather(batch.is_redundant[idx], 1, query_kv_index)
    shared = torch.gather(batch.is_shared[idx], 1, query_kv_index)

    def masked_mean(mask: Tensor) -> float:
        n = int(mask.sum())
        return float(correct[mask].float().mean()) if n else float("nan")

    return {
        "accuracy": float(correct.float().mean()),
        "accuracy_redundant": masked_mean(redundant),
        "accuracy_non_redundant": masked_mean(~redundant),
        "accuracy_shared": masked_mean(shared),
        "num_queries": int(correct.numel()),
        "num_redundant": int(redundant.sum()),
        "num_shared": int(shared.sum()),
    }

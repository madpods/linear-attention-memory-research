# Addendum 02 — State accounting and the query→cluster read path (settle before Stage 4b)

Companion to `linear_attention_memory_research_plan.md` and `addendum_01_residual_matrix.md`. These two decisions must be fixed on paper before Stage 4b (clustering + residual matrix combined) is implemented — both are cheap to settle now and expensive to discover mid-implementation.

## Decision 1 — R is the S-equivalent, not an addition to a separate S

`R` (the residual matrix from Addendum 01) is the same shape as, and functionally replaces, the plain state `S` from vanilla delta rule — same `d_k × d_v` shape, same delta-rule update rule, only the write target differs (`r` instead of raw `v`). It is **not** maintained alongside a separate full-size `S`.

The overhead beyond a plain-`S` baseline (e.g. Gated DeltaNet) is the codebook, not `R` itself. State size, stated as a formula rather than a fixed claim:

```
baseline (plain S):        d_k × d_v
this design (R + codebook): d_k × d_v  +  num_clusters × (d_k + d_v)
```

**Consequence for experiment design:** "negligible overhead" and "double the state" are both possible outcomes depending on `num_clusters` — this is not fixed by the architecture, it is set by a hyperparameter. Report codebook size explicitly (`num_clusters × (d_k + d_v)`) in every Stage 4b run's logged config, alongside `R`'s fixed `d_k × d_v`. **The honest control is a Gated DeltaNet baseline matched on the total (`R` + codebook), not matched on `R` vs. `S` alone** — a run with a large `num_clusters` must be compared against a correspondingly larger baseline, or the comparison silently favors this design.

## Decision 2 — the codebook needs two matrices, not one, because write-time and read-time addressing use different signals

**The asymmetry:** cluster assignment at write time compares the incoming `v` against existing cluster values — `v` is available, it's part of the input pair. At read time, `v` is exactly what's being retrieved — it is not available to match against. A query only has `q`, which lives in key-space, not value-space. The write-time lookup mechanism (value similarity) cannot be reused at read time. This was unresolved in the original plan (flagged generally in §10/§11 as "does the read step need to know the regime it's reading from") — Stage 4b forces a concrete answer.

**Resolution:** the codebook is two matrices, not one:
- `C_keys` — `num_clusters × d_k`, the running average of keys that have been routed to each cluster.
- `C_values` — `num_clusters × d_v`, the cluster's representative value (what Addendum 01 called `c`).

Write path (per token, `k`, `v`):
1. Assign to nearest cluster by comparing `v` against `C_values` (as in Addendum 01).
2. Update `C_keys` for that cluster with a running average incorporating `k` (new — was previously unspecified).
3. Update `C_values` for that cluster with a running average incorporating `v` (as in Addendum 01).
4. Compute residual `r = v − c` and delta-rule-write it into `R` as before.

Read path (per query `q`), replacing the single-lookup version in Addendum 01:
```
weights = softmax(q · C_keys^T)      # or hard argmax — a (1 × num_clusters) lookup
c        = weights @ C_values
v̂        = c + q^T R
```

This is structurally a small nested key/value memory lookup — the same key-indexes-value pattern the rest of the architecture already uses, just at codebook scale. Because `num_clusters` is small, batch this the same way as every other chunked operation in the plan (§15): for a whole chunk of queries, `Q_chunk @ C_keys^T` is one matmul producing an `n × num_clusters` similarity matrix, no per-token loop required — this does not reintroduce the sequential/search cost problem as long as `num_clusters` stays bounded.

## Why these two decisions reinforce the unbounded-centroids caution (already flagged separately)

Unbounded cluster growth (`num_clusters` growing without a cap) now has three compounding costs, not one:
1. Write-side: comparing each new token against a growing `C_values`.
2. Read-side (new, from Decision 2): every query now also pays `O(num_clusters)` against `C_keys`.
3. State-size (new, from Decision 1): the codebook term in the state-size formula above grows without bound, eventually rivaling or exceeding `R`'s fixed cost.

All three degrade together if `num_clusters` isn't capped or actively consolidated by the slow pass. This is not a new risk, but it is now three separate, compounding reasons for the same mitigation (bounded `num_clusters`, or unbounded creation paired with mandatory periodic consolidation) rather than one.

## What to implement in Stage 4b given these decisions

- [ ] Codebook is `(C_keys, C_values)`, both maintained per head, with the running-average update specified above.
- [ ] Write path: nearest-cluster lookup by `v`-similarity against `C_values` (as in Addendum 01), plus the `C_keys` update this addendum adds.
- [ ] Read path: `q`-similarity lookup against `C_keys` (softmax or hard argmax — implement both, compare), producing `c` from `C_values`, combined with `q^T R`.
- [ ] Log, per run: `num_clusters` actually in use over time (ties to the cluster-count-over-time metric from the unbounded-centroids discussion), and total state size using the Decision 1 formula.
- [ ] Baseline comparison: Gated DeltaNet sized to match total state (`R` + codebook), not just `R` alone, for every `num_clusters` setting tested.

# Addendum 03 — Key routing must be delta-rule-trained, not a running mean (revises Addendum 02, Decision 2)

Companion to `linear_attention_memory_research_plan.md`, `addendum_01_residual_matrix.md`, and `addendum_02_state_accounting_and_read_path.md`. This revises the `C_keys` design from Addendum 02's Decision 2. Do not implement Addendum 02's running-mean version of `C_keys` — implement this version instead. Decision 1 (state-size accounting) from Addendum 02 is unaffected and still holds.

## The bug: running-mean centroids collapse exactly where the mechanism is supposed to help

§2's premise is cross-key aliasing — different keys, unrelated in key-space, whose *values* happen to be similar enough to share a cluster. So by construction, the `m` keys routed to one cluster are close to random/orthogonal with respect to each other.

Averaging `m` near-orthogonal unit vectors produces a mean whose norm shrinks as `1/√m` (sum of `m` roughly-independent unit vectors has norm `~√m`; the average is `~√m/m`). At read time, a query matching one member contributes signal `~1/m` to its own cluster's score, while a competing cluster returns noise of order `1/√(m'·d_k)` from `m'` unrelated dot products. Solving for when the true cluster's signal beats the worst of `num_clusters − 1` competitors:

```
m  ≲  d_k / (2 · ln(num_clusters))
```

Example: `d_k = 16`, `num_clusters = 16` → `m ≲ 3` keys per cluster before routing degrades toward noise. **Many keys per cluster is the entire benefit of the mechanism — it's what saves capacity — and it's exactly the regime this scaling kills.** The design as specified in Addendum 02 works best precisely where it buys the least.

**Compounding effect, not in the base derivation:** real cluster sizes won't be equal — usage/frequency evidence discussed earlier in this project (e.g. attention mass following a power-law distribution) suggests some clusters will be sparsely populated. A cluster with small `m'` has a *larger*-magnitude, noisier mean (`1/√(m'·d_k)` grows as `m'` shrinks), so sparse clusters are systematically louder at read time than well-populated ones — biasing routing toward small clusters on top of the base signal-vs-noise problem, working against exactly the well-populated, high-value clusters the mechanism is meant to protect.

## The fix: a delta-rule-trained routing matrix, not an average

Replace the running-mean `C_keys` with a routing matrix `M` (`d_k × num_clusters` — same shape and same state-size cost as the running-mean version it replaces, see Addendum 02 Decision 1, unaffected by this change).

**Write path**, per token `(k, v)` (extends Addendum 02's write path):
1. Assign to nearest cluster by comparing `v` against `C_values` (as before — administrative purpose only, see below).
2. Update `M` via ordinary delta rule, targeting a one-hot vector for the assigned cluster `c`:
   ```
   target = one_hot(c, num_clusters)
   error  = target − k^T M_prev
   M      = M_prev + β · k · error^T
   ```
3. Update `C_values` for that cluster as before.
4. Compute the residual and write `R` — **see the write/read consistency fix below before implementing this step.**

**Read path**, per query `q` (replaces the `C_keys`-based lookup in Addendum 02):
```
logits  = q^T M                       # (1 × num_clusters)
weights = softmax(logits)             # or hard argmax — implement and compare both
c       = weights @ C_values
v̂       = c + q^T R
```
Batches the same way as every other chunked operation in the plan: `Q_chunk @ M` for a whole chunk is one matmul producing an `n × num_clusters` logit matrix.

**Why this doesn't inherit the same scaling failure:** averaging is solving *recall* — reconstructing `m` individually distinguishable vectors from one summary, which is exactly the class of problem plain accumulation (no error correction) is already known to be bad at, for the same reason established throughout this project. The delta-rule router reframes the problem as *classification* — mapping `m` inputs to one shared target. Keys sharing a cluster reinforce the same decision boundary instead of competing for capacity, so more members per cluster should sharpen the boundary rather than degrade it — the opposite direction from the averaging failure.

**This is not a free pass — flag as an open empirical question, not a solved one:** a linear multi-class separator's difficulty still grows with `num_clusters` (more classes to keep separable), so some degradation curve should be expected, just a much more forgiving one than `1/√m`. Do not assume it away.

## Required test before trusting any Stage 4b recall number

Build and run this in isolation, before wiring `M` into the full clustering + residual pipeline:
- [ ] Synthetic routing-accuracy test: generate `num_clusters` random cluster assignments, `m` near-orthogonal keys per cluster, train `M` via the delta-rule update above, measure query→cluster routing accuracy.
- [ ] Sweep **both** `m` (keys per cluster) and `num_clusters` independently — confirm routing accuracy holds up (or characterize how it degrades) well past the point where the running-mean version was predicted to fail (`m ≳ d_k / (2 ln num_clusters)`).
- [ ] Do not proceed to end-to-end MQAR recall numbers for Stage 4b until this test passes at the `m` / `num_clusters` values the redundancy-sweep experiments actually intend to use. A broken router produces the same symptom as "clustering doesn't help" in an end-to-end recall curve — this isolation step is what prevents misattributing a routing bug to the mechanism itself.

## Write/read residual consistency fix (Addendum 02 Decision 2, second issue)

The residual stored during write was `r = v − c_write`, where `c_write` came from `v`-similarity assignment. Reconstruction at read time uses `c_read`, produced by the `q`-similarity routing above (Decision 2's read path, whether hard or soft). Nothing forces these to agree — a softmax blend at read time in particular cannot reproduce a hard single-cluster assignment made at write time, so the mismatch is not just a training-noise issue, it's a structural bias.

**Fix:** compute the residual against the read-time reconstruction path, not the write-time assignment path. `v`-similarity is used **only** to decide the administrative question — which cluster's `C_values` and `M` entry to update, or whether to spawn a new cluster — never to define what gets subtracted for the residual target. Concretely, at write time, after (or using a consistent snapshot of) the current `M` and `C_values`, compute `c_read` by running the same query mechanism the read path will later use, with `k` playing the role of the query:
```
c_read_at_write = softmax(k^T M) @ C_values      # same computation the read path performs
r = v − c_read_at_write
```
This makes reconstruction exact by construction (up to `R`'s own delta-rule convergence) rather than coincidentally correct only when write-time and read-time addressing happen to agree.

## Summary of what changes vs. Addendum 02

| | Addendum 02 (superseded) | This addendum |
|---|---|---|
| `C_keys` update | running mean of routed keys | delta-rule write, one-hot(cluster) target |
| Read routing | `q · C_keys^T` similarity | `softmax(q^T M)` or argmax |
| Residual target | `v − c_write` (`v`-similarity) | `v − c_read` (same path as reconstruction) |
| State size | `num_clusters × (d_k + d_v)` codebook + `R` | unchanged — `M` is the same shape as the `C_keys` it replaces |
| Validated before Stage 4b full pipeline | — | isolated routing-accuracy test, swept over `m` and `num_clusters` |

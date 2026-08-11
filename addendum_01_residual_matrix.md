# Addendum 01 — Residual Matrix for Cluster Collapse (Stage 4 extension)

Companion to `linear_attention_memory_research_plan.md`. Read that file first — this extends **Stage 4** (clustering / pointer write-gating, §2 of the research log) rather than replacing it. **Insert this into the plan immediately after Stage 4's basic pointer mechanism is correctness-validated in sequential mode, and build/test it before Stage 5 (entropy-based eviction).**

## The problem this solves

The clustering mechanism from Stage 4 has an unaddressed failure mode: aliasing multiple keys to the same cluster centroid means those keys all retrieve the *identical* value. Two real costs follow from that:
1. **Collapse** — any per-key nuance the raw value carried beyond its cluster gets thrown away entirely, not just compressed.
2. **Drift fragility** — if two keys' true values only stay similar for a while and then diverge, the aliasing has no graceful way to absorb that; the only fix in the original plan was a hard, periodic, expensive reassignment/split pass (Stage 5's slow path).

## The idea — a second, residual state matrix

Real precedent: **Residual Vector Quantization (RVQ)**, the technique behind modern neural audio codecs (SoundStream, EnCodec) and VQ-VAE-2's multi-level codebooks — same problem (a coarse codebook collapses fine detail), same fix (a second stage that encodes only what the first stage missed).

Add a matrix `R`, same shape as `S` (`d_k × d_v`), maintained alongside the existing cluster codebook. At every token:

```
c  = nearest_cluster(k)            # coarse lookup, shared across aliased keys (existing Stage 4 logic)
r  = v - c                         # residual: what THIS key needed beyond its cluster's value
R  = R_prev + β · k · (r - k^T R_prev)^T   # write the residual via ordinary delta rule
```

Read becomes two-stage instead of one:

```
v̂ = c(k) + k^T R
```

— coarse cluster prediction plus a per-key correction pulled from `R`. Two keys aliased to the same cluster no longer retrieve identical values; each still gets its own individualized correction.

## Why this is worth the added compute (intuition, for context in code comments / writeup)

- `R` is not compressing raw values — it's compressing **residuals**, which have much lower variance and much less redundancy across keys than the raw values did, *if* clustering is doing its job. A rank-limited associative memory represents a low-variance, decorrelated source far more faithfully than a high-variance one at equal capacity. This is the same argument that makes RVQ beat a single flat codebook of equivalent total size at equal bit budget in the audio-codec literature — a reproduced empirical result in a closely analogous compression setting, not just a plausible story.
- `r = v - c` is already an error/correlation signal, so writing it via delta rule means: if a value matches its cluster tightly, `r ≈ 0` and the residual write is automatically near-zero. The "don't waste capacity on redundant info" property from the original correlation-gating idea falls out for free — no separate gating logic needed on top.
- **Two-for-one with drift (the real reason to prioritize this above Stage 5):** the original drift problem — aliased keys' true values diverging over time — no longer needs a hard reassignment/split to fix. Divergence gets absorbed automatically as a growing residual, up to `R`'s own capacity, instead of requiring the expensive periodic re-evaluation pass. **This may reduce or eliminate the need for Stage 5's SVD-based eviction entirely — test this before building Stage 5's more expensive machinery.**
- **Where this is a bad trade:** at low redundancy (`r` near 0 in the MQAR redundancy sweep — not to be confused with the residual matrix `R`, unfortunate notation collision, keep these clearly distinguished in code/config), the residual collapses back to ≈ the raw value and the cluster contributes nothing — you're paying for two writes to do the job one plain delta-rule write already did. The payoff is entirely conditional on real redundancy existing in the data, same as the base clustering mechanism it extends.

## Implementation spec

**Naming collision warning:** the MQAR redundancy sweep parameter is `redundancy_r` (float, 0–1, controls the data). The residual matrix here is `R` (a `d_k × d_v` matrix). Do not name any variable `r` in code that touches both — use `redundancy_r` and `residual` / `resid_matrix` explicitly.

- [ ] Extend the Stage 4 clustering module with a second `d_k × d_v` state matrix `R`, initialized to zero, one per head (same head structure as `S`).
- [ ] Per-token forward logic:
  1. Run existing Stage 4 cluster lookup to get `c` (the matched centroid's value vector, or the fresh-write path if no match).
  2. Compute `resid_target = v - c`.
  3. Apply delta rule to `R` using `k` and `resid_target` exactly as in the base delta-rule update from the original plan's §1 — same `β`, same error-correction structure, just targeting the residual instead of the raw value.
  4. Read: `v_pred = c + k^T R` (or `φ(q)^T R` at actual query time, consistent with the normalized read step in §11 of the main plan).
- [ ] Config knobs to expose (add to the same experiment-config system used for Stage 4/5 knobs): enable/disable residual matrix (so it's directly A/B-able against plain Stage 4), `R`'s own `β` (can differ from the cluster-write `β`), whether `R` gets its own decay/gating (optional; start without one and add only if plain accumulation in `R` proves unstable).
- [ ] **Test plan:** reuse the exact redundancy-swept MQAR eval from Stage 2/4 (recall accuracy vs. `redundancy_r`, vs. `num_kv_pairs`). Report **with** and **without** the residual matrix, holding everything else fixed. Specifically check:
  - Does adding `R` close most of the accuracy gap between plain clustering (Stage 4 alone) and the Gated DeltaNet baseline at low `redundancy_r`, where plain clustering was expected to underperform?
  - Does it preserve plain clustering's advantage at high `redundancy_r`?
  - Track throughput cost (tokens/sec) of the added matrix explicitly — this is the concrete number that answers "is it worth it" empirically rather than by intuition.
  - **Drift test specifically:** construct a variant of the redundancy-swept task where a cluster's true member values slowly diverge partway through a long sequence (two keys start near-identical, drift apart over time). Compare recall accuracy on this variant with `R` enabled vs. with Stage 5's periodic reassignment pass enabled vs. neither. This is the direct test of the "two-for-one" claim above — if `R` alone matches or beats the Stage 5 slow-pass on this task, Stage 5 can be deprioritized or dropped.

## Priority note for the coding agent

Given current progress is at Stage 2 in the main plan: build and validate this residual-matrix extension as part of finishing Stage 4, **before** starting Stage 5. If the drift test above shows `R` alone handles drift adequately, treat Stage 5 (entropy/SVD-based eviction) as optional follow-up work rather than a required stage, and note that explicitly in results before moving to Stage 6.

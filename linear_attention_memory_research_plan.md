# Linear Attention Memory Research — Log & Implementation Plan

This document has two parts. **Part 1** is a detailed narrative record of the research ideas developed in conversation, in the order they came up, with the math formalized and the real-world precedent for each piece named. **Part 2** is a staged, concrete implementation plan meant to be handed to a coding agent for rapid prototyping. Each stage in Part 2 is scoped to be independently buildable and testable, so progress can be tracked stage by stage rather than needing the whole system built before anything runs.

The throughline: this is fundamentally an **online lossy compression problem**. A linear-attention state matrix `S` has fixed size but the token stream is unbounded, so every mechanism below is either (a) a smarter *encoder* — deciding what gets written and how much space it takes — or (b) a way to raise the *capacity ceiling* itself.

---

## Part 1: Research Log

### 1. Origin — independently deriving the delta rule

Starting point was a whiteboard sketch of a state matrix `S`, a key `K`, a value `v`, and a translation matrix `T`, with the intuition: *"if [the key] already has a similar value in the state, store a translation of that key instead in matrix T."*

This is the **delta rule** (Schlag, Irie & Schmidhuber, 2021), the same update rule underlying DeltaNet, Gated DeltaNet, and Kimi Delta Attention (KDA).

**Formalization.** State `S` is a `d_k × d_v` matrix built by accumulating outer products:

```
S_t = S_{t-1} + k_t v_t^T          (plain linear attention, no correction)
```

Delta rule replaces the raw write with an error-corrected write:

```
v_read  = k_t^T S_{t-1}                 # what the state currently predicts for this key
e_t     = v_t - v_read                  # prediction error ("translation")
T_t     = β · (k_t ⊗ e_t) = β k_t e_t^T # rank-1 correction, β = write strength / learning rate
S_t     = S_{t-1} + T_t
```

If the new value already matches what the state predicts (`e_t ≈ 0`), the write is nearly a no-op — no redundant information gets added. This directly reduces interference versus plain linear attention, which just keeps superimposing new outer products regardless of redundancy.

**Worked numeric example** (from the original sketch): `S = [[0,9],[9,0]]`, `k = [1,0]`, `v = [5,9]`.
- `v_read = k^T S = [0,9]`
- `e = v - v_read = [5,0]`
- `T = k^T e = [[5,0],[0,0]]`
- `S_new = S + T = [[5,9],[9,0]]`

Row 0 (addressed by `k=[1,0]`) updates from `[0,9]` to `[5,9]`; row 1 is untouched since `k` has zero component there. This matches delta rule exactly.

### 2. Refinement — cross-key aliasing ("pointer") memory

Delta rule only checks *temporal* consistency: does the *same* key's value change over time. The next idea was different in kind: when a **new, different** key arrives, check whether *any existing, different key* already holds a similar value. If so, don't spend state capacity writing a fresh pattern — just record that the new key **points at** the existing one.

This is deduplication in the key dimension, not error-correction over time. Real precedent:
- **Vector-quantization / codebook memory** (VQ-VAE style): maintain a small set of representative value clusters; new (k,v) gets matched to nearest cluster; only a genuinely novel value spends a fresh write.
- **Product-key memory** (Lample et al., 2019): large sparse memory layers using exactly this kind of clustered addressing.

Mechanically: a separate structure `T` (the "pointer table") maps `key → cluster id`. `S` only ever holds as many unique patterns as there are clusters, regardless of how many keys arrive.

**Open problem raised immediately:** if two keys alias to the same cluster because their values were only *approximately* similar, and the true values later diverge, the aliasing becomes wrong for one of them ("drift").

### 3. Drift correction — periodic, more expensive re-evaluation

Proposed fix for aliasing drift: run a cheap approximate assignment step every token (fast path), and periodically run a more expensive exact re-evaluation that rechecks every existing pointer assignment and splits/reassigns clusters that have drifted apart (slow path).

Real precedent, three fields independently arrive at the same two-tier structure:
1. **k-means / Lloyd's algorithm** — cheap nearest-centroid assignment, periodic expensive centroid recomputation + reassignment.
2. **Product-key memory** — codebooks get periodically refreshed by re-clustering keys that accumulated against them.
3. **Complementary Learning Systems** (McClelland, neuroscience) — hippocampus does fast approximate binding of new experience; slow replay during sleep reconsolidates it into neocortex, correcting/reorganizing associations. This is the closest conceptual analogue: two systems because a system cheap enough to run constantly can't also be accurate enough to trust forever.

**Open design question:** fixed-interval slow pass (simple, possibly wasteful) vs. drift-triggered threshold (cheaper on average, but requires tracking drift continuously, partially defeating the "barely touch anything" savings of the fast path).

### 4. Capacity limits, and how to raise them

`S` is `d_k × d_v`, so `rank(S) ≤ min(d_k, d_v)`. Past that many independent associations, interference is mathematically guaranteed regardless of update rule — this is a property of the matrix, not the learning rule.

**Two levers to reduce interference within the existing ceiling:**
- **Orthogonality of keys** — interference comes from `k_i · k_j ≠ 0`. Random high-dim vectors are nearly orthogonal by default (concentration of measure); *learned* keys can collapse toward each other, silently eating capacity.
- **Kernel feature maps** — project `k` into a higher-dimensional `φ(k)` before writing, buying more effective separation without inflating the base vector width used elsewhere in the model. (Expanded in section 17.)

**The exponential escape hatch:** Modern Hopfield networks (Ramsauer et al., *"Hopfield Networks is All You Need"* — the paper that reframes softmax attention as associative memory) show that capacity scales **exponentially** in `d` if retrieval is exponential/softmax rather than plain linear dot-product. The nonlinearity in the **read** step, not the write step, is what buys softmax attention its extra effective capacity over plain linear attention with the same state size.

**What decay/forget gates actually buy:** Mamba2, GLA, Gated DeltaNet decay gates don't raise `rank(S)`'s ceiling — they increase *effective* capacity by freeing slots from stale patterns (eviction, not expansion).

The clustering/pointer idea from section 2 is a fourth lever distinct from all of these: instead of raising capacity, it **lowers demand** on it by refusing to let redundant values consume a slot at all.

### 5. Frequency-weighted retention vs. pure recency

Extension of the goal: older-but-still-relevant tokens should be able to stay in state longer than pure recency-decay allows, as long as semantically similar things keep recurring. This is **frequency-weighted retention** layered on top of recency-based decay.

Real precedent:
- **LRU vs. LFU caching**, and **Adaptive Replacement Cache (ARC)**, which dynamically balances recency and frequency rather than committing to one — pure LRU evicts a genuinely important item right before it's needed again if it hasn't been touched recently.
- **Spaced repetition** (cognitive science): items recalled correctly get their retention interval extended; unreviewed items decay on a normal curve. Semantic recurrence = successful "recall," which should push back the eviction clock.

**Contrast with Titans' surprise-based retention** (see section 7): surprise protects things because they're *rare and unexpected*; this mechanism protects things because they're *recurring and consistent*. These guard against opposite failure modes (losing rare anomalies vs. losing common-but-important themes) and are natural to stack rather than pick between.

**Risk flagged:** without weighting hit-count by match tightness, a frequent-but-shallow pattern (filler, generic transitions) could permanently squat on a slot through sheer repetition, crowding out a rare pattern that only appears twice but matters both times.

### 6. Sanity check #1 (honest assessment)

- **Solid:** the core instinct — decoupling "does this need to change the state" from "write it regardless" — is correct and is the actual reason delta-rule architectures beat plain linear attention.
- **Not individually novel:** delta rule (2021), RLS/Kalman adaptive gain (decades old), Titans (Dec 2024), product-key/codebook memory (2019), ARC (2003), spaced repetition (1970s) all have direct precedent for the pieces proposed so far. Reconstructing them from first principles is a legitimate way to deeply understand *why* they're shaped the way they are.
- **Genuinely unsettled:** the specific *combination* of cross-key clustering + periodic re-evaluation + frequency-weighted retention in one coherent mechanism has no direct precedent found. That combination is the real novel surface area — and the hardest part, since it means debugging interactions between three feedback loops with no existing ablation data to lean on.
- **Two concrete risks that could kill it in practice:**
  1. **Nearest-neighbor search cost** — every new key needs a similarity check against existing clusters; naive search breaks the linear-time property that's the whole point of this architecture family (this is exactly why product-key memory needed a specific factored structure).
  2. **Hyperparameter surface** — similarity threshold, β decay curve, re-evaluation interval/trigger, frequency-counter decay, split-threshold for de-aliasing. Many interacting knobs; small-scale toy tests often hide instabilities that only appear at scale.
- **Recommendation:** don't design the whole system on paper first — build the smallest synthetic testbed, implement just the pointer/aliasing piece on top of vanilla delta rule, and see if it beats DeltaNet on a task specifically designed to reward deduplication of near-duplicate values across distinct keys.

### 7. Titans' surprise mechanism, formalized

Titans (Behrouz et al., Google, Dec 2024) argues prior fast-weight rules are based on *momentary* surprise, miss the flow of surprise across the sequence, and mostly lack a forgetting gate. Their fix, formalized (renaming their "S" for surprise to `Su` here to avoid clashing with the state matrix `S`):

```
loss:              ℓ(M; k_t, v_t) = ‖M(k_t) − v_t‖²
momentary surprise: raw = −θ_t · ∇ℓ(M_{t-1}; k_t, v_t)
with momentum:      Su_t = η_t · Su_{t-1} + raw
forget gate:        M_t  = (1 − α_t) · M_{t-1} + Su_t
```

`η_t` (momentum decay) and `α_t` (forget rate) are both **learned, data-dependent, and independent of each other** — this independence is the key design choice: a token that isn't individually shocking but arrives right after a run of surprising tokens still gets written strongly (via momentum), and forgetting is a genuinely separate control, not derived purely from the surprise signal driving the write.

### 8. Combining surprise + correlation

Surprise (section 7) and the correlation/frequency signal (section 5) protect against **opposite failure modes** — rare-and-unexpected vs. recurring-and-consistent — so combining them is complementary, not redundant:

|                     | Recurring (correlation high) | Rare (correlation low) |
|---------------------|-------------------------------|--------------------------|
| **Surprising**      | protect hardest — important theme | protect short-term via momentum, may fade if truly one-off |
| **Unsurprising**     | protect moderately — cap to avoid filler lock-in | decay fastest — correctly forgotten |

**Mechanism:** write strength becomes a function of *both* signals as **independent inputs** to the same gate:

```
β_t = f(surprise_t, correlation_t)
```

Critically, **not** one derived from the other — mirroring Titans' choice to keep momentum and the forget gate independent rather than coupled, which avoids a specific instability: any signal that's self-referential (built from comparing against its own past predictions, then used to gate its own future updates) risks a "rich-get-richer" lock-in, where an early lucky match inflates confidence and then future *mismatches* get underweighted exactly when they should be correcting the error. RLS/Kalman filtering handles this with a hard forgetting-factor ceiling; Titans handles it by keeping the two gates structurally separate.

### 9. Multi-head split — heterogeneous update rules per head

Rather than fusing surprise-gating and correlation-gating into one scalar `β`, split them across heads: standard multi-head linear attention already gives each head its own small state `S_h`, computed independently and concatenated at the end. So some heads run the surprise+momentum+forget update rule, other heads run the correlation/clustering update rule, each with their own capacity. The read step doesn't need to change — still `q^T S_h` per head, concatenated, projected.

**Why this is stronger than fusing into one gate:** physically separate state matrices mean a frequent-but-shallow pattern in a correlation-head literally cannot compete for capacity with a rare-but-critical pattern living in a surprise-head's separate memory. That's a structural fix, not a hyperparameter compromise.

**Real precedent:** Hymba (NVIDIA, 2024) runs attention heads and SSM (Mamba) heads in parallel within the same layer on the same input, specifically because the two mechanisms are good at different things and forcing one to do both is a worse compromise than letting each specialize.

**What's actually new:** Hymba mixes two different *architectures* (attention vs. SSM). Mixing surprise-gated and correlation-gated heads is mixing two different *update rules within the same linear-attention framework* — narrower and, as far as could be determined, not directly precedented.

**Open design question:** fixed head-type ratio (e.g., 3 surprise-heads / 5 correlation-heads, set architecturally) vs. learned soft assignment (a head drifts toward whichever behavior the data rewards, mixture-of-experts-over-memory-dynamics). Fixed is cheaper and easier to reason about; learned is more powerful but reintroduces a coarser-grained version of the same lock-in risk (a whole head collapsing to one regime, rather than one gate value drifting).

### 10. Sanity check #2 (honest assessment, multi-head split)

- **Solid:** separating incompatible objectives into separate heads instead of fusing them into one gate is a documented-sound pattern, not just a nice idea.
- **Overclaimed previously:** the *general* pattern (type-heterogeneous heads) has precedent (Hymba). What's untested is the *specific pairing* — surprise-gated heads next to correlation/clustering heads. No ablation data anywhere confirms these two mechanisms are complementary rather than redundant or actively unhelpful together. Reasonable hypothesis, not validated.
- **Underweighted risk — fixed ratio:** assumes in advance how much of each behavior a task needs, which almost certainly varies by layer and task; get it wrong and the "wrong" heads for that task contribute close to nothing, yielding a worse version of plain Gated DeltaNet with fewer effective heads doing real work.
- **Underweighted risk — learned ratio:** reopens lock-in risk at a coarser granularity (a whole head going dead is a bigger capacity loss than one continuous gate value drifting).
- **Unaddressed until this point — read-side cost:** every idea so far only changed the *write* side. A query currently does one dot product against `S`. If half the heads are clustering/pointer-based and half are raw associative memory, does the query treat them identically at read time, or does retrieval need to know which regime it's reading from? Treating them identically may leave the benefit of the split on the table; treating them differently adds complexity to both sides of the mechanism.
- **Where this stands:** plausible, well-motivated, **not validated**. Requires actually building both head types and testing on a task designed to need both (rare critical facts *and* recurring themes in the same sequence), compared against a same-parameter-budget plain Gated DeltaNet baseline.

### 11. How querying (reading) works in linear attention

```
S_t = Σ_{i≤t} k_i v_i^T                     # accumulated state
output_t = q_t^T S_t = Σ_{i≤t} (q_t · k_i) v_i
```

Compare to softmax attention: `output_t = Σ softmax(q_t·k_i / √d) v_i`. Softmax attention computes every `q·k_i` explicitly at read time (the O(n²) cost) then exponentiates/normalizes into weights. Linear attention never computes `q·k_i` individually at read time — everything's already pre-compressed into `S`. Expanding `q^T S` shows it's the *same* weighted sum of values as softmax attention, just with `(q·k_i)` as the raw weight instead of `softmax(q·k_i)` — no exponential, so nothing forces weights positive or normalized, which is the origin of classic linear-attention instability (unbounded growth, negative weights).

**Standard fixes, used in real implementations:**
- **Kernel feature maps** — replace raw `q,k` with `φ(q), φ(k)` where `φ(x) ≥ 0` elementwise (e.g. `elu(x)+1`), keeping weights non-negative.
- **Normalization** — track `z_t = Σ φ(k_i)` alongside `S`; divide output by `q_t^T z_t` at read time (the linear-attention analogue of softmax's denominator).

```
output_t = (φ(q_t)^T S_t) / (φ(q_t)^T z_t)
```

**Key structural fact:** the read step `q^T S` is completely agnostic to *how* `S` was built — it doesn't know or care whether `S` came from plain accumulation, delta rule, clustering, or surprise-gating. This is good news (every write-side idea above bolts onto the same read mechanism, no new read machinery required) but leaves an open question: is treating a clustering-built `S_h` and a surprise-built `S_h` identically at read time throwing away information the query could use?

### 12. Getting more out of the read step (Hymba's fusion mechanism)

Direct precedent for combining structurally different head types at read time — Hymba's actual solution, not a naive concatenation:

```
Y = W_out( β1 · norm(Y_attn) + β2 · norm(Y_ssm) )
```

Two points transfer directly to the surprise/correlation split:
1. **Normalize before combining.** Hymba found SSM-head output magnitudes were consistently larger than attention-head magnitudes and normalized each before combining, purely for training stability. Surprise-heads (sparse, abrupt overwrites) and correlation-heads (slowly-changing, high-confidence) will very likely have different output scales too — skipping this normalization is a plausible silent failure mode.
2. **β is learned per-channel, globally, via ordinary gradient descent** — not derived from any live confidence/surprise signal computed inside the head. This is a meaningful simplification: real, working head-type fusion benefit from a much dumber combination rule than a dynamic router. Worth building as the first baseline before anything fancier.

**Further than Hymba:** some recent work critiques head-level hybrids (including Hymba-style designs) for keeping mechanisms fully compartmentalized — each head produces its output independently and never references the other during computation — and proposes fusing *at the score level* instead of only post-hoc. That's the more ambitious version: let one head-type's confidence gate how much another head-type's output is trusted, before either finishes computing, not just as a post-hoc reweight.

**Recommendation:** start with the Hymba-style fixed learned-β combination; only reach for score-level fusion or confidence-conditioned routing if the simple version shows a gap a dumb linear combination can't close.

### 13. Entropy-based / compression-aware eviction

Reframe: when a new low-correlation token needs to be admitted and state is full, choose **which existing slot to overwrite** based on the *marginal information value* of that slot — not recency, not similarity to the new item. Three real precedents, each defining "informativeness" differently:

1. **Usage-based — H2O (Zhang et al., 2023).** For KV-cache pruning, accumulated attention scores across tokens follow a power-law: a small set of "heavy hitters" account for most attention mass, and removing them destroys quality, while the long tail can be evicted safely. Mechanism: track a running sum of attention received per token, evict lowest scorers (with a small recency window blended in). This is the read-side analogue of "entropy" — a key rarely retrieved by any query has low information value regardless of how it got written. Cheap: just a running scalar sum per slot.

2. **Importance-weighted plasticity — Elastic Weight Consolidation (continual learning).** Computes a Fisher-information score per parameter estimating importance to previously learned tasks; constrains updates so high-importance parameters stay put while low-importance ones absorb new learning freely. Transferable as a running importance score per key-slot gating how much a new low-correlation token can overwrite it.

3. **Sharpest formalization — SVD / K-SVD dictionary learning.** If `S = Σ σ_i u_i v_i^T`, the singular values literally measure how much information each stored direction carries; low-`σ` components contribute almost nothing to retrieval. K-SVD replaces underused dictionary atoms by re-deriving them from the SVD of the current residual error. Translated: when a new key doesn't fit anything well, decompose `S` and write the new pair predominantly into the **lowest-singular-value direction**, disturbing well-represented, high-value directions as little as possible.

**Honest cost note:** Fisher estimation and live SVD are not cheap per token — doing them every step defeats the point of staying linear-attention-cheap. H2O's running-sum approach is the pragmatic middle ground (cheap enough for every step, has production-scale validation). The natural design: keep something H2O-like cheap every token, and reserve the SVD-based reallocation for the periodic slow pass from section 3.

### 14. Reframing everything as rate-distortion compression

Precise statement of the actual problem: `S` has fixed size (bounded by `rank(S) ≤ min(d_k, d_v)`), the token stream is unbounded — this is **online lossy compression under a fixed rate constraint**, i.e. rate-distortion theory applied to an associative-memory matrix. Every mechanism above slots into a standard role:

| Role | Standard term | This project's mechanism |
|---|---|---|
| Encoder (write rule) | how new data enters the fixed-size code | delta rule (gradient-step encoder) / clustering (VQ-style encoder) |
| Distortion metric | measures encoding quality | `e = v − v̂` |
| Rate allocation over time | which parts of the code stay funded | decay/forget gates + frequency-weighted retention |
| Optimal component selection under a full code | what to evict when full | entropy / lowest-singular-value eviction (§13) |
| Decoder (read rule) | pulling an estimate back out | `q^T S` / normalization (§11) |
| Efficient batch encoding | encode many symbols at once | chunkwise WY trick (§15) |

Practical value of the reframe: rate-distortion theory has **hard bounds** on how well any fixed-rate encoder can do for a given source distribution — worth knowing whether a given design is chasing achievable gains or already near the theoretical ceiling for a given `d_k`.

### 15. Batched correlation computation via chunked matmuls

**Easy part:** against the state entering a chunk (`S_0`, before any of the chunk's own tokens have written anything), a whole sentence's worth of correlation checks batch into one matmul:

```
V̂ = K_chunk @ S_0        # (n × d_k)(d_k × d_v) → n × d_v, all n predictions at once
E  = V_chunk − V̂          # all n error signals at once
```

**The catch, specific to delta rule:** plain linear attention's read never depends on other tokens in the same chunk, so the above is exactly correct there. Delta rule breaks this — token 5's correlation check should reflect that tokens 1–4, earlier in the *same* chunk, already corrected the state. A naive single matmul against `S_0` for the whole chunk is only exactly right for the first token in it; everything after reintroduces sequential dependency within the chunk.

**How DeltaNet solves this** (Yang, Wang, Zhang, Shen, Kim — *"Parallelizing Linear Transformers with the Delta Rule over Sequence Length,"* NeurIPS 2024): reparameterize the recurrence as a matrix-valued RNN driven by a generalized Householder transformation, which admits a memory-efficient **WY representation** for products of Householder matrices — this avoids materializing the full matrix-sized state at every single step. Within each chunk, all projections, gating, and delta updates compile into batched dense matmuls (GEMMs), with the within-chunk sequential dependency handled via this WY/UT factorization (a triangular solve expressed as dense matmuls) rather than a token-by-token loop.

**Takeaway for this project:** correlating a whole chunk against the incoming state in one shot is correct and cheap as a first-order approximation; getting it exactly right within a chunk isn't an open problem to solve from scratch — it's the same WY-style correction DeltaNet already implements, which the `flash-linear-attention` library already provides. Bolt correlation/clustering logic onto that existing chunked machinery rather than rebuilding parallelism.

### 16. Distillation / JEPA-inspired amortized update

Idea: run the full expensive mechanism (delta rule + clustering + entropy eviction + periodic re-evaluation) as a "teacher" during training to generate ground-truth state deltas, then train a small, cheap predictor network to imitate it, and deploy **only** the cheap predictor at inference — never running the expensive mechanism again.

**JEPA connection, precisely.** A JEPA model has a state encoder, an action encoder, and a predictor, trained by minimizing distance between a predicted representation and the true representation of the resulting state after an action — the premise being that predicting *future states* is easier in learned representation space than raw input space. Mapped here: state `S` = the latent, new token `(k,v)` = the "action," predictor `f_θ(S, k, v) → S'` (or `ΔS`).

**Where the analogy is exact, and where it isn't.** JEPA's efficiency comes mainly from *not decoding back to input space* — predicting embeddings is cheap because embeddings are lower-dimensional, not because the predictor was trained against something heavier than itself (JEPA's target encoder is usually just an EMA copy of the same-size context encoder). What's actually being proposed here is closer to:

- **Learned optimizers (L2O)** — *"Learning to learn by gradient descent by gradient descent"* (Andrychowicz et al., 2016): train a small network to predict parameter updates, replacing a hand-designed rule (Adam-like) with a learned one. Same shape as replacing the hand-designed delta/clustering/eviction update with a learned `f_θ`.
- **Policy distillation from an expensive controller** — run a slow, exact model-predictive-control optimizer offline to generate optimal actions, train a small fast network via behavior cloning to imitate it, discard the expensive optimizer at deployment. Structurally the cleanest match: expensive-but-correct process generates training targets, small network deployed alone.

**Honest tension.** JEPA's predictor stays useful because it's predicting in a space specifically regularized to be predictable (anti-collapse terms exist precisely so the predictor can't cheat toward a trivial constant). `S` here isn't a representation learned to be predictable — it's a hand-designed compression target with fairly intricate rule dynamics (clustering, entropy eviction, decay), a harder distillation target than JEPA's, closer to the known difficulty of L2O/MPC-distillation students, which are known to degrade on states the teacher never visited during training (out-of-distribution generalization risk).

**Open question, unresolved:** replace the hand-designed rule entirely with the learned predictor, or keep the exact rule as the periodic slow-pass correction (§3) and use the predictor only for the cheap per-token step in between? The hybrid version is lower-risk and ties directly back to the fast/slow two-tier structure already designed in section 3.

### 17. Kernel feature maps — raising the capacity ceiling itself

Everything above operates *within* the fixed ceiling `rank(S) ≤ min(d_k, d_v)`. This is the one lever that changes the ceiling directly.

**Mechanism:** a feature map `φ: R^{d_k} → R^{d_φ}` with `d_φ > d_k` projects `k` into a higher-dimensional space before it touches the state. Write becomes `S += φ(k) v^T`, read becomes `φ(q)^T S` (with normalization as in §11). `S` is now `d_φ × d_v`, so the ceiling becomes `min(d_φ, d_v)` — moved up directly by however much `d_φ` is inflated.

**Why it helps even before hitting the new ceiling:** in high dimensions, random vectors become nearly orthogonal automatically (concentration of measure / Johnson–Lindenstrauss). Projecting into a higher-dimensional space pushes different keys apart from each other, reducing `k_i · k_j` crosstalk between stored patterns, even with zero learning or clever design — a partial, free win on the "orthogonal keys" goal from section 4.

**Concrete feature maps:**
- `elu(x) + 1` — original linear-attention paper's choice; mainly ensures non-negativity (mimicking softmax positivity) rather than expanding dimension.
- **Performer / FAVOR+** (positive random features) — random projections constructed so `φ(q)·φ(k)` approximates the softmax kernel in expectation.
- **DPFP** (Schlag et al. — the same "fast weight programmers" paper that introduced delta rule) — most directly relevant here, since it was explicitly designed to reduce interference in exactly this kind of associative fast-weight memory rather than approximate softmax. Builds `φ(k)` by concatenating ReLU'd, shifted/permuted copies of `k`, deliberately inflating dimension and sparsity to separate keys cleanly. Parameter-free, fixed transform.

**Honest cost:** `S`'s size is `d_φ × d_v`, so both write and read cost scale with `d_φ` — a direct trade of per-token compute for capacity, not a free lunch. Needs empirical measurement (Stage 3 below) to find the smallest `d_φ` inflation that buys the needed interference reduction, since past some point the same budget is likely better spent on the clustering/eviction mechanisms instead.

---

## Part 2: Implementation Plan

### Guiding principles for the coding agent

1. **Don't reimplement DeltaNet / Gated DeltaNet from scratch.** Use the [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) (`fla-org/flash-linear-attention`) repository, which already has working, chunk-parallel (WY/UT-factorized, §15) implementations of linear attention, DeltaNet, and Gated DeltaNet. All new mechanisms should be built as extensions/hooks on top of this, not a parallel reimplementation.
2. **One new mechanism per experiment.** Never combine two untested mechanisms in the same run before each has an isolated result — otherwise a win or a regression can't be attributed to anything.
3. **Always compare against the same fixed baseline**: Gated DeltaNet at matched parameter count, same training budget.
4. **Start tiny.** Everything through Stage 5 should run on a single GPU (or even CPU for correctness debugging) in minutes, not on a cluster. Only move to the OSU HPC Slurm partitions once sweeping many configs in parallel (Stage 8).
5. **Correctness before speed.** For any new mechanism that's awkward to express in the chunked/parallel kernel (most likely: clustering/pointer write-gating, Stage 4), first prototype it in `fla`'s slower sequential/recurrent mode to validate correctness, *then* worry about writing an efficient chunked or custom Triton kernel.

### Stage 0 — Environment & repo setup

- [ ] Clone `flash-linear-attention` (`fla-org/flash-linear-attention`).
- [ ] Set up Python env: PyTorch (CUDA build matching whatever GPU is available), `einops`, `triton`, plus `fla`'s own requirements.
- [ ] Sanity check: run `fla`'s existing DeltaNet and Gated DeltaNet modules on a toy forward/backward pass to confirm the environment works before writing any new code.
- [ ] If targeting OSU's HPC Slurm cluster eventually: confirm which partitions expose GPUs suitable for this (check with `sinfo`), but do **not** move there until Stage 8 — everything before that is small enough for a single local/lab GPU.

### Stage 1 — Data: redundancy-parameterized MQAR generator

Base task: **Multi-Query Associative Recall (MQAR)** — the standard synthetic benchmark used in the DeltaNet paper itself to stress-test associative recall. Sequences interleave `(key, value)` pairs with filler/noise tokens, then later include query tokens asking "what value went with this key," scored by recall accuracy.

**New addition needed — a redundancy parameter `r ∈ [0, 1]`:** controls what fraction of keys get values drawn from a small set of clusters (near-duplicate values) rather than fully unique random values. `r = 0` reproduces vanilla MQAR (sanity-checkable against published curves). `r → 1` is the regime where clustering/pointer-based mechanisms (§2, Stage 4) should show a measurable advantage, if they're going to show one anywhere.

**Generator interface (pseudocode):**

```python
def gen_redundant_mqar(
    num_examples: int,
    seq_len: int,
    vocab_size: int,
    num_kv_pairs: int,        # unique keys per sequence
    num_queries: int,         # query positions per sequence
    redundancy_r: float,      # fraction of kv pairs using a shared cluster value
    num_value_clusters: int,  # size of the shared-value pool when r > 0
    seed: int,
) -> dict:
    """
    Returns dict with:
      input_ids: (num_examples, seq_len) token ids, interleaving kv pairs,
                 filler tokens, and query tokens
      query_positions: (num_examples, num_queries) positions of query tokens
      labels: (num_examples, num_queries) correct value token id for each query
      is_redundant: (num_examples, num_kv_pairs) bool mask, which kv pairs
                    were assigned a shared-cluster value (for later analysis
                    by mechanism — does clustering help specifically on
                    these positions)
    """
```

**Sampling logic per sequence:**
1. Sample `num_kv_pairs` distinct key tokens from `vocab_size`.
2. Sample `num_value_clusters` "cluster representative" value tokens.
3. For each key: with probability `redundancy_r`, assign it a value equal to one of the `num_value_clusters` cluster representatives (chosen uniformly); otherwise assign a fresh unique random value token.
4. Interleave `(key, value)` pairs into the sequence at random non-overlapping positions, with filler tokens in between (standard MQAR format — reuse `fla`'s or the original MQAR paper's interleaving logic if available rather than reinventing it).
5. Place `num_queries` query tokens after their corresponding key's position (a query for a key can only appear after that key has been seen), label = the correct value token.
6. Output format compatible with whatever training loop format `fla`'s existing training scripts expect (check `fla`'s examples/training scripts for the expected batch format before finalizing this).

**Deliverable for this stage:** a standalone, testable data generator module with unit tests confirming: (a) `r=0` output is statistically indistinguishable from vanilla MQAR, (b) at `r>0`, the fraction of kv pairs sharing a cluster value matches `r` within sampling noise, (c) no key ever appears with two different values in the same sequence (would break the recall task's well-posedness).

### Stage 2 — Baseline training runs

- [ ] Configs to run, unmodified from `fla`: **linear attention** (no delta — floor), **DeltaNet**, **Gated DeltaNet** (current best-in-class baseline). Optionally a small softmax transformer as a ceiling reference.
- [ ] Fix architecture hyperparameters across all baseline and later experimental runs for comparability: `d_model` (start ~64–256), number of layers (2–4), number of heads, chunk size, optimizer + LR schedule, number of training steps.
- [ ] First run at `r=0` and confirm the recall-accuracy-vs-`num_kv_pairs` curve reproduces the known shape from the DeltaNet paper's MQAR results (sanity check that the setup is correct before trusting any later comparison against it).
- [ ] Then sweep `r ∈ {0, 0.25, 0.5, 0.75, 0.9}`, recording recall accuracy vs. `num_kv_pairs` for each `r`, for each of the three baseline models. **This produces the reference curves every later stage is compared against — do not skip or shortcut this.**

### Stage 3 — Kernel feature map ablation

Cheapest test, no new gating logic — isolates how much of the capacity problem (§4, §17) is solvable without any of the novel mechanisms.

- [ ] Implement a configurable feature-map module `φ(x)` applied to `k` (and `q`) before entering `fla`'s attention/delta computation. Support at minimum: `elu(x)+1` (already likely present in `fla`), and **DPFP** (concatenated ReLU'd shifted/permuted copies of `k` — implement from the Schlag et al. paper's definition).
- [ ] Sweep the feature-map output dimension `d_φ` (expansion factor relative to `d_k`) at a fixed total parameter budget.
- [ ] Measure: recall accuracy vs. `r`, recall accuracy vs. `num_kv_pairs`, and throughput (tokens/sec) cost of each `d_φ`.
- [ ] Deliverable: a plot/table of accuracy vs. `d_φ` vs. throughput, identifying the smallest `d_φ` that meaningfully closes the gap to the ceiling — this number informs parameter budgeting for every later stage.

### Stage 4 — Clustering / pointer write-gating (§2)

The trickiest mechanism to implement efficiently in the chunked/parallel setting — follow the "correctness before speed" principle here specifically.

- [ ] **First**, prototype in `fla`'s sequential/recurrent (non-chunked) mode for correctness:
  - Maintain a small fixed-size codebook per head: `num_clusters` centroid vectors in value-space, updated online (simple exponential-moving-average centroid update, or basic online k-means).
  - For each incoming `(k, v)`: compute similarity of `v` against all current centroids (cosine or dot-product).
  - If `max_similarity > τ` (configurable threshold): treat as a match — update the pointer table (`key → cluster id`) and only lightly update the matched centroid (small `β`); barely touch `S`.
  - Else: normal delta-rule write into `S`; if under `num_clusters` capacity, spawn a new centroid from this value.
- [ ] Config knobs to expose: `num_clusters`, similarity threshold `τ`, β-scale function (how strongly similarity suppresses the write), whether matching is done in raw value space or a projected space.
- [ ] **Then**, once correctness is validated, look at whether this can be expressed as a chunked/batched operation (recall §15 — correlating a whole chunk against the incoming state is a single matmul; the cluster-similarity lookup step will likely need its own batched formulation, e.g. a single `(n × num_clusters)` similarity matrix per chunk via one matmul against the centroid matrix). If a fully chunked version isn't tractable in the time available, it's acceptable to first report results using the slower sequential mode and flag chunked-kernel optimization as follow-up work.
- [ ] Test against the `r` sweep from Stage 2: this mechanism should specifically help at high `r` (redundant values) without hurting accuracy at `r=0` relative to the Gated DeltaNet baseline. If it hurts at `r=0`, that's a real negative result worth reporting, not a bug to hide.

### Stage 5 — Entropy-based eviction (§13)

Implement the **cheap proxy first**, exact SVD version second (cost tradeoff is itself a thing to measure).

- [ ] **Cheap proxy (H2O-style):** maintain a running usage counter per cluster/key-slot, incremented based on how much that slot contributed to read outputs (e.g. accumulate `|q·k|` or the slot's contribution magnitude to `q^T S` at each read). When a new value needs a slot and the codebook (from Stage 4) is at capacity, evict/reassign the lowest-cumulative-usage slot.
- [ ] **Exact version (periodic slow pass, ties back to §3):** every `N` tokens (configurable), compute (or approximate via power iteration, for speed) the SVD of `S` per head; identify the lowest-singular-value component; for any pending "doesn't fit anywhere" writes accumulated since the last slow pass, write preferentially into that low-information subspace instead of an arbitrary slot.
- [ ] Compare cheap-proxy vs. exact-SVD versions on: recall accuracy (does the more expensive version actually buy measurably better retention decisions), and wall-clock cost of the periodic pass at a few different `N` (re-evaluation intervals).
- [ ] Deliverable: a recommendation on whether the exact SVD pass is worth its cost given the measured accuracy delta, and at what interval `N` it should run if so.

### Stage 6 — Multi-head split: surprise-heads vs. correlation-heads (§9)

- [ ] Implement the Titans-style update rule (§7) as one head-type module: needs a momentum buffer per head (`Su_t`) and a forget gate `α_t` (small learned projection of the input, per the Titans design).
- [ ] Use the correlation/clustering head-type module from Stage 4 as the other head type.
- [ ] Split total heads into two groups with a **configurable ratio** (start with 50/50, then sweep e.g. 25/75, 50/50, 75/25) — treat the ratio itself as a hyperparameter to sweep, per the open question in §9/§10.
- [ ] Combine head outputs Hymba-style (§12): normalize each head-type's output, combine with learned per-channel `β` weights, concatenate + project. Implement the simple fixed-`β` version first; only attempt score-level fusion (§12, the more ambitious variant) if the simple version shows a measurable gap.
- [ ] **Task extension needed:** design a task variant combining a "rare critical fact" signal (a few unique kv pairs that must be recalled, similar to low-`r` MQAR) with a "recurring theme" signal (many redundant/clustered kv pairs, similar to high-`r` MQAR) *in the same sequence*, so that neither head type alone should be expected to win outright — this is the specific regime the combination is meant to help.
- [ ] Compare: combined heterogeneous-head model vs. all-surprise-heads vs. all-correlation-heads vs. plain Gated DeltaNet, all at matched total parameter count.

### Stage 7 (stretch, lower priority) — Distilled predictor (§16)

Only attempt after Stages 4–5 produce a working "expensive full mechanism" to distill from.

- [ ] During training runs of the combined mechanism (clustering + entropy eviction, from Stages 4–5), log `(S_t, k_t, v_t) → ΔS_true` pairs.
- [ ] Train a small predictor network `f_θ` (2–3 layer MLP) to regress `ΔS_pred` from `(S_t, k_t, v_t)` via MSE against the logged `ΔS_true`. Optionally follow up with end-to-end fine-tuning against the downstream task loss.
- [ ] Compare three configurations at inference: (a) full expensive mechanism, (b) distilled-predictor-only, (c) plain Gated DeltaNet baseline — on recall accuracy and tokens/sec throughput.
- [ ] Specifically check for the OOD-degradation risk flagged in §16: does the distilled predictor's accuracy degrade disproportionately on sequences with a different `r` value or `num_kv_pairs` range than what the teacher process was logged on during training?

### Cross-cutting: metrics & logging harness

Build once, reuse identically across every stage for apples-to-apples comparison:

- [ ] Recall accuracy vs. `num_kv_pairs` (the standard MQAR curve — where does it collapse).
- [ ] Recall accuracy vs. redundancy `r`, at fixed `num_kv_pairs` (the new axis this project adds).
- [ ] Throughput: tokens/sec, measured identically across all configs.
- [ ] Effective capacity: max `num_kv_pairs` recallable at ≥95% accuracy, compared at matched total parameter count across mechanisms.
- [ ] Simple CSV or Weights & Biases logging; one shared eval script invoked identically by every stage's training run, so results are directly comparable without post-hoc reconciliation.

### Stage 8 — Scale-up plan (OSU HPC Slurm)

- [ ] Only move here once the ablation matrix (mechanism × `r` × scale) from Stages 3–6 is fully defined and there are many configs to run in parallel — the queue/allocation overhead isn't worth it for single small runs.
- [ ] Use the Slurm partition / GPU survey already done for this project to pick appropriate partitions.
- [ ] Scale up `d_model`, layer count, and sequence length once small-scale results identify which mechanisms are worth the larger compute investment.

---

## Consolidated risks & open questions (carried forward from the sanity checks)

Keep these visible throughout implementation — several apply to more than one stage:

1. **Self-referential confidence lock-in** (§6, §8): any signal built by comparing against its own past predictions, then used to gate its own future updates, risks rich-get-richer lock-in. Any confidence/correlation trace implemented in Stage 4 or 6 should have an explicit ceiling or forgetting factor, not be allowed to grow unbounded.
2. **Nearest-neighbor search cost at scale** (§6): the clustering lookup in Stage 4 must stay cheap (bounded `num_clusters`, batched similarity matmul) or it defeats the linear-time property of the whole architecture family.
3. **Hyperparameter surface** (§6): similarity threshold, β-decay curve, re-evaluation interval, frequency-decay rate, split threshold, head-type ratio — track all of these explicitly in experiment configs rather than hardcoding, since the sweep over them is itself part of the deliverable.
4. **Read-side regime-awareness** (§10, §11): decide explicitly, and document the decision, on whether the read step treats heterogeneous head types identically or differently — don't let this be an accidental default.
5. **Distillation OOD risk** (§16): if Stage 7 is attempted, explicitly test generalization outside the teacher's logged training distribution before trusting the distilled predictor's numbers.
6. **Filler/shallow-pattern lock-in** (§5): frequency-based retention should weight by match tightness (correlation strength), not raw hit count, or a common-but-shallow pattern can crowd out a rare-but-important one.

## Novelty assessment, stated plainly

Nearly every individual mechanism above (delta rule, RLS/Kalman adaptive gain, Titans' surprise+momentum+forget gates, product-key/codebook memory, ARC cache eviction, spaced repetition, H2O eviction, K-SVD dictionary learning, Hymba head fusion, JEPA/L2O-style distillation) has direct precedent in the literature — reconstructing them independently is a strong sign of genuine understanding of the problem, not evidence of novelty by itself. The part of this project without a found precedent is the **specific combination**: cross-key pointer-based deduplication + periodic drift correction + frequency-weighted retention + surprise/correlation head-type separation, built on top of the existing delta-rule chunked-parallel machinery. That combination is untested, is where the real implementation risk and the real potential contribution both live, and is exactly what Stages 4–6 are designed to validate or falsify empirically.

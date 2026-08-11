# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Stages 0 and 1 are complete; Stage 2's machinery is built and the reference curves are being generated.

- `src/lamr/data/mqar.py` — redundancy-parameterized MQAR generator (Stage 1).
- `src/lamr/layers/recurrent.py` — sequential reference update rules; correctness ground truth.
- `src/lamr/layers/chunked.py` — chunk-parallel delta rule in portable PyTorch (§15), ~8.5× the sequential reference on CPU.
- `src/lamr/layers/linear_attn.py` — the layer; modes `linear` / `delta` / `gated_delta` differ *only* in update rule.
- `src/lamr/models/lm.py` — small stacked LM; architecture fixed here per Stage 2.
- `src/lamr/metrics.py` — shared recall metrics, sliced by redundancy. Every stage calls this.
- `src/lamr/train.py` — the single training entry point (`python -m lamr.train`).
- `scripts/stage2_sweep.py` — baseline sweeps; appends CSV and skips completed runs, so it resumes.

Smoke results at `r=0`, `num_kv_pairs=8`, `d_model=128`: gated_delta 99.3% @500 steps, delta 30% @300, linear 1.9% @300 — the expected ordering, with plain linear attention failing as the floor.

**Output convention: no next-token shift.** Logits at position `p` are scored directly against the answer for a query at `p`. The generator never writes answers into the sequence, so a shifted objective would be predicting filler. `test_metrics_read_query_positions_without_a_shift` pins this.

**Default `chunk_size=64`** — measured optimum on this CPU (8.4–9.5× across seq_len 128–512; 16 and 128 are both meaningfully worse). Chunk size never changes results, only speed, and `test_chunk_size_does_not_change_the_result` enforces that — so treat it as a performance knob, never a hyperparameter to sweep for accuracy.

Read `linear_attention_memory_research_plan.md` before doing anything non-trivial. It is the single source of truth for both the research design (Part 1, sections §1–§17) and the build order (Part 2, Stages 0–8). Part 2 stages cite the Part 1 sections that motivate them (e.g. Stage 4 implements §2); follow those cross-references rather than re-deriving the design.

## Commands

```bash
.venv/Scripts/python.exe -m pytest tests/                  # full suite (~4s)
.venv/Scripts/python.exe -m pytest tests/test_mqar.py      # one file
.venv/Scripts/python.exe -m pytest -k worked_example       # one test by name
.venv/Scripts/python.exe scripts/bench_backends.py         # tokens/sec by backend
.venv/Scripts/python.exe -m lamr.train --mode delta --steps 500   # single run
.venv/Scripts/python.exe scripts/stage2_sweep.py --preset kv-curve  # baseline curves
.venv/Scripts/python.exe -m pip install -e .               # after editing pyproject
```

Use the venv interpreter explicitly. The system `python` on this box is the Windows Store build and does not have the project's dependencies.

## Hardware constraint (drives the backend split)

This workstation has an **AMD Radeon RX 5700 and no CUDA**. Triton's AMD backend needs ROCm, which does not support RDNA1 (gfx1010) and is Linux-only regardless, so **`flash-linear-attention` cannot run here** — installed torch is `2.13.0+cpu`.

This does not change plan principle #1 (don't reimplement DeltaNet). It splits the work:

- **Locally (CPU)** — data generation, correctness, unit tests, the sequential reference rules in `lamr/layers/recurrent.py`, and the portable chunked backend in `lamr/layers/chunked.py` for runs that need to finish in reasonable time. This is precisely the "prototype in sequential/recurrent mode first" path the plan prescribes.
- **On the OSU HPC (CUDA)** — `pip install -e ".[gpu]"` and swap in `fla`'s fused Triton kernels for the real training runs.

There are therefore **three** implementations of the same update rule, and they must agree numerically:

| | Speed | Runs on | Role |
|---|---|---|---|
| `recurrent.py` | 1× | anywhere | ground truth; O(T) python loop |
| `chunked.py` | ~8.5× | anywhere | local sweeps; same math, dense matmuls + triangular solve |
| `fla` Triton | fastest | CUDA only | real training runs |

`tests/test_chunked_parity.py` enforces the first-to-second agreement; `tests/test_fla_parity.py` enforces the third. `fla` returns state as `[N, H, K, V]`, which already matches the plan's `(d_k, d_v)` — no transpose needed, though the gated kernel's `state_v_first` flag would flip it. What *does* need converting is the time/head axis order; see `fla_backend.py`.

`chunked.py` is also where Stage 4 lands. Cluster/pointer write-gating has to be expressed against the within-chunk triangular structure eventually, and the derivation in that module's docstring is the thing to extend.

## HPC handoff (first session on the GPU box)

The cluster is the **Oregon State CoE HPC** (Slurm).

Remote: <https://github.com/madpods/linear-attention-memory-research> (public, so the clone needs no credentials and no `gh` on the cluster).

```bash
git clone https://github.com/madpods/linear-attention-memory-research.git
cd linear-attention-memory-research
srun --partition=dgxh --gres=gpu:1 --time=01:00:00 --pty bash   # interactive GPU node
bash scripts/slurm/setup_env.sh          # venv + CUDA torch + fla, runs both suites
python scripts/stage2_sweep.py --preset full --count   # must equal the sbatch array size
sbatch scripts/slurm/sweep_array.sbatch
python scripts/merge_results.py results/parts results/stage2.csv
```

`scripts/slurm/survey_cluster.sh` (read-only, login node) re-derives the table below plus the available module versions.

**Modules must match between build and run.** The cluster uses Lmod, and the CoE docs are explicit that modules used to build software into a python environment have to be loaded again to run it. Triton compiles kernels against the CUDA toolkit on first call, so a batch job missing the cuda module fails deep inside `fla` rather than at import — a confusing failure to debug mid-sweep. `setup_env.sh` therefore resolves modules (preferring Lmod's `(D)` default, overridable via `MODULE_PYTHON` / `MODULE_CUDA`), records them to `$VENV/modules.env`, and `sweep_array.sbatch` replays that file. If you rebuild the venv by hand, re-record that file too.

**Partitions accepting any Slurm account** — no `--account` line needed; every other partition in the CoE table is restricted to one research group.

| partition | GPU | time limit | concurrent GPU cap |
|---|---|---|---|
| `preempt` | any | 7 days | none (preemptible) |
| `dgxh` | H100/H200 | 2 days | 8 |
| `dgx2` | V100 | 7 days | 16 |
| `share` | M60/A40 | 14 days | 2 |
| `ampere` | A40 | 2 days | 2 |

The array defaults to `preempt` with `--requeue`. These are short jobs and the sweep skips configurations already in its CSV, so a preempted task resumes instead of repeating — which makes an uncapped preemptible queue strictly better than waiting on `share`'s 2-GPU cap. If preempt starves, `dgxh` is the fastest fallback; drop the array throttle to `%8` there, `%2` on `share`/`ampere`. User-wide limits are 1000 submitted / 400 running.

**`src/lamr/layers/fla_backend.py` has never been executed**, though its signatures are now verified against `fla`'s source. Three conventions are reconciled there:

- **Layout** — `fla` requires `(B, T, H, D)` and has *removed* `head_first` (passing it raises `DeprecationWarning`). This project works in `(B, H, T, D)`, so the adapter transposes unconditionally. This is the dangerous one: a wrong layout does not raise, it treats heads as timesteps and returns plausible nonsense.
- **Scale** — `fla` defaults to `k.shape[-1] ** -0.5`; we pass `scale=1.0` because the layer L2-normalizes q/k itself.
- **Decay** — the gated kernel takes `g`, log-space, positioned *before* `beta`. Passed by keyword so the order mismatch cannot bite.

`tests/test_fla_parity.py` is the gate. Run it before any training on the GPU — until it passes, no GPU number is comparable to any CPU number, and interchangeability is the premise the whole backend split rests on. Each assertion names the convention it pins. **Fix `fla_backend.py`; never relax the test.** Note the gate skips itself when `fla` is unimportable, so `setup_env.sh --verify` asserts `fla_available()` first; otherwise pytest would exit 0 with the gate never having run.

Two operational notes: `sweep_array.sbatch` deliberately leaves `--partition` and `--account` unset (run `sinfo -o "%P %G %m %l"` first, as the plan's Stage 0 says). And each array task writes its own CSV — concurrent appends to one file interleave mid-row — which `merge_results.py` stitches back together, keeping the last row per configuration so a re-run task overrides its partial.

## Project

Research into raising the effective memory capacity of linear-attention state matrices. The framing that ties every mechanism together: a state matrix `S` has fixed size while the token stream is unbounded, so this is **online lossy compression under a fixed rate constraint**. Every proposed mechanism is either a smarter *encoder* (what gets written, how much space it takes) or a way to raise the *capacity ceiling* `rank(S) ≤ min(d_k, d_v)` itself. §14 tabulates how each mechanism maps onto a standard rate-distortion role — use that table to place any new idea before building it.

Structurally, all proposed mechanisms except §17 (kernel feature maps) modify the **write** side only. The read step `q^T S` is agnostic to how `S` was built, which is why mechanisms bolt onto existing delta-rule machinery instead of requiring new retrieval code. §10/§11 flag the one unresolved consequence: whether heterogeneous head types should be read identically. Decide and document that explicitly rather than letting it default.

## Non-negotiable working rules (from the plan's guiding principles)

These override normal instincts and are the most common way this project gets derailed:

1. **Never reimplement DeltaNet / Gated DeltaNet from scratch.** Build on [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention), which already provides chunk-parallel (WY/UT-factorized) linear attention, DeltaNet, and Gated DeltaNet. New mechanisms are extensions and hooks on top of it. Likewise reuse `fla`'s or the original MQAR paper's sequence-interleaving logic rather than reinventing it.
2. **One new mechanism per experiment.** Never combine two untested mechanisms in one run before each has an isolated result — attribution becomes impossible.
3. **Always compare against the same fixed baseline:** Gated DeltaNet at matched parameter count and matched training budget. Architecture hyperparameters (`d_model`, layers, heads, chunk size, optimizer, LR schedule, steps) are fixed once in Stage 2 and held constant across every later stage.
4. **Correctness before speed.** Prototype awkward mechanisms in `fla`'s sequential/recurrent mode first, validate, *then* attempt a chunked or Triton version. Reporting results from the slow sequential path with chunked optimization flagged as follow-up is explicitly acceptable.
5. **Start tiny.** Stages 0–5 must run on a single GPU (or CPU for correctness debugging) in minutes. Do not move to OSU HPC Slurm until Stage 8, when there are many configs to sweep in parallel.
6. **Negative results are deliverables.** If a mechanism hurts at `r=0`, that is a result to report, not a bug to hide.

## Stage dependencies

```
Stage 0  env setup (clone fla, PyTorch/CUDA, einops, triton; toy fwd/bwd sanity check)
Stage 1  redundant-MQAR data generator          ─┐
Stage 2  baseline curves (linear attn, DeltaNet, ├─ Stage 2 curves are the reference
         Gated DeltaNet) over the r sweep       ─┘  every later stage compares against
Stage 3  kernel feature-map ablation (§17)  — independent of 4/5; sets d_φ budget
Stage 4  clustering / pointer write-gating (§2)  — hardest to parallelize
Stage 4b residual matrix R (addendum 01)         — build before Stage 5, not after
Stage 5  entropy-based eviction (§13)  — CONDITIONAL: may be dropped, see below
Stage 6  surprise-heads × correlation-heads split (§9) — needs Stage 4 + new task variant
Stage 7  distilled predictor (§16) — stretch; needs Stages 4–5 working as the teacher
Stage 8  Slurm scale-up
```

Stage 2's `r ∈ {0, 0.25, 0.5, 0.75, 0.9}` sweep is a hard prerequisite, not a formality — it produces the reference curves for everything downstream. Before trusting it, confirm the `r=0` recall-vs-`num_kv_pairs` curve reproduces the shape published in the DeltaNet paper.

The **cross-cutting metrics/logging harness** (recall vs. `num_kv_pairs`, recall vs. `r`, tokens/sec, effective capacity at ≥95% accuracy) is built once and invoked identically by every stage — do not let stages grow their own eval paths, or results stop being comparable.

## Notation

The plan uses a consistent notation that code should follow:

- `S` — the state matrix (`d_k × d_v`, or `d_φ × d_v` with a feature map). Never reuse `S` for Titans' surprise term; the plan renames that to `Su` (§7) specifically to avoid the clash.
- `k`, `v`, `q` — key, value, query. `e = v − v_read` — the delta-rule error/"translation".
- `β` — write strength. Note §12 uses `β1`/`β2` for Hymba's *learned per-channel output-fusion* weights; different quantity, same letter, as in the source papers.
- `r` — redundancy parameter of the new MQAR generator (fraction of keys assigned near-duplicate cluster values). This is the novel evaluation axis this project adds; `r=0` must reproduce vanilla MQAR.
- `τ` — cluster-match similarity threshold. `d_φ` — feature-map output dimension. `N` — slow-pass re-evaluation interval.
- `R` — the Stage 4b residual state matrix (`d_k × d_v`), distinct from `S`.

**Three-way collision on the letter r.** `redundancy_r` is the data parameter; `R` is the residual matrix; and the residual quantity `v − c` is the same *kind* of object as the delta-rule error `e = v − v_read`. Never name a variable `r` in code touching any two of these — use `redundancy_r`, `resid_matrix`, and `resid_target`.

## Addendum 01 — residual matrix (Stage 4b)

`addendum_01_residual_matrix.md` extends Stage 4 with a second state matrix `R` holding the per-key residual `v − c` that the coarse cluster prediction misses, read as `v̂ = c(k) + q^T R`. It is RVQ's coarse/fine split applied to associative memory, and it reorders the build: **4b lands before Stage 5, and if its drift test succeeds Stage 5 becomes optional.** Read that file before touching Stage 4.

`addendum_02_state_accounting_and_read_path.md` settles the two structural questions. **Decision 1:** `R` replaces `S`; overhead over a plain-`S` baseline is `num_clusters × (d_k + d_v)`, so it is set by a hyperparameter, not fixed by the architecture, and the matched-state control must be sized to `R` + codebook at every `num_clusters`. **Decision 2:** the codebook splits into `C_keys` (`num_clusters × d_k`, running mean of keys routed to each cluster) and `C_values`; write addresses by `v`-similarity against `C_values`, read addresses by `q`-similarity against `C_keys`. Both are right, and Decision 2's batching argument holds — `Q_chunk @ C_keys^T` is one matmul, so the §6 nearest-neighbour cost risk stays contained.

Two problems remain, and the first is serious:

**1. `C_keys` averages vectors that are dissimilar by construction.** §2's whole premise is cross-key aliasing: *different* keys whose *values* happen to be similar. So the keys routed to one cluster are unrelated in key space, and their running mean is not a centroid of anything — it is the mean of `m` roughly-orthogonal unit vectors. At read time the correct cluster still wins, but only barely: `q ≈ k_i` contributes `1/m` to its own cluster's mean while competing clusters return noise of order `1/√(m·d_k)`, giving separation `√(d_k / m)`. Beating `num_clusters − 1` competitors needs roughly `m ≲ d_k / (2 ln num_clusters)` — at `d_k = 16` and `num_clusters = 16`, about **three keys per cluster**. The mechanism's benefit is many keys per cluster, and that is precisely what destroys its read path. **Suggested fix:** carry the assignment in a delta-rule-written `d_k × num_clusters` memory (target = one-hot cluster) instead of a running mean. Error-corrected superposition is what associative memory is *for*; plain averaging has no mechanism to keep aliased keys separable. Test the scaling directly — sweep keys-per-cluster and measure read-time routing accuracy in isolation, before trusting any end-to-end recall number.

**2. Write-time `c` and read-time `c` are different quantities.** The residual stored is `v − c_write`, where `c_write` was selected by `v`-similarity; reconstruction computes `c_read` from `q`-similarity. The residual only cancels if the two agree, and nothing forces them to — the soft/hard choice in Decision 2 widens the gap further, since a softmax blend at read cannot reproduce a hard assignment at write. **Fix:** compute the residual against the *read-time* addressing (`c` from the key-addressed lookup), and use `v`-similarity only to decide which cluster to update or spawn. Then reconstruction is exact by construction rather than by hoping the two lookups coincide.

**4. The slow pass and `R` conflict; as drawn, running both is incorrect.** `R` stores `v − c_old`. When Stage 5's slow pass reassigns clusters, every residual already in `R` was computed against a centroid that no longer applies, so `c_new + q^T R` reconstructs the wrong value. This is stronger than the addendum's "`R` may make Stage 5 unnecessary" — the two are not merely redundant, they corrupt each other. Either the slow pass must rewrite `R` by the centroid delta (`R += k(c_old − c_new)^T` for affected keys, which is not cheap), or commit to `R`-absorbs-drift and drop the slow pass. Decide before building, and note that this makes the drift test a fork in the design, not just a benchmark.

**3. The RVQ precedent is directional, not dispositive.** The variance argument is sound: residuals are lower-variance and less correlated across keys than raw values, and rank-limited memory represents them more faithfully. But RVQ's second stage is another *codebook* quantizing the same vector, whereas `R` is rank-limited associative storage holding per-key corrections for many keys at once — a different and harder load. The bit-budget accounting that makes RVQ beat a flat codebook does not transfer directly. Treat it as motivation, not as evidence, and let the `redundancy_r` sweep settle it.

The addendum's own honest note stands and is testable immediately: at `redundancy_r = 0` the residual collapses to the raw value, the cluster contributes nothing, and the mechanism pays two writes for what one plain delta-rule write already did. Stage 2's `r=0` curves are the reference for exactly that check.

## Risks to hold in view while implementing

Carried from §6/§8/§10 and consolidated at the end of the plan; each has bitten a comparable design before:

- **Self-referential confidence lock-in** — any signal built from comparing against its own past predictions and then used to gate its own future updates needs an explicit ceiling or forgetting factor. Applies to any correlation/confidence trace in Stages 4 and 6.
- **Nearest-neighbor cost** — the Stage 4 cluster lookup must stay a bounded, batched similarity matmul; a naive search destroys the linear-time property that is the entire point of this architecture family.
- **Hyperparameter surface** — `τ`, β-decay curve, re-evaluation interval, frequency-decay rate, split threshold, head-type ratio. Keep all of these in experiment configs, never hardcoded; the sweep over them is itself part of the deliverable.
- **Filler lock-in** — frequency-based retention must weight by match tightness, not raw hit count, or shallow recurring patterns squat on slots.

## Novelty boundary

Nearly every individual mechanism here has direct published precedent (delta rule 2021, Titans 2024, product-key memory 2019, ARC, H2O, K-SVD, Hymba, L2O/JEPA-style distillation). The plan states this plainly and it should stay stated plainly — do not describe individual pieces as novel in code comments, write-ups, or commit messages. The untested contribution is the *specific combination*: cross-key pointer deduplication + periodic drift correction + frequency-weighted retention + surprise/correlation head separation on top of existing chunked delta-rule machinery. Stages 4–6 exist to validate or falsify exactly that.

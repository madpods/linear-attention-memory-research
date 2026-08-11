# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Stages 0–2 are complete. The full 75-config grid ran on the cluster on 2026-08-11 (A40, `backend=fla`, seed 0) and is merged to `results/stage2_gpu.csv`. **Four findings from it govern everything downstream — read these before designing Stage 3 or 4.**

**1. The single-seed baseline is not trustworthy at the capacity cliff, and one seed reversed a headline.** The CPU-vs-GPU `r=0` control matched to ±0.2 points everywhere except `delta` at kv=32, which moved **65.5% → 75.5%**. That is not bf16 noise — every other cell moved <1.5 points. kv=32 sits mid-collapse, where the run is partly-converged and small perturbations move the endpoint a lot. The consequence: on CPU `gated_delta` beat `delta` at kv=32 by 9.6 points; on GPU `delta` beat `gated_delta` by 1.8. **The "decay gates buy capacity" reading at kv=32 was noise, and the ordering is not resolved at n=1.** `capacity@95%` is unaffected (16 for both) because it is a threshold crossing away from the noisy region — prefer it over raw mid-cliff accuracy. Seed replication is now mandatory before any mechanism is compared against this baseline; a GPU run is 27–69 s, so a whole grid is ~30 min and there is no longer any excuse for n=1.

**2. Redundancy destroys effective capacity — the headline Stage 2 result.** `capacity@95%` for both error-corrected rules goes `16, 16, 8, 8, <4` across `r = 0, 0.25, 0.5, 0.75, 0.9`. Halved by `r=0.5`, gone by `r=0.9`. This is the curve every later stage is trying to lift, and it is measured on the axis this project added.

**3. `gated_delta` beats `delta` robustly at `r>0`, which is a different claim from the noisy `r=0` one.** At kv=32 the gaps are large and consistent in one direction: 72.1 vs 9.8 (`r=0.25`), 40.8 vs 20.1 (`r=0.5`), 50.0 vs 20.0 (`r=0.75`), 59.0 vs 22.4 (`r=0.9`). Many cells, one direction, margins far outside the ±10 seen at the cliff. Treat *this* as the baseline's real advantage, not the `r=0` kv=32 difference. Caveat: `delta`'s 9.8% at `r=0.25`/kv=32 is out of line with its own neighbours (~20% at higher r) and looks like a single failed run — another thing seeds settle.

**4. The redundant/non-redundant gap is partly an artifact, and must not be read as pure retrieval.** Redundant queries beat non-redundant ones by +8 to +62 points, widening with `r`. But `linear` — which cannot retrieve at all — scores **34.9% on redundant queries at `r=0.9`** against 0.0% on non-redundant. So a large part of redundant-query accuracy is reachable by learning the shared-value prior over `num_value_clusters` representatives without doing retrieval. The honest measure is the margin over `linear` at the same `r` (at `r=0.9`: 75.1 gated vs 34.9 linear, so real retrieval is happening) — never the raw redundant accuracy, and never the redundant-minus-non-redundant gap on its own. **Stage 4's dedup mechanism will be evaluated on exactly these queries, so this confound is a trap set directly in its path.**

- `src/lamr/data/mqar.py` — redundancy-parameterized MQAR generator (Stage 1).
- `src/lamr/layers/recurrent.py` — sequential reference update rules; correctness ground truth.
- `src/lamr/layers/chunked.py` — chunk-parallel delta rule in portable PyTorch (§15), ~8.5× the sequential reference on CPU.
- `src/lamr/layers/linear_attn.py` — the layer; modes `linear` / `delta` / `gated_delta` differ *only* in update rule.
- `src/lamr/models/lm.py` — small stacked LM; architecture fixed here per Stage 2.
- `src/lamr/metrics.py` — shared recall metrics, sliced by redundancy. Every stage calls this.
- `src/lamr/train.py` — the single training entry point (`python -m lamr.train`).
- `scripts/stage2_sweep.py` — baseline sweeps; appends CSV and skips completed runs, so it resumes.

Smoke results at `r=0`, `num_kv_pairs=8`, `d_model=128`: gated_delta 99.3% @500 steps, delta 30% @300, linear 1.9% @300 — the expected ordering, with plain linear attention failing as the floor.

**GPU throughput, measured over the full grid** (A40, 1500 steps, `seq_len=256`, `d_model=64`, batch 32): `linear` 454k tok/s (27 s/run), `delta` 181k (68 s), `gated_delta` 181k (69 s) — against ~26k tok/s for `delta`/`gated_delta` and ~40k for `linear` on this workstation's CPU. So `fla` gives ~7× over the CPU chunked backend, and the linear/delta spread is the **update rule** (delta's within-chunk sequential structure), not the backend.

Beware short timing runs. The same `gated_delta` config measured **1,068 tok/s over 20 steps** and **28,884 over 200**, against 181k over 1500 — Triton's first-call JIT (~45 s) swamps everything short. Any throughput number from fewer than several hundred steps is measuring the compiler, and an early reading of it here led to a wrong conclusion that `fla` was slower than the portable chunked path (it was comparing `linear`-on-chunked against `gated_delta`-on-fla, i.e. two different update rules, with the fla figure still compile-dominated). There is no evidence the chunked backend beats `fla` on GPU; that comparison has not been run.

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

**Modules must match between build and run.** The cluster uses Lmod, and the CoE docs are explicit that modules used to build software into a python environment have to be loaded again to run it. Triton compiles kernels against the CUDA toolkit on first call, so a batch job missing the cuda module fails deep inside `fla` rather than at import — a confusing failure to debug mid-sweep. `setup_env.sh` therefore resolves modules, records them to `$VENV/modules.env`, and `sweep_array.sbatch` replays that file. If you rebuild the venv by hand, re-record that file too. `setup_env.sh --verify` compiles and runs a trivial Triton kernel for exactly this reason: `import triton` succeeding proves nothing about the failure mode, which is *first-call compilation*.

### Cluster environment (module list captured 2026-08-11)

Rocky Linux **8 and 9, mixed** (~60/40). Slurm (`slurm/current` is loaded by default, so the `module load slurm` fallback is belt-and-braces).

**Module trees are per-OS-generation.** The captured `module avail` came from `/usr/local/apps/modulefiles-8` — note the `-8` — and carries explicitly suffixed variants like `Rstudio/2024.04_el8` alongside `Rstudio/2025.09_el9`. So a name recorded in `modules.env` on one generation is not guaranteed to resolve on the other, which is a second, independent reason the `build_os` check exists.

What the resolver actually picks on this cluster, given Lmod's `(D)` defaults:

**`module -t avail` emits no `(D)` markers on this cluster.** The columnar `module avail` shows them, terse output does not — so `pick_module`'s `(D)` branch never fires here and its *fallback* is what actually runs. That matters because the fallback is "highest version", which is right for one module and wrong for another:

| module | available | what gets loaded | why |
|---|---|---|---|
| `python` | 3.8 – 3.13 | `python/3.13` | highest is right. Note `python3` on PATH is 3.6.8, so the module is mandatory, and `sort -V` is required — a lexical sort picks 3.9 over 3.13 |
| `cuda` | 9.2 – 13.3 | `cuda/12.9` or `cuda/13.3` | neither highest nor default: follows `torch.version.cuda`, then newest of *that major* |
| `gcc` | 4.9.4 – 15.2 | `gcc/12.5` | highest is **wrong** — `gcc/15.2` is newer than any host compiler CUDA 12.x/13.x accepts, so `pick_capped_module` caps the major at `GCC_MAX_MAJOR` (12). Triton compiles a C launcher stub at runtime and Rocky 8 ships gcc 8.5, so a module is still needed |

Both the CUDA and gcc rows are arguments against trusting either `(D)` or "newest". The real captured lists are test cases in `test_module_resolution.sh`, pinning `cu12 → cuda/12.9`, `cu13 → cuda/13.3`, and `python/3.13 + gcc/12.5` from submit-a's exact terse output. `MODULE_GCC=none` opts out; `GCC_MAX_MAJOR` raises the cap.

There is **no `pytorch`/`conda` module in play** — the venv is built from the python module, which is what makes the module record and the OS check load-bearing.

One EL8 caveat to watch on first install: PyPI's torch wheels are `manylinux_2_28`, and Rocky 8's glibc 2.28 is exactly that floor. If pip ever fails to resolve a torch wheel there, the fallback is to build on EL9 and constrain the array to EL9 rather than to fight it.

**The mixed OS is a live hazard, not trivia**, and the rule is *directional*. glibc is backward compatible but not forward compatible:

- **built on EL8, running on EL9 → fine.** Binaries linked against the older glibc resolve against the newer one.
- **built on EL9, running on EL8 → fatal.** `GLIBC_2.34 not found` from the dynamic loader, since EL8 ships 2.28.

**The settled configuration: build on the EL8 login node, run on EL9 nodes.** The survey resolved this. `submit-a` is Rocky 8.10 / glibc 2.28, so anything built there is EL8 — and that is *fine*, because glibc is backward compatible and an EL8 venv runs on EL9 nodes. Meanwhile the array targets EL9 because the EL8 GPU nodes are M60 and GTX980 hardware modern Triton cannot use at all; every useful accelerator (A40, L40S, V100, H100, H200, RTX 6000/8000) is EL9. So no EL9 submit host is needed and `--build-in-alloc` stays unused.

This is exactly the direction `check_os_compat.sh` permits while rejecting the reverse, and `day1.sh`'s preflight now allows it too — it fails only when the login node is *newer* than the target. Earlier drafts of both blocked any mismatch, which would have refused the best configuration available here.

Feature names are confirmed **`el8` / `el9`**, e.g. `srun -p preempt --constraint=el8 --pty tcsh`.

**The array pins a single GPU class: `--constraint=el9&a40`.** Both halves matter, and the second is methodological rather than operational. `preempt`'s EL9 GPU nodes span `gtx1080` (sm_61) to `h200` (sm_90); Triton requires sm_70+, so the four gtx1080 nodes would fail outright and `--requeue` could return a task to one. More importantly, **tokens/sec is a Stage 2 deliverable**, and measured across V100/A40/H100 it is not a comparable column — which defeats the matched-baseline rule (principle 3) that every later stage depends on. Homogeneous hardware is a correctness requirement for that metric. A40 has the most `preempt` nodes (13, ~26 GPUs, comfortably over the `%20` throttle) and is sm_86. If a class starves, **switch to another single class rather than widening** — and note that switching mid-sweep breaks comparability against rows already recorded.

Accounting: this user's associations are `coehpc` and `eecs`; `preempt`/`share` accept any account, so no `--account` line is needed. `MaxArraySize` is 1001 and `MaxJobCount` 5000, so 75 tasks is unremarkable.

**Home is 25 GB** (`/nfs/stak/users`), which a torch venv fits but does not swim in. `/scratch` is 347 GB but node-local (`/dev/sda2`), so it cannot hold a venv shared across array tasks. This is also what rules out the container route in practice: `apptainer` 1.5.1 is installed but there is no `/etc/subuid` entry, so images can only be pulled, not built — and a ~20 GB NGC image against a 25 GB quota is not viable. The venv path is the right one here, not merely the incumbent.

`scripts/slurm/check_os_compat.sh` owns this rule so it exists once rather than in both callers, and encodes the direction — an EL8-built venv is explicitly *not* blocked on EL9 nodes. Getting that backwards is expensive both ways: too strict silently refuses nodes the venv would have run on, too loose lets tasks die in the loader. Nine cases in `test_module_resolution.sh` pin it, including dotted minors: a naive digit-strip turns `rocky8.10` into `810`, which compares as *newer* than `rocky9` and waves through a venv that cannot run. The residual risk on the EL8→EL9 path is not glibc but the per-generation module tree, which surfaces as a legible module-load warning rather than a loader error.

`setup_env.sh` records the build generation to `$VENV/build_os`; both `--verify` and every array task compare against it and abort early with the fix rather than dying in the loader mid-sweep.

**The CUDA module is chosen from what torch reports, not from Lmod's `(D)` default.** With 11.x through 13.x on the menu the default is very unlikely to be the major torch was built against, and a toolkit *newer* than torch's bundled runtime is precisely where Triton's first compile breaks. So the install phase now installs torch first, reads `torch.version.cuda`, then loads the highest matching `cuda/<major>.x`. `MODULE_CUDA` still overrides. Ten cases in `scripts/slurm/test_module_resolution.sh` pin this, including the one that matters: a cu12 torch must pick `cuda/12.4` over a `cuda/13.0(D)` default.

**torch installs from PyPI by default**, no `--index-url`. Its linux wheels are CUDA-enabled and bundle their own runtime, and this keeps the cluster's torch in step with the CPU box's 2.13.x — which matters because `fla` tracks recent torch closely, and the old hardcoded `cu124` index lags well behind. Set `TORCH_INDEX` to pin a specific build if a driver ever requires it.

**Partitions accepting any Slurm account** — no `--account` line needed; every other partition in the CoE table is restricted to one research group.

| partition | GPU | time limit | concurrent GPU cap |
|---|---|---|---|
| `preempt` | any | 7 days | none (preemptible) |
| `dgxh` | H100/H200 | 2 days | 8 |
| `dgx2` | V100 | 7 days | 16 |
| `share` | M60/A40 | 14 days | 2 |
| `ampere` | A40 | 2 days | 2 |

The array defaults to `preempt` with `--requeue`. These are short jobs and the sweep skips configurations already in its CSV, so a preempted task resumes instead of repeating — which makes an uncapped preemptible queue strictly better than waiting on `share`'s 2-GPU cap. If preempt starves, `dgxh` is the fastest fallback; drop the array throttle to `%8` there, `%2` on `share`/`ampere`. User-wide limits are 1000 submitted / 400 running.

**`src/lamr/layers/fla_backend.py` is verified on GPU as of 2026-08-11** — H100 80GB (MIG 4g.40gb), `torch 2.13.0+cu130`, `triton 3.7.1`, `fla` from PyPI. The parity gate passes 9/9. Of the three conventions guessed without a device, **all three were correct**; the gate found a fourth (dtype) that source-reading had not surfaced. Specifically confirmed by execution, not inspection:

- layout — the unconditional `(B,H,T,D) → (B,T,H,D)` transpose is right, and the negative control shows the untransposed version is catastrophically wrong even when shapes match;
- `scale=1.0` — right; fla's default `d_k ** -0.5` is wrong here by ~0.82 relative, so the choice is load-bearing rather than cosmetic;
- state orientation — `[N, H, K, V]` is already `(d_k, d_v)`, so *not* transposing is right, tested with `d_k=32, d_v=16` so a transpose could not hide;
- `initial_state` round-trips, which Stage 5's slow pass depends on;
- and all three implementations agree — recurrent reference, portable chunked, and fla Triton — which is the premise the whole backend split rests on.

Four conventions are reconciled there:

- **Layout** — `fla` requires `(B, T, H, D)` and has *removed* `head_first` (passing it raises `DeprecationWarning`). This project works in `(B, H, T, D)`, so the adapter transposes unconditionally. This is the dangerous one: a wrong layout does not raise, it treats heads as timesteps and returns plausible nonsense.
- **Scale** — `fla` defaults to `k.shape[-1] ** -0.5`; we pass `scale=1.0` because the layer L2-normalizes q/k itself.
- **Decay** — the gated kernel takes `g`, log-space, positioned *before* `beta`. Passed by keyword so the order mismatch cannot bite.

Re-verified against upstream `main` on 2026-08-11: both import paths resolve (`chunk_delta_rule`, `chunk_gated_delta_rule`), the parameter orders match, `scale=None` still means `k.shape[-1] ** -0.5`, `g` is still log-space while `use_gate_in_kernel` is False (the default), and the final state is still `[N, H, K, V]` — i.e. already `(d_k, d_v)`, so the adapter is right *not* to transpose it. Note the gated kernel types `v`/`g`/`beta` against `HV` rather than `H`, which matters only if query and value head counts ever diverge; they don't here.

- **dtype — a fourth convention, found by running the gate on 2026-08-11.** `chunk_delta_rule` *asserts* `q.dtype != torch.float32`: "does not support float32. Please use bfloat16." So the adapter casts to `KERNEL_DTYPE` (bf16) on the way in and back to the caller's dtype on the way out. bf16 rather than fp16 because it is what `fla`'s own examples use, and the state matrix accumulates over the whole sequence, where the wider exponent range matters more than fp16's extra mantissa bits.

  The gated kernel does **not** carry that assert and ran fine in fp32 — it is cast anyway, deliberately. Stage 2's primary comparison is delta against gated_delta, and if one ran bf16 while the other ran fp32, that comparison would be partly about precision rather than about the update rule it is supposed to isolate.

**GPU runs are therefore bf16 where the CPU baselines were fp32.** This is a real numerical difference and not a formality. It is also *measured*, not assumed: the sweep re-runs every `r=0` configuration on GPU, so those 15 rows sit directly against the recorded CPU curves as a built-in control. Check them before trusting any `r>0` row.

**`linear` mode never touches `fla`** — `linear_attn.py` routes `backend="fla"` to the portable chunked path, because the floor baseline has no within-chunk sequential dependency and so nothing for a Triton kernel to fuse. Consequence: in a GPU sweep, `linear` runs fp32 while `delta`/`gated_delta` run bf16. Left as-is deliberately, and the reasoning should survive: bf16 is the *less* precise setting, so an error-corrected rule beating the linear floor while handicapped that way is a conservative result, not a flattered one. It is recorded here so nobody later reads part of that gap as a precision artifact.

`REL_TOL` in the parity test was re-derived for this: 1e-3 was set assuming fp32 and is unreachable for bf16 kernels against an fp64 reference, since rounding q/k/v alone costs ~2e-3 before any arithmetic. It is now 3e-2 — and because loosening a tolerance is only honest if the gate still discriminates, three `test_negative_control_*` cases feed the wrong conventions (fla's default `scale`, raw `alpha` as `g`, and the untransposed layout with `H == T` so no shape error can catch it) and assert each lands 10× outside the bound. **If a negative control ever fails, the tolerance has stopped discriminating and the positive results mean nothing**, which is the property that keeps "never relax the test" intact rather than merely asserted.

**The parity gate is necessary but not sufficient — it tests the kernels' forward pass only.** It never builds a model, never calls backward through `fla`'s autograd, and never moves a batch to a device, so it cannot catch a failure in the training loop itself. `setup_env.sh --verify` therefore ends with a 20-step `python -m lamr.train --backend fla` run. That check exists because the codebase had **no device handling at all** until 2026-08-11: `TrainConfig` had no `device`, nothing called `.to()`, and so the sweep would have run CPU-only on A40 nodes — except that `backend="fla"` would have failed first, since `fla_available()` tests only that a CUDA device *exists*, not that the tensors are on it. All 75 tasks would have died. `resolve_device` now raises that mismatch explicitly, and `MQARBatch.to()` moves every tensor field (walked over the dataclass fields, so a field added later is not left behind — `recall_metrics` gathers five of them against the logits).

Two details in that path are load-bearing for comparability with the recorded CPU baselines: the batch-sampling `Generator` stays on the **host** and only the drawn indices are moved, because a CUDA generator would draw a different sequence for the same seed and the `r=0` rows are meant to be a controlled comparison; and `torch.cuda.synchronize()` runs before the final clock read, since async launches would otherwise report a `tokens_per_sec` that is simply wrong — and that column is a Stage 2 deliverable.

`tests/test_fla_parity.py` is the gate. Run it before any training on the GPU — until it passes, no GPU number is comparable to any CPU number, and interchangeability is the premise the whole backend split rests on. Each assertion names the convention it pins. **Fix `fla_backend.py`; never relax the test.** Note the gate skips itself when `fla` is unimportable, so `setup_env.sh --verify` asserts `fla_available()` first; otherwise pytest would exit 0 with the gate never having run.

One lesson from running it, worth keeping if the negative controls are ever extended: a wrong convention can fail by *diverging* rather than by being numerically far off. Raw `alpha` passed as log-space `g` makes the per-step factor `exp(0.95) ≈ 2.59`, and `2.59**128 ≈ 1e52` overflows bf16 to `nan`. An `err > floor` assertion reads `nan > 0.3` as False and then reports "the tolerance has stopped discriminating" when the opposite was just demonstrated. `assert_detectably_wrong` therefore counts non-finite output as the stronger pass. The positive tests need no such handling — `nan < REL_TOL` is False, so a diverged result can never sneak through one.

Two operational notes: `sweep_array.sbatch` deliberately leaves `--partition` and `--account` unset (run `sinfo -o "%P %G %m %l"` first, as the plan's Stage 0 says). And each array task writes its own CSV — concurrent appends to one file interleave mid-row — which `merge_results.py` stitches back together, keeping the last row per configuration so a re-run task overrides its partial.

**`logs/` must exist before `sbatch`.** Slurm opens the array's `--output`/`--error` files itself at task launch, so the `mkdir -p logs` inside the job script runs far too late — a missing directory fails all 75 tasks instantly with no output explaining why. `logs/.gitkeep` is tracked (with a `!logs/` negation in `.gitignore`) so a fresh clone already has it, and `day1.sh` re-creates it before submitting. Keep both true if those paths are ever edited.

The module-resolution path in `setup_env.sh` is exercised by a stubbed-Lmod harness rather than trusted by inspection — it previously died silently on its own happy path, because a function whose last command is `[ test ] && echo` returns that test's status, and `set -e` kills the script when the test is false. Two more of the same class were in `pick_module` (a `grep` for `(D)` finding nothing, inside a `$(...)` assignment under `pipefail`). If you edit that function, re-run the harness; every one of these failures presents as an exit with no message at all. Related: re-running `--install` over an existing venv used to blank `$VENV/modules.env`, which is worse than deleting it — `sweep_array.sbatch` found the file, loaded nothing, reported nothing wrong, and left Triton to fail on first kernel compile. It now distinguishes missing / empty / populated.

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

`addendum_03_key_routing_via_delta_rule.md` supersedes Addendum 02's Decision 2: `C_keys` becomes a delta-rule-trained routing matrix `M` (`d_k × num_clusters`, one-hot target), read as `softmax(q^T M) @ C_values`, and the residual is computed against the read-time path so reconstruction is exact by construction. **Implement 03's version, not 02's.** Addendum 02 Decision 1 (state accounting) is unaffected.

Two open issues remain on the router:

**1. The router is itself rank-limited, so it may relocate the ceiling rather than raise it.** `M` is `d_k × num_clusters` and must map `N` keys — arbitrarily labelled, since cluster identity comes from *value* similarity and is uncorrelated with key geometry — onto `num_clusters` classes. That is a linear classifier over `d_k` features learning an arbitrary labelling of `N` random points, so by Cover's-theorem-style counting it saturates around `N ≈ 2·d_k`, *regardless of how those keys are distributed across clusters*. Meanwhile `R` is `d_k × d_v` and holds per-key residuals, also rank `≤ d_k`. So the composite stores `num_clusters` values plus two `O(d_k)`-capacity structures — and plain delta rule already gave `O(d_k)`. **The mechanism has not obviously raised the ceiling; it may have moved it into the router.** Addendum 03's sweep over `m` and `num_clusters` does cover this (since `N = m × num_clusters`), but the predicted failure axis should be **total keys `N` against `d_k`**, not keys-per-cluster `m`. Reporting routing accuracy against `N / d_k` is what will show this; reporting against `m` alone could hide it.

**2. This is a strong argument for running Stage 3 before 4b, as already sequenced.** If the router saturates at `N ≈ 2·d_k`, then a feature map (§17) that gives `M` shape `d_φ × num_clusters` with `d_φ > d_k` raises routing capacity directly — the same lever, applied to the new bottleneck. Stage 3's `d_φ` budget therefore constrains Stage 4b's viable `N`, which is a dependency the plan's stage ordering already implies but does not state.

**3. `R` chases a moving target.** The consistency fix computes the residual as `v − softmax(k^T M) @ C_values` using the *current* `M`. But `M` is still training, so a residual written at step `t` reconstructs against a router that has since moved, and early-training residuals are written against near-random routing. The delta rule self-corrects only for keys that recur — and in MQAR most keys appear once or twice per sequence, so there is little correction opportunity. This is consolidated risk #1 (self-referential signals gating their own future updates) in a new place; a stop-gradient, a warmup period where only `M` trains, or an explicit `R` decay are the usual mitigations. Decide deliberately rather than discovering it as instability.

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

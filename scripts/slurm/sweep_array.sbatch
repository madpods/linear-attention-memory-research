#!/usr/bin/env bash
#SBATCH --job-name=lamr-stage2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# NOTE: logs/ must exist BEFORE submitting. Slurm opens these files itself when
# the task launches, and the `mkdir -p logs` further down runs too late to help
# -- a missing directory fails every task instantly with no output to explain
# why. logs/.gitkeep is tracked so a fresh clone already has it; day1.sh also
# creates it before submitting. If you hand-edit these paths, keep that true.
#SBATCH --partition=preempt
#SBATCH --constraint=el9&a40
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-74%20
#
# Stage 2 full grid on the OSU CoE HPC: 3 modes x 5 redundancies x 5 kv sizes
# = 75 runs. Verify the array size still matches before submitting:
#
#     python scripts/stage2_sweep.py --preset full --count
#
# Submit:  sbatch scripts/slurm/sweep_array.sbatch
#
# --------------------------------------------------------------------------
# --constraint=el9&a40 -- both halves are load-bearing
# --------------------------------------------------------------------------
# el9: the cluster mixes Rocky 8 and Rocky 9 and glibc is not forward
# compatible, so the venv and the tasks must agree. EL9 because the EL8 GPU
# nodes are M60 and GTX980 hardware that modern Triton cannot use at all; the
# useful accelerators are all EL9. Building the venv on an EL8 login node and
# running here is fine and expected -- glibc is backward compatible, and
# check_os_compat.sh permits exactly that direction while rejecting the reverse.
#
# a40: this is NOT just about avoiding slow nodes. Two reasons:
#
#   1. preempt's EL9 GPU nodes span gtx1080 (sm_61) to h200 (sm_90), and Triton
#      requires sm_70+. The four gtx1080 nodes (cn-e07..cn-e10) would fail
#      outright, and --requeue could land a task back on one.
#   2. tokens/sec is a Stage 2 deliverable. Measured across V100, A40 and H100
#      it is not a comparable column, which defeats the fixed-baseline rule the
#      whole experiment design rests on (plan principle 3). Homogeneous hardware
#      is a correctness requirement for the metric, not a preference.
#
# A40 has the most nodes on preempt (cn-r-1..6, cn-s-1..5, cn-t-1, sail-gpu0 =
# 13 nodes, ~26 GPUs, comfortably above the %20 throttle) and is sm_86.
#
# To change class, keep it to ONE: l40s (4 nodes), v100 (dgx2-*, 4 nodes),
# h100/h200 (dgxh, but see the partition note). If a class starves, switch
# rather than widen -- and if you do switch mid-sweep, the tokens/sec column
# stops being comparable across the rows already recorded.
#
#     sbatch --constraint=el9&l40s scripts/slurm/sweep_array.sbatch
#
# Accounting: this user's associations are coehpc and eecs, and preempt/share
# accept any account, so no --account line is needed.
#
# --------------------------------------------------------------------------
# --time=04:00:00
# --------------------------------------------------------------------------
# Each task is 1500 steps x batch 32 x seq_len 256 = 12.29M tokens. The CPU
# baselines ran at ~26k tok/s (~8 min/run), so a GPU should be far under an
# hour -- but the first observed GPU throughput was 1,068 tok/s on a 20-step
# smoke run, which is 3.2 hours at that rate. That number is dominated by
# Triton's first-call JIT compilation rather than steady state, and 20 steps is
# far too short to amortize it. 4 hours is headroom against being wrong about
# that: on preempt the limit is 7 days, the queue is uncapped, and --requeue
# plus CSV-skip makes a longer reservation nearly free. Hitting a wall clock
# mid-sweep costs a resubmit; over-reserving costs nothing measurable.
#
# --------------------------------------------------------------------------
# Partition choice
# --------------------------------------------------------------------------
# These accept ANY Slurm account (per the CoE Slurm Howto), so no --account
# line is needed; the department account assigned at signup will do. Everything
# else in the partition table is restricted to a specific research group.
#
#   partition  GPU          time limit  concurrent GPU cap   notes
#   preempt    any          7 days      none                 preemptible
#   dgxh       H100/H200    2 days      8                    fastest
#   dgx2       V100         7 days      16
#   share      M60/A40      14 days     2                    open to all users
#   ampere     A40          2 days      2
#
# preempt is the default here because it has no resource caps and these are
# short jobs: --requeue means a preempted task simply restarts, and the sweep
# skips configurations already present in its CSV, so a restart resumes rather
# than repeats. That combination makes preemption nearly free.
#
# If preempt is starved, dgxh is the best alternative (H100s), but drop the
# array throttle to %8 to respect its concurrent-GPU cap. On share or ampere
# use %2. Exceeding a cap does not fail the job -- tasks just sit pending --
# but the throttle makes the queue legible.
#
# User-wide limits: 1000 jobs submitted, 400 running. 75 is comfortably inside.
#
# --------------------------------------------------------------------------
# Each task writes its OWN csv. Concurrent appends to one file interleave and
# corrupt rows; scripts/merge_results.py stitches them afterwards.
set -euo pipefail

VENV="${VENV:-.venv-gpu}"
PRESET="${PRESET:-full}"

# Slurm is normally auto-loaded on submit nodes. If it was missing, the CoE
# docs say to reset Unix config files via TEACH; this is the short-term fix.
command -v srun >/dev/null 2>&1 || module load slurm

# Replay the exact modules setup_env.sh built the venv against. The CoE Lmod
# docs are explicit that build-time modules must also be present at run time;
# Triton compiles kernels against the CUDA toolkit on first call, so skipping
# this fails deep inside fla rather than at import.
if [ -s "$VENV/modules.env" ]; then
    while read -r m; do
        [ -n "$m" ] || continue
        module load "$m" || echo "WARN: could not load $m"
    done < "$VENV/modules.env"
    echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
elif [ -f "$VENV/modules.env" ]; then
    # Present but empty. Distinguished from missing on purpose: an empty record
    # means setup_env.sh loaded nothing, so CUDA is whatever the bare PATH
    # provides. Triton compiles against the toolkit on first call, so this is
    # the point to notice it -- not 40 minutes into a 75-task array.
    echo "WARN: $VENV/modules.env is empty -- no modules will be loaded."
    echo "      If fla fails to compile kernels, re-run setup_env.sh with"
    echo "      MODULE_PYTHON / MODULE_CUDA set explicitly."
else
    echo "WARN: $VENV/modules.env missing -- run scripts/slurm/setup_env.sh first"
fi

# The cluster mixes Rocky 8 and Rocky 9 (~60/40). glibc is backward but not
# forward compatible, so an EL9-built venv cannot run on an EL8 node -- which
# left unchecked surfaces as "GLIBC_2.34 not found" from the dynamic loader
# partway through a 75-task array. The rule is directional and lives in
# check_os_compat.sh; an EL8-built venv is fine everywhere and is not blocked.
if [ -r "$VENV/build_os" ]; then
    bash scripts/slurm/check_os_compat.sh "$(cat "$VENV/build_os")" || exit 1
else
    echo "WARN: no $VENV/build_os -- cannot check OS compatibility."
    echo "      Rebuild with scripts/slurm/setup_env.sh --install to record it."
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

mkdir -p logs results/parts

IDX="${SLURM_ARRAY_TASK_ID:?must run as a job array}"
OUT="results/parts/stage2_${IDX}.csv"

# Triton's kernel cache goes to NODE-LOCAL storage, not its default ~/.triton.
# Two reasons, both specific to this sweep:
#   1. 20 tasks start concurrently and compile the SAME kernels. The default
#      cache sits on NFS-mounted home, and concurrent cross-node writes to it
#      are a known source of contention and corrupted entries.
#   2. Home here is a 25GB quota; a kernel cache is not what it is for.
# The cost is that each task recompiles from cold, measured at ~45s against a
# ~7min task -- cheap next to debugging a poisoned shared cache mid-array.
# Slurm sets TMPDIR to a per-job node-local directory on most configurations;
# /scratch (347G, node-local here) is the fallback.
TRITON_BASE="${TMPDIR:-/scratch/$USER}"
if mkdir -p "$TRITON_BASE/triton-$SLURM_JOB_ID-$IDX" 2>/dev/null; then
    export TRITON_CACHE_DIR="$TRITON_BASE/triton-$SLURM_JOB_ID-$IDX"
    echo "triton cache: $TRITON_CACHE_DIR (node-local)"
    # Node-local, so it dies with the node anyway; remove it explicitly so a
    # requeued task on the same node does not accumulate copies.
    trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT
else
    echo "WARN: could not create a node-local triton cache under $TRITON_BASE;"
    echo "      falling back to the default (~/.triton, on NFS home)."
fi

echo "task $IDX -> $OUT  (host $(hostname), partition ${SLURM_JOB_PARTITION:-?})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --only-index runs exactly one grid point. Safe to re-run: completed rows are
# skipped, which is what makes --requeue harmless.
srun python scripts/stage2_sweep.py \
    --preset "$PRESET" \
    --backend fla \
    --only-index "$IDX" \
    --out "$OUT"

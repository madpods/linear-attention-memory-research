#!/usr/bin/env bash
#SBATCH --job-name=lamr-stage2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --time=02:00:00
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
if [ -f "$VENV/modules.env" ]; then
    while read -r m; do
        [ -n "$m" ] && module load "$m"
    done < "$VENV/modules.env"
    echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
else
    echo "WARN: $VENV/modules.env missing -- run scripts/slurm/setup_env.sh first"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

mkdir -p logs results/parts

IDX="${SLURM_ARRAY_TASK_ID:?must run as a job array}"
OUT="results/parts/stage2_${IDX}.csv"

echo "task $IDX -> $OUT  (host $(hostname), partition ${SLURM_JOB_PARTITION:-?})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --only-index runs exactly one grid point. Safe to re-run: completed rows are
# skipped, which is what makes --requeue harmless.
srun python scripts/stage2_sweep.py \
    --preset "$PRESET" \
    --backend fla \
    --only-index "$IDX" \
    --out "$OUT"

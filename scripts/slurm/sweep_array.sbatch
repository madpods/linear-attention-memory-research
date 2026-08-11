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

# The cluster mixes Rocky 8 and Rocky 9 (~60/40) and glibc is not forward
# compatible, so a venv built against an EL9 python module cannot run on an EL8
# node. Left unchecked that surfaces as "GLIBC_2.34 not found" from the dynamic
# loader partway through a 75-task array. Fail fast and name the fix instead.
if [ -r "$VENV/build_os" ] && [ -r /etc/os-release ]; then
    BUILD_OS="$(cat "$VENV/build_os")"
    # shellcheck disable=SC1091
    RUN_OS="$(. /etc/os-release && printf '%s%s' "${ID:-unknown}" "${VERSION_ID%%.*}")"
    if [ "$BUILD_OS" != "$RUN_OS" ]; then
        echo "FATAL: venv was built on $BUILD_OS but this node is $RUN_OS." >&2
        echo "glibc is not forward compatible, so this task would fail in the" >&2
        echo "dynamic loader. Pin the array to one OS generation and resubmit:" >&2
        echo >&2
        echo "    sbatch --constraint=<feature> scripts/slurm/sweep_array.sbatch" >&2
        echo >&2
        echo "Find the feature name for this cluster with:" >&2
        echo "    sinfo -o '%20N %20P %40f'      # per-node features" >&2
        echo "    scripts/slurm/survey_cluster.sh  # includes a deduped list" >&2
        echo >&2
        echo "Building on an EL8 node instead makes the venv run on both." >&2
        exit 1
    fi
    echo "os: $RUN_OS (matches build)"
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

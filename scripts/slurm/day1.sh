#!/usr/bin/env bash
# One-shot first-day setup on the OSU CoE HPC.
#
# Run from the repo root on a LOGIN node (it allocates its own GPU node for the
# parts that need one):
#
#     bash scripts/slurm/day1.sh                 # set up and stop before submitting
#     bash scripts/slurm/day1.sh --submit        # ... and submit the 75-job array
#     bash scripts/slurm/day1.sh --partition dgx2
#
# It surveys the cluster, builds the GPU environment, runs both test suites,
# checks the array size, and prints the submit command. It deliberately does
# NOT submit unless asked: the fla adapter has never executed, and queueing 75
# GPU jobs against an unverified kernel binding wastes an allocation.
set -uo pipefail

PARTITION="${PARTITION:-dgxh}"
VENV="${VENV:-.venv-gpu}"
SETUP_TIME="${SETUP_TIME:-01:00:00}"
SUBMIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --submit) SUBMIT=1 ;;
        --partition) PARTITION="$2"; shift ;;
        --partition=*) PARTITION="${1#*=}" ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

step()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()    { printf '    \033[32mok\033[0m %s\n' "$1"; }
warn()  { printf '    \033[33mwarn\033[0m %s\n' "$1"; }
die()   { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

# --- 0. preflight -------------------------------------------------------

step "preflight"
[ -f pyproject.toml ] && [ -d src/lamr ] || die "run this from the repo root"
command -v srun >/dev/null 2>&1 || module load slurm 2>/dev/null || true
command -v srun >/dev/null 2>&1 || die "no srun; is this a submit node? try: module load slurm"
ok "repo root, slurm available"

if ! sinfo -h -p "$PARTITION" >/dev/null 2>&1; then
    warn "partition '$PARTITION' not found. Available:"
    sinfo -h -o "      %P" | sort -u
    die "pick one with --partition <name>"
fi
ok "partition $PARTITION exists"

# --- 1. survey ----------------------------------------------------------

step "surveying cluster -> cluster_survey.txt"
bash scripts/slurm/survey_cluster.sh > cluster_survey.txt 2>&1 \
    && ok "written ($(wc -l < cluster_survey.txt) lines)" \
    || warn "survey had errors; see cluster_survey.txt"

# --- 2. build the environment on a GPU node -----------------------------

step "building GPU environment (allocates a node on $PARTITION; ~10-20 min)"
echo "    installing torch+cuda, fla, then running both test suites"
srun --partition="$PARTITION" --gres=gpu:1 --time="$SETUP_TIME" \
     --cpus-per-task=4 --mem=16G \
     bash scripts/slurm/setup_env.sh 2>&1 | tee setup.log
SETUP_RC=${PIPESTATUS[0]}

if [ "$SETUP_RC" -ne 0 ]; then
    echo
    if grep -q "test_fla_parity" setup.log 2>/dev/null; then
        cat <<'EOF'
The fla parity gate failed. This is the EXPECTED failure mode, not a surprise:
src/lamr/layers/fla_backend.py was written without a CUDA device and encodes
three unverified guesses. The failing assertion names which one is wrong:

  "output differs"  -> query scaling. We pass scale=1.0 because the layer
                       L2-normalizes q/k; fla defaults to d_k ** -0.5.
  "transposed state"-> fla returns (d_v, d_k). Transpose it INSIDE
                       fla_backend.py; do not change the convention
                       downstream, Stage 4 indexes that matrix by key.
  "gated ... differs"-> decay parameterization; fla takes g = log(alpha).

Fix fla_backend.py, then re-run just the gate:
EOF
        printf '\n  srun --partition=%s --gres=gpu:1 --time=00:20:00 \\\n' "$PARTITION"
        printf "       bash -c 'source %s/bin/activate && python -m pytest tests/test_fla_parity.py -q'\n\n" "$VENV"
        echo "Do not submit the array until it passes -- GPU results are not"
        echo "comparable to the CPU baselines until they agree."
    else
        echo "Environment build failed before the parity gate. See setup.log."
    fi
    exit 1
fi
ok "environment built, all tests passed (including the fla parity gate)"

# --- 3. verify the array size matches the grid --------------------------

step "checking array size"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
GRID=$(python scripts/stage2_sweep.py --preset full --count) || die "could not count the grid"
SBATCH_FILE=scripts/slurm/sweep_array.sbatch
# POSIX sed rather than grep -oP: -P is a GNU extension and not guaranteed.
ARRAY=$(sed -n 's/^#SBATCH --array=[0-9]*-\([0-9]*\).*/\1/p' "$SBATCH_FILE" | head -1)
SB_PART=$(sed -n 's/^#SBATCH --partition=\([^ ]*\).*/\1/p' "$SBATCH_FILE" | head -1)
[ -n "$ARRAY" ] || die "could not parse --array from $SBATCH_FILE"
EXPECTED=$((ARRAY + 1))

if [ "$GRID" != "$EXPECTED" ]; then
    die "grid has $GRID runs but sbatch array covers $EXPECTED (0-$ARRAY).
     Fix --array in scripts/slurm/sweep_array.sbatch to 0-$((GRID - 1))."
fi
ok "grid=$GRID matches array 0-$ARRAY"

# --- 4. submit ----------------------------------------------------------

if [ "$SUBMIT" -eq 1 ]; then
    step "submitting"
    sbatch scripts/slurm/sweep_array.sbatch
    echo
    echo "Watch:   squeue -u \$USER"
    echo "Merge:   python scripts/merge_results.py results/parts results/stage2.csv"
else
    step "ready -- not submitting (pass --submit to go ahead)"
    cat <<EOF
    Submit:  sbatch scripts/slurm/sweep_array.sbatch
    Watch:   squeue -u \$USER
    Merge:   python scripts/merge_results.py results/parts results/stage2.csv

    $GRID runs on partition '$SB_PART'.
    Change the partition or throttle by editing $SBATCH_FILE.
EOF
fi

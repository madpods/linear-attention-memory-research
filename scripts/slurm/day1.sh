#!/usr/bin/env bash
# One-shot first-day setup on the OSU CoE HPC.
#
# Run from the repo root on a LOGIN node (it allocates its own GPU node for the
# parts that need one):
#
#     bash scripts/slurm/day1.sh                 # set up and stop before submitting
#     bash scripts/slurm/day1.sh --submit        # ... and submit the 75-job array
#     bash scripts/slurm/day1.sh --partition dgx2
#     bash scripts/slurm/day1.sh --constraint el8   # pin to one OS generation
#
# The cluster mixes Rocky 8 and Rocky 9, and glibc is backward but not forward
# compatible. Building on an EL8 login node is therefore the better default: the
# venv then runs on every node and needs no --constraint. An EL9-built venv must
# be pinned with --constraint=el9, which costs ~60% of the nodes. Feature names
# are el8 / el9; cluster_survey.txt lists what each node advertises under "node
# features". scripts/slurm/check_os_compat.sh enforces the direction.
#
# It surveys the cluster, builds the GPU environment, runs both test suites,
# checks the array size, and prints the submit command. It deliberately does
# NOT submit unless asked: the fla adapter has never executed, and queueing 75
# GPU jobs against an unverified kernel binding wastes an allocation.
set -uo pipefail

PARTITION="${PARTITION:-dgxh}"
VENV="${VENV:-.venv-gpu}"
SETUP_TIME="${SETUP_TIME:-01:00:00}"
CONSTRAINT="${CONSTRAINT:-}"
SUBMIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --submit) SUBMIT=1 ;;
        --partition) PARTITION="$2"; shift ;;
        --partition=*) PARTITION="${1#*=}" ;;
        --constraint) CONSTRAINT="$2"; shift ;;
        --constraint=*) CONSTRAINT="${1#*=}" ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

# The cluster mixes Rocky 8 and Rocky 9 and glibc is not forward compatible, so
# the venv must be built and run on one generation. --constraint pins both the
# verify allocation and the array to the same feature. Feature names are
# site-specific; cluster_survey.txt lists them (see the "node features"
# section). Passed as a real srun/sbatch argument only when set, so the default
# behaviour is unchanged.
CONSTRAINT_ARGS=()
[ -n "$CONSTRAINT" ] && CONSTRAINT_ARGS=(--constraint="$CONSTRAINT")

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

# Installing here rather than inside srun: compute nodes often have no outbound
# network, and holding a GPU through a multi-GB torch download wastes the
# allocation. Only the GPU-dependent checks are srun'd, below.
step "installing on the login node (~10-20 min; no GPU held)"
bash scripts/slurm/setup_env.sh --install 2>&1 | tee setup.log
INSTALL_RC=${PIPESTATUS[0]}
[ "$INSTALL_RC" -eq 0 ] || die "install phase failed; see setup.log"
ok "venv built, CPU test suite passed"

step "verifying on a GPU node ($PARTITION)"
srun --partition="$PARTITION" --gres=gpu:1 --time="$SETUP_TIME" \
     --cpus-per-task=4 --mem=16G "${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"}" \
     bash scripts/slurm/setup_env.sh --verify 2>&1 | tee -a setup.log
SETUP_RC=${PIPESTATUS[0]}

if [ "$SETUP_RC" -ne 0 ]; then
    echo
    if grep -q "test_fla_parity" setup.log 2>/dev/null; then
        cat <<'EOF'
The fla parity gate failed. src/lamr/layers/fla_backend.py has never executed
on a GPU, so this is a likely failure mode rather than a surprise. Its import
names and signatures ARE verified against fla's source; what is unverified is
numerical agreement. The failing assertion names the convention at fault:

  "output differs"  -> query scaling. We pass scale=1.0 because the layer
                       L2-normalizes q/k; fla defaults to d_k ** -0.5. If fla
                       ever stops honouring scale=1.0, normalize in the kernel
                       via use_qk_l2norm_in_kernel=True instead.
  "transposed state"-> upstream documents [N, H, K, V], i.e. already (d_k, d_v),
                       so the adapter deliberately does NOT transpose. Seeing
                       this means upstream changed. Add the transpose INSIDE
                       fla_backend.py -- do not change the convention
                       downstream, Stage 4 indexes that matrix by key. Check
                       state_v_first on the gated kernel first; it flips this.
  "gated ... differs"-> decay parameterization. fla takes g = log(alpha) while
                       use_gate_in_kernel is False (the default).

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
    # Slurm opens the array's --output/--error files itself at task launch, so
    # logs/ must exist now; the mkdir inside the job script runs too late.
    mkdir -p logs results/parts
    sbatch "${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"}" scripts/slurm/sweep_array.sbatch
    echo
    echo "Watch:   squeue -u \$USER"
    echo "Merge:   python scripts/merge_results.py results/parts results/stage2.csv"
else
    step "ready -- not submitting (pass --submit to go ahead)"
    mkdir -p logs results/parts
    BUILT_ON="$( [ -r "$VENV/build_os" ] && cat "$VENV/build_os" || echo unknown )"
    cat <<EOF
    Submit:  sbatch ${CONSTRAINT:+--constraint=$CONSTRAINT }scripts/slurm/sweep_array.sbatch
    Watch:   squeue -u \$USER
    Merge:   python scripts/merge_results.py results/parts results/stage2.csv

    $GRID runs on partition '$SB_PART'.
    Change the partition or throttle by editing $SBATCH_FILE.

    The venv was built on '$BUILT_ON'. glibc is backward but not forward
    compatible, so an el8 build runs on every node and needs no --constraint,
    while an el9 build must use --constraint=el9. Tasks that would fail abort
    early naming the fix instead of dying in the dynamic loader.
EOF
fi

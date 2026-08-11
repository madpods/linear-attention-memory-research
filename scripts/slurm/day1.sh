#!/usr/bin/env bash
# One-shot first-day setup on the OSU CoE HPC.
#
# Run from the repo root on a LOGIN node (it allocates its own GPU node for the
# parts that need one):
#
#     bash scripts/slurm/day1.sh                 # set up and stop before submitting
#     bash scripts/slurm/day1.sh --submit        # ... and submit the 75-job array
#     bash scripts/slurm/day1.sh --partition dgx2
#     bash scripts/slurm/day1.sh --constraint el8   # build/run on EL8 instead
#     bash scripts/slurm/day1.sh --build-in-alloc   # build on a compute node
#
# The cluster mixes Rocky 8 and Rocky 9. glibc is backward but not forward
# compatible, so the venv's generation and the array's --constraint must agree;
# check_os_compat.sh enforces the direction that actually breaks.
#
# EL9 is the default. Node share favours EL8, but that is not the relevant
# quantity: only GPU nodes matter, the newer accelerators are on EL9, and EL8's
# glibc 2.28 is exactly the floor for PyPI torch's manylinux_2_28 wheels with no
# headroom as torch moves to manylinux_2_34. Feature names are el8 / el9.
#
# The venv's generation is fixed by where it is BUILT, and this script builds on
# the login node -- so if that node is EL8, it stops and offers the three ways
# out rather than producing a venv that cannot run where you asked.
#
# It surveys the cluster, builds the GPU environment, runs both test suites,
# checks the array size, and prints the submit command. It deliberately does
# NOT submit unless asked: queueing 75 GPU jobs before the parity gate has
# passed in THIS environment wastes an allocation. The adapter is verified as of
# 2026-08-11, but a rebuilt venv can pick up a different torch/triton/fla.
set -uo pipefail

PARTITION="${PARTITION:-dgxh}"
VENV="${VENV:-.venv-gpu}"
SETUP_TIME="${SETUP_TIME:-01:00:00}"
# Empty by default ON PURPOSE. A --constraint passed to sbatch REPLACES the
# script's own "#SBATCH --constraint=el9&a40" rather than adding to it, so
# defaulting this to "el9" made the printed submit command silently drop the a40
# pin -- losing both the sm_70+ guarantee and the homogeneous hardware that makes
# tokens_per_sec a comparable column. Set it only to deliberately override.
CONSTRAINT="${CONSTRAINT:-}"
BUILD_IN_ALLOC=0
SUBMIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --submit) SUBMIT=1 ;;
        --partition) PARTITION="$2"; shift ;;
        --partition=*) PARTITION="${1#*=}" ;;
        --constraint) CONSTRAINT="$2"; shift ;;
        --constraint=*) CONSTRAINT="${1#*=}" ;;
        --build-in-alloc) BUILD_IN_ALLOC=1 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
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
# The venv's OS generation is fixed by wherever it is built, and glibc is not
# forward compatible, so building on the wrong generation is not recoverable
# later -- it has to be rebuilt. Check before spending 10-20 minutes on it.
LOGIN_OS="$( [ -r /etc/os-release ] && ( . /etc/os-release && printf '%s%s' "${ID:-unknown}" "${VERSION_ID%%.*}" ) || echo unknown )"
# The generation the ARRAY will target: an explicit --constraint if given,
# otherwise whatever the sbatch file itself pins. Read from the file so the
# preflight keeps working without this script having to restate the default.
SBATCH_FILE_NAME=scripts/slurm/sweep_array.sbatch
SB_CONSTRAINT="$(sed -n 's/^#SBATCH --constraint=\([^ ]*\).*/\1/p' \
    "$SBATCH_FILE_NAME" | head -1)"
TARGET_CONSTRAINT="${CONSTRAINT:-$SB_CONSTRAINT}"
# First digit run only; see the same note in check_os_compat.sh.
WANT_GEN="$(printf '%s' "$TARGET_CONSTRAINT" | sed -n 's/^[^0-9]*\([0-9][0-9]*\).*/\1/p')"
LOGIN_GEN="$(printf '%s' "$LOGIN_OS" | sed -n 's/^[^0-9]*\([0-9][0-9]*\).*/\1/p')"
echo "    login node: $LOGIN_OS   array targets: ${TARGET_CONSTRAINT:-none}$([ -z "$CONSTRAINT" ] && [ -n "$SB_CONSTRAINT" ] && echo "  (from $SBATCH_FILE_NAME)")"

# Only the NEWER-built-than-target direction is fatal; glibc is backward
# compatible, so building on EL8 and running on EL9 is fine and is in fact the
# best configuration on this cluster (the login nodes are EL8, and the EL8 GPU
# nodes are ancient M60/GTX980 hardware that modern Triton cannot use anyway).
if [ -n "$WANT_GEN" ] && [ -n "$LOGIN_GEN" ] && [ "$LOGIN_GEN" -gt "$WANT_GEN" ] \
   && [ "$BUILD_IN_ALLOC" -eq 0 ]; then
    cat >&2 <<EOF

This login node is $LOGIN_OS but you asked for --constraint=$CONSTRAINT. A venv built
here carries EL${LOGIN_GEN}'s glibc, which cannot run on an EL${WANT_GEN} node -- and where it is
built is not fixable afterwards, only rebuildable. Three options:

  1. Log in to an EL${WANT_GEN} submit host and re-run. Preferred: pip needs outbound
     network, which login nodes reliably have and compute nodes may not.

  2. Build inside an allocation on an EL${WANT_GEN} node:
         bash scripts/slurm/day1.sh --constraint $CONSTRAINT --build-in-alloc

  3. Target EL${LOGIN_GEN} instead:
         bash scripts/slurm/day1.sh --constraint el${LOGIN_GEN}
EOF
    die "OS generation mismatch; pick one of the three above"
fi

if [ -n "$WANT_GEN" ] && [ -n "$LOGIN_GEN" ] && [ "$LOGIN_GEN" -lt "$WANT_GEN" ]; then
    ok "building on EL$LOGIN_GEN for EL$WANT_GEN nodes -- fine, glibc is backward compatible"
fi

if [ "$BUILD_IN_ALLOC" -eq 1 ]; then
    step "installing inside an allocation on $CONSTRAINT (~10-20 min)"
    warn "compute nodes may lack outbound network; if pip cannot reach PyPI,"
    warn "use an EL${WANT_GEN} login node instead (option 1 above)"
    srun --partition="$PARTITION" --time="$SETUP_TIME" --cpus-per-task=4 --mem=16G \
         "${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"}" \
         bash scripts/slurm/setup_env.sh --install 2>&1 | tee setup.log
else
    step "installing on the login node (~10-20 min; no GPU held)"
    bash scripts/slurm/setup_env.sh --install 2>&1 | tee setup.log
fi
INSTALL_RC=${PIPESTATUS[0]}
[ "$INSTALL_RC" -eq 0 ] || die "install phase failed; see setup.log"
ok "venv built on $(cat "$VENV/build_os" 2>/dev/null || echo '?'), CPU test suite passed"

step "verifying on a GPU node ($PARTITION)"
srun --partition="$PARTITION" --gres=gpu:1 --time="$SETUP_TIME" \
     --cpus-per-task=4 --mem=16G "${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"}" \
     bash scripts/slurm/setup_env.sh --verify 2>&1 | tee -a setup.log
SETUP_RC=${PIPESTATUS[0]}

if [ "$SETUP_RC" -ne 0 ]; then
    echo
    if grep -q "test_fla_parity" setup.log 2>/dev/null; then
        cat <<'EOF'
The fla parity gate failed. It passed 9/9 on an H100 with torch 2.13.0+cu130,
triton 3.7.1 (2026-08-11), so a failure here means this environment differs --
most likely a torch, triton or fla version picked up by a rebuilt venv. The
failing assertion names the convention at fault:

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
  "does not support   -> dtype. chunk_delta_rule asserts against fp32; the
   float32"              adapter casts to bf16 and back. If you see this, the
                         cast was removed or bypassed.
  negative_control    -> NOT a convention bug. REL_TOL has stopped separating
   failure               correct from wrong-convention, so the positive tests
                         prove nothing. Fix the tolerance derivation, and do
                         not trust a green suite until it passes.

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

    $GRID runs on partition '$SB_PART', constraint '${TARGET_CONSTRAINT:-none}'.
    Change the partition or throttle by editing $SBATCH_FILE.

    Do NOT add --constraint on the command line unless you mean to REPLACE the
    script's own '$SB_CONSTRAINT' -- sbatch overrides rather than combines, so
    passing just the OS generation would drop the GPU-class pin that keeps
    tokens_per_sec comparable and tasks off the sm_61 nodes.

    The venv was built on '$BUILT_ON'. glibc is backward but not forward
    compatible, so an el8 build runs on every node and needs no --constraint,
    while an el9 build must use --constraint=el9. Tasks that would fail abort
    early naming the fix instead of dying in the dynamic loader.
EOF
fi

#!/usr/bin/env bash
# Environment setup on the OSU CoE HPC, in two phases.
#
#   bash scripts/slurm/setup_env.sh --install   # login node: modules, venv, pip
#   bash scripts/slurm/setup_env.sh --verify    # GPU node:  sanity + test gates
#   bash scripts/slurm/setup_env.sh             # both (only valid on a GPU node)
#
# The phases are split because compute nodes on many clusters have no outbound
# network, so pip must run on the login node; and because holding a GPU through
# a multi-GB torch download wastes an allocation. scripts/slurm/day1.sh drives
# both in the right places.
#
# Module choices are recorded to $VENV/modules.env and replayed by
# sweep_array.sbatch. The CoE Lmod docs are explicit that modules used to build
# software into a python environment must also be loaded to run it -- Triton
# compiles kernels at runtime against the CUDA toolkit, so a batch job that
# skips the cuda module fails inside fla rather than at import.
set -euo pipefail

VENV="${VENV:-.venv-gpu}"
DO_INSTALL=1
DO_VERIFY=1

case "${1:-}" in
    --install) DO_VERIFY=0 ;;
    --verify)  DO_INSTALL=0 ;;
    "")        ;;
    *) echo "usage: $0 [--install|--verify]" >&2; exit 2 ;;
esac

# Every non-zero status in here is deliberately swallowed. This function runs
# under `set -e`, and module systems return failure for ordinary things (a
# `module -t avail` for a name the cluster does not carry, a `grep` that finds
# no (D) marker). An unguarded one aborts the script with no message at all --
# which is exactly what it did before, on the happy path.
REPLAYED_MODULES=0
LOADED=()

load_modules() {
    command -v srun >/dev/null 2>&1 || module load slurm 2>/dev/null || true
    if [ -f "$VENV/modules.env" ]; then
        # Replaying a previous run's choices. Do NOT re-derive them, and do not
        # let the install phase overwrite the record with what it did not pick.
        while read -r m; do
            [ -n "$m" ] || continue
            echo "    loading $m (recorded)"
            module load "$m" || echo "    WARN: could not load recorded module $m"
        done < "$VENV/modules.env"
        REPLAYED_MODULES=1
        return 0
    fi
    # Lmod marks its default with (D), which is what a bare `module load python`
    # picks. Prefer that; fall back to the highest version.
    pick_module() {
        local avail default
        avail="$(module -t avail "$1" 2>&1 | tr -d '\r' | grep -E "^$1/" || true)"
        [ -n "$avail" ] || return 0
        default="$(printf '%s\n' "$avail" | grep -F '(D)' | head -1 | sed 's/(.*)//' || true)"
        if [ -n "$default" ]; then printf '%s\n' "$default"
        else printf '%s\n' "$avail" | sed 's/(.*)//' | sort -V | tail -1; fi
    }
    MODULE_PYTHON="${MODULE_PYTHON:-$(pick_module python || true)}"
    MODULE_CUDA="${MODULE_CUDA:-$(pick_module cuda || true)}"
    for m in "$MODULE_PYTHON" "$MODULE_CUDA"; do
        if [ -n "$m" ]; then
            echo "    loading $m"
            if module load "$m"; then LOADED+=("$m")
            else echo "    WARN: could not load $m"; fi
        fi
    done
    if [ "${#LOADED[@]}" -eq 0 ]; then
        echo "    WARN: no modules loaded; using PATH as-is"
        echo "    If python/cuda are module-provided here, set them explicitly:"
        echo "        export MODULE_PYTHON=python/<ver> MODULE_CUDA=cuda/<ver>"
    fi
    return 0
}

# ---------------------------------------------------------------- install

if [ "$DO_INSTALL" -eq 1 ]; then
    echo "==> modules"
    load_modules
    python3 --version || { echo "FATAL: no python3 on PATH"; exit 1; }

    # pyproject requires >=3.10; catching it here beats a confusing pip error.
    python3 - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    print(f"FATAL: python {sys.version.split()[0]} is too old; need >= 3.10.")
    print("Pick a newer module: module -t avail python")
    sys.exit(1)
PY

    echo "==> venv: $VENV"
    python3 -m venv "$VENV"
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip

    # Only write the record if this run actually chose the modules. On a repeat
    # --install the venv already carries a modules.env, load_modules replayed it
    # and LOADED is empty -- writing here would blank the file, and a blank
    # modules.env makes sweep_array.sbatch load nothing while reporting nothing
    # wrong. Triton would then fail deep inside fla on first kernel compile.
    if [ "$REPLAYED_MODULES" -eq 1 ]; then
        echo "    kept existing $VENV/modules.env ($(tr '\n' ' ' < "$VENV/modules.env"))"
    elif [ "${#LOADED[@]}" -gt 0 ]; then
        printf '%s\n' "${LOADED[@]}" > "$VENV/modules.env"
        echo "    recorded modules -> $VENV/modules.env"
    else
        : > "$VENV/modules.env"
        echo "    no modules to record; wrote empty $VENV/modules.env"
    fi

    echo "==> torch (CUDA build)"
    pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

    echo "==> project + gpu extras"
    pip install -e ".[gpu,dev]"

    echo "==> CPU-only correctness suite (no GPU needed)"
    python -m pytest tests/ -q -x --ignore=tests/test_fla_parity.py
fi

# ----------------------------------------------------------------- verify

if [ "$DO_VERIFY" -eq 1 ]; then
    echo "==> modules"
    load_modules
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"

    echo "==> device sanity"
    python - <<'PY'
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: no CUDA device. Run the verify phase on a GPU node.")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import triton
    print("triton", getattr(triton, "__version__", "unknown"))
except Exception as exc:
    print("triton import FAILED:", exc)
PY

    # The parity tests skip themselves when fla is unimportable. On a GPU node
    # that is a FAILURE, not a skip: it would let pytest exit 0 with the gate
    # never having run, and every downstream GPU number would be unvalidated.
    echo "==> gate integrity (the parity tests must be able to run at all)"
    python - <<'PY'
import sys
from lamr.layers.fla_backend import fla_available, unavailable_reason
if not fla_available():
    print("FATAL: the fla parity gate cannot run:", unavailable_reason())
    print()
    print("If fla is installed but not importable, the import path in")
    print("src/lamr/layers/fla_backend.py is wrong. Find the real one:")
    print("    python -c \"import fla, pkgutil; print([m.name for m in pkgutil.iter_modules(fla.ops.__path__)])\"")
    sys.exit(1)
print("fla importable and CUDA present -- gate will actually execute")
PY

    echo "==> fla parity gate"
    echo "    If this fails, fix src/lamr/layers/fla_backend.py -- it was written"
    echo "    without a GPU available and its conventions are unverified guesses."
    python -m pytest tests/test_fla_parity.py -q -rs

    echo
    echo "==> done. Submit the sweep with:"
    echo "    python scripts/stage2_sweep.py --preset full --count   # expect 75"
    echo "    sbatch scripts/slurm/sweep_array.sbatch"
fi

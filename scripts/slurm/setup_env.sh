#!/usr/bin/env bash
# One-time environment setup on the OSU CoE HPC.
#
#   bash scripts/slurm/setup_env.sh
#
# Run this on a *GPU node* (srun --partition=dgxh --gres=gpu:1 --pty bash), not
# the login node: the last step runs the fla parity test, which needs a real
# device and is the gate for trusting any GPU result this project produces.
#
# Module choices are recorded to $VENV/modules.env and replayed by
# sweep_array.sbatch. The CoE Lmod docs are explicit that modules used to build
# software in a python environment must also be loaded to run it -- Triton
# compiles kernels at runtime against the CUDA toolkit, so a batch job that
# skips the cuda module fails inside fla rather than at import.
set -euo pipefail

VENV="${VENV:-.venv-gpu}"

echo "==> modules"
# Slurm is normally auto-loaded on submit nodes; if missing, the CoE docs say
# to log into TEACH and click "Reset Unix Config Files".
command -v srun >/dev/null 2>&1 || module load slurm || true

# Lmod marks its default with (D), which is what a bare `module load python`
# picks. Prefer that; fall back to the highest version.
pick_module() {
    local avail
    avail="$(module -t avail "$1" 2>&1 | tr -d '\r' | grep -E "^$1/" || true)"
    local default
    default="$(printf '%s\n' "$avail" | grep -F '(D)' | head -1 | sed 's/(.*)//')"
    if [ -n "$default" ]; then
        printf '%s\n' "$default"
    else
        printf '%s\n' "$avail" | sed 's/(.*)//' | sort -V | tail -1
    fi
}

MODULE_PYTHON="${MODULE_PYTHON:-$(pick_module python)}"
MODULE_CUDA="${MODULE_CUDA:-$(pick_module cuda)}"

LOADED=()
for m in "$MODULE_PYTHON" "$MODULE_CUDA"; do
    if [ -n "$m" ]; then
        echo "    loading $m"
        if module load "$m"; then
            LOADED+=("$m")
        else
            echo "    WARN: could not load $m"
        fi
    fi
done
[ ${#LOADED[@]} -eq 0 ] && echo "    WARN: no modules loaded; using whatever is on PATH"

python3 --version || { echo "FATAL: no python3 on PATH"; exit 1; }
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "    note: nvcc not on PATH"

echo "==> venv: $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# Replayed by sweep_array.sbatch so batch jobs run under the same modules the
# environment was built against.
printf '%s\n' "${LOADED[@]}" > "$VENV/modules.env"
echo "    recorded modules -> $VENV/modules.env"

echo "==> torch (CUDA build)"
# Match the wheel to the loaded CUDA major version where possible; the cu124
# wheel bundles its own runtime and works against newer drivers.
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "==> project + gpu extras"
pip install -e ".[gpu,dev]"

echo "==> sanity"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
try:
    import triton
    print("triton", getattr(triton, "__version__", "unknown"))
except Exception as exc:
    print("triton import FAILED:", exc)
try:
    import fla
    print("fla", getattr(fla, "__version__", "unknown"))
except Exception as exc:
    print("fla import FAILED:", exc)
PY

echo "==> CPU-only correctness suite (must pass before trusting the GPU path)"
python -m pytest tests/ -q -x --ignore=tests/test_fla_parity.py

echo "==> fla parity gate"
echo "    If this fails, fix src/lamr/layers/fla_backend.py -- it was written"
echo "    without a GPU available and its conventions are unverified guesses."
python -m pytest tests/test_fla_parity.py -q -rs

echo
echo "==> done. Submit the sweep with:"
echo "    python scripts/stage2_sweep.py --preset full --count   # expect 75"
echo "    sbatch scripts/slurm/sweep_array.sbatch"

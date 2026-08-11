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

# EL8 vs EL9. The cluster is a mix (roughly 60/40 Rocky 8 / Rocky 9), which
# matters because glibc is not forward compatible: a venv whose interpreter came
# from an EL9 python module fails on an EL8 node with "GLIBC_2.34 not found",
# and array tasks land wherever the scheduler puts them. Recorded at build time
# and checked at run time so that mismatch reports itself instead of surfacing
# as a cryptic loader error 40 tasks into a sweep.
os_generation() {
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        printf '%s%s\n' "${ID:-unknown}" "${VERSION_ID%%.*}"
    else
        echo unknown
    fi
}

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

load_one() {
    local m="$1"
    [ -n "$m" ] || return 0
    echo "    loading $m"
    if module load "$m"; then LOADED+=("$m")
    else echo "    WARN: could not load $m"; fi
    return 0
}

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
    load_one "${MODULE_PYTHON:-$(pick_module python || true)}"
    if [ "${#LOADED[@]}" -eq 0 ]; then
        echo "    WARN: no python module loaded; using PATH as-is"
        echo "    If python is module-provided here, set it explicitly:"
        echo "        export MODULE_PYTHON=python/<ver>"
    fi
    # A host compiler, because Triton does not only emit PTX: it compiles a
    # small C launcher stub at runtime, with the system compiler. Rocky 8 ships
    # gcc 8.5, old enough to be a plausible source of a first-call failure that
    # would look like a CUDA problem. The Lmod default here is gcc/12.5, which
    # is also inside the host-compiler range both CUDA 12.x and 13.x accept.
    # Set MODULE_GCC=none to skip, or to a specific version to pin.
    if [ "${MODULE_GCC:-}" != "none" ]; then
        load_one "${MODULE_GCC:-$(pick_module gcc || true)}"
    fi
    return 0
}

# The CUDA module is chosen AFTER torch is installed, from what torch itself
# reports, rather than from Lmod's (D) default. This cluster carries CUDA 11.x
# through 13.x, so the default is not necessarily the major torch was built
# against -- and a toolkit newer than torch's bundled runtime is exactly where
# Triton's first kernel compile goes wrong. torch ships its own CUDA runtime;
# the module is here for Triton's ptxas, so matching the major is what matters.
load_cuda_module() {
    local want="$1" avail match
    if [ -n "${MODULE_CUDA:-}" ]; then load_one "$MODULE_CUDA"; return 0; fi
    if [ -z "$want" ]; then
        echo "    WARN: torch reports no CUDA version; skipping the cuda module"
        return 0
    fi
    avail="$(module -t avail cuda 2>&1 | tr -d '\r' | grep -E '^cuda/' | sed 's/(.*)//' || true)"
    if [ -z "$avail" ]; then
        echo "    WARN: no cuda modules found; relying on torch's bundled runtime"
        return 0
    fi
    match="$(printf '%s\n' "$avail" | grep -E "^cuda/${want}\." | sort -V | tail -1 || true)"
    if [ -n "$match" ]; then
        echo "    torch wants CUDA ${want}.x -> $match"
        load_one "$match"
    else
        echo "    WARN: no cuda/${want}.x module (torch was built against it)."
        printf '    available: %s\n' "$(printf '%s\n' "$avail" | tr '\n' ' ')"
        echo "    Continuing on torch's bundled runtime. If Triton fails to"
        echo "    compile, set MODULE_CUDA explicitly and rebuild."
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

    # torch's linux wheels on PyPI are CUDA-enabled and bundle their own runtime,
    # so no --index-url is needed and the version stays in step with the CPU
    # box's 2.13.x -- which matters because fla tracks recent torch closely.
    # Set TORCH_INDEX to pin a specific CUDA build (e.g. .../whl/cu126).
    echo "==> torch (CUDA build)"
    if [ -n "${TORCH_INDEX:-}" ]; then
        echo "    index: $TORCH_INDEX"
        pip install torch --index-url "$TORCH_INDEX"
    else
        pip install torch
    fi

    # Ask torch which CUDA it was built against rather than guessing. Done here,
    # after the install, because that answer is what selects the module.
    echo "==> cuda module (matched to the installed torch)"
    if [ "$REPLAYED_MODULES" -eq 1 ]; then
        echo "    replayed from $VENV/modules.env; not re-deriving"
    else
        TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
        echo "    torch $(python -c 'import torch; print(torch.__version__)') built against CUDA ${TORCH_CUDA:-none}"
        load_cuda_module "${TORCH_CUDA%%.*}"
    fi

    echo "==> project + gpu extras"
    pip install -e ".[gpu,dev]"

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

    # Record the OS generation this venv was built on. See os_generation().
    os_generation > "$VENV/build_os"
    echo "    built on $(cat "$VENV/build_os") -> $VENV/build_os"
    if [ "$(cat "$VENV/build_os")" = "rocky9" ]; then
        echo "    NOTE: this cluster mixes Rocky 8 and 9, and glibc is not"
        echo "    forward compatible. A venv built here will NOT run on an EL8"
        echo "    node. Either constrain the array to EL9 nodes, or build on an"
        echo "    EL8 login node so it runs on both. sweep_array.sbatch checks"
        echo "    this per task and fails fast rather than dying in the loader."
    fi

    echo "==> CPU-only correctness suite (no GPU needed)"
    python -m pytest tests/ -q -x --ignore=tests/test_fla_parity.py
fi

# ----------------------------------------------------------------- verify

if [ "$DO_VERIFY" -eq 1 ]; then
    echo "==> modules"
    load_modules
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"

    echo "==> os generation"
    if [ -r "$VENV/build_os" ]; then
        bash "$(dirname "$0")/check_os_compat.sh" "$(cat "$VENV/build_os")" \
            | sed 's/^/    /' || exit 1
    else
        echo "    WARN: no $VENV/build_os; rebuild with setup_env.sh --install"
    fi

    echo "==> device sanity"
    python - <<'PY'
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torch built against CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    print("FATAL: no CUDA device. Run the verify phase on a GPU node.")
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
print("device:", torch.cuda.get_device_name(0), cap)
# The kernels are Triton; an architecture torch's build does not target will
# compile but not run. sm_90 = H100/H200, sm_80/86 = A40/A100, sm_70 = V100.
arches = torch.cuda.get_arch_list()
print("arch list:", ", ".join(arches))
if f"sm_{cap[0]}{cap[1]}" not in arches:
    print(f"WARN: sm_{cap[0]}{cap[1]} is not in this torch build's arch list.")
    print("      Kernels may fall back to JIT or fail outright.")
try:
    import triton
    print("triton", getattr(triton, "__version__", "unknown"))
except Exception as exc:
    print("triton import FAILED:", exc)
    sys.exit(1)
PY

    # Importing triton proves nothing about the failure this whole module-
    # recording dance exists to prevent: Triton compiles kernels on FIRST CALL,
    # against the CUDA toolkit. So actually compile and run one, here, where the
    # error is legible -- rather than discovering it deep inside fla, mid-sweep.
    echo "==> triton kernel compile (the real toolkit check)"
    python - <<'PY'
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def _add1(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(y_ptr + off, tl.load(x_ptr + off, mask=mask) + 1.0, mask=mask)

try:
    x = torch.zeros(128, device="cuda")
    y = torch.empty_like(x)
    _add1[(1,)](x, y, x.numel(), BLOCK=128)
    torch.cuda.synchronize()
    assert torch.allclose(y, torch.ones_like(y)), y
except Exception as exc:
    print("FATAL: Triton could not compile/run a trivial kernel:")
    print(f"  {type(exc).__name__}: {exc}")
    print()
    print("This is the CUDA-toolkit mismatch the module recording guards")
    print("against. torch was built against CUDA", torch.version.cuda,
          "-- load a matching cuda/<major>.x module and rebuild:")
    print("    export MODULE_CUDA=cuda/<version>")
    print("    rm -rf", sys.prefix, "&& bash scripts/slurm/setup_env.sh --install")
    sys.exit(1)
print("triton compiled and ran a kernel on", torch.cuda.get_device_name(0))
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

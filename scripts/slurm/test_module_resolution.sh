#!/usr/bin/env bash
# Test setup_env.sh's module resolution against a stubbed Lmod. Runs anywhere,
# including the CUDA-less dev workstation:
#
#     bash scripts/slurm/test_module_resolution.sh
#
# Why this exists as a test rather than a careful reading: every failure in this
# code path presents as an exit with no message. `load_modules` runs under
# `set -e`, and module systems return non-zero for entirely ordinary things --
# `module -t avail` for a name the cluster does not carry, a `grep` for the (D)
# default marker on a cluster whose Lmod does not emit one. An unguarded one of
# those aborts setup_env.sh mid-sentence. Three such bugs were live here, all on
# paths a real cluster hits first.
#
# It tests the real function, spliced out of setup_env.sh, so it cannot drift
# from the thing it validates.
set -uo pipefail

SETUP="${1:-$(dirname "$0")/setup_env.sh}"
[ -f "$SETUP" ] || { echo "cannot find setup_env.sh at $SETUP" >&2; exit 2; }
SETUP="$(cd "$(dirname "$SETUP")" && pwd)/$(basename "$SETUP")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILURES=0

# One case: stub `module -t avail` output, an optional pre-existing modules.env,
# and the modules.env content we require afterwards.
run_case() {
    local name="$1" avail="$2" existing="$3" expect="$4"
    local dir="$WORK/case"
    rm -rf "$dir"; mkdir -p "$dir/.venv-gpu"
    [ -n "$existing" ] && printf '%s\n' "$existing" > "$dir/.venv-gpu/modules.env"

    {
        echo 'set -euo pipefail'
        echo 'VENV=.venv-gpu'
        # Stub Lmod. `module -t avail <name>` is the only query under test.
        cat <<'STUB'
module() {
    case "${1:-}" in
        -t) printf '%s\n' "${AVAIL_OUT:-}" ;;
        load) echo "  [stub loaded ${2:-}]" ;;
        *) : ;;
    esac
}
STUB
        # Splice the real helpers (everything from REPLAYED_MODULES=0 up to the
        # install section) plus the real record-writing block, so this test
        # cannot drift from the code it validates.
        sed -n '/^REPLAYED_MODULES=0/,/^# -*  *install/p' "$SETUP" | sed '$d'
        echo 'load_modules'
        sed -n '/^    # Only write the record if this run actually chose/,/^    fi$/p' "$SETUP"
        echo 'echo "REACHED_END"'
    } > "$dir/harness.sh"

    local out rc
    out="$(cd "$dir" && AVAIL_OUT="$avail" bash harness.sh 2>&1)"; rc=$?

    local got=""
    [ -f "$dir/.venv-gpu/modules.env" ] &&
        got="$(tr '\n' ' ' < "$dir/.venv-gpu/modules.env" | sed 's/ *$//')"

    printf '%-28s' "$name"
    if [ "$rc" -ne 0 ] || ! grep -q REACHED_END <<<"$out"; then
        # The historical bug: set -e aborts and prints nothing.
        echo "FAIL (exited $rc before finishing -- set -e abort?)"
        sed 's/^/      /' <<<"$out"
        FAILURES=$((FAILURES + 1))
    elif [ "$got" != "$expect" ]; then
        echo "FAIL (modules.env = [$got], expected [$expect])"
        sed 's/^/      /' <<<"$out"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok   modules.env=[$got]"
    fi
}

echo "testing $SETUP"
echo

# Normal Lmod: terse output with the default marked (D). Must pick the default,
# not the highest version.
run_case "picks the (D) default" \
    'python/3.10
python/3.11(D)
python/3.12' '' 'python/3.11'

# Some Lmod configurations emit no (D) at all. Falls back to highest version.
# This case used to die at the `default=` assignment: grep found nothing,
# pipefail propagated it, and the assignment's status tripped set -e.
run_case "no (D) marker" \
    'python/3.10
python/3.11' '' 'python/3.11'

# A module name the cluster does not carry. Must warn, not abort.
run_case "no such module" '' '' ''

# Repeat --install over an existing venv. Must replay and PRESERVE the record:
# blanking it makes sweep_array.sbatch load nothing while reporting nothing
# wrong, and Triton then fails on first kernel compile deep inside fla.
run_case "preserves prior record" \
    'python/3.12(D)' \
    'python/3.9
cuda/12.1' \
    'python/3.9 cuda/12.1'

# A host compiler is loaded alongside python: Triton compiles a C launcher stub
# at runtime, and Rocky 8's system gcc is 8.5. Both must land in the record so
# sweep_array.sbatch replays them.
run_case "loads gcc with python" \
    'python/3.13(D)
gcc/12.5(D)
gcc/15.2' '' 'python/3.13 gcc/12.5'

# MODULE_GCC=none opts out without disturbing the python pick.
MODULE_GCC=none run_case "MODULE_GCC=none skips gcc" \
    'python/3.13(D)
gcc/12.5(D)' '' 'python/3.13'

# --- load_cuda_module -------------------------------------------------------
#
# The cluster carries CUDA 11.x through 13.x, so Lmod's (D) default is not
# necessarily the major the installed torch was built against. A toolkit newer
# than torch's bundled runtime is where Triton's first kernel compile goes
# wrong, so selection follows torch, not the default.

echo
echo "load_cuda_module (cluster carries cuda 11.x-13.x)"

# $1 name, $2 stubbed `module -t avail cuda` output, $3 torch's CUDA major,
# $4 MODULE_CUDA override, $5 expected module recorded (empty = none loaded)
run_cuda_case() {
    local name="$1" avail="$2" want="$3" override="$4" expect="$5"
    local dir="$WORK/cuda"
    rm -rf "$dir"; mkdir -p "$dir/.venv-gpu"

    {
        echo 'set -euo pipefail'
        echo 'VENV=.venv-gpu'
        cat <<'STUB'
module() {
    case "${1:-}" in
        -t) printf '%s\n' "${AVAIL_OUT:-}" ;;
        load) echo "  [stub loaded ${2:-}]" ;;
        *) : ;;
    esac
}
STUB
        sed -n '/^REPLAYED_MODULES=0/,/^# -*  *install/p' "$SETUP" | sed '$d'
        echo 'load_cuda_module "$WANT"'
        echo 'printf "%s\n" "${LOADED[@]:-}" | grep -E "^cuda/" || true'
        echo 'echo "REACHED_END"'
    } > "$dir/harness.sh"

    local out rc got
    out="$(cd "$dir" && AVAIL_OUT="$avail" WANT="$want" MODULE_CUDA="$override" \
        bash harness.sh 2>&1)"; rc=$?
    got="$(grep -E '^cuda/' <<<"$out" | tail -1 || true)"

    printf '%-28s' "$name"
    if [ "$rc" -ne 0 ] || ! grep -q REACHED_END <<<"$out"; then
        echo "FAIL (exited $rc before finishing -- set -e abort?)"
        sed 's/^/      /' <<<"$out"; FAILURES=$((FAILURES + 1))
    elif [ "$got" != "$expect" ]; then
        echo "FAIL (chose [$got], expected [$expect])"
        sed 's/^/      /' <<<"$out"; FAILURES=$((FAILURES + 1))
    else
        echo "ok   chose=[${got:-none}]"
    fi
}

ALL_CUDA='cuda/11.8
cuda/12.1
cuda/12.4
cuda/13.0(D)'

# torch built against 12.x must NOT take the 13.0 default.
run_cuda_case "torch cu12 -> newest 12.x" "$ALL_CUDA" 12 '' 'cuda/12.4'
# ... and a cu13 torch should take 13.0 even though it is also the default.
run_cuda_case "torch cu13 -> 13.0"        "$ALL_CUDA" 13 '' 'cuda/13.0'
# An older torch still resolves to its own major rather than the default.
run_cuda_case "torch cu11 -> 11.8"        "$ALL_CUDA" 11 '' 'cuda/11.8'
# No module for torch's major: warn and continue on the bundled runtime.
run_cuda_case "no matching major"  'cuda/11.8
cuda/12.4' 13 '' ''
# Explicit override always wins.
run_cuda_case "MODULE_CUDA override"      "$ALL_CUDA" 12 'cuda/11.8' 'cuda/11.8'
# torch reports no CUDA at all (a CPU wheel slipped in): must not abort.
run_cuda_case "torch has no cuda"         "$ALL_CUDA" '' '' ''

# The real module list from the OSU CoE HPC (modulefiles-8, captured
# 2026-08-11). Note cuda/13.0 carries (D) while 13.1/13.2/13.3 also exist --
# so Lmod's default is neither the newest nor, for a cu12 torch, the right
# major. Both facts are why selection follows torch instead of the default.
REAL_CUDA='cuda/9.2
cuda/10.1
cuda/10.2
cuda/11.0
cuda/11.1
cuda/11.2
cuda/11.3
cuda/11.4
cuda/11.5
cuda/11.6
cuda/11.7
cuda/11.8
cuda/12.0
cuda/12.1
cuda/12.2
cuda/12.3
cuda/12.4
cuda/12.5
cuda/12.6
cuda/12.8
cuda/12.9
cuda/13.0(D)
cuda/13.1
cuda/13.2
cuda/13.3'

# sort -V must order 12.9 above 12.10-style strings and above 12.1; a plain
# lexical sort would pick cuda/12.6 here, and cuda/11.8 over cuda/11.10.
run_cuda_case "real list, cu12 -> 12.9"  "$REAL_CUDA" 12 '' 'cuda/12.9'
run_cuda_case "real list, cu13 -> 13.3"  "$REAL_CUDA" 13 '' 'cuda/13.3'
run_cuda_case "real list, cu11 -> 11.8"  "$REAL_CUDA" 11 '' 'cuda/11.8'

# --- check_os_compat.sh -----------------------------------------------------
#
# The rule is DIRECTIONAL, and getting it backwards is expensive in both
# directions: too strict wastes ~40% of the cluster's nodes on a venv that
# would have run fine, too loose lets tasks die in the dynamic loader.

echo
echo "check_os_compat (glibc is backward but not forward compatible)"

COMPAT="$(dirname "$SETUP")/check_os_compat.sh"

run_os_case() {
    local name="$1" build="$2" run="$3" want_rc="$4"
    local rc
    bash "$COMPAT" "$build" "$run" >/dev/null 2>&1; rc=$?
    printf '%-28s' "$name"
    if [ "$rc" -ne "$want_rc" ]; then
        echo "FAIL (build=$build run=$run gave exit $rc, wanted $want_rc)"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok   build=$build run=$run exit=$rc"
    fi
}

run_os_case "el8 venv on el8"   rocky8  rocky8  0
# The case that must NOT be blocked: an EL8 build is usable everywhere, which
# is why building on the EL8 login node needs no --constraint at all.
run_os_case "el8 venv on el9"   rocky8  rocky9  0
# The case that must be blocked: newer glibc cannot run on older.
run_os_case "el9 venv on el8"   rocky9  rocky8  1
run_os_case "el9 venv on el9"   rocky9  rocky9  0
# Unknown either side: warn and continue rather than blocking a whole sweep on
# an unparseable /etc/os-release.
run_os_case "unknown build"     unknown rocky8  0
run_os_case "unknown run"       rocky8  unknown 0
run_os_case "empty build"       ''      rocky8  0

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "all cases passed"
else
    echo "$FAILURES case(s) failed"
    exit 1
fi

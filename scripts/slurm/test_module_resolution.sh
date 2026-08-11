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
        # Splice the real function and the real record-writing block.
        sed -n '/^REPLAYED_MODULES=0/,/^}/p' "$SETUP"
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

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "all cases passed"
else
    echo "$FAILURES case(s) failed"
    exit 1
fi

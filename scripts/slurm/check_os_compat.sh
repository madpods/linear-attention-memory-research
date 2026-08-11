#!/usr/bin/env bash
# Is a venv built on one EL generation usable on this node?
#
#     bash scripts/slurm/check_os_compat.sh <build_os> [run_os]
#
# Exits 0 if usable (silently, or with a warning), 1 if not. Both callers --
# setup_env.sh --verify and sweep_array.sbatch -- go through here so the rule
# lives in one place and can be tested without a cluster.
#
# The rule is DIRECTIONAL, which is the whole reason this is not an equality
# check. glibc is backward compatible but not forward compatible:
#
#   built on el8, running on el9  -> fine. Binaries linked against the older
#                                    glibc resolve against the newer one.
#   built on el9, running on el8  -> fatal. "GLIBC_2.34 not found" from the
#                                    dynamic loader, because EL8 ships 2.28.
#
# So building on EL8 is strictly the better choice on this mixed cluster: the
# venv is then usable on every node and the array needs no --constraint. An
# EL9-built venv must be pinned with --constraint=el9.
#
# The residual risk on the el8 -> el9 path is not glibc but the module tree:
# modules live in per-generation paths (/usr/local/apps/modulefiles-8) with
# _el8/_el9 suffixed variants, so a name recorded in modules.env need not
# resolve on the other generation. That surfaces as a module load warning and,
# if the venv's base interpreter is genuinely absent, a python startup failure
# -- both legible, unlike the loader error this guards.
set -uo pipefail

BUILD_OS="${1:-}"
RUN_OS="${2:-}"

if [ -z "$RUN_OS" ]; then
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        RUN_OS="$(. /etc/os-release && printf '%s%s' "${ID:-unknown}" "${VERSION_ID%%.*}")"
    else
        RUN_OS=unknown
    fi
fi

if [ -z "$BUILD_OS" ] || [ "$BUILD_OS" = unknown ] || [ "$RUN_OS" = unknown ]; then
    echo "os: build=${BUILD_OS:-?} run=$RUN_OS -- cannot compare, continuing"
    exit 0
fi

if [ "$BUILD_OS" = "$RUN_OS" ]; then
    echo "os: $RUN_OS (matches build)"
    exit 0
fi

# Trailing digits are the EL generation: rocky8 -> 8.
build_major="$(printf '%s' "$BUILD_OS" | sed 's/[^0-9]*//g')"
run_major="$(printf '%s' "$RUN_OS" | sed 's/[^0-9]*//g')"

if [ -z "$build_major" ] || [ -z "$run_major" ]; then
    echo "os: build=$BUILD_OS run=$RUN_OS -- unparseable generation, continuing"
    exit 0
fi

if [ "$run_major" -ge "$build_major" ]; then
    echo "os: built on $BUILD_OS, running on $RUN_OS -- newer, glibc is"
    echo "    backward compatible so this is expected to work."
    echo "    If python or a recorded module is missing here, the per-generation"
    echo "    module tree is the likely cause; pin the array with"
    echo "    --constraint=el${build_major} and resubmit."
    exit 0
fi

cat >&2 <<EOF
FATAL: venv was built on $BUILD_OS but this node is $RUN_OS.

glibc is not forward compatible: binaries linked against EL${build_major}'s glibc
cannot run on EL${run_major}. This task would fail in the dynamic loader with
something like "GLIBC_2.34 not found". Two fixes:

  1. Pin the array to the generation you built on:
         sbatch --constraint=el${build_major} scripts/slurm/sweep_array.sbatch

  2. Or rebuild on the OLDER generation, which then runs everywhere:
         srun -p preempt --constraint=el${run_major} --pty bash
         rm -rf .venv-gpu && bash scripts/slurm/setup_env.sh --install

Feature names on this cluster are el8 / el9; survey_cluster.sh lists what
each node actually advertises under "node features".
EOF
exit 1

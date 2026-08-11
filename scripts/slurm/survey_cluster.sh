#!/usr/bin/env bash
# First-login survey of the Oregon State HPC. Run on the LOGIN node:
#
#     bash scripts/slurm/survey_cluster.sh 2>&1 | tee cluster_survey.txt
#
# Answers everything scripts/slurm/sweep_array.sbatch needs and cannot guess:
# partition names, which partitions actually expose GPUs, what account to
# charge, time limits, and the module names for python/cuda.
#
# Nothing here submits a job, allocates a node, or writes outside the current
# directory -- it is all read-only queries.
set -uo pipefail   # deliberately not -e: a missing command should not abort

section() { printf '\n\n========== %s ==========\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "identity"
echo "user:    $(whoami)"
echo "host:    $(hostname -f 2>/dev/null || hostname)"
echo "home:    $HOME"
echo "groups:  $(id -Gn 2>/dev/null)"

section "partitions (name, GPUs, memory, timelimit, nodes, state)"
# %P partition  %G generic resources (gpu:type:count)  %m memory  %l timelimit
if have sinfo; then
    sinfo -o "%20P %20G %10m %12l %6D %10t" | sort -u
else
    echo "sinfo not found -- is this a Slurm cluster / is the module loaded?"
fi

section "GPU nodes only"
if have sinfo; then
    sinfo -N -o "%20N %20P %30G %10m %8t" | grep -i gpu || echo "(no nodes advertise a gpu GRES)"
fi

section "node features -- for --constraint (OS generation lives here)"
# The cluster mixes Rocky 8 and Rocky 9, and glibc is backward but not forward
# compatible: a venv built on EL9 will not run on an EL8 node, though an EL8
# build runs on both. The feature names here are el8 / el9, e.g.
#     srun -p preempt --constraint=el8 --pty bash
#     sbatch --constraint=el9 scripts/slurm/sweep_array.sbatch
# What is worth reading below is which GPU partitions carry which generation --
# constraining to el9 on a partition whose GPU nodes are all el8 just pends
# forever.
if have sinfo; then
    # %f needs room: the dgxh rows run to ~50 chars and a narrow field silently
    # truncates the el8/el9 suffix off the end -- the one token being looked up.
    echo "--- per node (name, partition, features) ---"
    sinfo -N -o "%20N %20P %70f" | sort -u | sed 's/ *$//'
    echo
    echo "--- deduped feature list ---"
    sinfo -h -o "%f" | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$' | sort -u
    echo
    echo "--- features on GPU nodes only ---"
    sinfo -N -h -o "%N %G %f" | grep -i gpu | awk '{print $NF}' \
        | tr ',' '\n' | sort -u | grep -v '^$' || echo "(none reported)"
else
    echo "sinfo not found"
fi
echo
echo "This login node reports:"
if [ -r /etc/os-release ]; then
    ( . /etc/os-release && echo "    $PRETTY_NAME  (ID=$ID VERSION_ID=$VERSION_ID)" )
else
    echo "    /etc/os-release unreadable"
fi
echo "    glibc: $(ldd --version 2>/dev/null | head -1)"
echo
echo "Build the venv on the OLDER generation (EL8) and it runs on both;"
echo "build on EL9 and the array must be constrained to EL9 nodes."

section "accounts this user may charge (--account)"
if have sacctmgr; then
    sacctmgr -n show associations user="$(whoami)" \
        format=Cluster,Account,Partition,QOS%30 2>/dev/null \
        || echo "(no associations returned; cluster may not require --account)"
else
    echo "sacctmgr not found; accounting may be disabled -- omit --account"
fi

section "QOS limits"
have sacctmgr && sacctmgr -n show qos format=Name,MaxWall,MaxTRESPU%30,Priority 2>/dev/null | head -20

section "default job limits"
have scontrol && scontrol show config 2>/dev/null | grep -Ei "MaxArraySize|MaxJobCount|DefMemPerNode" || true

# Lmod: -t gives one module per line (plain `avail` is columnar and mangles
# under tr). A "(D)" suffix marks the version a bare `module load <name>` picks.
list_modules() {
    if have module; then
        module -t avail "$1" 2>&1 | tr -d '\r' | grep -E "^$1/" | sort -V || echo "(none)"
    else
        echo "no 'module' command"
    fi
}

section "modules: python  ((D) = default)"
list_modules python
have python3 && echo "on PATH already: $(python3 --version 2>&1)"

section "modules: cuda"
list_modules cuda

section "modules: gcc  (needed at runtime if used at build time)"
list_modules gcc

section "modules: torch/conda, if the cluster provides them"
for m in pytorch conda miniconda anaconda mamba; do list_modules "$m"; done

section "apptainer / singularity (container alternative to the venv path)"
# A container sidesteps the EL8/EL9 glibc split and module resolution entirely,
# and an NGC PyTorch image ships torch+CUDA+Triton tested together. The three
# things that decide whether it is viable here:
#   1. is the runtime present, and what version;
#   2. can this user BUILD images, or only run them -- unprivileged build needs
#      --fakeroot, which many sites restrict to admins. If build is blocked,
#      `apptainer pull` of a prebuilt image still works;
#   3. is there disk for a ~20GB image (see the storage section below).
for c in apptainer singularity; do
    if have "$c"; then
        echo "--- $c ---"
        "$c" --version 2>&1 | head -1
        echo "config (cache/tmp dirs honoured via APPTAINER_CACHEDIR / TMPDIR):"
        env | grep -Ei '^(APPTAINER|SINGULARITY)_' | sort || echo "  (no env overrides set)"
    else
        echo "$c: not on PATH here"
    fi
done
echo
echo "fakeroot / unprivileged build capability:"
if have apptainer; then
    # Does not build anything; asks whether the subuid mapping exists at all.
    grep -q "^$(whoami):" /etc/subuid 2>/dev/null \
        && echo "  /etc/subuid has an entry for $(whoami) -- --fakeroot build likely allowed" \
        || echo "  no /etc/subuid entry -- expect to PULL prebuilt images, not build them"
else
    echo "  (apptainer absent; skipping)"
fi
echo
echo "If containers are viable, the base image to want is an NGC PyTorch one"
echo "(torch + CUDA + Triton tested together), with flash-linear-attention"
echo "layered on top -- NGC does not ship it. Check the image's torch version"
echo "against what fla requires before committing to a tag."

section "storage and quota"
echo "--- home ---"; df -h "$HOME" 2>/dev/null
for d in /scratch /nfs /data "$HOME/../scratch"; do
    [ -d "$d" ] && { echo "--- $d ---"; df -h "$d" 2>/dev/null; }
done
have quota && quota -s 2>/dev/null || echo "(quota command unavailable)"

section "current queue"
have squeue && squeue -u "$(whoami)" 2>/dev/null

section "GPU visibility from THIS node (expected: none on a login node)"
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null \
        || echo "nvidia-smi present but no GPU here (normal on a login node)"
else
    echo "nvidia-smi not on PATH here (normal on a login node)"
fi

section "NEXT"
cat <<'EOF'
Fill these into scripts/slurm/sweep_array.sbatch:

  #SBATCH --partition=<a partition from the GPU-nodes section>
  #SBATCH --account=<from the accounts section, if accounting is enabled>

and export the module names if they differ from the defaults:

  export MODULE_PYTHON=python/<version>
  export MODULE_CUDA=cuda/<version>

Then get an interactive GPU node and run the environment setup there --
the parity test needs a real device:

  srun --partition=<gpu partition> --gres=gpu:1 --time=01:00:00 --pty bash
  bash scripts/slurm/setup_env.sh
EOF

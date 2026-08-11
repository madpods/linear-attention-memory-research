"""Compile and run one trivial Triton kernel. The real CUDA-toolkit check.

    python scripts/slurm/triton_smoke.py

Run by ``setup_env.sh --verify`` before the fla parity gate, and safe to run by
hand when debugging. Exits 0 on success, 1 with a diagnosis otherwise.

This exists as a FILE, not a heredoc, and that is load-bearing:
``@triton.jit`` calls ``inspect.getsourcelines`` on the decorated function, so a
kernel defined on stdin fails at decoration time with "@jit functions should be
defined in a Python file" -- before any compilation is attempted, and therefore
without testing the thing this script is for.

Why bother at all, when ``import triton`` already succeeded: Triton compiles
kernels on FIRST CALL, against the CUDA toolkit, and emits a host launcher stub
through the system C compiler. Neither happens at import. A toolkit/torch
mismatch or a too-old gcc therefore surfaces deep inside fla mid-sweep rather
than here, which is the entire reason setup_env.sh records its module choices.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"FATAL: cannot import torch: {exc}")
        return 1

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device. Run this on a GPU node.")
        return 1

    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        print(f"FATAL: cannot import triton: {exc}")
        return 1

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
    except Exception as exc:
        print("FATAL: Triton could not compile/run a trivial kernel:")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("This is the CUDA-toolkit mismatch the module recording guards")
        print(f"against. torch was built against CUDA {torch.version.cuda};")
        print("load a matching cuda/<major>.x module and rebuild:")
        print("    export MODULE_CUDA=cuda/<version>")
        print(f"    rm -rf {sys.prefix} && bash scripts/slurm/setup_env.sh --install")
        print()
        print("If the error mentions a compiler or a missing header, the host")
        print("compiler is the problem instead -- Triton builds a launcher stub")
        print("with it. Try a different gcc: export GCC_MAX_MAJOR=13")
        return 1

    if not torch.allclose(y, torch.ones_like(y)):
        # Compiled and ran, but produced the wrong answer. Worse than failing:
        # every downstream number would be quietly wrong.
        print(f"FATAL: kernel ran but returned {y[:8].tolist()}, expected ones.")
        print("Do not proceed -- this GPU/toolkit combination is miscompiling.")
        return 1

    print(f"triton {getattr(triton, '__version__', '?')} compiled and ran a "
          f"kernel on {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Linear Attention Memory Research (lamr).

Staged testbed for the mechanisms described in
``linear_attention_memory_research_plan.md``.

Backend policy
--------------
Every state-update rule has two implementations:

``naive``
    Pure-PyTorch sequential/recurrent reference. Runs on CPU, is the
    correctness ground truth, and is what the plan's "correctness before
    speed" principle asks for.

``fla``
    ``flash-linear-attention``'s chunk-parallel Triton kernels. Requires an
    NVIDIA CUDA GPU and is therefore unavailable on the development
    workstation; it is selected automatically when importable.

The two must agree numerically. ``tests/test_backend_parity.py`` enforces that
whenever the ``fla`` backend is importable, and skips otherwise.
"""

__version__ = "0.0.1"

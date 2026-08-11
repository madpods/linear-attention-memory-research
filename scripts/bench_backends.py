"""Throughput comparison: sequential reference vs chunked backend.

Tokens/sec measured identically across backends, per the plan's cross-cutting
metrics requirement. Run before and after any kernel change; a chunk size that
wins here still has to pass tests/test_chunked_parity.py.

    python scripts/bench_backends.py
"""

from __future__ import annotations

import time

import torch

from lamr.layers import chunk_delta_rule, delta_rule_recurrent


def timed(fn, *args, repeats: int = 3, **kwargs) -> float:
    fn(*args, **kwargs)  # warmup
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    torch.manual_seed(0)
    b, h, d = 8, 4, 64
    print(f"batch={b} heads={h} d_k=d_v={d}  device=cpu  dtype=float32")
    print(f"{'seq_len':>8} {'backend':>12} {'chunk':>6} {'sec':>8} {'tok/s':>12} {'speedup':>8}")

    for seq_len in (128, 256, 512):
        q = torch.randn(b, h, seq_len, d)
        k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, d), dim=-1)
        v = torch.randn(b, h, seq_len, d)
        beta = torch.rand(b, h, seq_len)
        tokens = b * seq_len

        with torch.no_grad():
            base = timed(delta_rule_recurrent, q, k, v, beta)
            print(f"{seq_len:>8} {'sequential':>12} {'-':>6} {base:>8.3f} "
                  f"{tokens / base:>12,.0f} {1.0:>8.1f}x")

            for chunk in (16, 32, 64, 128):
                if chunk > seq_len:
                    continue
                dt = timed(chunk_delta_rule, q, k, v, beta, chunk_size=chunk)
                print(f"{seq_len:>8} {'chunked':>12} {chunk:>6} {dt:>8.3f} "
                      f"{tokens / dt:>12,.0f} {base / dt:>8.1f}x")


if __name__ == "__main__":
    main()

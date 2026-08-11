"""Training loop for the MQAR testbed.

One entry point used identically by every stage, so a Stage 4 number and a
Stage 2 number differ only by the mechanism under test.

    python -m lamr.train --mode gated_delta --redundancy-r 0.5 --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lamr.data import gen_redundant_mqar
from lamr.data.mqar import IGNORE_INDEX, MQARBatch
from lamr.metrics import recall_metrics
from lamr.models import LinearAttentionLM, LMConfig


@dataclass
class TrainConfig:
    # --- data ---
    seq_len: int = 128
    vocab_size: int = 256
    num_kv_pairs: int = 8
    num_queries: int = 4
    redundancy_r: float = 0.0
    num_value_clusters: int = 4
    num_train: int = 8192
    num_eval: int = 1024
    data_seed: int = 0

    # --- model (fixed across runs per plan Stage 2) ---
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mode: str = "gated_delta"
    chunk_size: int = 64
    backend: str = "chunked"
    use_short_conv: bool = True
    # "auto" -> cuda when present, else cpu. Explicit "cpu" forces the host even
    # on a GPU node, which is what reproduces the Stage 2 CPU baselines.
    device: str = "auto"

    # --- optimization ---
    steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0

    # --- logging ---
    eval_every: int = 250
    log_every: int = 100
    results_csv: str | None = None
    tag: str = ""

    def model_config(self) -> LMConfig:
        return LMConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mode=self.mode,
            chunk_size=self.chunk_size,
            backend=self.backend,
            use_short_conv=self.use_short_conv,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_data(cfg: TrainConfig, num_examples: int, seed: int) -> MQARBatch:
    return gen_redundant_mqar(
        num_examples=num_examples,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        num_kv_pairs=cfg.num_kv_pairs,
        num_queries=cfg.num_queries,
        redundancy_r=cfg.redundancy_r,
        num_value_clusters=cfg.num_value_clusters,
        seed=seed,
    )


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup into cosine decay."""
    warmup = max(1, int(cfg.steps * cfg.warmup_frac))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, batch: MQARBatch, batch_size: int = 64) -> dict[str, float]:
    """Accuracy over a held-out set, accumulated in minibatches."""
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for start in range(0, len(batch), batch_size):
        sl = slice(start, min(start + batch_size, len(batch)))
        logits = model(batch.input_ids[sl])
        m = recall_metrics(logits, batch, slice_index=sl)
        for key, count_key in (
            ("accuracy", "num_queries"),
            ("accuracy_redundant", "num_redundant"),
            ("accuracy_non_redundant", "num_queries"),
            ("accuracy_shared", "num_shared"),
        ):
            n = m[count_key]
            if n and not math.isnan(m[key]):
                totals[key] = totals.get(key, 0.0) + m[key] * n
                counts[key] = counts.get(key, 0) + n

    model.train()
    return {k: totals[k] / counts[k] for k in totals}


def resolve_device(spec: str) -> torch.device:
    """``"auto"`` -> cuda when available, else cpu. Anything else is taken as-is.

    The ``fla`` backend is CUDA-only: its Triton kernels cannot run on host
    tensors, and ``fla_available()`` tests only that a CUDA *device exists*, not
    that the tensors are on it. So a CPU-resident model with ``backend="fla"``
    passes that check and then fails inside the kernel -- which is why this
    raises instead, naming the actual problem.
    """
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else spec
    )
    return device


def train(cfg: TrainConfig, verbose: bool = True) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    if cfg.backend == "fla" and device.type != "cuda":
        raise RuntimeError(
            f"backend='fla' needs CUDA tensors but device resolved to {device}. "
            "fla's Triton kernels cannot run on the host. Either run on a GPU "
            "node (device='auto' finds it) or use backend='chunked'."
        )

    # Data is generated on the host and moved once, up front: these are small
    # fixed corpora, not streamed, so per-step transfers would be pure overhead.
    train_data = _make_data(cfg, cfg.num_train, cfg.data_seed).to(device)
    eval_data = _make_data(cfg, cfg.num_eval, cfg.data_seed + 10_000).to(device)

    model = LinearAttentionLM(cfg.model_config()).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    n_params = model.num_parameters()
    if verbose:
        print(
            f"mode={cfg.mode} r={cfg.redundancy_r} kv={cfg.num_kv_pairs} "
            f"params={n_params:,} steps={cfg.steps} device={device} "
            f"backend={cfg.backend}"
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)

        # Generator stays on the HOST deliberately, then the indices are moved.
        # A CUDA generator would draw a different sequence for the same seed, so
        # GPU runs would see a different batch order than the recorded CPU
        # baselines -- and the r=0 rows are supposed to be a controlled
        # comparison against those.
        idx = torch.randint(0, len(train_data), (cfg.batch_size,), generator=generator)
        idx = idx.to(device)
        logits = model(train_data.input_ids[idx])
        # No next-token shift: position p carries the answer to the query at p.
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            train_data.lm_labels[idx].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if verbose and cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"  step {step + 1:>5}  loss {loss.item():.4f}")

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            metrics = evaluate(model, eval_data)
            history.append({"step": step + 1, "loss": loss.item(), **metrics})
            if verbose:
                acc = metrics["accuracy"]
                red = metrics.get("accuracy_redundant", float("nan"))
                non = metrics.get("accuracy_non_redundant", float("nan"))
                print(
                    f"  step {step + 1:>5}  acc {acc:.3f}  "
                    f"redundant {red:.3f}  non-redundant {non:.3f}"
                )

    final = evaluate(model, eval_data)
    # CUDA launches are asynchronous, so the host clock would otherwise stop
    # while kernels are still running and report a tokens_per_sec that is simply
    # wrong. That column is a Stage 2 deliverable, so synchronize before reading
    # the clock. No-op on CPU.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        **cfg.as_dict(),
        **{f"final_{k}": v for k, v in final.items()},
        "num_parameters": n_params,
        "wall_clock_sec": round(elapsed, 2),
        "tokens_per_sec": round(cfg.steps * cfg.batch_size * cfg.seq_len / elapsed),
        "history": history,
    }

    if cfg.results_csv:
        _append_csv(Path(cfg.results_csv), result)
    if verbose:
        print(
            f"  done in {elapsed:.1f}s  final acc {final['accuracy']:.3f}  "
            f"({result['tokens_per_sec']:,} tok/s)"
        )
    return result


def _append_csv(path: Path, result: dict[str, Any]) -> None:
    """Append one row; ``history`` is JSON-encoded so the row stays flat."""
    row = {k: (json.dumps(v) if k == "history" else v) for k, v in result.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _cli() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() == "true", default=value)
        else:
            arg_type = type(value) if value is not None else str
            parser.add_argument(flag, type=arg_type, default=value)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_cli())

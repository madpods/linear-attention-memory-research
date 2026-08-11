"""Stage 2 baseline sweeps -- the reference curves every later stage cites.

Two presets matter:

``kv-curve``
    r=0, recall vs num_kv_pairs for all three baselines. This is the plan's
    hard prerequisite: before any of these numbers can be trusted, the shape
    has to match the DeltaNet paper's published MQAR result -- high recall for
    few pairs, collapsing as pairs exceed what the state can hold, with
    linear attention collapsing earliest and Gated DeltaNet latest.

``full``
    The same, crossed with r in {0, 0.25, 0.5, 0.75, 0.9}. This is the
    reference grid Stages 3-6 compare against.

Architecture hyperparameters are fixed here once and must not drift between
stages (plan principle 3). Rows are appended to CSV and completed
configurations are skipped, so a long sweep can be interrupted and resumed.

    python scripts/stage2_sweep.py --preset kv-curve
    python scripts/stage2_sweep.py --preset full --steps 4000
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from lamr.train import TrainConfig, train

MODES = ("linear", "delta", "gated_delta")

PRESETS = {
    "quick": dict(kv_pairs=(4, 8), redundancies=(0.0, 0.5), steps=200),
    "kv-curve": dict(kv_pairs=(4, 8, 16, 32, 64), redundancies=(0.0,), steps=1500),
    "full": dict(
        kv_pairs=(4, 8, 16, 32, 64),
        redundancies=(0.0, 0.25, 0.5, 0.75, 0.9),
        steps=1500,
    ),
}

# Fixed across every run in every stage. Changing these invalidates comparison
# against previously recorded curves.
FIXED = dict(
    seq_len=256,
    vocab_size=512,
    d_model=64,
    num_layers=2,
    num_heads=4,
    num_value_clusters=8,
    num_train=8192,
    num_eval=1024,
    batch_size=32,
    lr=1e-3,
)


def already_done(path: Path, cfg: TrainConfig) -> bool:
    if not path.exists():
        return False
    keys = ("mode", "redundancy_r", "num_kv_pairs", "steps", "seed")
    want = {k: str(getattr(cfg, k)) for k in keys}
    with path.open(newline="", encoding="utf-8") as fh:
        return any(
            all(row.get(k) == v for k, v in want.items()) for row in csv.DictReader(fh)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="kv-curve")
    parser.add_argument("--out", default="results/stage2.csv")
    parser.add_argument("--steps", type=int, default=None, help="override preset steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--backend", default="chunked", help="chunked|sequential|fla")
    parser.add_argument(
        "--count",
        action="store_true",
        help="print the number of runs and exit (for sizing a Slurm array)",
    )
    parser.add_argument(
        "--only-index",
        type=int,
        default=None,
        help="run a single grid element by 0-based index. One Slurm array task "
        "per index; give each task its own --out to avoid concurrent CSV writes.",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    steps = args.steps or preset["steps"]
    out = Path(args.out)

    grid = list(itertools.product(args.modes, preset["redundancies"], preset["kv_pairs"]))

    if args.count:
        print(len(grid))
        return
    if args.only_index is not None:
        if not 0 <= args.only_index < len(grid):
            raise SystemExit(
                f"--only-index {args.only_index} out of range for {len(grid)} runs"
            )
        grid = [grid[args.only_index]]

    print(f"preset={args.preset} runs={len(grid)} steps={steps} -> {out}")

    for i, (mode, r, kv) in enumerate(grid, start=1):
        cfg = TrainConfig(
            mode=mode,
            redundancy_r=r,
            num_kv_pairs=kv,
            # Held fixed across kv so the eval set size does not vary with it.
            num_queries=min(8, kv),
            steps=steps,
            seed=args.seed,
            backend=args.backend,
            results_csv=str(out),
            eval_every=0,
            log_every=0,
            tag=args.preset,
            **FIXED,
        )
        if already_done(out, cfg):
            print(f"[{i}/{len(grid)}] skip (done): {mode} r={r} kv={kv}")
            continue
        print(f"[{i}/{len(grid)}] {mode} r={r} kv={kv}")
        train(cfg, verbose=True)


if __name__ == "__main__":
    main()

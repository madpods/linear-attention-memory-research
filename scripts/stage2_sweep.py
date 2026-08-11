"""Stage 2 baseline sweeps -- the reference curves every later stage cites.

Three presets matter:

``kv-curve``
    r=0, recall vs num_kv_pairs for all three baselines. This is the plan's
    hard prerequisite: before any of these numbers can be trusted, the shape
    has to match the DeltaNet paper's published MQAR result -- high recall for
    few pairs, collapsing as pairs exceed what the state can hold, with
    linear attention collapsing earliest and Gated DeltaNet latest.

``full``
    The same, crossed with r in {0, 0.25, 0.5, 0.75, 0.9}. This is the
    reference grid Stages 3-6 compare against. **Run it at several seeds.**
    Mid-cliff cells are bimodal, not noisy -- kv=32 spreads ~47 points over 4
    seeds -- so a single seed there is a coin flip, and one already reversed the
    delta / gated_delta ordering between two otherwise-matched runs.

``cliff``
    steps as a grid axis, to test whether the capacity cliff is a capacity
    limit or a training-budget limit. See the comment on the preset.

Architecture hyperparameters are otherwise fixed here once and must not drift
between stages (plan principle 3); ``cliff`` varies ``steps`` deliberately and
is a diagnostic, not a source of reference curves. Rows are appended to CSV and
completed configurations are skipped, so a long sweep can be interrupted and
resumed.

    python scripts/stage2_sweep.py --preset kv-curve
    python scripts/stage2_sweep.py --preset full --steps 4000
    python scripts/stage2_sweep.py --preset cliff --modes delta gated_delta
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
    # THE reference grid. Re-fixed 2026-08-11 after the cliff and converge
    # sweeps showed the original (1500 steps, kv up to 64) was measuring
    # convergence rather than capacity. Both changes are load-bearing:
    #
    # steps 1500 -> 8000. capacity@95% moved 16 -> 16 -> 32 -> 32 over
    #   1500/3000/6000/12000 and then held at 32 through 24000 and 48000, so it
    #   converges by 6000; 8000 is that with margin. At 1500 the mid-cliff cells
    #   were bimodal (47-point spreads) purely from undertraining.
    #
    # kv gains 48. The converged 95% crossings are ~35 (linear), ~56 (delta),
    #   ~57 (gated_delta) -- ALL inside the old grid's 32 -> 64 jump, so
    #   capacity@95% read 32 for every mode and could not discriminate at all.
    #   kv=48 splits them (linear ~80%, delta ~97%). 96 stays as headroom; the
    #   hard ceiling is 124, since seq_len=256 must hold 2*kv + num_queries.
    "full": dict(
        kv_pairs=(4, 8, 16, 32, 48, 64, 96),
        redundancies=(0.0, 0.25, 0.5, 0.75, 0.9),
        steps=8000,
    ),
    # Is the capacity cliff a CAPACITY limit or a TRAINING-BUDGET limit?
    #
    # Over 4 seeds the mid-cliff cells are bimodal, not noisy: kv=32 at r=0 has
    # a 47-point spread around a 62% mean, i.e. some seeds learn it and some do
    # not. If more steps collapse that spread upward, then 1500 steps was the
    # binding constraint and "effective capacity" was measuring the optimizer,
    # not the state matrix -- which would move every reference curve and change
    # what Stages 3-6 are compared against. Cheap to settle: ~11 min of wall
    # clock at 14-way concurrency.
    #
    #   sbatch --array=0-35%14 --export=ALL,PRESET=cliff,SEED=0 \
    #          scripts/slurm/sweep_array.sbatch
    #
    # Run it for several seeds; a single seed cannot distinguish a collapsed
    # spread from a lucky draw, which is the whole point.
    "cliff": dict(
        kv_pairs=(16, 32, 64),
        redundancies=(0.0,),
        steps=1500,
        steps_list=(1500, 3000, 6000, 12000),
    ),
    # ANSWERED, and the answer was "training budget". capacity@95% moved
    # 16 -> 16 -> 32 -> 32 over those four step counts and had not stopped;
    # kv=64 reached 91% at 12000 steps, just under the threshold. This preset
    # pushes until it stops moving, which is the precondition for fixing the
    # budget honestly. kv=96 is included because 64 is no longer the top of the
    # curve -- seq_len=256 admits up to 124 pairs (2*124 + 8 queries).
    #
    #   PARTS=results/parts_converge sbatch --array=0-17%14 \
    #       --export=ALL,PRESET=converge,SEED=0,PARTS=results/parts_converge,MODES="delta gated_delta" \
    #       scripts/slurm/sweep_array.sbatch
    #
    # ~18 min per 24000-step run, ~36 min at 48000. Raise --time if the array
    # starts hitting the 4h wall at the larger kv.
    "converge": dict(
        kv_pairs=(32, 64, 96),
        redundancies=(0.0,),
        steps=24000,
        steps_list=(24000, 48000),
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
    out = Path(args.out)

    # steps is normally fixed (plan principle 3) and so not a grid axis. The
    # cliff preset makes it one deliberately, to ask whether the fixed budget is
    # what the capacity numbers are actually measuring. An explicit --steps
    # overrides either way.
    if args.steps:
        steps_list: tuple[int, ...] = (args.steps,)
    else:
        steps_list = tuple(preset.get("steps_list", (preset["steps"],)))

    grid = list(
        itertools.product(
            args.modes, preset["redundancies"], preset["kv_pairs"], steps_list
        )
    )

    if args.count:
        print(len(grid))
        return
    if args.only_index is not None:
        if not 0 <= args.only_index < len(grid):
            raise SystemExit(
                f"--only-index {args.only_index} out of range for {len(grid)} runs"
            )
        grid = [grid[args.only_index]]

    print(f"preset={args.preset} runs={len(grid)} steps={sorted(steps_list)} -> {out}")

    for i, (mode, r, kv, steps) in enumerate(grid, start=1):
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
            print(f"[{i}/{len(grid)}] skip (done): {mode} r={r} kv={kv} steps={steps}")
            continue
        print(f"[{i}/{len(grid)}] {mode} r={r} kv={kv} steps={steps}")
        train(cfg, verbose=True)


if __name__ == "__main__":
    main()

"""Read a sweep CSV into the plan's cross-cutting metrics.

    python scripts/analyze_stage2.py                      # results/stage2.csv
    python scripts/analyze_stage2.py results/stage2.csv --csv summary.csv

Reports the four things the plan asks every stage to produce identically:
recall vs num_kv_pairs, recall vs redundancy r, throughput, and effective
capacity (the largest num_kv_pairs still recalled at >= 95%).

Effective capacity is the headline number: it compresses a whole curve into
one figure comparable across mechanisms at matched parameter count, which is
what "does this mechanism buy capacity" actually means.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MODE_ORDER = ["linear", "delta", "gated_delta"]
CAPACITY_THRESHOLD = 0.95


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no results at {path} -- run scripts/stage2_sweep.py first")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def by_mode(rows: list[dict]) -> list[str]:
    present = {r["mode"] for r in rows}
    return [m for m in MODE_ORDER if m in present] + sorted(present - set(MODE_ORDER))


def fmt_pct(x: float | None) -> str:
    return "     -" if x is None else f"{100 * x:5.1f}%"


def effective_capacity(points: dict[int, float]) -> str:
    """Largest kv still at >= threshold, with a note if the curve never drops.

    Reported as a bound rather than a point estimate when the sweep never
    reaches the threshold in either direction -- claiming a capacity outside
    the measured range would overstate what the data supports.
    """
    if not points:
        return "n/a"
    kvs = sorted(points)
    passing = [kv for kv in kvs if points[kv] >= CAPACITY_THRESHOLD]
    if not passing:
        return f"< {kvs[0]}"
    best = max(passing)
    return f">= {best}" if best == kvs[-1] else str(best)


def _group_of(row: dict, group_key: str) -> float | int:
    value = float(row[group_key])
    return int(value) if group_key == "num_kv_pairs" else value


def collect(rows: list[dict], mode: str, group_key: str, value_key: str) -> dict:
    """``{group: [value, ...]}`` for one mode. A LIST, not a scalar.

    Assigning a scalar here silently kept only the last matching row, which made
    the "pooled over kv" table report the kv=64 column alone -- and that column
    is fully collapsed, so it read as recall *improving* with redundancy when it
    was showing chance improving as the answer pool shrank. Multiple rows per
    cell are now the normal case anyway, since seeds are replicated.
    """
    points: dict = defaultdict(list)
    for row in rows:
        if row["mode"] != mode:
            continue
        if not row.get(value_key):
            continue
        points[_group_of(row, group_key)].append(float(row[value_key]))
    return points


def table(rows: list[dict], group_key: str, title: str, value_key: str) -> None:
    """Recall table: one row per mode, one column per distinct group_key.

    Cells are means over whatever else varies (seeds always; kv as well when
    grouping by r). When any cell pools more than one run, a companion spread
    table follows -- at the capacity cliff the spread is the whole story, and a
    mean alone there invites reading noise as an effect.
    """
    groups = sorted({_group_of(r, group_key) for r in rows})
    if len(groups) < 2 and group_key == "redundancy_r":
        return  # nothing to compare yet

    print(f"\n{title}")
    header = "  ".join(f"{g:>7}" for g in groups)
    print(f"{'mode':<13} {header}   capacity@{int(CAPACITY_THRESHOLD * 100)}%")
    print("-" * (14 + 9 * len(groups) + 16))

    spreads: dict[str, dict] = {}
    for mode in by_mode(rows):
        points = collect(rows, mode, group_key, value_key)
        means = {g: sum(v) / len(v) for g, v in points.items()}
        cells = "  ".join(fmt_pct(means.get(g)) for g in groups)
        # Capacity is only defined against a kv curve. For the r-grouped table
        # it is reported separately, per r, by capacity_vs_r().
        cap = effective_capacity(means) if group_key == "num_kv_pairs" else ""
        print(f"{mode:<13} {cells}   {cap:>12}")
        spread = {g: max(v) - min(v) for g, v in points.items() if len(v) > 1}
        if spread:
            spreads[mode] = (spread, {g: len(v) for g, v in points.items()})

    if spreads:
        n_max = max(max(counts.values()) for _, counts in spreads.values())
        print(f"  spread (max-min across {n_max} runs per cell)")
        for mode, (spread, _counts) in spreads.items():
            cells = "  ".join(
                "     -" if g not in spread else f"{100 * spread[g]:5.1f} "
                for g in groups
            )
            print(f"  {mode:<11} {cells}")


def capacity_vs_r(rows: list[dict]) -> None:
    """Effective capacity at each redundancy level.

    This is the headline cross-cutting number the plan asks every stage to
    produce, and it is the one figure that compresses a curve into something
    comparable across mechanisms. Reported per r because capacity is defined
    against a kv curve -- there is no single capacity for a mode across all r,
    which is why the r-grouped recall table leaves that column blank.
    """
    r_values = sorted({float(x["redundancy_r"]) for x in rows})
    print(f"\nEffective capacity@{int(CAPACITY_THRESHOLD * 100)}% vs redundancy r")
    header = "  ".join(f"{r:>7}" for r in r_values)
    print(f"{'mode':<13} {header}")
    print("-" * (14 + 9 * len(r_values)))
    for mode in by_mode(rows):
        cells = []
        for r_val in r_values:
            subset = [x for x in rows if float(x["redundancy_r"]) == r_val]
            points = collect(subset, mode, "num_kv_pairs", "final_accuracy")
            means = {g: sum(v) / len(v) for g, v in points.items()}
            cells.append(f"{effective_capacity(means):>7}")
        print(f"{mode:<13} {'  '.join(cells)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results/stage2.csv", type=Path)
    parser.add_argument("--csv", type=Path, help="also write a tidy summary csv")
    parser.add_argument(
        "--tag", help="keep only rows with this tag (the preset that produced them)"
    )
    parser.add_argument(
        "--steps", type=int, help="keep only rows with this step count"
    )
    args = parser.parse_args()

    rows = load(args.path)
    if args.tag:
        rows = [r for r in rows if r.get("tag") == args.tag]
        if not rows:
            raise SystemExit(f"no rows with tag={args.tag!r} in {args.path}")
    if args.steps:
        rows = [r for r in rows if int(float(r["steps"])) == args.steps]
        if not rows:
            raise SystemExit(f"no rows with steps={args.steps} in {args.path}")

    # Cells group by (mode, r, kv) and average everything else, which is right
    # for seeds and WRONG for step counts -- averaging a 1500-step run with a
    # 12000-step one produces a number describing neither. The cliff preset
    # varies steps deliberately, so refuse to average silently.
    step_values = sorted({int(float(r["steps"])) for r in rows})
    if len(step_values) > 1:
        raise SystemExit(
            f"{args.path} mixes {len(step_values)} step counts: {step_values}.\n"
            "Cells average over everything except (mode, r, kv), so this would\n"
            "average different training budgets together. Split them first:\n"
            f"    python {Path(__file__).name} {args.path} --steps {step_values[0]}\n"
            "or filter by the preset that produced them, e.g. --tag full / --tag cliff.\n"
            f"Tags present: {sorted({r.get('tag', '') for r in rows})}"
        )
    n_r = len({r["redundancy_r"] for r in rows})
    print(f"{len(rows)} runs from {args.path}")
    print(f"steps={rows[0]['steps']}  d_model={rows[0]['d_model']}  "
          f"layers={rows[0]['num_layers']}  heads={rows[0]['num_heads']}  "
          f"seq_len={rows[0]['seq_len']}")

    # Recall vs num_kv_pairs, at each redundancy level separately: pooling
    # across r would average away the axis the project exists to measure.
    for r_val in sorted({float(x["redundancy_r"]) for x in rows}):
        subset = [x for x in rows if float(x["redundancy_r"]) == r_val]
        table(subset, "num_kv_pairs", f"Recall vs num_kv_pairs  (r={r_val})",
              "final_accuracy")

    if n_r > 1:
        table(rows, "redundancy_r",
              "Recall vs redundancy r (mean over kv -- mixes saturated and "
              "collapsed regimes;\nread the capacity table below instead for a "
              "single comparable number)",
              "final_accuracy")
        capacity_vs_r(rows)
        print("\nRedundant vs non-redundant queries (r>0 only)")
        print(f"{'mode':<13} {'r':>6} {'redundant':>11} {'non-redund':>11} {'gap':>8}")
        print("-" * 52)
        for mode in by_mode(rows):
            for r_val in sorted({float(x["redundancy_r"]) for x in rows}):
                if r_val == 0:
                    continue
                sel = [x for x in rows
                       if x["mode"] == mode and float(x["redundancy_r"]) == r_val]
                red = [float(x["final_accuracy_redundant"]) for x in sel
                       if x.get("final_accuracy_redundant")]
                non = [float(x["final_accuracy_non_redundant"]) for x in sel
                       if x.get("final_accuracy_non_redundant")]
                if not red or not non:
                    continue
                a, b = sum(red) / len(red), sum(non) / len(non)
                print(f"{mode:<13} {r_val:>6} {fmt_pct(a):>11} {fmt_pct(b):>11} "
                      f"{100 * (a - b):>+7.1f}")

    print("\nThroughput")
    print(f"{'mode':<13} {'tok/s':>10} {'sec/run':>9} {'params':>10}")
    print("-" * 45)
    for mode in by_mode(rows):
        sel = [r for r in rows if r["mode"] == mode]
        tps = sum(float(r["tokens_per_sec"]) for r in sel) / len(sel)
        sec = sum(float(r["wall_clock_sec"]) for r in sel) / len(sel)
        print(f"{mode:<13} {tps:>10,.0f} {sec:>9.0f} {int(sel[0]['num_parameters']):>10,}")

    if args.csv:
        fields = ["mode", "redundancy_r", "num_kv_pairs", "final_accuracy",
                  "final_accuracy_redundant", "final_accuracy_non_redundant",
                  "num_parameters", "tokens_per_sec"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["mode"],
                                                    float(r["redundancy_r"]),
                                                    int(r["num_kv_pairs"]))))
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

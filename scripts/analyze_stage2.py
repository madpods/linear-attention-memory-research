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


def table(rows: list[dict], group_key: str, title: str, value_key: str) -> None:
    """Recall table: one row per mode, one column per distinct group_key."""
    groups = sorted({int(float(r[group_key])) if group_key == "num_kv_pairs"
                     else float(r[group_key]) for r in rows})
    if len(groups) < 2 and group_key == "redundancy_r":
        return  # nothing to compare yet

    print(f"\n{title}")
    header = "  ".join(f"{g:>7}" for g in groups)
    print(f"{'mode':<13} {header}   capacity@{int(CAPACITY_THRESHOLD * 100)}%")
    print("-" * (14 + 9 * len(groups) + 16))

    for mode in by_mode(rows):
        points: dict = {}
        for r in rows:
            if r["mode"] != mode:
                continue
            g = int(float(r[group_key])) if group_key == "num_kv_pairs" else float(r[group_key])
            points[g] = float(r[value_key])
        cells = "  ".join(fmt_pct(points.get(g)) for g in groups)
        cap = effective_capacity(points) if group_key == "num_kv_pairs" else ""
        print(f"{mode:<13} {cells}   {cap:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results/stage2.csv", type=Path)
    parser.add_argument("--csv", type=Path, help="also write a tidy summary csv")
    args = parser.parse_args()

    rows = load(args.path)
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
        table(rows, "redundancy_r", "Recall vs redundancy r (pooled over kv)",
              "final_accuracy")
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

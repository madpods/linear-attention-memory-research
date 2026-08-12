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


def by_group(rows: list[dict], key: str = "mode") -> list[str]:
    """Row labels, in a stable order.

    Stage 2 groups by mode; Stage 3 groups by feature_map (or d_model) so each
    ablation arm gets its own row instead of being averaged into one.
    """
    present = {r[key] for r in rows}
    known = [m for m in MODE_ORDER if m in present]
    return known + sorted(present - set(known), key=_sort_key)


def _sort_key(label: str):
    """Numeric-aware so d_model 128 sorts before 512 and dpfp2 before dpfp3."""
    digits = "".join(ch for ch in label if ch.isdigit())
    return (int(digits) if digits else 0, label)


def fmt_pct(x: float | None) -> str:
    return "     -" if x is None else f"{100 * x:5.1f}%"


def _as_float(value: str | None) -> float | None:
    """Parse a cell, or None if it is blank or not a number."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_of(rows: list[dict], key: str) -> float | None:
    values = [v for v in (_as_float(r.get(key)) for r in rows) if v is not None]
    return sum(values) / len(values) if values else None


def effective_capacity(points: dict[int, float]) -> str:
    """Effective capacity: the kv at which recall crosses the threshold.

    Interpolated between grid points rather than snapped to one, because
    snapping destroys the measurement. At 8000 steps every mode's 95% crossing
    falls inside the 32->48 gap, so the snapped value reads a flat 32 at every
    redundancy level and the redundancy effect -- a ~24% capacity loss from r=0
    to r=0.9 -- is completely invisible. Interpolating recovers it (44.0 -> 33.4)
    and costs nothing, since it is the same definition read at finer resolution.

    Uses the LAST crossing, because these curves are not always monotone: linear
    attention at r=0 dips below the threshold at kv=4, rises above it at kv=8,
    then falls for good. The last downward crossing is the capacity; an early
    dip at a trivially small kv is a training artifact, not a ceiling.

    Still reported as a bound when the curve never crosses inside the measured
    range, since claiming a capacity outside it would overstate the data.
    """
    if not points:
        return "n/a"
    kvs = sorted(points)
    crossing = None
    for lo, hi in zip(kvs, kvs[1:]):
        if points[lo] >= CAPACITY_THRESHOLD > points[hi]:
            span = points[lo] - points[hi]
            frac = (points[lo] - CAPACITY_THRESHOLD) / span if span else 0.0
            crossing = lo + frac * (hi - lo)
    if crossing is not None:
        return f"{crossing:.1f}"
    if points[kvs[-1]] >= CAPACITY_THRESHOLD:
        return f">= {kvs[-1]}"
    return f"< {kvs[0]}"


def _group_of(row: dict, group_key: str) -> float | int:
    value = float(row[group_key])
    return int(value) if group_key == "num_kv_pairs" else value


def collect(rows: list[dict], mode: str, group_key: str, value_key: str,
            row_key: str = "mode") -> dict:
    """``{group: [value, ...]}`` for one mode. A LIST, not a scalar.

    Assigning a scalar here silently kept only the last matching row, which made
    the "pooled over kv" table report the kv=64 column alone -- and that column
    is fully collapsed, so it read as recall *improving* with redundancy when it
    was showing chance improving as the answer pool shrank. Multiple rows per
    cell are now the normal case anyway, since seeds are replicated.
    """
    points: dict = defaultdict(list)
    for row in rows:
        if row[row_key] != mode:
            continue
        if not row.get(value_key):
            continue
        points[_group_of(row, group_key)].append(float(row[value_key]))
    return points


def table(rows: list[dict], group_key: str, title: str, value_key: str,
          row_key: str = "mode") -> None:
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
    print(f"{row_key:<13} {header}   capacity@{int(CAPACITY_THRESHOLD * 100)}%")
    print("-" * (14 + 9 * len(groups) + 16))

    spreads: dict[str, dict] = {}
    for mode in by_group(rows, row_key):
        points = collect(rows, mode, group_key, value_key, row_key)
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


def capacity_vs_r(rows: list[dict], row_key: str = "mode") -> None:
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
    print(f"{row_key:<13} {header}")
    print("-" * (14 + 9 * len(r_values)))
    for mode in by_mode(rows):
        cells = []
        for r_val in r_values:
            subset = [x for x in rows if float(x["redundancy_r"]) == r_val]
            points = collect(subset, mode, "num_kv_pairs", "final_accuracy", row_key)
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
    parser.add_argument("--feature-map", help="keep only rows with this feature map")
    parser.add_argument("--d-model", help="keep only rows with this d_model")
    parser.add_argument(
        "--group-by", default="mode",
        choices=("mode", "feature_map", "d_model", "arm"),
        help="row key for every table. Stage 3 wants 'arm', a derived "
             "feature_map/d_model label -- its wide control arms are ALSO named "
             "identity, so grouping on feature_map alone would merge "
             "identity at d_model=64 with identity at d_model=256.",
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
    # Derived row key for Stage 3: the arm is (feature_map, d_model) together.
    for row in rows:
        if row.get("feature_map") and row.get("d_model"):
            row["arm"] = f"{row['feature_map']}/d{row['d_model']}"

    for flag, column in (("feature_map", "feature_map"), ("d_model", "d_model")):
        want = getattr(args, flag)
        if want:
            rows = [r for r in rows if r.get(column) == want]
            if not rows:
                raise SystemExit(f"no rows with {column}={want!r} in {args.path}")

    # Cells group by (group_key, r, kv) and average everything else. That is right
    # for seeds and wrong for every other axis: averaging a 1500-step run with a
    # 12000-step one, or a dpfp2 arm with an identity arm, produces a number
    # describing neither. Refuse rather than average silently -- this guard exists
    # because the steps version of the mistake was actually made.
    CONFOUNDS = ("steps", "feature_map", "d_model", "mode")
    encoded = {"arm": ("feature_map", "d_model")}.get(args.group_by, ())
    for column in CONFOUNDS:
        if column == args.group_by or column in encoded:
            continue
        values = sorted({r[column] for r in rows if r.get(column) not in (None, "")})
        if len(values) > 1:
            flag = f"--{column.replace('_', '-')}"
            raise SystemExit(
                f"{args.path} mixes {len(values)} values of {column!r}: {values}.\n"
                f"Cells average over everything except ({args.group_by}, r, kv), so\n"
                "this would average them together. Either filter:\n"
                f"    python {Path(__file__).name} {args.path} {flag} {values[0]}\n"
                f"or make it the row key:\n"
                f"    python {Path(__file__).name} {args.path} --group-by {column}\n"
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
              "final_accuracy", args.group_by)

    if n_r > 1:
        table(rows, "redundancy_r",
              "Recall vs redundancy r (mean over kv -- mixes saturated and "
              "collapsed regimes;\nread the capacity table below instead for a "
              "single comparable number)",
              "final_accuracy", args.group_by)
        capacity_vs_r(rows, args.group_by)
        print("\nRedundant vs non-redundant queries (r>0 only)")
        print(f"{'mode':<13} {'r':>6} {'redundant':>11} {'non-redund':>11} {'gap':>8}")
        print("-" * 52)
        for mode in by_group(rows, args.group_by):
            for r_val in sorted({float(x["redundancy_r"]) for x in rows}):
                if r_val == 0:
                    continue
                sel = [x for x in rows
                       if x[args.group_by] == mode
                       and float(x["redundancy_r"]) == r_val]
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
    print(f"{'mode':<13} {'tok/s':>10} {'sec/run':>9} {'params':>10} {'n':>5}")
    print("-" * 51)
    for mode in by_group(rows, args.group_by):
        sel = [r for r in rows if r[args.group_by] == mode]
        # Skip unparseable cells rather than dying. Timing columns can be blank
        # or misaligned in older data (see the _append_csv note in train.py);
        # losing a throughput average is not a reason to lose the accuracy
        # tables, which sit earlier in the row and are unaffected.
        tps = _mean_of(sel, "tokens_per_sec")
        sec = _mean_of(sel, "wall_clock_sec")
        params = _mean_of(sel, "num_parameters")
        n = sum(1 for r in sel if _as_float(r.get("tokens_per_sec")) is not None)
        cells = (
            f"{tps:>10,.0f}" if tps is not None else f"{'-':>10}",
            f"{sec:>9.0f}" if sec is not None else f"{'-':>9}",
            f"{params:>10,.0f}" if params is not None else f"{'-':>10}",
        )
        print(f"{mode:<13} {' '.join(cells)} {n:>5}")
    if any(_as_float(r.get("tokens_per_sec")) is None for r in rows):
        bad = sum(1 for r in rows if _as_float(r.get("tokens_per_sec")) is None)
        print(f"  note: {bad} of {len(rows)} rows have no usable timing column.")
        print("  Accuracy is unaffected (it precedes the timing columns in the")
        print("  row), but re-run those configs for trustworthy throughput.")

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

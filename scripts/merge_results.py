"""Merge per-task result CSVs into one file.

The Slurm array gives each task its own CSV because concurrent appends to a
shared file interleave mid-row. This stitches them back together.

    python scripts/merge_results.py results/parts results/stage2.csv

Duplicate configurations (same mode / r / kv / steps / seed) keep the last
occurrence, so re-running a failed array task overrides its earlier partial row.

Part files do NOT all carry the same columns, which is why the header is a union
rather than whatever the first file happened to have. ``evaluate()`` only reports
a metric when its denominator is non-zero, so an ``r=0`` run has no redundant
pairs and omits ``final_accuracy_redundant`` / ``final_accuracy_shared``, while
an ``r>0`` run includes them. Rows missing a column get an empty value, matching
how the Stage 2 CPU summary already represents those cells.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

#: What makes a row a distinct run. Every axis a stage can sweep has to be in
#: here, or rows that differ only on a missing axis collapse into one and the
#: survivor is whichever file happened to be read last.
#:
#: This bit once: with the Stage 2 key alone, Stage 3's 305 rows merged down to
#: 28, because its eleven feature-map arms share (mode, r, kv, steps, seed) and
#: only differ on feature_map and d_model. The output looked like a clean result
#: for a single arm rather than like data loss. Anything a later stage adds as a
#: config axis belongs here on the day it is added.
KEY = (
    "mode",
    "redundancy_r",
    "num_kv_pairs",
    "steps",
    "seed",
    "feature_map",  # Stage 3
    "d_model",      # Stage 3's matched-state control arms
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    # Natural sort: a plain sort gives stage2_10 before stage2_2, which makes the
    # merged file's row order arbitrary to read. Every digit run is kept as its
    # own component, because names carry a seed as well as a task index
    # ("stage2_s1_7"); concatenating the digits would make that 17 and collide
    # with seed 0's task 17.
    def task_index(path: Path) -> tuple[tuple[int, ...], str]:
        return (tuple(int(g) for g in re.findall(r"\d+", path.stem)), path.stem)

    parts = sorted(args.parts_dir.glob("*.csv"), key=task_index)
    if not parts:
        raise SystemExit(f"no CSVs found in {args.parts_dir}")

    rows: dict[tuple[str, ...], dict[str, str]] = {}
    fieldnames: list[str] = []
    seen: set[str] = set()
    for part in parts:
        with part.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                # Union of columns, in first-seen order. Taking the first row's
                # header instead would drop every metric that only appears in
                # later, wider rows -- and DictWriter raises rather than
                # truncating, so this fails loudly instead of losing data.
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
                rows[tuple(row.get(k, "") for k in KEY)] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows.values())

    widths = {len(r) for r in rows.values()}
    note = f", {len(fieldnames)} columns unioned from row widths {sorted(widths)}"
    print(f"merged {len(parts)} files -> {args.out} ({len(rows)} unique runs{note})")


if __name__ == "__main__":
    main()

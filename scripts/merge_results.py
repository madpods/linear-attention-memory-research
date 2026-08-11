"""Merge per-task result CSVs into one file.

The Slurm array gives each task its own CSV because concurrent appends to a
shared file interleave mid-row. This stitches them back together.

    python scripts/merge_results.py results/parts results/stage2.csv

Duplicate configurations (same mode / r / kv / steps / seed) keep the last
occurrence, so re-running a failed array task overrides its earlier partial row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEY = ("mode", "redundancy_r", "num_kv_pairs", "steps", "seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    parts = sorted(args.parts_dir.glob("*.csv"))
    if not parts:
        raise SystemExit(f"no CSVs found in {args.parts_dir}")

    rows: dict[tuple[str, ...], dict[str, str]] = {}
    fieldnames: list[str] = []
    for part in parts:
        with part.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not fieldnames:
                    fieldnames = list(row)
                rows[tuple(row.get(k, "") for k in KEY)] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.values())

    print(f"merged {len(parts)} files -> {args.out} ({len(rows)} unique runs)")


if __name__ == "__main__":
    main()

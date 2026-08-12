"""Merging per-task sweep CSVs.

The case that matters is schema drift between part files. ``evaluate()`` reports a
metric only when its denominator is non-zero, so an ``r=0`` run omits
``final_accuracy_redundant`` / ``final_accuracy_shared`` while an ``r>0`` run
includes them. Taking the header from the first part file therefore raises once a
wider row shows up -- which is invisible until someone merges a grid that spans
both, i.e. not until the first full sweep.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_results.py"


def load_script():
    spec = importlib.util.spec_from_file_location("merge_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_part(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(monkeypatch, parts_dir: Path, out: Path) -> None:
    module = load_script()
    monkeypatch.setattr(sys, "argv", ["merge_results.py", str(parts_dir), str(out)])
    module.main()


def base_row(**over):
    row = {
        "mode": "delta",
        "redundancy_r": "0.0",
        "num_kv_pairs": "4",
        "steps": "1500",
        "seed": "0",
        "final_accuracy": "0.5",
    }
    row.update(over)
    return row


def test_unions_columns_across_narrow_and_wide_parts(tmp_path, monkeypatch):
    """An r=0 part (narrow) merged with an r>0 part (wide) must not raise.

    Order matters: the narrow file sorts first, so the header is established
    before the wider row is seen. That is exactly the real failure.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(parts / "stage2_0.csv", [base_row()])
    write_part(
        parts / "stage2_1.csv",
        [
            base_row(
                redundancy_r="0.5",
                final_accuracy_redundant="0.4",
                final_accuracy_shared="0.3",
            )
        ],
    )

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == 2
    header = merged[0].keys()
    assert "final_accuracy_redundant" in header
    assert "final_accuracy_shared" in header

    narrow = next(r for r in merged if r["redundancy_r"] == "0.0")
    wide = next(r for r in merged if r["redundancy_r"] == "0.5")
    # Absent in the source, so empty here -- which is how the Stage 2 CPU
    # summary already represents an r=0 redundant cell.
    assert narrow["final_accuracy_redundant"] == ""
    assert wide["final_accuracy_redundant"] == "0.4"


def test_wide_part_first_also_works(tmp_path, monkeypatch):
    """The reverse order must behave identically, not merely not crash."""
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(
        parts / "stage2_0.csv",
        [base_row(redundancy_r="0.5", final_accuracy_redundant="0.4")],
    )
    write_part(parts / "stage2_1.csv", [base_row(num_kv_pairs="8")])

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == 2
    assert all("final_accuracy_redundant" in r for r in merged)


def test_last_occurrence_wins_for_a_repeated_configuration(tmp_path, monkeypatch):
    """A requeued task overrides its earlier partial row."""
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(parts / "stage2_0.csv", [base_row(final_accuracy="0.1")])
    write_part(parts / "stage2_1.csv", [base_row(final_accuracy="0.9")])

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == 1, "same mode/r/kv/steps/seed should collapse to one row"
    assert merged[0]["final_accuracy"] == "0.9"


def test_parts_are_read_in_task_order_not_lexicographic(tmp_path, monkeypatch):
    """stage2_10 must not sort before stage2_2.

    Only affects readability of the merged file, but a 75-task array is exactly
    where lexicographic order stops being reasonable to scan.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    for i in (0, 2, 10):
        write_part(parts / f"stage2_{i}.csv", [base_row(num_kv_pairs=str(i))])

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [r["num_kv_pairs"] for r in merged] == ["0", "2", "10"]


def test_seeded_part_names_do_not_collide_or_misorder(tmp_path, monkeypatch):
    """Seed replication names parts stage2_s<seed>_<idx>.

    Concatenating the digits would turn "stage2_s1_7" into 17 and collide with
    seed 0's task 17, so each digit run is a separate sort component. The rows
    themselves are keyed on seed, so both seeds must survive as distinct rows.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(parts / "stage2_s0_17.csv", [base_row(seed="0", num_kv_pairs="16")])
    write_part(parts / "stage2_s1_7.csv", [base_row(seed="1", num_kv_pairs="8")])
    write_part(parts / "stage2_s0_2.csv", [base_row(seed="0", num_kv_pairs="4")])

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == 3, "seeds must not collapse into one another"
    # seed 0 task 2, then seed 0 task 17, then seed 1 task 7.
    assert [(r["seed"], r["num_kv_pairs"]) for r in merged] == [
        ("0", "4"), ("0", "16"), ("1", "8"),
    ]


def test_missing_parts_dir_is_a_clean_error(tmp_path, monkeypatch):
    empty = tmp_path / "parts"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no CSVs found"):
        run(monkeypatch, empty, tmp_path / "merged.csv")


def test_rows_differing_only_on_a_stage3_axis_are_not_collapsed(tmp_path, monkeypatch):
    """The Stage 3 arms share (mode, r, kv, steps, seed).

    With only the Stage 2 key, eleven feature-map arms merged into ONE row and the
    survivor was whichever part file was read last -- 305 rows became 28, and the
    output looked like a clean single-arm result rather than like data loss.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    arms = [
        ("identity", "64"), ("elu", "64"), ("relu", "64"),
        ("dpfp1", "64"), ("dpfp2", "64"), ("dpfp3", "64"), ("dpfp4", "64"),
        ("identity", "128"), ("identity", "256"),
        ("identity", "384"), ("identity", "512"),
    ]
    for i, (fm, dm) in enumerate(arms):
        write_part(
            parts / f"stage3_s0_{i}.csv",
            [base_row(feature_map=fm, d_model=dm, final_accuracy=f"0.{i:02d}")],
        )

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == len(arms), (
        f"expected {len(arms)} distinct arms, got {len(merged)} -- rows collapsed"
    )
    assert {(r["feature_map"], r["d_model"]) for r in merged} == set(arms)


def test_the_wide_control_is_distinguished_from_the_phi_arm(tmp_path, monkeypatch):
    """identity/d64 and identity/d256 share a feature_map name.

    d_model has to be in the key too, or the matched-state control arms -- which
    are the ones that actually decide whether phi beats simply widening the
    projections -- merge into the d_k=16 baseline.
    """
    parts = tmp_path / "parts"
    parts.mkdir()
    write_part(parts / "stage3_s0_0.csv",
               [base_row(feature_map="identity", d_model="64", final_accuracy="0.44")])
    write_part(parts / "stage3_s0_1.csv",
               [base_row(feature_map="identity", d_model="256", final_accuracy="0.99")])

    out = tmp_path / "merged.csv"
    run(monkeypatch, parts, out)

    merged = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert len(merged) == 2, "the wide control collapsed into the baseline"
    assert {r["d_model"] for r in merged} == {"64", "256"}

"""Appending result rows to a CSV.

The bug worth a test: rows do not all have the same columns. :func:`evaluate`
reports a metric only when its denominator is non-zero, so an ``r=0`` run has no
``final_accuracy_redundant`` while an ``r>0`` run does. Building the writer from
the incoming row's own key order wrote a 34-value row into a 36-column file,
shifting every column after ``final_accuracy``. CSV cannot detect that, so it
corrupted the timing columns silently -- found only when a downstream float()
tripped over a JSON-encoded history that had landed in ``tokens_per_sec``.
"""

from __future__ import annotations

import csv

from lamr.train import _append_csv

WIDE = {
    "mode": "delta",
    "redundancy_r": 0.5,
    "final_accuracy": 0.90,
    "final_accuracy_redundant": 0.80,
    "final_accuracy_non_redundant": 0.70,
    "num_parameters": 133864,
    "wall_clock_sec": 69.0,
    "tokens_per_sec": 180504,
    "history": [],
}
NARROW = {
    "mode": "delta",
    "redundancy_r": 0.0,
    "final_accuracy": 0.98,
    "final_accuracy_non_redundant": 0.98,
    "num_parameters": 133864,
    "wall_clock_sec": 70.0,
    "tokens_per_sec": 181000,
    "history": [],
}


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_narrow_row_appended_to_wide_file_stays_aligned(tmp_path):
    path = tmp_path / "out.csv"
    _append_csv(path, WIDE)
    _append_csv(path, NARROW)

    rows = read(path)
    assert len(rows) == 2
    narrow = rows[1]
    # The column that used to absorb the shift.
    assert narrow["tokens_per_sec"] == "181000"
    assert narrow["wall_clock_sec"] == "70.0"
    assert narrow["num_parameters"] == "133864"
    assert narrow["final_accuracy"] == "0.98"
    # Absent from the row, so blank -- not borrowed from the next column.
    assert narrow["final_accuracy_redundant"] == ""


def test_wide_row_appended_to_narrow_file_widens_the_file(tmp_path):
    """The reverse order needs the file rewritten; appending cannot add a column."""
    path = tmp_path / "out.csv"
    _append_csv(path, NARROW)
    _append_csv(path, WIDE)

    rows = read(path)
    assert len(rows) == 2
    assert "final_accuracy_redundant" in rows[0]
    assert rows[0]["final_accuracy_redundant"] == ""      # the earlier narrow row
    assert rows[1]["final_accuracy_redundant"] == "0.8"   # the new wide row
    # And the pre-existing row's other columns did not shift while widening.
    assert rows[0]["tokens_per_sec"] == "181000"
    assert rows[0]["final_accuracy"] == "0.98"


def test_history_is_json_encoded_and_stays_in_its_own_column(tmp_path):
    path = tmp_path / "out.csv"
    _append_csv(path, {**WIDE, "history": [{"step": 250, "loss": 1.5}]})
    row = read(path)[0]
    assert row["history"] == '[{"step": 250, "loss": 1.5}]'
    assert row["tokens_per_sec"] == "180504"


def test_repeated_appends_do_not_duplicate_the_header(tmp_path):
    path = tmp_path / "out.csv"
    for _ in range(3):
        _append_csv(path, NARROW)
    with path.open(encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.startswith("mode,")) == 1
    assert len(read(path)) == 3

"""Effective capacity: the kv at which recall crosses the threshold.

Snapping this to a grid point destroyed the Stage 2 result. At 8000 steps every
mode's 95% crossing falls inside the kv 32->48 gap, so a snapped metric read a
flat 32 at every redundancy level and the ~24% capacity loss from r=0 to r=0.9
was invisible. These cases pin the interpolation and its edges.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_stage2.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_stage2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load()
capacity = analyze.effective_capacity


def test_interpolates_between_grid_points():
    """The real delta curve at r=0. Snapped this reads 32; the crossing is 44."""
    points = {4: 0.998, 8: 1.0, 16: 0.999, 32: 0.983, 48: 0.939, 64: 0.900, 96: 0.808}
    assert capacity(points) == "44.0"


def test_resolves_the_redundancy_effect_a_snapped_metric_hides():
    """r=0 and r=0.9 both snap to 32 but differ by ~24% interpolated."""
    at_r0 = {4: 0.998, 8: 1.0, 16: 0.999, 32: 0.983, 48: 0.939, 64: 0.900, 96: 0.808}
    at_r9 = {4: 0.997, 8: 0.992, 16: 0.976, 32: 0.954, 48: 0.907, 64: 0.819, 96: 0.629}
    lo, hi = float(capacity(at_r9)), float(capacity(at_r0))
    assert hi > lo, "capacity must fall with redundancy"
    assert (hi - lo) / hi > 0.2, f"expected a >20% drop, got {(hi - lo) / hi:.1%}"


def test_takes_the_last_crossing_when_the_curve_is_not_monotone():
    """linear at r=0 dips below 95% at kv=4, recovers at kv=8, then falls.

    An early dip at a trivially small kv is a training artifact, not a ceiling,
    so the capacity is the LAST downward crossing. Taking the first would report
    ~4 here instead of ~8.
    """
    points = {4: 0.937, 8: 0.951, 16: 0.829, 32: 0.285, 48: 0.096}
    result = float(capacity(points))
    assert 8 <= result < 16, f"expected the kv=8->16 crossing, got {result}"


def test_never_reaching_the_threshold_is_a_lower_bound():
    points = {4: 0.90, 8: 0.50, 16: 0.10}
    assert capacity(points) == "< 4"


def test_never_dropping_below_the_threshold_is_an_upper_bound():
    """Claiming a capacity outside the measured range would overstate the data."""
    points = {4: 0.999, 8: 0.998, 16: 0.997}
    assert capacity(points) == ">= 16"


def test_exactly_at_threshold_counts_as_passing():
    points = {4: 0.99, 8: 0.95, 16: 0.80}
    assert float(capacity(points)) == 8.0


def test_empty_is_not_a_crash():
    assert capacity({}) == "n/a"


def test_flat_span_does_not_divide_by_zero():
    """Equal values either side cannot happen with a strict crossing, but the
    guard exists so a future threshold change cannot turn it into a ZeroDivision."""
    points = {4: 0.95, 8: 0.95, 16: 0.10}
    assert float(capacity(points)) == 8.0

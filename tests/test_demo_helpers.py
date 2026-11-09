"""Tests for the small render helpers shared across interactive_demo scenarios.

The demo currently inlines the same ``_sim_bar`` closure five times in
five different scenarios. This file pins the contract of the extracted
module-level helper so any future tweak to the colour bands or width
defaults gets caught immediately.
"""

from __future__ import annotations

import re

import pytest

from scripts.demo_helpers import format_observer_sample, sim_bar


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — pure-Python helpers, no Neo4j."""
    yield


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _filled_count(s: str) -> int:
    return _strip(s).count("█")


def _empty_count(s: str) -> int:
    return _strip(s).count("░")


class TestSimBarShape:
    def test_default_width_is_20(self):
        bar = sim_bar(0.5)
        assert _filled_count(bar) + _empty_count(bar) == 20

    def test_custom_width_respected(self):
        bar = sim_bar(0.5, width=40)
        assert _filled_count(bar) + _empty_count(bar) == 40

    def test_zero_similarity_renders_empty(self):
        bar = sim_bar(0.0)
        assert _filled_count(bar) == 0
        assert _empty_count(bar) == 20

    def test_full_similarity_renders_full(self):
        bar = sim_bar(1.0)
        assert _filled_count(bar) == 20
        assert _empty_count(bar) == 0


class TestSimBarClamp:
    """Cosine similarities >1.0 or <0 must not break the layout."""

    def test_above_one_clamps_to_full(self):
        bar = sim_bar(1.7, width=10)
        assert _filled_count(bar) == 10
        assert _empty_count(bar) == 0

    def test_below_zero_clamps_to_empty(self):
        bar = sim_bar(-0.3, width=10)
        assert _filled_count(bar) == 0
        assert _empty_count(bar) == 10


class TestSimBarColours:
    """Three colour bands: green > 0.4, yellow (0.2, 0.4], red <= 0.2."""

    def test_high_similarity_is_green(self):
        bar = sim_bar(0.6)
        assert "\x1b[32m" in bar       # GREEN

    def test_mid_similarity_is_yellow(self):
        bar = sim_bar(0.3)
        assert "\x1b[33m" in bar       # YELLOW

    def test_low_similarity_is_red(self):
        bar = sim_bar(0.1)
        assert "\x1b[31m" in bar       # RED

    def test_boundary_at_0_4_is_yellow(self):
        # > 0.4 is strict; sim == 0.4 falls into the yellow band.
        bar = sim_bar(0.4)
        assert "\x1b[33m" in bar
        assert "\x1b[32m" not in bar

    def test_boundary_at_0_2_is_red(self):
        # > 0.2 is strict; sim == 0.2 falls into the red band.
        bar = sim_bar(0.2)
        assert "\x1b[31m" in bar
        assert "\x1b[33m" not in bar


class TestFormatObserverSample:
    """The observer's sample() returns a Unix-epoch float plus useful counts.

    The first iteration of the demo printed ``sampled_at`` raw, which surfaces
    as e.g. ``1714572345.612`` — readable for nobody. ``new_anomalies`` and
    ``clusters_sampled`` were silently discarded even though they are what
    makes the sample meaningful in the demo.
    """

    def test_formats_unix_epoch_as_iso(self):
        # 2026-05-09 12:00:00 UTC = 1778328000
        out = format_observer_sample(
            "Sample #1",
            {"sampled_at": 1778328000.0, "new_anomalies": 0,
             "clusters_sampled": 2, "regions_sampled": 1},
        )
        assert "2026-05-09" in out
        # And not the raw float
        assert "1778328000" not in out

    def test_includes_anomaly_count(self):
        out = format_observer_sample(
            "Sample #2",
            {"sampled_at": 1778328000.0, "new_anomalies": 3,
             "clusters_sampled": 4, "regions_sampled": 1, "total_anomalies": 5},
        )
        assert "3" in out  # new anomalies this sample
        assert "5" in out  # total anomalies running

    def test_includes_clusters_sampled(self):
        out = format_observer_sample(
            "Sample #1",
            {"sampled_at": 1778328000.0, "new_anomalies": 0,
             "clusters_sampled": 7, "regions_sampled": 2},
        )
        assert "7" in out  # cluster count visible
        assert "2" in out  # region count visible

    def test_includes_label(self):
        out = format_observer_sample(
            "Sample #1",
            {"sampled_at": 1778328000.0, "new_anomalies": 0,
             "clusters_sampled": 2, "regions_sampled": 1},
        )
        assert "Sample #1" in out

    def test_handles_missing_fields_gracefully(self):
        # Older API versions or test doubles may omit fields.
        out = format_observer_sample("Sample #1", {})
        # Must not raise; should produce something usable.
        assert "Sample #1" in out
        assert isinstance(out, str)
        assert len(out) > 0

    def test_handles_none_sample(self):
        # A failed sample call may pass through as None.
        out = format_observer_sample("Sample #1", None)
        assert "Sample #1" in out
        assert isinstance(out, str)

    def test_includes_duration_when_present(self):
        out = format_observer_sample(
            "Sample #1",
            {"sampled_at": 1778328000.0, "sample_duration_ms": 12.34,
             "new_anomalies": 0, "clusters_sampled": 1, "regions_sampled": 1},
        )
        assert "12.3" in out  # duration shows in ms with reasonable precision

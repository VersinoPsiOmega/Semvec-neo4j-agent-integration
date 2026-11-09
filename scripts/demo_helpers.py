"""Pure-Python render helpers shared across interactive_demo scenarios.

Kept separate from ``interactive_demo.py`` so they can be unit-tested
without spinning up the demo's environment guard (which requires a
fully populated ``.env`` and a reachable Neo4j).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

# ANSI colour codes used by the demo. Duplicated here on purpose so the
# helpers stay importable even before the main demo module loads.
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_FILLED = "█"
_EMPTY = "░"


def sim_bar(sim: float, width: int = 20) -> str:
    """Render a cosine-similarity bar with three semantic colour bands.

    Bands (matching the demo's existing visual language):

    - ``sim > 0.4``     → green  (on-topic)
    - ``0.2 < sim ≤ 0.4`` → yellow (shifting)
    - ``sim ≤ 0.2``     → red    (off-topic / drifted)

    Cosine similarity can drift slightly outside ``[0, 1]`` due to
    floating-point normalisation; the bar clamps so the layout never
    breaks.
    """
    clamped = max(0.0, min(sim, 1.0))
    filled = int(clamped * width)
    bar = f"{_FILLED * filled}{_EMPTY * (width - filled)}"
    if sim > 0.4:
        colour = _GREEN
    elif sim > 0.2:
        colour = _YELLOW
    else:
        colour = _RED
    return f"{colour}{bar}{_RESET}"


def format_observer_sample(label: str, sample: Mapping[str, Any] | None) -> str:
    """Render an ``observer_sample()`` result as a single human-readable line.

    The raw payload from ``semvec.observer_sample`` returns ``sampled_at``
    as a Unix-epoch float (e.g. ``1714572345.612``) plus a handful of
    counts the demo previously dropped on the floor (``new_anomalies``,
    ``clusters_sampled``, ``regions_sampled``, ``total_anomalies``,
    ``sample_duration_ms``).

    The previous demo-side rendering — ``f"sampled at {sample.get('sampled_at')}"``
    — printed the raw float and threw away every metric that made the
    sample interesting. This helper formats a tight one-liner that names
    the sample, shows an ISO timestamp, and includes the counts that
    actually move during a drift event.

    Robust to ``None`` and to missing keys (older API versions or test
    doubles): always returns a non-empty string starting with ``label``.
    """
    if not sample:
        return f"{label}: (no sample data)"

    sampled_at = sample.get("sampled_at")
    if isinstance(sampled_at, (int, float)) and sampled_at > 0:
        ts = (
            datetime.fromtimestamp(float(sampled_at), tz=timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S UTC")
        )
    else:
        ts = "unknown time"

    parts: list[str] = [label, "@", ts]

    clusters_sampled = sample.get("clusters_sampled")
    regions_sampled = sample.get("regions_sampled")
    if clusters_sampled is not None or regions_sampled is not None:
        parts.append(
            f"— {clusters_sampled or 0} clusters / {regions_sampled or 0} regions"
        )

    new_anomalies = sample.get("new_anomalies")
    total_anomalies = sample.get("total_anomalies")
    if new_anomalies is not None or total_anomalies is not None:
        anomaly_seg = f"new anomalies: {new_anomalies if new_anomalies is not None else 0}"
        if total_anomalies is not None:
            anomaly_seg += f" (total {total_anomalies})"
        parts.append(f"— {anomaly_seg}")

    duration = sample.get("sample_duration_ms")
    if isinstance(duration, (int, float)):
        parts.append(f"[{duration:.1f} ms]")

    return " ".join(parts)

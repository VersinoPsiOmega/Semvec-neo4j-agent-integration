# ABOUTME: The one drift rule of the integration: Semvec's verdict OR (drift_score >= threshold AND sim <= ceiling).
# ABOUTME: DriftGate adds an optional per-session cooldown; DriftRule.from_env reads DRIFT_* settings.

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T", int, float)


@dataclass(frozen=True)
class DriftRule:
    """Thresholds that decide whether a turn counts as a topic switch.

    A turn is flagged when Semvec itself reports ``drift_detected`` or when
    ``drift_score >= drift_threshold`` and ``top_similarity <= sim_ceiling``.
    The similarity condition filters out sub-topic shifts that stay close to
    the accumulated context. ``cooldown_turns`` suppresses that many turns
    after a flagged one (0 = off); it is applied by :class:`DriftGate`.
    """

    drift_threshold: float
    sim_ceiling: float
    cooldown_turns: int = 0

    def __post_init__(self) -> None:
        if self.cooldown_turns < 0:
            raise ValueError(f"cooldown_turns must be >= 0, got {self.cooldown_turns}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> DriftRule:
        """Build the rule from DRIFT_THRESHOLD, DRIFT_SIM_CEILING and DRIFT_COOLDOWN_TURNS.

        The two thresholds are required: they are calibrated per LLM, so a
        baked-in fallback would silently apply another model's calibration.
        DRIFT_COOLDOWN_TURNS is optional and defaults to 0 (cooldown off).
        """
        required = ("DRIFT_THRESHOLD", "DRIFT_SIM_CEILING")
        missing = [name for name in required if not environ.get(name)]
        if missing:
            raise ValueError(
                f"Missing drift settings: {', '.join(missing)}. "
                "Set them in .env (see .env.example)."
            )
        return cls(
            drift_threshold=_parse(environ, "DRIFT_THRESHOLD", float),
            sim_ceiling=_parse(environ, "DRIFT_SIM_CEILING", float),
            cooldown_turns=_parse(environ, "DRIFT_COOLDOWN_TURNS", int) if environ.get("DRIFT_COOLDOWN_TURNS") else 0,
        )

    def matches(self, drift_score: float, top_similarity: float, drift_detected: bool) -> bool:
        """Apply the rule to one turn, ignoring the cooldown."""
        if drift_detected:
            return True
        return drift_score >= self.drift_threshold and top_similarity <= self.sim_ceiling


@dataclass
class DriftGate:
    """Applies a :class:`DriftRule` turn by turn, with its cooldown, per session."""

    rule: DriftRule
    _turns_since_flag: dict[str, int] = field(default_factory=dict)

    def check(self, session_id: str, drift_score: float, top_similarity: float, drift_detected: bool) -> bool:
        """Return True when this turn of ``session_id`` is flagged as drift.

        Every call counts as one turn of that session, whether it matches or
        not; a match within ``cooldown_turns`` turns of the last flag is
        suppressed.
        """
        turns_since_flag = self._turns_since_flag.get(session_id)
        in_cooldown = turns_since_flag is not None and turns_since_flag < self.rule.cooldown_turns
        flagged = not in_cooldown and self.rule.matches(drift_score, top_similarity, drift_detected)
        if flagged:
            self._turns_since_flag[session_id] = 0
        elif turns_since_flag is not None:
            self._turns_since_flag[session_id] = turns_since_flag + 1
        return flagged

    def forget(self, session_id: str) -> None:
        """Drop the cooldown state of a session that was reset or deleted."""
        self._turns_since_flag.pop(session_id, None)


def _parse(environ: Mapping[str, str], name: str, convert: Callable[[str], T]) -> T:
    """Convert one environment value, naming the variable when it is malformed."""
    try:
        return convert(environ[name])
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {environ[name]!r}") from exc

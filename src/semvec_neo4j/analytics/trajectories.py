"""Temporal path analysis on semantic state evolution.

Analyzes how semantic states evolve over time, identifies drift points,
and computes trajectory stability metrics.
"""

from __future__ import annotations

import numpy as np

from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore


class TrajectoryAnalyzer:
    def __init__(self, state_store: Neo4jStateStore):
        self._state_store = state_store

    def get_trajectory(self, session_id: str, steps: int = 20) -> list[dict]:
        """Get the semantic state trajectory for a session."""
        history = self._state_store.get_state_history(session_id, limit=steps)
        return [
            {
                "step": state.step,
                "beta": state.beta,
                "mean_similarity": state.mean_similarity,
                "variance": state.variance,
                "cosine_similarity": sim,
                "timestamp": state.timestamp.isoformat() if state.timestamp else None,
            }
            for state, sim in history
        ]

    def find_drift_points(
        self, session_id: str, threshold: float = 0.3
    ) -> list[dict]:
        """Find points in the trajectory where cosine similarity dropped below threshold."""
        history = self._state_store.get_state_history(session_id, limit=100)
        drift_points = []
        for state, sim in history:
            if sim < (1.0 - threshold):
                drift_points.append({
                    "step": state.step,
                    "cosine_similarity": sim,
                    "drop": 1.0 - sim,
                    "timestamp": state.timestamp.isoformat() if state.timestamp else None,
                })
        return drift_points

    def compute_stability(self, session_id: str) -> float:
        """Compute overall trajectory stability (0=unstable, 1=perfectly stable).

        Based on mean and variance of cosine similarities in the trajectory.
        """
        history = self._state_store.get_state_history(session_id, limit=50)
        if len(history) < 2:
            return 1.0

        similarities = [sim for _, sim in history]
        mean_sim = float(np.mean(similarities))
        var_sim = float(np.var(similarities))

        # Stability = high mean similarity + low variance
        stability = mean_sim * (1.0 - min(var_sim * 10, 1.0))
        return max(0.0, min(1.0, stability))

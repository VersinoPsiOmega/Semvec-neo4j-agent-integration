"""Drift detection bridge: Semvec runtime → Neo4j Graph.

Semvec (the bundled Python package) does ALL computation — this module
is a thin bridge that:
  1. Calls SemvecClient.run (with inline-store of previous LLM response)
  2. Mirrors the results into Neo4j for graph queries

All drift signals come directly from Semvec:
  - drift_score     (0.0–1.0)
  - drift_detected  (True when drift_score >= threshold)
  - drift_phase     (stable / shifting / drifted)
  - top_similarity  (cosine between query and accumulated context)
"""

from __future__ import annotations

from typing import Optional

from semvec_neo4j.core.semvec_client import SemvecClient
from semvec_neo4j.persistence.models import (
    DriftEvent,
    DriftPhase,
    DriftSeverity,
    Phase,
    PhaseName,
    SemanticState,
)
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_phase_store import Neo4jPhaseStore
from semvec_neo4j.persistence.neo4j_drift_event_store import Neo4jDriftEventStore


_DRIFT_PHASE_MAP = {
    "stable": DriftPhase.STABLE,
    "shifting": DriftPhase.SHIFTING,
    "drifted": DriftPhase.DRIFTED,
}


def _classify_severity(drift_score: float) -> DriftSeverity:
    if drift_score < 0.3:
        return DriftSeverity.LOW
    elif drift_score < 0.5:
        return DriftSeverity.MEDIUM
    elif drift_score < 0.7:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


class DriftDetector:
    """Thin bridge: Semvec runtime → Neo4j persistence.

    Semvec does all drift computation. This class just calls the
    in-process client, reads the response, and mirrors the results
    into Neo4j nodes and relationships.
    """

    def __init__(
        self,
        semvec_client: SemvecClient,
        state_store: Neo4jStateStore,
        phase_store: Neo4jPhaseStore,
        drift_event_store: Neo4jDriftEventStore,
    ):
        self._semvec = semvec_client
        self._state_store = state_store
        self._phase_store = phase_store
        self._drift_event_store = drift_event_store

        # A DriftEvent node is persisted when the client's ``drift`` verdict
        # is set: its DriftRule (Semvec's verdict OR the combined
        # drift_score/similarity rule, with optional cooldown) when the client
        # has one, otherwise Semvec's own binary drift_detected (>= 0.5).

        self._semvec_sessions: dict[str, str] = {}
        self._prev_responses: dict[str, str | None] = {}

    def process_input(self, session_id: str, message: str) -> dict:
        """Send message to Semvec, mirror results to Neo4j."""
        semvec_session_id = self._semvec_sessions.get(session_id)
        prev_response = self._prev_responses.pop(session_id, None)

        # === Call Semvec (inline-store pattern) ===
        semvec_result = self._semvec.run(
            message=message,
            session_id=semvec_session_id,
            response=prev_response,
        )

        if session_id not in self._semvec_sessions:
            self._semvec_sessions[session_id] = semvec_result["session_id"]
        semvec_session_id = semvec_result["session_id"]

        # Read Semvec signals directly
        drift_score = semvec_result.get("drift_score", 0.0)
        drift_detected = semvec_result.get("drift_detected", False)
        drift = semvec_result.get("drift", drift_detected)
        drift_phase_str = semvec_result.get("drift_phase", "stable")
        context = semvec_result.get("context", "")
        top_similarity = semvec_result.get("top_similarity", 0.0)
        short_circuit = semvec_result.get("short_circuit", False)

        drift_phase = _DRIFT_PHASE_MAP.get(drift_phase_str, DriftPhase.STABLE)
        severity = _classify_severity(drift_score)

        # === Mirror to Neo4j: Semantic State ===
        prev_state = self._state_store.get_current_state(session_id)
        step = (prev_state.step + 1) if prev_state else 0

        state = SemanticState(
            step=step,
            beta=drift_score,
            mean_similarity=top_similarity,
            variance=drift_score,
            vector=[],
        )
        self._state_store.append_state(
            session_id, state, cosine_similarity=top_similarity
        )

        # === Mirror to Neo4j: Phase ===
        self._update_phase(session_id, drift_phase, drift_score, top_similarity)

        # === Mirror to Neo4j: Drift Event ===
        if drift:
            event = DriftEvent(
                drift_score=drift_score,
                drift_phase=drift_phase,
                topic_switch=True,
                cosine_drop=max(0.0, 1.0 - top_similarity),
                mean_sim=top_similarity,
                variance=drift_score,
                severity=severity,
            )
            self._drift_event_store.create_drift_event(
                session_id, state.state_id, event
            )

        return {
            "context": context,
            "drift_score": drift_score,
            "drift_detected": drift_detected,
            "drift": drift,
            "drift_phase": drift_phase,
            "top_similarity": top_similarity,
            "short_circuit": short_circuit,
            "severity": severity,
            "state_id": state.state_id,
            "step": step,
            "semvec_session_id": semvec_session_id,
        }

    def store_response(self, session_id: str, response: str) -> dict:
        """Buffer LLM response for inline-store on next /run call."""
        self._prev_responses[session_id] = response
        semvec_session_id = self._semvec_sessions.get(session_id)
        if not semvec_session_id:
            return {"error": "No Semvec session — call process_input first"}
        return self._semvec.store(semvec_session_id, response)

    def get_semvec_session_id(self, session_id: str) -> Optional[str]:
        return self._semvec_sessions.get(session_id)

    def _update_phase(
        self, session_id: str, drift_phase: DriftPhase,
        drift_score: float, similarity: float,
    ) -> None:
        current = self._phase_store.get_current_phase(session_id)
        new_phase = self._infer_phase(drift_phase, drift_score, similarity, current)
        if current is None or new_phase != current.name:
            self._phase_store.set_phase(session_id, Phase(
                name=new_phase,
                srs_score=similarity,
                tc_score=similarity,
                be_score=drift_score,
            ))

    @staticmethod
    def _infer_phase(
        drift_phase: DriftPhase, drift_score: float,
        similarity: float, current: Optional[Phase],
    ) -> PhaseName:
        if current is None:
            return PhaseName.INITIALIZATION
        if drift_phase == DriftPhase.DRIFTED:
            return PhaseName.INSTABILITY
        if drift_phase == DriftPhase.SHIFTING:
            return PhaseName.EXPLORATION
        if drift_score < 0.1 and similarity > 0.5:
            return PhaseName.STABILITY
        if drift_score < 0.2 and similarity > 0.3:
            return PhaseName.RESONANCE
        return PhaseName.CONVERGENCE

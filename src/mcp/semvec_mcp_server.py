"""Semvec MCP server — exposes drift detection to agent frameworks.

Architecture:
  Semvec runtime (in-process) ←→ SemvecMCPServer ←→ Neo4j (persistence + graph queries)
                                       ↕
                                MCP protocol → LangGraph, CrewAI, OpenAI, Claude, ...

Semvec computes. Neo4j persists. MCP exposes tools.
"""

from __future__ import annotations

from typing import Optional

from neo4j import Driver

from src.core.semvec_client import SemvecClient
from src.core.drift_detector import DriftDetector
from src.persistence.models import (
    AgentSession,
    DriftSeverity,
    Memory,
    MemoryTier,
)
from src.persistence.neo4j_session_store import Neo4jSessionStore
from src.persistence.neo4j_state_store import Neo4jStateStore
from src.persistence.neo4j_phase_store import Neo4jPhaseStore
from src.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from src.persistence.neo4j_memory_store import Neo4jMemoryStore


class SemvecMCPServer:
    """MCP tool server bridging the Semvec runtime and a Neo4j graph.

    Each method is an MCP tool consumable by any framework that speaks
    the Model Context Protocol.
    """

    def __init__(
        self,
        driver: Driver,
        database: str = "neo4j",
        semvec_client: Optional[SemvecClient] = None,
    ):
        self._driver = driver
        self._database = database
        if semvec_client is None:
            raise TypeError(
                "semvec_client is required: semvec >= 0.8.7 ties every session "
                "and network partition to a license subject, so the caller must "
                "construct SemvecClient(owner_subject=...) explicitly"
            )
        self._semvec = semvec_client

        self._sessions = Neo4jSessionStore(driver, database)
        self._states = Neo4jStateStore(driver, database)
        self._phases = Neo4jPhaseStore(driver, database)
        self._drift_events = Neo4jDriftEventStore(driver, database)
        self._memories = Neo4jMemoryStore(driver, database)

        self._detector = DriftDetector(
            semvec_client=self._semvec,
            state_store=self._states,
            phase_store=self._phases,
            drift_event_store=self._drift_events,
        )

    # === Session management =================================================

    def create_agent_session(
        self, agent_id: str, cluster_id: Optional[str] = None
    ) -> dict:
        """Open a fresh agent session and persist it in Neo4j."""
        session = AgentSession(agent_id=agent_id)
        result = self._sessions.create_session(session)
        return {
            "session_id": result.session_id,
            "agent_id": result.agent_id,
            "status": result.status.value,
            "created_at": result.created_at.isoformat() if result.created_at else None,
        }

    def end_agent_session(self, session_id: str) -> dict:
        """Close a session and return a summary for the audit trail."""
        self._sessions.close_session(session_id)
        phase = self._phases.get_current_phase(session_id)
        drift_count = self._drift_events.get_drift_event_count(session_id)
        state_count = self._states.get_state_count(session_id)

        return {
            "status": "closed",
            "final_phase": phase.name.value if phase else None,
            "total_drift_events": drift_count,
            "total_states": state_count,
        }

    # === Drift detection ====================================================

    def detect_drift(self, session_id: str, message: str) -> dict:
        """Run the message through Semvec, mirror the result into Neo4j.

        Semvec computes the semantic state, drift score, and compressed
        context. The Neo4j stores receive the persisted view used for
        graph queries.
        """
        result = self._detector.process_input(session_id, message)
        return {
            "context": result["context"],
            "drift_score": result["drift_score"],
            "drift_detected": result["drift_detected"],
            "drift_phase": result["drift_phase"].value,
            "severity": result["severity"].value,
            "top_similarity": result["top_similarity"],
            "short_circuit": result["short_circuit"],
            "state_id": result["state_id"],
            "step": result["step"],
        }

    def store_response(self, session_id: str, response: str) -> dict:
        """Feed the LLM answer back to Semvec so it learns the turn."""
        return self._detector.store_response(session_id, response)

    # === Phase + drift queries (Neo4j) ======================================

    def get_phase(self, session_id: str) -> dict:
        """Return the current conversation phase from Neo4j."""
        phase = self._phases.get_current_phase(session_id)
        if phase is None:
            return {"phase": None, "entered_at": None}
        return {
            "phase": phase.name.value,
            "entered_at": phase.entered_at.isoformat() if phase.entered_at else None,
            "srs_score": phase.srs_score,
            "tc_score": phase.tc_score,
            "be_score": phase.be_score,
        }

    def get_drift_score(self, session_id: str) -> dict:
        """Return the latest drift score from the persisted state chain."""
        state = self._states.get_current_state(session_id)
        if state is None:
            return {
                "drift_score": 0.0,
                "components": {
                    "topic_switch_magnitude": 0.0,
                    "mean_similarity": 1.0,
                    "variance": 0.0,
                },
            }
        return {
            "drift_score": state.variance,
            "components": {
                "topic_switch_magnitude": state.variance,
                "mean_similarity": state.mean_similarity,
                "variance": state.variance,
            },
        }

    def query_drift_history(
        self, session_id: str, limit: int = 20, min_severity: Optional[str] = None,
    ) -> list[dict]:
        """Query drift events from Neo4j."""
        sev = DriftSeverity(min_severity) if min_severity else None
        events = self._drift_events.get_drift_events(session_id, limit=limit, min_severity=sev)
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "drift_score": e.drift_score,
                "drift_phase": e.drift_phase.value,
                "topic_switch": e.topic_switch,
                "severity": e.severity.value,
            }
            for e in events
        ]

    def get_state_trajectory(self, session_id: str, steps: int = 10) -> list[dict]:
        """Return the semantic-state trajectory recorded in Neo4j."""
        history = self._states.get_state_history(session_id, limit=steps)
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

    # === Cluster operations =================================================

    def create_cluster(
        self, name: str, aggregation_mode: str = "weighted_average", coupling_factor: float = 0.0,
    ) -> dict:
        return self._semvec.create_cluster(name, aggregation_mode, coupling_factor)

    def cluster_run(self, cluster_id: str, message: str) -> dict:
        return self._semvec.cluster_run(cluster_id, message)

    def cluster_feedback(self, cluster_id: str) -> dict:
        return self._semvec.cluster_feedback(cluster_id)

    # === Region operations ==================================================

    def create_region(self, name: str, consensus_threshold: float = 0.5) -> dict:
        return self._semvec.create_region(name, consensus_threshold)

    def get_region_events(self, region_id: str, limit: int = 20) -> list[dict]:
        return self._semvec.get_region_events(region_id, limit)

    # === Global observer ====================================================

    def get_anomalies(self, limit: int = 20) -> list[dict]:
        return self._semvec.get_anomalies(limit)

    def observer_sample(self) -> dict:
        return self._semvec.observer_sample()

    # === Memory (Neo4j persistence) =========================================

    def store_memory(
        self, session_id: str, text: str, importance: float = 0.5,
        vector: Optional[list[float]] = None, tier: str = "short",
    ) -> dict:
        mem = Memory(
            tier=MemoryTier(tier), content_vector=vector or [],
            importance=importance, text_summary=text,
        )
        result = self._memories.store_memory(session_id, mem)
        return {"memory_id": result.memory_id, "tier": result.tier.value, "importance": result.importance}

    def memory_query(
        self, session_id: str, query_vector: list[float], limit: int = 5, tier: Optional[str] = None,
    ) -> list[dict]:
        t = MemoryTier(tier) if tier else None
        results = self._memories.search_similar_memories(session_id, query_vector, limit=limit, tier=t)
        return [
            {"memory_id": mem.memory_id, "text": mem.text_summary, "tier": mem.tier.value,
             "importance": mem.importance, "similarity": score}
            for mem, score in results
        ]

    def memory_consolidate(self, session_id: str) -> dict:
        short_count = self._memories.count_memories(session_id, MemoryTier.SHORT)
        promoted = 0
        if short_count > 15:
            short_mems = self._memories.get_memories_by_tier(session_id, MemoryTier.SHORT, limit=100)
            for mem in short_mems[15:]:
                self._memories.promote_memory(mem.memory_id, MemoryTier.MEDIUM)
                promoted += 1
        medium_count = self._memories.count_memories(session_id, MemoryTier.MEDIUM)
        if medium_count > 50:
            medium_mems = self._memories.get_memories_by_tier(session_id, MemoryTier.MEDIUM, limit=100)
            for mem in medium_mems[50:]:
                self._memories.promote_memory(mem.memory_id, MemoryTier.LONG)
                promoted += 1
        return {
            "consolidated": promoted,
            "short": self._memories.count_memories(session_id, MemoryTier.SHORT),
            "medium": self._memories.count_memories(session_id, MemoryTier.MEDIUM),
            "long": self._memories.count_memories(session_id, MemoryTier.LONG),
        }

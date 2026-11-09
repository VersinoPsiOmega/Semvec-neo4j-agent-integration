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

from semvec_neo4j.core.semvec_client import SemvecClient
from semvec_neo4j.core.drift_detector import DriftDetector
from semvec_neo4j.core.drift_rule import DriftRule
from semvec_neo4j.persistence.models import (
    Agent,
    AgentSession,
    AnomalyEvent,
    AnomalyType,
    ConsensusRound,
    ConsensusVote,
    DriftSeverity,
    GlobalObserver,
    LiteralFact,
    Memory,
    MemoryTier,
)
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_phase_store import Neo4jPhaseStore
from semvec_neo4j.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from semvec_neo4j.persistence.neo4j_memory_store import Neo4jMemoryStore
from semvec_neo4j.persistence.neo4j_agent_store import Neo4jAgentStore
from semvec_neo4j.persistence.neo4j_global_observer_store import (
    Neo4jGlobalObserverStore,
)
from semvec_neo4j.persistence.neo4j_anomaly_event_store import (
    Neo4jAnomalyEventStore,
)
from semvec_neo4j.persistence.neo4j_literal_fact_store import Neo4jLiteralFactStore
from semvec_neo4j.persistence.neo4j_consensus_store import Neo4jConsensusStore


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
        # An injected client brings its own drift rule; a client built here
        # takes it from DRIFT_* in the environment and fails loudly without.
        self._semvec = semvec_client or SemvecClient(drift_rule=DriftRule.from_env())

        self._sessions = Neo4jSessionStore(driver, database)
        self._states = Neo4jStateStore(driver, database)
        self._phases = Neo4jPhaseStore(driver, database)
        self._drift_events = Neo4jDriftEventStore(driver, database)
        self._memories = Neo4jMemoryStore(driver, database)
        self._agents = Neo4jAgentStore(driver, database)
        self._observers = Neo4jGlobalObserverStore(driver, database)
        self._anomaly_events = Neo4jAnomalyEventStore(driver, database)
        self._literal_facts = Neo4jLiteralFactStore(driver, database)
        self._consensus = Neo4jConsensusStore(driver, database)

        # DriftEvent nodes follow the client's drift verdict, so the persisted
        # nodes match what the demo reports and what reaches the regions.
        self._detector = DriftDetector(
            semvec_client=self._semvec,
            state_store=self._states,
            phase_store=self._phases,
            drift_event_store=self._drift_events,
        )

    @property
    def semvec_client(self) -> SemvecClient:
        """The in-process Semvec client this server drives."""
        return self._semvec

    # === Session management =================================================

    def create_agent_session(
        self, agent_id: str, cluster_id: Optional[str] = None
    ) -> dict:
        """Open a fresh agent session and persist it in Neo4j.

        Also ensures an :class:`Agent` node exists and links it to the new
        session via ``(Agent)-[:RAN]->(AgentSession)`` so sessions can be
        grouped per agent for cross-session analytics.
        """
        session = AgentSession(agent_id=agent_id)
        result = self._sessions.create_session(session)
        self._agents.ensure_agent(Agent(agent_id=agent_id, name=agent_id))
        self._agents.link_to_session(agent_id, result.session_id)
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
            "drift": result["drift"],
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

    def persist_observer(
        self,
        region_ids: Optional[list[str]] = None,
        sample_interval_seconds: float = 30.0,
    ) -> dict:
        """Create the in-process observer and mirror it into Neo4j.

        Persists a ``GlobalObserver`` node and links it to each region it
        watches via ``(GlobalObserver)-[:OBSERVES]->(Region)``.
        """
        info = self._semvec.create_observer(sample_interval_seconds, region_ids)
        observer_id = info["observer_id"]
        self._observers.create_observer(GlobalObserver(observer_id=observer_id))
        for rid in region_ids or []:
            self._observers.observe_region(observer_id, rid)
        return info

    def persist_anomalies(self, observer_id: str, limit: int = 20) -> dict:
        """Mirror the observer's detected anomalies into Neo4j.

        Each anomaly becomes an ``AnomalyEvent`` node linked to the
        detecting observer via ``(GlobalObserver)-[:DETECTED]->(AnomalyEvent)``.
        """
        self._observers.create_observer(GlobalObserver(observer_id=observer_id))
        stored = 0
        for a in self._semvec.get_anomalies(limit):
            event = AnomalyEvent(
                event_id=a["anomaly_id"],
                anomaly_type=a.get("anomaly_type", AnomalyType.SYSTEMIC_DRIFT),
                severity=a.get("severity", 0.0),
                affected_clusters=a.get("affected_cluster_ids", []),
                description=a.get("description", ""),
            )
            self._anomaly_events.create_anomaly_event(observer_id, event)
            stored += 1
        return {"observer_id": observer_id, "stored": stored}

    def get_stored_anomalies(self, observer_id: str, limit: int = 20) -> list[dict]:
        """Read AnomalyEvent nodes previously persisted for an observer."""
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "anomaly_type": e.anomaly_type.value,
                "severity": e.severity,
                "affected_clusters": e.affected_clusters,
                "description": e.description,
            }
            for e in self._anomaly_events.get_anomaly_events(observer_id, limit)
        ]

    # === Consensus (Neo4j persistence) ======================================

    def run_consensus_round(
        self,
        region_id: str,
        local_id: str,
        network_id: str,
        level: str,
        voters: dict[str, float],
        proposal_type: str,
        rationale: str,
        votes: dict[str, bool],
    ) -> dict:
        """Run one ConsensusEngine vote in Semvec and mirror it under the region.

        ``voters`` maps each voting instance to its weight, ``votes`` maps
        instances to approve (True) or reject (False). The round is stored as
        (Region)-[:HAS_CONSENSUS_ENGINE]->(ConsensusEngine)-[:PROPOSED]->
        (ConsensusProposal)-[:RECEIVED_VOTE]->(ConsensusVote), with votes of
        agent sessions linked via (AgentSession)-[:CAST]->(ConsensusVote), and
        the verdict as (ConsensusProposal)-[:DECIDED]->(ConsensusEvent).
        Returns Semvec's verdict plus the proposal's status at submission.
        """
        engine_id = self._semvec.create_consensus_engine(
            local_id=local_id, network_id=network_id, level=level
        )["engine_id"]
        for voter, weight in voters.items():
            self._semvec.register_consensus_voter(engine_id, voter, weight=weight)
        proposal = self._semvec.submit_consensus_proposal(
            engine_id, proposal_type=proposal_type, proposed_state=[0.0] * 8, rationale=rationale
        )
        for voter, approve in votes.items():
            self._semvec.vote_on_consensus(engine_id, proposal["proposal_id"], approve, voting_instance=voter)
        verdict = self._semvec.evaluate_consensus(engine_id, proposal["proposal_id"])
        self._consensus.store_round(
            region_id,
            ConsensusRound(
                engine_id=engine_id,
                level=level,
                proposal_id=proposal["proposal_id"],
                proposal_type=proposal_type,
                rationale=rationale,
                votes=[ConsensusVote(voter=v, approve=a, weight=voters.get(v, 1.0)) for v, a in votes.items()],
                accepted=verdict["accepted"],
                ratio=verdict["ratio"],
            ),
        )
        return {**verdict, "submitted_status": proposal["status"]}

    # === Literal facts (Neo4j persistence) ==================================

    def extract_and_store_facts(self, session_id: str, text: str) -> dict:
        """Extract verbatim facts from *text* and mirror them into Neo4j.

        Each fact becomes a ``LiteralFact`` node linked to its session via
        ``(AgentSession)-[:EXTRACTED]->(LiteralFact)``.
        """
        stored = 0
        for f in self._semvec.extract_facts(text):
            fact = LiteralFact(
                session_id=session_id,
                key=f["raw"],
                kind=f.get("kind", ""),
                value=str(f.get("value", "")),
                unit=f.get("unit", ""),
                id_type=f.get("id_type", ""),
            )
            self._literal_facts.store_fact(session_id, fact)
            stored += 1
        return {"session_id": session_id, "stored": stored}

    def get_stored_facts(self, session_id: str, limit: int = 50) -> list[dict]:
        """Read LiteralFact nodes previously persisted for a session."""
        return [
            {
                "fact_id": f.fact_id,
                "key": f.key,
                "kind": f.kind,
                "value": f.value,
                "unit": f.unit,
                "id_type": f.id_type,
                "importance": f.importance,
            }
            for f in self._literal_facts.get_facts(session_id, limit)
        ]

    # === Agent (cross-session) ==============================================

    def get_agent_sessions(self, agent_id: str) -> list[dict]:
        """List every session a given agent has run."""
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in self._agents.get_sessions(agent_id)
        ]

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

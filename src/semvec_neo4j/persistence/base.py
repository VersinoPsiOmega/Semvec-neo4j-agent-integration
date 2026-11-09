"""Abstract base classes for the Semvec persistence stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from semvec_neo4j.persistence.models import (
    Agent,
    AgentSession,
    AnomalyEvent,
    Cluster,
    ConsensusEvent,
    ConsensusRound,
    DriftEvent,
    DriftSeverity,
    GlobalObserver,
    LiteralFact,
    Memory,
    MemoryTier,
    Phase,
    PhaseName,
    Region,
    SemanticState,
    SessionStatus,
)


class SessionStore(ABC):
    """Manages AgentSession lifecycle."""

    @abstractmethod
    def create_session(self, session: AgentSession) -> AgentSession:
        """Create a new agent session."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID."""

    @abstractmethod
    def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status and last_active timestamp."""

    @abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Close a session."""

    @abstractmethod
    def list_active_sessions(self, agent_id: Optional[str] = None) -> list[AgentSession]:
        """List active sessions, optionally filtered by agent_id."""


class StateStore(ABC):
    """Manages SemanticState persistence as a linked chain."""

    @abstractmethod
    def append_state(
        self,
        session_id: str,
        state: SemanticState,
        cosine_similarity: float,
    ) -> SemanticState:
        """Append a new state to the session's state chain.

        Creates the node, links it to the previous CURRENT_STATE via STATE_HISTORY,
        and moves the CURRENT_STATE pointer.
        """

    @abstractmethod
    def get_current_state(self, session_id: str) -> Optional[SemanticState]:
        """Get the current (latest) semantic state for a session."""

    @abstractmethod
    def get_state_history(
        self, session_id: str, limit: int = 50
    ) -> list[tuple[SemanticState, float]]:
        """Get recent state history as (state, cosine_similarity_to_next) pairs.

        Returns states in reverse chronological order (newest first).
        """

    @abstractmethod
    def get_state_count(self, session_id: str) -> int:
        """Count total states for a session."""


class PhaseStore(ABC):
    """Manages Phase nodes and transitions."""

    @abstractmethod
    def set_phase(self, session_id: str, phase: Phase) -> Phase:
        """Set the current phase for a session.

        If there is a previous phase, creates a TRANSITIONED_TO relationship
        and sets exited_at on the previous phase.
        """

    @abstractmethod
    def get_current_phase(self, session_id: str) -> Optional[Phase]:
        """Get the current phase for a session."""

    @abstractmethod
    def get_phase_history(self, session_id: str, limit: int = 20) -> list[Phase]:
        """Get phase transition history (newest first)."""

    @abstractmethod
    def count_transitions(
        self, session_id: str, from_phase: PhaseName, to_phase: PhaseName
    ) -> int:
        """Count transitions between two specific phases."""

    @abstractmethod
    def get_transition_matrix(self, session_id: str) -> dict[str, dict[str, float]]:
        """Compute Markov transition probabilities from phase history."""


class DriftEventStore(ABC):
    """Manages DriftEvent nodes."""

    @abstractmethod
    def create_drift_event(
        self, session_id: str, state_id: str, event: DriftEvent
    ) -> DriftEvent:
        """Create a drift event linked to the triggering state."""

    @abstractmethod
    def get_drift_events(
        self,
        session_id: str,
        limit: int = 20,
        min_severity: Optional[DriftSeverity] = None,
        since: Optional[datetime] = None,
    ) -> list[DriftEvent]:
        """Query drift events for a session with optional filters."""

    @abstractmethod
    def get_drift_event_count(self, session_id: str) -> int:
        """Count total drift events for a session."""


class MemoryStore(ABC):
    """Manages Memory nodes across tiers."""

    @abstractmethod
    def store_memory(self, session_id: str, memory: Memory) -> Memory:
        """Store a memory unit in the specified tier."""

    @abstractmethod
    def get_memories_by_tier(
        self, session_id: str, tier: MemoryTier, limit: int = 50
    ) -> list[Memory]:
        """Get memories for a session in a specific tier, ordered by importance."""

    @abstractmethod
    def promote_memory(self, memory_id: str, new_tier: MemoryTier) -> bool:
        """Promote a memory to a higher tier."""

    @abstractmethod
    def update_memory_access(self, memory_id: str) -> bool:
        """Increment access_count and update last_accessed."""

    @abstractmethod
    def count_memories(self, session_id: str, tier: Optional[MemoryTier] = None) -> int:
        """Count memories, optionally filtered by tier."""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory node."""

    @abstractmethod
    def search_similar_memories(
        self,
        session_id: str,
        query_vector: list[float],
        limit: int = 5,
        tier: Optional[MemoryTier] = None,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search across memories. Returns (memory, score) pairs."""


class ClusterStore(ABC):
    """Manages Cluster topology."""

    @abstractmethod
    def create_cluster(self, cluster: Cluster) -> Cluster:
        """Create a new cluster."""

    @abstractmethod
    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""

    @abstractmethod
    def add_member(self, cluster_id: str, session_id: str, weight: float = 1.0) -> bool:
        """Add a session as a cluster member."""

    @abstractmethod
    def remove_member(self, cluster_id: str, session_id: str) -> bool:
        """Remove a session from a cluster."""

    @abstractmethod
    def get_members(self, cluster_id: str) -> list[tuple[AgentSession, float]]:
        """Get cluster members with weights. Returns (session, weight) pairs."""

    @abstractmethod
    def get_member_states(self, cluster_id: str) -> list[tuple[SemanticState, float]]:
        """Get current states of all members with weights. Returns (state, weight) pairs."""

    @abstractmethod
    def set_aggregated_state(self, cluster_id: str, state: SemanticState) -> bool:
        """Set the cluster's aggregated state."""


class RegionStore(ABC):
    """Manages Region topology."""

    @abstractmethod
    def create_region(self, region: Region) -> Region:
        """Create a new region."""

    @abstractmethod
    def get_region(self, region_id: str) -> Optional[Region]:
        """Get region by ID."""

    @abstractmethod
    def add_cluster_to_region(self, region_id: str, cluster_id: str) -> bool:
        """Add a cluster to a region."""

    @abstractmethod
    def get_clusters_in_region(self, region_id: str) -> list[Cluster]:
        """Get all clusters in a region."""

    @abstractmethod
    def store_consensus_event(
        self, region_id: str, event: ConsensusEvent
    ) -> ConsensusEvent:
        """Store a consensus event for the region."""

    @abstractmethod
    def get_consensus_events(
        self, region_id: str, limit: int = 10
    ) -> list[ConsensusEvent]:
        """Get recent consensus events for a region."""


class AgentStore(ABC):
    """Manages Agent nodes and their link to sessions.

    An agent runs many sessions over time; the Agent node groups them so
    cross-session analytics (trust, influence over time) become possible.
    """

    @abstractmethod
    def ensure_agent(self, agent: Agent) -> Agent:
        """Create the agent node if absent (idempotent MERGE on agent_id)."""

    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""

    @abstractmethod
    def link_to_session(self, agent_id: str, session_id: str) -> bool:
        """Link an agent to one of its sessions via (Agent)-[:RAN]->(AgentSession)."""

    @abstractmethod
    def get_sessions(self, agent_id: str) -> list[AgentSession]:
        """Get all sessions an agent has run."""


class GlobalObserverStore(ABC):
    """Manages GlobalObserver nodes — the anchor for network anomalies."""

    @abstractmethod
    def create_observer(self, observer: GlobalObserver) -> GlobalObserver:
        """Create the observer node (idempotent MERGE on observer_id)."""

    @abstractmethod
    def get_observer(self, observer_id: str) -> Optional[GlobalObserver]:
        """Get an observer by ID."""

    @abstractmethod
    def observe_region(self, observer_id: str, region_id: str) -> bool:
        """Link an observer to a region via (GlobalObserver)-[:OBSERVES]->(Region)."""


class AnomalyEventStore(ABC):
    """Manages AnomalyEvent nodes detected by the global observer."""

    @abstractmethod
    def create_anomaly_event(
        self, observer_id: str, event: AnomalyEvent
    ) -> AnomalyEvent:
        """Create an anomaly event linked to the detecting observer."""

    @abstractmethod
    def get_anomaly_events(
        self, observer_id: str, limit: int = 20
    ) -> list[AnomalyEvent]:
        """Query anomaly events for an observer (newest first)."""

    @abstractmethod
    def get_anomaly_event_count(self, observer_id: str) -> int:
        """Count total anomaly events for an observer."""


class LiteralFactStore(ABC):
    """Manages LiteralFact nodes — verbatim facts mirrored from a session."""

    @abstractmethod
    def store_fact(self, session_id: str, fact: LiteralFact) -> LiteralFact:
        """Store (idempotent upsert) a literal fact linked to its session."""

    @abstractmethod
    def get_facts(self, session_id: str, limit: int = 50) -> list[LiteralFact]:
        """Get literal facts for a session (newest first)."""

    @abstractmethod
    def get_fact_count(self, session_id: str) -> int:
        """Count literal facts for a session."""


class ConsensusStore(ABC):
    """Manages ConsensusEngine vote rounds — engine, proposal, votes and outcome."""

    @abstractmethod
    def store_round(self, region_id: str, vote_round: ConsensusRound) -> ConsensusEvent:
        """Store one vote round under the region and return its outcome event."""

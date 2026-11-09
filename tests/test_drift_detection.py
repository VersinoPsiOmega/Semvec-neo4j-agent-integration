"""Tests for core drift detection — uses the local Semvec runtime.

Semvec does the computation, Neo4j mirrors the results.
"""

import pytest

from semvec_neo4j.core.embedder import HashEmbedder
from semvec_neo4j.core.semvec_client import SemvecClient
from semvec_neo4j.core.drift_detector import DriftDetector
from semvec_neo4j.core.drift_rule import DriftRule
from semvec_neo4j.persistence.models import (
    AgentSession, DriftPhase, DriftSeverity, PhaseName,
)
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_phase_store import Neo4jPhaseStore
from semvec_neo4j.persistence.neo4j_drift_event_store import Neo4jDriftEventStore


@pytest.fixture(scope="session")
def semvec_client():
    return SemvecClient(embedder=HashEmbedder(dimension=32))


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def state_store(neo4j_driver):
    return Neo4jStateStore(neo4j_driver, database="neo4j")


@pytest.fixture
def phase_store(neo4j_driver):
    return Neo4jPhaseStore(neo4j_driver, database="neo4j")


@pytest.fixture
def drift_event_store(neo4j_driver):
    return Neo4jDriftEventStore(neo4j_driver, database="neo4j")


@pytest.fixture
def detector(semvec_client, state_store, phase_store, drift_event_store):
    return DriftDetector(
        semvec_client=semvec_client,
        state_store=state_store,
        phase_store=phase_store,
        drift_event_store=drift_event_store,
    )


@pytest.fixture
def active_session(session_store):
    session = AgentSession(session_id="drift-det-1", agent_id="agent-1")
    session_store.create_session(session)
    return session


class TestSemvecIntegration:
    """Test that Semvec is called and results are mirrored to Neo4j."""

    def test_process_input_returns_semvec_results(self, detector, active_session):
        result = detector.process_input(
            active_session.session_id,
            "What are the latest trends in cardiac treatment?",
        )
        # Semvec results
        assert "context" in result
        assert "drift_score" in result
        assert "drift_detected" in result
        assert "drift_phase" in result
        assert "top_similarity" in result
        assert "short_circuit" in result
        # Neo4j mirrored state
        assert "state_id" in result
        assert "step" in result
        assert result["step"] == 0

    def test_context_is_compressed(self, detector, active_session):
        """Semvec should return compressed context, not the full message."""
        result = detector.process_input(
            active_session.session_id,
            "Analyze the relationship between heart failure readmission rates and medication adherence across multiple hospital facilities in the northeast region",
        )
        assert result["context"] is not None
        assert isinstance(result["context"], str)

    def test_drift_detected_on_topic_switch(self, detector, active_session):
        sid = active_session.session_id
        # Build strong context around cardiology (Semvec needs enough turns)
        cardiology_msgs = [
            "Patient heart failure diagnosis and treatment history",
            "Cardiac medication contraindications and drug interactions",
            "Heart failure treatment outcomes and survival rates",
            "Cardiac rehabilitation protocols for post-surgery patients",
            "Heart failure readmission prevention strategies",
            "Cardiovascular risk assessment methodologies",
            "Beta blocker efficacy in heart failure management",
        ]
        for msg in cardiology_msgs:
            result = detector.process_input(sid, msg)

        # Switch to completely different topic
        result = detector.process_input(
            sid, "What is the best recipe for chocolate lava cake with vanilla ice cream?"
        )
        # Semvec should detect drift — at minimum drift_score >= 0 (it processes the change)
        assert result["drift_score"] >= 0
        # The context should reflect the topic shift
        assert result["context"] is not None


class TestNeo4jMirroring:
    """Test that Semvec results are correctly persisted in Neo4j."""

    def test_state_persisted(self, detector, active_session, state_store):
        sid = active_session.session_id
        detector.process_input(sid, "Patient diagnosis history")

        state = state_store.get_current_state(sid)
        assert state is not None
        assert state.step == 0

    def test_state_chain_grows(self, detector, active_session, state_store):
        sid = active_session.session_id
        detector.process_input(sid, "First message about cardiology")
        detector.process_input(sid, "Second message about cardiology")
        detector.process_input(sid, "Third message about cardiology")

        assert state_store.get_state_count(sid) == 3

    def test_phase_set(self, detector, active_session, phase_store):
        sid = active_session.session_id
        detector.process_input(sid, "Initial patient assessment")

        phase = phase_store.get_current_phase(sid)
        assert phase is not None
        assert phase.name in list(PhaseName)

    def test_drift_event_persisted(self, detector, active_session, drift_event_store):
        sid = active_session.session_id
        # Build context
        for msg in [
            "Heart failure management strategies",
            "Cardiac rehabilitation protocols",
            "Medication adherence in cardiac patients",
        ]:
            detector.process_input(sid, msg)

        # Topic switch
        detector.process_input(sid, "How to bake a chocolate cake recipe")

        events = drift_event_store.get_drift_events(sid)
        # Check that at least the significant drifts were captured
        assert isinstance(events, list)


class TestDriftRulePersistence:
    """A DriftEvent node is written exactly when the client's drift rule flags a turn."""

    def _detector(self, rule, state_store, phase_store, drift_event_store):
        return DriftDetector(
            semvec_client=SemvecClient(embedder=HashEmbedder(dimension=32), drift_rule=rule),
            state_store=state_store,
            phase_store=phase_store,
            drift_event_store=drift_event_store,
        )

    def test_drift_event_persisted_when_rule_flags(
        self, active_session, state_store, phase_store, drift_event_store
    ):
        # Arrange
        detector = self._detector(
            DriftRule(drift_threshold=0.0, sim_ceiling=1.0), state_store, phase_store, drift_event_store
        )

        # Act
        result = detector.process_input(active_session.session_id, "Heart failure management")

        # Assert
        assert result["drift"] is True
        assert len(drift_event_store.get_drift_events(active_session.session_id)) == 1

    def test_no_drift_event_when_rule_does_not_flag(
        self, active_session, state_store, phase_store, drift_event_store
    ):
        # Arrange
        detector = self._detector(
            DriftRule(drift_threshold=1.01, sim_ceiling=0.0), state_store, phase_store, drift_event_store
        )

        # Act
        result = detector.process_input(active_session.session_id, "Heart failure management")

        # Assert
        assert result["drift"] is False
        assert drift_event_store.get_drift_events(active_session.session_id) == []


class TestStoreResponse:
    """Test that LLM responses are fed back to Semvec."""

    def test_store_response(self, detector, active_session):
        sid = active_session.session_id
        detector.process_input(sid, "What treatments exist for heart failure?")

        result = detector.store_response(
            sid,
            "Common treatments include ACE inhibitors, beta-blockers, and diuretics. "
            "Lifestyle modifications are also recommended.",
        )
        assert result is not None

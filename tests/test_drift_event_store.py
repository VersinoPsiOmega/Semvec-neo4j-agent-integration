"""Tests for Neo4j DriftEventStore."""

import pytest
from datetime import datetime, timezone, timedelta

from semvec_neo4j.persistence.models import (
    AgentSession, DriftEvent, DriftPhase, DriftSeverity, SemanticState,
)
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from tests.conftest import make_random_vector


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def state_store(neo4j_driver):
    return Neo4jStateStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jDriftEventStore(neo4j_driver, database="neo4j")


@pytest.fixture
def session_with_state(session_store, state_store):
    """Create a session with one semantic state."""
    session = AgentSession(session_id="drift-sess-1", agent_id="agent-1")
    session_store.create_session(session)
    state = SemanticState(
        state_id="drift-state-1",
        vector=make_random_vector(768, seed=1),
        step=0, beta=0.1,
    )
    state_store.append_state(session.session_id, state, cosine_similarity=1.0)
    return session, state


class TestDriftEventCreate:
    def test_create_drift_event(self, store, session_with_state):
        session, state = session_with_state
        event = DriftEvent(
            drift_score=0.75,
            drift_phase=DriftPhase.DRIFTED,
            topic_switch=True,
            cosine_drop=0.35,
            severity=DriftSeverity.HIGH,
        )
        result = store.create_drift_event(session.session_id, state.state_id, event)
        assert result.event_id == event.event_id
        assert result.timestamp is not None

    def test_drift_event_linked_to_state(self, store, session_with_state, db_session):
        session, state = session_with_state
        event = DriftEvent(drift_score=0.5, severity=DriftSeverity.MEDIUM)
        store.create_drift_event(session.session_id, state.state_id, event)

        # Verify relationship exists
        result = db_session.run("""
            MATCH (s:SemanticState {state_id: $state_id})-[:TRIGGERED]->(d:DriftEvent {event_id: $event_id})
            RETURN d.event_id AS eid
        """, state_id=state.state_id, event_id=event.event_id)
        assert result.single() is not None


class TestDriftEventQuery:
    def test_get_drift_events(self, store, session_with_state):
        session, state = session_with_state
        for i in range(5):
            event = DriftEvent(
                drift_score=0.1 * (i + 1),
                severity=DriftSeverity.LOW if i < 3 else DriftSeverity.HIGH,
            )
            store.create_drift_event(session.session_id, state.state_id, event)

        events = store.get_drift_events(session.session_id, limit=10)
        assert len(events) == 5

    def test_filter_by_severity(self, store, session_with_state):
        session, state = session_with_state
        store.create_drift_event(
            session.session_id, state.state_id,
            DriftEvent(drift_score=0.3, severity=DriftSeverity.LOW),
        )
        store.create_drift_event(
            session.session_id, state.state_id,
            DriftEvent(drift_score=0.8, severity=DriftSeverity.HIGH),
        )
        store.create_drift_event(
            session.session_id, state.state_id,
            DriftEvent(drift_score=0.9, severity=DriftSeverity.CRITICAL),
        )

        events = store.get_drift_events(
            session.session_id, min_severity=DriftSeverity.HIGH
        )
        assert len(events) == 2
        assert all(e.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL) for e in events)

    def test_get_event_count(self, store, session_with_state):
        session, state = session_with_state
        for i in range(3):
            store.create_drift_event(
                session.session_id, state.state_id,
                DriftEvent(drift_score=0.5),
            )
        assert store.get_drift_event_count(session.session_id) == 3

    def test_empty_events(self, store, session_with_state):
        session, _ = session_with_state
        events = store.get_drift_events(session.session_id)
        assert events == []
        assert store.get_drift_event_count(session.session_id) == 0

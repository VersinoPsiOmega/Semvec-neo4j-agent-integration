"""Tests for Neo4j PhaseStore — phase tracking and transition matrix."""

import pytest

from semvec_neo4j.persistence.models import AgentSession, Phase, PhaseName
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_phase_store import Neo4jPhaseStore


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jPhaseStore(neo4j_driver, database="neo4j")


@pytest.fixture
def active_session(session_store):
    session = AgentSession(session_id="phase-sess-1", agent_id="agent-1")
    session_store.create_session(session)
    return session


class TestPhaseStoreSetPhase:
    def test_set_first_phase(self, store, active_session):
        phase = Phase(name=PhaseName.INITIALIZATION)
        result = store.set_phase(active_session.session_id, phase)
        assert result.phase_id == phase.phase_id
        assert result.entered_at is not None

    def test_set_phase_transitions(self, store, active_session):
        sid = active_session.session_id
        p1 = Phase(name=PhaseName.INITIALIZATION)
        p2 = Phase(name=PhaseName.EXPLORATION)
        store.set_phase(sid, p1)
        store.set_phase(sid, p2)

        current = store.get_current_phase(sid)
        assert current.name == PhaseName.EXPLORATION

    def test_previous_phase_gets_exited_at(self, store, active_session):
        sid = active_session.session_id
        p1 = Phase(name=PhaseName.INITIALIZATION)
        p2 = Phase(name=PhaseName.EXPLORATION)
        store.set_phase(sid, p1)
        store.set_phase(sid, p2)

        history = store.get_phase_history(sid)
        # p1 should have exited_at set
        old_phase = [p for p in history if p.name == PhaseName.INITIALIZATION][0]
        assert old_phase.exited_at is not None


class TestPhaseStoreGetCurrent:
    def test_no_phase_returns_none(self, store, active_session):
        result = store.get_current_phase(active_session.session_id)
        assert result is None

    def test_get_current_after_multiple_transitions(self, store, active_session):
        sid = active_session.session_id
        for name in [PhaseName.INITIALIZATION, PhaseName.EXPLORATION,
                     PhaseName.CONVERGENCE, PhaseName.RESONANCE]:
            store.set_phase(sid, Phase(name=name))

        current = store.get_current_phase(sid)
        assert current.name == PhaseName.RESONANCE


class TestPhaseStoreHistory:
    def test_empty_history(self, store, active_session):
        result = store.get_phase_history(active_session.session_id)
        assert result == []

    def test_history_in_reverse_order(self, store, active_session):
        sid = active_session.session_id
        names = [PhaseName.INITIALIZATION, PhaseName.EXPLORATION, PhaseName.CONVERGENCE]
        for name in names:
            store.set_phase(sid, Phase(name=name))

        history = store.get_phase_history(sid)
        assert len(history) == 3
        assert history[0].name == PhaseName.CONVERGENCE
        assert history[2].name == PhaseName.INITIALIZATION


class TestPhaseStoreTransitionMatrix:
    def test_empty_transition_matrix(self, store, active_session):
        result = store.get_transition_matrix(active_session.session_id)
        assert result == {}

    def test_transition_matrix_probabilities(self, store, active_session):
        sid = active_session.session_id
        # Create sequence: INIT -> EXPL -> CONV -> EXPL -> CONV -> STABILITY
        sequence = [
            PhaseName.INITIALIZATION,
            PhaseName.EXPLORATION,
            PhaseName.CONVERGENCE,
            PhaseName.EXPLORATION,
            PhaseName.CONVERGENCE,
            PhaseName.STABILITY,
        ]
        for name in sequence:
            store.set_phase(sid, Phase(name=name))

        matrix = store.get_transition_matrix(sid)
        # INIT -> EXPL: 1 transition = probability 1.0
        assert abs(matrix["initialization"]["exploration"] - 1.0) < 1e-5
        # EXPL -> CONV: 2 transitions out of 2 = 1.0
        assert abs(matrix["exploration"]["convergence"] - 1.0) < 1e-5
        # CONV -> EXPL: 1/2, CONV -> STABILITY: 1/2
        assert abs(matrix["convergence"]["exploration"] - 0.5) < 1e-5
        assert abs(matrix["convergence"]["stability"] - 0.5) < 1e-5

    def test_count_transitions(self, store, active_session):
        sid = active_session.session_id
        sequence = [
            PhaseName.INITIALIZATION, PhaseName.EXPLORATION,
            PhaseName.CONVERGENCE, PhaseName.EXPLORATION,
        ]
        for name in sequence:
            store.set_phase(sid, Phase(name=name))

        count = store.count_transitions(sid, PhaseName.EXPLORATION, PhaseName.CONVERGENCE)
        assert count == 1
        count = store.count_transitions(sid, PhaseName.CONVERGENCE, PhaseName.EXPLORATION)
        assert count == 1

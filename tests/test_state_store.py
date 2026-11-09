"""Tests for Neo4j StateStore — semantic state chain management."""

import math
import pytest

from semvec_neo4j.persistence.models import AgentSession, SemanticState
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from tests.conftest import make_random_vector


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jStateStore(neo4j_driver, database="neo4j")


@pytest.fixture
def active_session(session_store):
    """Create an active session for state tests."""
    session = AgentSession(session_id="state-sess-1", agent_id="agent-1")
    session_store.create_session(session)
    return session


class TestStateStoreAppend:
    def test_append_first_state(self, store, active_session):
        vec = make_random_vector(768, seed=1)
        state = SemanticState(vector=vec, step=0, beta=0.1)
        result = store.append_state(active_session.session_id, state, cosine_similarity=1.0)
        assert result.state_id == state.state_id
        assert result.step == 0
        assert result.timestamp is not None

    def test_append_creates_chain(self, store, active_session):
        sid = active_session.session_id
        s1 = SemanticState(vector=make_random_vector(768, seed=1), step=0, beta=0.1)
        s2 = SemanticState(vector=make_random_vector(768, seed=2), step=1, beta=0.15)
        s3 = SemanticState(vector=make_random_vector(768, seed=3), step=2, beta=0.2)

        store.append_state(sid, s1, cosine_similarity=1.0)
        store.append_state(sid, s2, cosine_similarity=0.85)
        store.append_state(sid, s3, cosine_similarity=0.72)

        current = store.get_current_state(sid)
        assert current is not None
        assert current.state_id == s3.state_id
        assert current.step == 2

    def test_append_preserves_vector(self, store, active_session):
        vec = make_random_vector(768, seed=42)
        state = SemanticState(vector=vec, step=0, beta=0.1)
        store.append_state(active_session.session_id, state, cosine_similarity=1.0)

        current = store.get_current_state(active_session.session_id)
        assert len(current.vector) == 768
        # Check vector values are close (float precision)
        for i in range(10):
            assert abs(current.vector[i] - vec[i]) < 1e-5


class TestStateStoreGetCurrent:
    def test_get_current_state_no_states(self, store, active_session):
        result = store.get_current_state(active_session.session_id)
        assert result is None

    def test_get_current_state_after_multiple_appends(self, store, active_session):
        sid = active_session.session_id
        for i in range(5):
            s = SemanticState(vector=make_random_vector(768, seed=i), step=i, beta=0.1 * i)
            store.append_state(sid, s, cosine_similarity=0.9)

        current = store.get_current_state(sid)
        assert current.step == 4


class TestStateStoreHistory:
    def test_get_history_empty(self, store, active_session):
        result = store.get_state_history(active_session.session_id)
        assert result == []

    def test_get_history_returns_chain(self, store, active_session):
        sid = active_session.session_id
        sims = [1.0, 0.9, 0.8, 0.7, 0.6]
        for i in range(5):
            s = SemanticState(vector=make_random_vector(768, seed=i), step=i, beta=0.1)
            store.append_state(sid, s, cosine_similarity=sims[i])

        history = store.get_state_history(sid, limit=5)
        # Should return newest first
        assert len(history) == 5
        steps = [state.step for state, _ in history]
        assert steps == [4, 3, 2, 1, 0]

    def test_get_history_with_limit(self, store, active_session):
        sid = active_session.session_id
        for i in range(10):
            s = SemanticState(vector=make_random_vector(768, seed=i), step=i, beta=0.1)
            store.append_state(sid, s, cosine_similarity=0.9)

        history = store.get_state_history(sid, limit=3)
        assert len(history) == 3
        assert history[0][0].step == 9  # Most recent

    def test_history_includes_cosine_similarity(self, store, active_session):
        sid = active_session.session_id
        s1 = SemanticState(vector=make_random_vector(768, seed=1), step=0, beta=0.1)
        s2 = SemanticState(vector=make_random_vector(768, seed=2), step=1, beta=0.2)

        store.append_state(sid, s1, cosine_similarity=1.0)
        store.append_state(sid, s2, cosine_similarity=0.75)

        history = store.get_state_history(sid, limit=2)
        # s2 has similarity 0.75 to s1
        assert len(history) == 2
        _, sim = history[0]  # newest (s2)
        assert abs(sim - 0.75) < 1e-5


class TestStateStoreCount:
    def test_count_empty(self, store, active_session):
        assert store.get_state_count(active_session.session_id) == 0

    def test_count_after_appends(self, store, active_session):
        sid = active_session.session_id
        for i in range(7):
            s = SemanticState(vector=make_random_vector(768, seed=i), step=i, beta=0.1)
            store.append_state(sid, s, cosine_similarity=0.9)
        assert store.get_state_count(sid) == 7

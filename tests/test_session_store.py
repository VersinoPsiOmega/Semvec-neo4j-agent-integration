"""Tests for Neo4j SessionStore — RED phase first."""

import pytest
from datetime import datetime

from semvec_neo4j.persistence.models import AgentSession, SessionStatus
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore


@pytest.fixture
def store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


class TestSessionStoreCreate:
    def test_create_session_returns_session_with_id(self, store):
        session = AgentSession(agent_id="agent-1")
        result = store.create_session(session)
        assert result.session_id == session.session_id
        assert result.agent_id == "agent-1"
        assert result.status == SessionStatus.ACTIVE

    def test_create_session_sets_timestamps(self, store):
        session = AgentSession(agent_id="agent-1")
        result = store.create_session(session)
        assert result.created_at is not None
        assert result.last_active is not None

    def test_create_duplicate_session_raises(self, store):
        session = AgentSession(session_id="dup-1", agent_id="agent-1")
        store.create_session(session)
        with pytest.raises(Exception):
            store.create_session(session)


class TestSessionStoreGet:
    def test_get_existing_session(self, store):
        session = AgentSession(session_id="get-1", agent_id="agent-1")
        store.create_session(session)
        result = store.get_session("get-1")
        assert result is not None
        assert result.session_id == "get-1"
        assert result.agent_id == "agent-1"

    def test_get_nonexistent_session_returns_none(self, store):
        result = store.get_session("nonexistent")
        assert result is None


class TestSessionStoreUpdate:
    def test_update_session_status(self, store):
        session = AgentSession(session_id="upd-1", agent_id="agent-1")
        store.create_session(session)
        result = store.update_session_status("upd-1", SessionStatus.ERROR)
        assert result is True
        updated = store.get_session("upd-1")
        assert updated.status == SessionStatus.ERROR

    def test_update_nonexistent_session_returns_false(self, store):
        result = store.update_session_status("nope", SessionStatus.CLOSED)
        assert result is False

    def test_close_session(self, store):
        session = AgentSession(session_id="close-1", agent_id="agent-1")
        store.create_session(session)
        result = store.close_session("close-1")
        assert result is True
        closed = store.get_session("close-1")
        assert closed.status == SessionStatus.CLOSED


class TestSessionStoreList:
    def test_list_active_sessions(self, store):
        store.create_session(AgentSession(session_id="a1", agent_id="agent-1"))
        store.create_session(AgentSession(session_id="a2", agent_id="agent-1"))
        store.create_session(AgentSession(session_id="a3", agent_id="agent-2"))

        result = store.list_active_sessions()
        assert len(result) == 3

    def test_list_active_sessions_filtered_by_agent(self, store):
        store.create_session(AgentSession(session_id="f1", agent_id="agent-1"))
        store.create_session(AgentSession(session_id="f2", agent_id="agent-1"))
        store.create_session(AgentSession(session_id="f3", agent_id="agent-2"))

        result = store.list_active_sessions(agent_id="agent-1")
        assert len(result) == 2
        assert all(s.agent_id == "agent-1" for s in result)

    def test_closed_sessions_not_listed(self, store):
        store.create_session(AgentSession(session_id="c1", agent_id="agent-1"))
        store.create_session(AgentSession(session_id="c2", agent_id="agent-1"))
        store.close_session("c2")

        result = store.list_active_sessions()
        assert len(result) == 1
        assert result[0].session_id == "c1"

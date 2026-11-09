"""Tests for Neo4j AgentStore."""

import pytest

from semvec_neo4j.persistence.models import Agent, AgentSession
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_agent_store import Neo4jAgentStore


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jAgentStore(neo4j_driver, database="neo4j")


class TestAgentEnsure:
    def test_ensure_agent_creates(self, store):
        result = store.ensure_agent(
            Agent(agent_id="agent-x", name="Dr. X", framework="langgraph")
        )
        assert result.agent_id == "agent-x"
        got = store.get_agent("agent-x")
        assert got is not None
        assert got.name == "Dr. X"
        assert got.framework == "langgraph"

    def test_ensure_agent_idempotent(self, store):
        store.ensure_agent(Agent(agent_id="agent-x", name="first"))
        # MERGE with ON CREATE: re-ensuring does not overwrite the name.
        store.ensure_agent(Agent(agent_id="agent-x", name="second"))
        got = store.get_agent("agent-x")
        assert got.name == "first"

    def test_get_missing_agent(self, store):
        assert store.get_agent("nope") is None


class TestAgentSessionLink:
    def test_link_and_get_sessions(self, store, session_store, db_session):
        store.ensure_agent(Agent(agent_id="agent-x", name="X"))
        for sid in ("agent-x-s1", "agent-x-s2"):
            session_store.create_session(AgentSession(session_id=sid, agent_id="agent-x"))
            assert store.link_to_session("agent-x", sid) is True

        sessions = store.get_sessions("agent-x")
        assert len(sessions) == 2

        result = db_session.run(
            "MATCH (:Agent {agent_id: 'agent-x'})-[:RAN]->(s:AgentSession) "
            "RETURN count(s) AS c"
        )
        assert result.single()["c"] == 2

    def test_link_idempotent(self, store, session_store):
        store.ensure_agent(Agent(agent_id="agent-x", name="X"))
        session_store.create_session(AgentSession(session_id="agent-x-s1", agent_id="agent-x"))
        store.link_to_session("agent-x", "agent-x-s1")
        store.link_to_session("agent-x", "agent-x-s1")  # MERGE rel — no duplicate
        assert len(store.get_sessions("agent-x")) == 1

    def test_get_sessions_empty(self, store):
        store.ensure_agent(Agent(agent_id="lonely", name="L"))
        assert store.get_sessions("lonely") == []

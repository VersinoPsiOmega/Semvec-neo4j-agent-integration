"""Tests for Neo4j LiteralFactStore."""

import pytest

from semvec_neo4j.persistence.models import AgentSession, LiteralFact
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_literal_fact_store import Neo4jLiteralFactStore


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jLiteralFactStore(neo4j_driver, database="neo4j")


@pytest.fixture
def session(session_store):
    s = AgentSession(session_id="fact-sess-1", agent_id="agent-1")
    session_store.create_session(s)
    return s


class TestLiteralFactStore:
    def test_store_and_get(self, store, session):
        fact = LiteralFact(key="5 mg", kind="numeric", value="5", unit="mg")
        result = store.store_fact(session.session_id, fact)
        assert result.created_at is not None
        assert result.session_id == session.session_id

        facts = store.get_facts(session.session_id)
        assert len(facts) == 1
        assert facts[0].value == "5"
        assert facts[0].unit == "mg"
        assert facts[0].kind == "numeric"

    def test_extracted_relationship(self, store, session, db_session):
        store.store_fact(
            session.session_id,
            LiteralFact(key="2026-05-07", kind="date", value="2026-05-07"),
        )
        result = db_session.run(
            "MATCH (s:AgentSession {session_id: $sid})-[:EXTRACTED]->(f:LiteralFact) "
            "RETURN f.key AS k",
            sid=session.session_id,
        )
        assert result.single()["k"] == "2026-05-07"

    def test_upsert_is_idempotent(self, store, session):
        f1 = store.store_fact(session.session_id, LiteralFact(key="5 mg", kind="numeric", value="5", unit="mg"))
        f2 = store.store_fact(session.session_id, LiteralFact(key="5 mg", kind="numeric", value="5", unit="mg"))
        # Same (session_id, key) -> one node, one EXTRACTED edge, stable id.
        assert store.get_fact_count(session.session_id) == 1
        assert f2.fact_id == f1.fact_id

    def test_count_and_empty(self, store, session):
        assert store.get_fact_count(session.session_id) == 0
        for k in ("a", "b", "c"):
            store.store_fact(
                session.session_id,
                LiteralFact(key=k, kind="identifier", value=k, id_type="uuid"),
            )
        assert store.get_fact_count(session.session_id) == 3
        assert len(store.get_facts(session.session_id)) == 3

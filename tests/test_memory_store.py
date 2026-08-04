"""Tests for Neo4j MemoryStore — multi-tier memory management."""

import pytest

from src.persistence.models import AgentSession, Memory, MemoryTier
from src.persistence.neo4j_session_store import Neo4jSessionStore
from src.persistence.neo4j_memory_store import Neo4jMemoryStore
from tests.conftest import make_random_vector


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jMemoryStore(neo4j_driver, database="neo4j")


@pytest.fixture
def active_session(session_store):
    session = AgentSession(session_id="mem-sess-1", agent_id="agent-1")
    session_store.create_session(session)
    return session


class TestMemoryStoreBasic:
    def test_store_memory(self, store, active_session):
        mem = Memory(
            tier=MemoryTier.SHORT,
            content_vector=make_random_vector(768, seed=1),
            importance=0.8,
            text_summary="User asked about Q3 earnings",
        )
        result = store.store_memory(active_session.session_id, mem)
        assert result.memory_id == mem.memory_id
        assert result.created_at is not None

    def test_get_memories_by_tier(self, store, active_session):
        sid = active_session.session_id
        for i in range(5):
            store.store_memory(sid, Memory(
                tier=MemoryTier.SHORT,
                content_vector=make_random_vector(768, seed=i),
                importance=0.1 * (i + 1),
                text_summary=f"Memory {i}",
            ))
        for i in range(3):
            store.store_memory(sid, Memory(
                tier=MemoryTier.MEDIUM,
                content_vector=make_random_vector(768, seed=100 + i),
                importance=0.5,
                text_summary=f"Medium memory {i}",
            ))

        short = store.get_memories_by_tier(sid, MemoryTier.SHORT)
        assert len(short) == 5
        medium = store.get_memories_by_tier(sid, MemoryTier.MEDIUM)
        assert len(medium) == 3

    def test_memories_ordered_by_importance(self, store, active_session):
        sid = active_session.session_id
        importances = [0.3, 0.9, 0.1, 0.7, 0.5]
        for i, imp in enumerate(importances):
            store.store_memory(sid, Memory(
                tier=MemoryTier.SHORT,
                content_vector=make_random_vector(768, seed=i),
                importance=imp,
                text_summary=f"Memory imp={imp}",
            ))

        result = store.get_memories_by_tier(sid, MemoryTier.SHORT)
        scores = [m.importance for m in result]
        assert scores == sorted(scores, reverse=True)


class TestMemoryPromote:
    def test_promote_memory(self, store, active_session):
        sid = active_session.session_id
        mem = Memory(
            tier=MemoryTier.SHORT,
            content_vector=make_random_vector(768, seed=1),
            importance=0.8,
            text_summary="Important memory",
        )
        store.store_memory(sid, mem)

        result = store.promote_memory(mem.memory_id, MemoryTier.MEDIUM)
        assert result is True

        short = store.get_memories_by_tier(sid, MemoryTier.SHORT)
        medium = store.get_memories_by_tier(sid, MemoryTier.MEDIUM)
        assert len(short) == 0
        assert len(medium) == 1
        assert medium[0].tier == MemoryTier.MEDIUM


class TestMemoryAccess:
    def test_update_access(self, store, active_session):
        sid = active_session.session_id
        mem = Memory(
            tier=MemoryTier.SHORT,
            content_vector=make_random_vector(768, seed=1),
            importance=0.5,
            text_summary="Accessed memory",
        )
        store.store_memory(sid, mem)

        store.update_memory_access(mem.memory_id)
        store.update_memory_access(mem.memory_id)
        store.update_memory_access(mem.memory_id)

        mems = store.get_memories_by_tier(sid, MemoryTier.SHORT)
        assert mems[0].access_count == 3
        assert mems[0].last_accessed is not None


class TestMemoryCount:
    def test_count_all(self, store, active_session):
        sid = active_session.session_id
        store.store_memory(sid, Memory(tier=MemoryTier.SHORT, content_vector=make_random_vector(768, seed=1), importance=0.5))
        store.store_memory(sid, Memory(tier=MemoryTier.SHORT, content_vector=make_random_vector(768, seed=2), importance=0.5))
        store.store_memory(sid, Memory(tier=MemoryTier.MEDIUM, content_vector=make_random_vector(768, seed=3), importance=0.5))

        assert store.count_memories(sid) == 3
        assert store.count_memories(sid, MemoryTier.SHORT) == 2
        assert store.count_memories(sid, MemoryTier.MEDIUM) == 1
        assert store.count_memories(sid, MemoryTier.LONG) == 0


class TestMemoryDelete:
    def test_delete_memory(self, store, active_session):
        sid = active_session.session_id
        mem = Memory(tier=MemoryTier.SHORT, content_vector=make_random_vector(768, seed=1), importance=0.5)
        store.store_memory(sid, mem)
        assert store.count_memories(sid) == 1

        result = store.delete_memory(mem.memory_id)
        assert result is True
        assert store.count_memories(sid) == 0

    def test_delete_nonexistent(self, store):
        assert store.delete_memory("nonexistent") is False


class TestMemoryVectorSearch:
    def test_search_similar(self, store, active_session):
        sid = active_session.session_id
        # Store memories with known vectors
        for i in range(5):
            store.store_memory(sid, Memory(
                tier=MemoryTier.SHORT,
                content_vector=make_random_vector(768, seed=i),
                importance=0.5,
                text_summary=f"Memory {i}",
            ))

        # Search with a vector similar to seed=0
        query_vec = make_random_vector(768, seed=0)
        results = store.search_similar_memories(sid, query_vec, limit=3)
        assert len(results) > 0
        # First result should be exact match (seed=0)
        mem, score = results[0]
        assert score > 0.99  # Near-perfect match
        assert mem.text_summary == "Memory 0"

    def test_search_with_tier_filter(self, store, active_session):
        sid = active_session.session_id
        store.store_memory(sid, Memory(
            tier=MemoryTier.SHORT,
            content_vector=make_random_vector(768, seed=1),
            importance=0.5, text_summary="Short mem",
        ))
        store.store_memory(sid, Memory(
            tier=MemoryTier.LONG,
            content_vector=make_random_vector(768, seed=1),  # Same vector
            importance=0.5, text_summary="Long mem",
        ))

        results = store.search_similar_memories(
            sid, make_random_vector(768, seed=1), limit=5, tier=MemoryTier.LONG
        )
        assert len(results) == 1
        assert results[0][0].text_summary == "Long mem"

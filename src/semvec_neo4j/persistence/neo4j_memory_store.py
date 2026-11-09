"""Neo4j-backed MemoryStore — multi-tier memory management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import MemoryStore
from semvec_neo4j.persistence.models import Memory, MemoryTier


class Neo4jMemoryStore(MemoryStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def store_memory(self, session_id: str, memory: Memory) -> Memory:
        now = datetime.now(timezone.utc)
        memory.created_at = now
        memory.last_accessed = now

        query = """
        MATCH (session:AgentSession {session_id: $session_id})
        CREATE (m:Memory {
            memory_id: $memory_id,
            tier: $tier,
            content_vector: $content_vector,
            importance: $importance,
            recency: $recency,
            access_count: $access_count,
            created_at: datetime($now),
            last_accessed: datetime($now),
            text_summary: $text_summary
        })
        CREATE (session)-[:HAS_MEMORY]->(m)
        RETURN m.memory_id AS mid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                session_id=session_id,
                memory_id=memory.memory_id,
                tier=memory.tier.value,
                content_vector=memory.content_vector,
                importance=memory.importance,
                recency=memory.recency,
                access_count=memory.access_count,
                now=now.isoformat(),
                text_summary=memory.text_summary,
            ).consume()
        return memory

    def get_memories_by_tier(
        self, session_id: str, tier: MemoryTier, limit: int = 50
    ) -> list[Memory]:
        query = """
        MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory {tier: $tier})
        RETURN m.memory_id AS memory_id,
               m.tier AS tier,
               m.content_vector AS content_vector,
               m.importance AS importance,
               m.recency AS recency,
               m.access_count AS access_count,
               m.created_at AS created_at,
               m.last_accessed AS last_accessed,
               m.text_summary AS text_summary
        ORDER BY m.importance DESC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, tier=tier.value, limit=limit)
            return [self._record_to_memory(r) for r in result]

    def promote_memory(self, memory_id: str, new_tier: MemoryTier) -> bool:
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        SET m.tier = $new_tier
        RETURN m.memory_id AS mid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, memory_id=memory_id, new_tier=new_tier.value)
            return result.single() is not None

    def update_memory_access(self, memory_id: str) -> bool:
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        SET m.access_count = m.access_count + 1,
            m.last_accessed = datetime($now)
        RETURN m.memory_id AS mid
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session(database=self._database) as db:
            result = db.run(query, memory_id=memory_id, now=now)
            return result.single() is not None

    def count_memories(self, session_id: str, tier: Optional[MemoryTier] = None) -> int:
        if tier is not None:
            query = """
            MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory {tier: $tier})
            RETURN count(m) AS cnt
            """
            params = {"session_id": session_id, "tier": tier.value}
        else:
            query = """
            MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory)
            RETURN count(m) AS cnt
            """
            params = {"session_id": session_id}

        with self._driver.session(database=self._database) as db:
            result = db.run(query, **params)
            record = result.single()
            return record["cnt"] if record else 0

    def delete_memory(self, memory_id: str) -> bool:
        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        DETACH DELETE m
        RETURN true AS deleted
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, memory_id=memory_id)
            return result.single() is not None

    def search_similar_memories(
        self,
        session_id: str,
        query_vector: list[float],
        limit: int = 5,
        tier: Optional[MemoryTier] = None,
    ) -> list[tuple[Memory, float]]:
        # Use Neo4j vector index for similarity search
        if tier is not None:
            query = """
            CALL db.index.vector.queryNodes('memory_content_vector', $top_k, $query_vector)
            YIELD node AS m, score
            MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m)
            WHERE m.tier = $tier
            RETURN m.memory_id AS memory_id,
                   m.tier AS tier,
                   m.content_vector AS content_vector,
                   m.importance AS importance,
                   m.recency AS recency,
                   m.access_count AS access_count,
                   m.created_at AS created_at,
                   m.last_accessed AS last_accessed,
                   m.text_summary AS text_summary,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {
                "session_id": session_id,
                "query_vector": query_vector,
                "tier": tier.value,
                "limit": limit,
                "top_k": limit * 5,  # Over-fetch to account for tier filtering
            }
        else:
            query = """
            CALL db.index.vector.queryNodes('memory_content_vector', $top_k, $query_vector)
            YIELD node AS m, score
            MATCH (session:AgentSession {session_id: $session_id})-[:HAS_MEMORY]->(m)
            RETURN m.memory_id AS memory_id,
                   m.tier AS tier,
                   m.content_vector AS content_vector,
                   m.importance AS importance,
                   m.recency AS recency,
                   m.access_count AS access_count,
                   m.created_at AS created_at,
                   m.last_accessed AS last_accessed,
                   m.text_summary AS text_summary,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {
                "session_id": session_id,
                "query_vector": query_vector,
                "limit": limit,
                "top_k": limit * 2,
            }

        with self._driver.session(database=self._database) as db:
            result = db.run(query, **params)
            return [
                (self._record_to_memory(r), r["score"])
                for r in result
            ]

    @staticmethod
    def _record_to_memory(record) -> Memory:
        created_at = record["created_at"]
        last_accessed = record["last_accessed"]
        if hasattr(created_at, "to_native"):
            created_at = created_at.to_native()
        if last_accessed is not None and hasattr(last_accessed, "to_native"):
            last_accessed = last_accessed.to_native()

        return Memory(
            memory_id=record["memory_id"],
            tier=MemoryTier(record["tier"]),
            content_vector=list(record["content_vector"]) if record["content_vector"] else [],
            importance=record["importance"],
            recency=record["recency"],
            access_count=record["access_count"],
            created_at=created_at,
            last_accessed=last_accessed,
            text_summary=record["text_summary"],
        )

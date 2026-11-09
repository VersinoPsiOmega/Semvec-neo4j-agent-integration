"""Neo4j-backed LiteralFactStore — mirrors verbatim extracted facts.

Persists the numeric/date/identifier facts pulled by Semvec's compliance
extractor (``SemvecClient.extract_facts``) into the graph, linked to the
session they were extracted from via ``(AgentSession)-[:EXTRACTED]->(LiteralFact)``.
The ``(session_id, key)`` pair is unique, so re-extracting the same span is
an idempotent upsert.
"""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import Driver

from semvec_neo4j.persistence.base import LiteralFactStore
from semvec_neo4j.persistence.models import LiteralFact


class Neo4jLiteralFactStore(LiteralFactStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def store_fact(self, session_id: str, fact: LiteralFact) -> LiteralFact:
        now = datetime.now(timezone.utc)
        fact.session_id = session_id
        fact.created_at = now

        # MERGE on the unique (session_id, key) pair: re-extracting the same
        # verbatim span updates the payload rather than duplicating the fact.
        # fact_id/created_at are pinned on first creation only.
        query = """
        MATCH (s:AgentSession {session_id: $session_id})
        MERGE (f:LiteralFact {session_id: $session_id, key: $key})
        ON CREATE SET f.fact_id = $fact_id,
                      f.created_at = datetime($created_at)
        SET f.kind = $kind,
            f.value = $value,
            f.unit = $unit,
            f.id_type = $id_type,
            f.importance = $importance
        MERGE (s)-[:EXTRACTED]->(f)
        RETURN f.fact_id AS fid, f.created_at AS created_at
        """
        params = dict(
            session_id=session_id,
            key=fact.key,
            fact_id=fact.fact_id,
            created_at=now.isoformat(),
            kind=fact.kind,
            value=fact.value,
            unit=fact.unit,
            id_type=fact.id_type,
            importance=fact.importance,
        )
        with self._driver.session(database=self._database) as db:
            record = db.run(query, **params).single()
            if record is not None:
                # Reflect the persisted identity (may pre-date this upsert).
                fact.fact_id = record["fid"]
                created = record["created_at"]
                if hasattr(created, "to_native"):
                    created = created.to_native()
                fact.created_at = created
        return fact

    def get_facts(self, session_id: str, limit: int = 50) -> list[LiteralFact]:
        query = """
        MATCH (f:LiteralFact {session_id: $session_id})
        RETURN f.fact_id AS fact_id,
               f.session_id AS session_id,
               f.key AS key,
               f.kind AS kind,
               f.value AS value,
               f.unit AS unit,
               f.id_type AS id_type,
               f.importance AS importance,
               f.created_at AS created_at
        ORDER BY f.created_at DESC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, limit=limit)
            return [self._record_to_fact(r) for r in result]

    def get_fact_count(self, session_id: str) -> int:
        query = """
        MATCH (f:LiteralFact {session_id: $session_id})
        RETURN count(f) AS cnt
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            record = result.single()
            return record["cnt"] if record else 0

    @staticmethod
    def _record_to_fact(record) -> LiteralFact:
        created = record["created_at"]
        if hasattr(created, "to_native"):
            created = created.to_native()
        return LiteralFact(
            fact_id=record["fact_id"],
            session_id=record["session_id"] or "",
            key=record["key"] or "",
            kind=record["kind"] or "",
            value=record["value"] or "",
            unit=record["unit"] or "",
            id_type=record["id_type"] or "",
            importance=record["importance"] if record["importance"] is not None else 1.0,
            created_at=created,
        )

"""Neo4j-backed StateStore — manages semantic state chains."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import StateStore
from semvec_neo4j.persistence.models import SemanticState


class Neo4jStateStore(StateStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def append_state(
        self,
        session_id: str,
        state: SemanticState,
        cosine_similarity: float,
    ) -> SemanticState:
        now = datetime.now(timezone.utc)
        state.timestamp = now

        # Check if session already has a CURRENT_STATE
        query_with_prev = """
        MATCH (session:AgentSession {session_id: $session_id})
               -[old_rel:CURRENT_STATE]->(prev:SemanticState)
        CREATE (new:SemanticState {
            state_id: $state_id,
            vector: $vector,
            timestamp: datetime($timestamp),
            step: $step,
            beta: $beta,
            mean_similarity: $mean_similarity,
            variance: $variance
        })
        CREATE (new)-[:STATE_HISTORY {cosine_similarity: $cosine_similarity}]->(prev)
        DELETE old_rel
        CREATE (session)-[:CURRENT_STATE]->(new)
        RETURN new.state_id AS state_id
        """

        query_first = """
        MATCH (session:AgentSession {session_id: $session_id})
        WHERE NOT EXISTS { (session)-[:CURRENT_STATE]->() }
        CREATE (new:SemanticState {
            state_id: $state_id,
            vector: $vector,
            timestamp: datetime($timestamp),
            step: $step,
            beta: $beta,
            mean_similarity: $mean_similarity,
            variance: $variance
        })
        CREATE (session)-[:CURRENT_STATE]->(new)
        RETURN new.state_id AS state_id
        """

        params = dict(
            session_id=session_id,
            state_id=state.state_id,
            vector=state.vector,
            timestamp=now.isoformat(),
            step=state.step,
            beta=state.beta,
            mean_similarity=state.mean_similarity,
            variance=state.variance,
            cosine_similarity=cosine_similarity,
        )

        with self._driver.session(database=self._database) as db:
            # Try with previous state first
            result = db.run(query_with_prev, **params)
            record = result.single()
            if record is None:
                # No previous state — this is the first
                result = db.run(query_first, **params)
                result.consume()

        return state

    def get_current_state(self, session_id: str) -> Optional[SemanticState]:
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_STATE]->(current:SemanticState)
        RETURN current.state_id AS state_id,
               current.vector AS vector,
               current.timestamp AS timestamp,
               current.step AS step,
               current.beta AS beta,
               current.mean_similarity AS mean_similarity,
               current.variance AS variance
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_state(record)

    def get_state_history(
        self, session_id: str, limit: int = 50
    ) -> list[tuple[SemanticState, float]]:
        # Walk the STATE_HISTORY chain from current.
        # For each node, the cosine_similarity is from the edge pointing TO its predecessor
        # (i.e. the relationship that was created when this state was appended).
        # For the oldest node (no outgoing STATE_HISTORY), similarity is 1.0.
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_STATE]->(current:SemanticState)
        MATCH path = (current)-[:STATE_HISTORY*0..]->(leaf:SemanticState)
        WITH path, length(path) AS depth
        ORDER BY depth DESC
        LIMIT 1
        WITH nodes(path) AS chain, relationships(path) AS rels
        UNWIND range(0, size(chain)-1) AS idx
        WITH chain[idx] AS node,
             CASE WHEN idx < size(rels) THEN rels[idx].cosine_similarity ELSE 1.0 END AS sim,
             idx AS depth
        ORDER BY depth
        LIMIT $limit
        RETURN node.state_id AS state_id,
               node.vector AS vector,
               node.timestamp AS timestamp,
               node.step AS step,
               node.beta AS beta,
               node.mean_similarity AS mean_similarity,
               node.variance AS variance,
               sim AS cosine_similarity
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, limit=limit)
            records = list(result)
            return [
                (self._record_to_state(r), r["cosine_similarity"])
                for r in records
            ]

    def get_state_count(self, session_id: str) -> int:
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_STATE]->(current:SemanticState)
        MATCH path = (current)-[:STATE_HISTORY*0..]->(ancestor:SemanticState)
        WITH last(nodes(path)) AS leaf
        RETURN count(DISTINCT leaf) AS total
        """
        # Simpler approach: just count all states owned by this session
        count_query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_STATE]->(current:SemanticState)
        OPTIONAL MATCH (current)-[:STATE_HISTORY*0..]->(s:SemanticState)
        RETURN count(DISTINCT s) AS total
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(count_query, session_id=session_id)
            record = result.single()
            if record is None:
                return 0
            return record["total"]

    @staticmethod
    def _record_to_state(record) -> SemanticState:
        ts = record["timestamp"]
        if hasattr(ts, "to_native"):
            ts = ts.to_native()

        return SemanticState(
            state_id=record["state_id"],
            vector=list(record["vector"]) if record["vector"] else [],
            timestamp=ts,
            step=record["step"],
            beta=record["beta"],
            mean_similarity=record["mean_similarity"],
            variance=record["variance"],
        )

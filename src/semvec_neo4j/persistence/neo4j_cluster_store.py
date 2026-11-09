"""Neo4j-backed ClusterStore — manages cluster topology."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import ClusterStore
from semvec_neo4j.persistence.models import (
    AgentSession, AggregationStrategy, Cluster, SemanticState, SessionStatus,
)


class Neo4jClusterStore(ClusterStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_cluster(self, cluster: Cluster) -> Cluster:
        now = datetime.now(timezone.utc)
        cluster.created_at = now

        query = """
        CREATE (c:Cluster {
            cluster_id: $cluster_id,
            name: $name,
            strategy: $strategy,
            coupling_strength: $coupling_strength,
            created_at: datetime($now)
        })
        RETURN c.cluster_id AS cid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                strategy=cluster.strategy.value,
                coupling_strength=cluster.coupling_strength,
                now=now.isoformat(),
            ).consume()
        return cluster

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        query = """
        MATCH (c:Cluster {cluster_id: $cluster_id})
        RETURN c.cluster_id AS cluster_id,
               c.name AS name,
               c.strategy AS strategy,
               c.coupling_strength AS coupling_strength,
               c.created_at AS created_at
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id)
            record = result.single()
            if record is None:
                return None
            created_at = record["created_at"]
            if hasattr(created_at, "to_native"):
                created_at = created_at.to_native()
            return Cluster(
                cluster_id=record["cluster_id"],
                name=record["name"],
                strategy=AggregationStrategy(record["strategy"]),
                coupling_strength=record["coupling_strength"],
                created_at=created_at,
            )

    def add_member(self, cluster_id: str, session_id: str, weight: float = 1.0) -> bool:
        query = """
        MATCH (c:Cluster {cluster_id: $cluster_id})
        MATCH (s:AgentSession {session_id: $session_id})
        CREATE (s)-[:MEMBER_OF {weight: $weight, joined_at: datetime()}]->(c)
        RETURN c.cluster_id AS cid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id, session_id=session_id, weight=weight)
            return result.single() is not None

    def remove_member(self, cluster_id: str, session_id: str) -> bool:
        query = """
        MATCH (s:AgentSession {session_id: $session_id})-[r:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
        DELETE r
        RETURN true AS deleted
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id, session_id=session_id)
            return result.single() is not None

    def get_members(self, cluster_id: str) -> list[tuple[AgentSession, float]]:
        query = """
        MATCH (s:AgentSession)-[r:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
        RETURN s.session_id AS session_id,
               s.agent_id AS agent_id,
               s.status AS status,
               s.created_at AS created_at,
               s.last_active AS last_active,
               r.weight AS weight
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id)
            members = []
            for r in result:
                created_at = r["created_at"]
                last_active = r["last_active"]
                if hasattr(created_at, "to_native"):
                    created_at = created_at.to_native()
                if hasattr(last_active, "to_native"):
                    last_active = last_active.to_native()
                session = AgentSession(
                    session_id=r["session_id"],
                    agent_id=r["agent_id"],
                    status=SessionStatus(r["status"]),
                    created_at=created_at,
                    last_active=last_active,
                )
                members.append((session, r["weight"]))
            return members

    def get_member_states(self, cluster_id: str) -> list[tuple[SemanticState, float]]:
        query = """
        MATCH (s:AgentSession)-[r:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
        MATCH (s)-[:CURRENT_STATE]->(state:SemanticState)
        RETURN state.state_id AS state_id,
               state.vector AS vector,
               state.timestamp AS timestamp,
               state.step AS step,
               state.beta AS beta,
               state.mean_similarity AS mean_similarity,
               state.variance AS variance,
               r.weight AS weight
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id)
            states = []
            for r in result:
                ts = r["timestamp"]
                if hasattr(ts, "to_native"):
                    ts = ts.to_native()
                state = SemanticState(
                    state_id=r["state_id"],
                    vector=list(r["vector"]) if r["vector"] else [],
                    timestamp=ts,
                    step=r["step"],
                    beta=r["beta"],
                    mean_similarity=r["mean_similarity"],
                    variance=r["variance"],
                )
                states.append((state, r["weight"]))
            return states

    def set_aggregated_state(self, cluster_id: str, state: SemanticState) -> bool:
        now = datetime.now(timezone.utc)
        state.timestamp = now

        query = """
        MATCH (c:Cluster {cluster_id: $cluster_id})
        OPTIONAL MATCH (c)-[old:AGGREGATED_STATE]->(:SemanticState)
        DELETE old
        CREATE (agg:SemanticState {
            state_id: $state_id,
            vector: $vector,
            timestamp: datetime($now),
            step: $step,
            beta: $beta,
            mean_similarity: $mean_similarity,
            variance: $variance
        })
        CREATE (c)-[:AGGREGATED_STATE {timestamp: datetime($now)}]->(agg)
        RETURN c.cluster_id AS cid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(
                query,
                cluster_id=cluster_id,
                state_id=state.state_id,
                vector=state.vector,
                now=now.isoformat(),
                step=state.step,
                beta=state.beta,
                mean_similarity=state.mean_similarity,
                variance=state.variance,
            )
            return result.single() is not None

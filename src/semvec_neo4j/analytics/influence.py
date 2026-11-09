"""PageRank-like influence scoring on agent drift networks.

Computes which agents are most influential in the network based on
their drift patterns, state evolution, and interaction counts.

Note: This is a pure-Cypher implementation that works without GDS.
When GDS is available, PageRank can be used for higher accuracy.
"""

from __future__ import annotations

from neo4j import Driver


class InfluenceAnalyzer:
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def compute_influence_scores(
        self, session_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Compute influence scores for given sessions.

        Influence is based on:
        - Number of semantic states (interaction depth) — 40%
        - Number of drift events triggered — 30%
        - Mean drift score of events — 30%

        Returns normalized scores summing to 1.0.
        """
        query = """
        UNWIND $session_ids AS sid
        MATCH (s:AgentSession {session_id: sid})
        OPTIONAL MATCH (s)-[:CURRENT_STATE]->(current:SemanticState)
        OPTIONAL MATCH (current)-[:STATE_HISTORY*0..]->(ancestor:SemanticState)
        WITH s, sid, count(DISTINCT ancestor) AS state_count
        OPTIONAL MATCH (d:DriftEvent {session_id: sid})
        WITH sid, state_count,
             count(d) AS drift_count,
             CASE WHEN count(d) > 0 THEN avg(d.drift_score) ELSE 0 END AS avg_drift
        RETURN sid AS session_id,
               state_count,
               drift_count,
               avg_drift
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_ids=session_ids)
            records = list(result)

        if not records:
            return []

        # Normalize each component to 0-1
        max_states = max(r["state_count"] for r in records) or 1
        max_drifts = max(r["drift_count"] for r in records) or 1

        raw_scores = []
        for r in records:
            state_norm = r["state_count"] / max_states
            drift_norm = r["drift_count"] / max_drifts
            avg_drift = r["avg_drift"]

            score = 0.4 * state_norm + 0.3 * drift_norm + 0.3 * avg_drift
            raw_scores.append((r["session_id"], score))

        # Normalize to sum to 1.0
        total = sum(s for _, s in raw_scores)
        if total == 0:
            n = len(raw_scores)
            return [(sid, 1.0 / n) for sid, _ in raw_scores]

        return [(sid, score / total) for sid, score in raw_scores]

    def compute_cluster_influence(
        self, cluster_id: str
    ) -> list[tuple[str, float]]:
        """Compute influence scores for all members of a cluster."""
        query = """
        MATCH (s:AgentSession)-[:MEMBER_OF]->(c:Cluster {cluster_id: $cluster_id})
        RETURN collect(s.session_id) AS session_ids
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, cluster_id=cluster_id)
            record = result.single()
            if record is None or not record["session_ids"]:
                return []
            return self.compute_influence_scores(record["session_ids"])

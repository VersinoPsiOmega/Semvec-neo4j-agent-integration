"""Neo4j-backed RegionStore — manages region topology and consensus."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import RegionStore
from semvec_neo4j.persistence.models import (
    AggregationStrategy, Cluster, ConsensusEvent, ConsensusOutcome, Region,
)


class Neo4jRegionStore(RegionStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_region(self, region: Region) -> Region:
        query = """
        CREATE (r:Region {
            region_id: $region_id,
            name: $name,
            consensus_threshold: $consensus_threshold,
            voting_weight_scheme: $voting_weight_scheme
        })
        RETURN r.region_id AS rid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                region_id=region.region_id,
                name=region.name,
                consensus_threshold=region.consensus_threshold,
                voting_weight_scheme=region.voting_weight_scheme,
            ).consume()
        return region

    def get_region(self, region_id: str) -> Optional[Region]:
        query = """
        MATCH (r:Region {region_id: $region_id})
        RETURN r.region_id AS region_id,
               r.name AS name,
               r.consensus_threshold AS consensus_threshold,
               r.voting_weight_scheme AS voting_weight_scheme
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, region_id=region_id)
            record = result.single()
            if record is None:
                return None
            return Region(
                region_id=record["region_id"],
                name=record["name"],
                consensus_threshold=record["consensus_threshold"],
                voting_weight_scheme=record["voting_weight_scheme"],
            )

    def add_cluster_to_region(self, region_id: str, cluster_id: str) -> bool:
        query = """
        MATCH (r:Region {region_id: $region_id})
        MATCH (c:Cluster {cluster_id: $cluster_id})
        CREATE (r)-[:CONTAINS_CLUSTER]->(c)
        RETURN r.region_id AS rid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, region_id=region_id, cluster_id=cluster_id)
            return result.single() is not None

    def get_clusters_in_region(self, region_id: str) -> list[Cluster]:
        query = """
        MATCH (r:Region {region_id: $region_id})-[:CONTAINS_CLUSTER]->(c:Cluster)
        RETURN c.cluster_id AS cluster_id,
               c.name AS name,
               c.strategy AS strategy,
               c.coupling_strength AS coupling_strength,
               c.created_at AS created_at
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, region_id=region_id)
            clusters = []
            for r in result:
                created_at = r["created_at"]
                if hasattr(created_at, "to_native"):
                    created_at = created_at.to_native()
                clusters.append(Cluster(
                    cluster_id=r["cluster_id"],
                    name=r["name"],
                    strategy=AggregationStrategy(r["strategy"]),
                    coupling_strength=r["coupling_strength"],
                    created_at=created_at,
                ))
            return clusters

    def store_consensus_event(
        self, region_id: str, event: ConsensusEvent
    ) -> ConsensusEvent:
        now = datetime.now(timezone.utc)
        event.timestamp = now

        query = """
        MATCH (r:Region {region_id: $region_id})
        CREATE (ce:ConsensusEvent {
            event_id: $event_id,
            timestamp: datetime($now),
            outcome: $outcome,
            vote_count: $vote_count,
            threshold_used: $threshold_used,
            drift_score_consensus: $drift_score_consensus
        })
        CREATE (r)-[:CONSENSUS_EVENT]->(ce)
        RETURN ce.event_id AS eid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                region_id=region_id,
                event_id=event.event_id,
                now=now.isoformat(),
                outcome=event.outcome.value,
                vote_count=event.vote_count,
                threshold_used=event.threshold_used,
                drift_score_consensus=event.drift_score_consensus,
            ).consume()
        return event

    def get_consensus_events(
        self, region_id: str, limit: int = 10
    ) -> list[ConsensusEvent]:
        query = """
        MATCH (r:Region {region_id: $region_id})-[:CONSENSUS_EVENT]->(ce:ConsensusEvent)
        RETURN ce.event_id AS event_id,
               ce.timestamp AS timestamp,
               ce.outcome AS outcome,
               ce.vote_count AS vote_count,
               ce.threshold_used AS threshold_used,
               ce.drift_score_consensus AS drift_score_consensus
        ORDER BY ce.timestamp DESC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, region_id=region_id, limit=limit)
            events = []
            for r in result:
                ts = r["timestamp"]
                if hasattr(ts, "to_native"):
                    ts = ts.to_native()
                events.append(ConsensusEvent(
                    event_id=r["event_id"],
                    timestamp=ts,
                    outcome=ConsensusOutcome(r["outcome"]),
                    vote_count=r["vote_count"],
                    threshold_used=r["threshold_used"],
                    drift_score_consensus=r["drift_score_consensus"],
                ))
            return events

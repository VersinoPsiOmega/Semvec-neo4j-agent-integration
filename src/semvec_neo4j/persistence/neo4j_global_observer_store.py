"""Neo4j-backed GlobalObserverStore — the anchor for network anomalies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import GlobalObserverStore
from semvec_neo4j.persistence.models import GlobalObserver


class Neo4jGlobalObserverStore(GlobalObserverStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_observer(self, observer: GlobalObserver) -> GlobalObserver:
        # MERGE: a single observer is created once and re-sampled over time.
        now = datetime.now(timezone.utc)
        observer.last_scan = now
        query = """
        MERGE (go:GlobalObserver {observer_id: $observer_id})
        SET go.pattern_vector = $pattern_vector,
            go.last_scan = datetime($last_scan)
        RETURN go.observer_id AS oid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                observer_id=observer.observer_id,
                pattern_vector=observer.pattern_vector,
                last_scan=now.isoformat(),
            ).consume()
        return observer

    def get_observer(self, observer_id: str) -> Optional[GlobalObserver]:
        query = """
        MATCH (go:GlobalObserver {observer_id: $observer_id})
        RETURN go.observer_id AS observer_id,
               go.pattern_vector AS pattern_vector,
               go.last_scan AS last_scan
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, observer_id=observer_id)
            record = result.single()
            if record is None:
                return None
            last_scan = record["last_scan"]
            if hasattr(last_scan, "to_native"):
                last_scan = last_scan.to_native()
            return GlobalObserver(
                observer_id=record["observer_id"],
                pattern_vector=list(record["pattern_vector"] or []),
                last_scan=last_scan,
            )

    def observe_region(self, observer_id: str, region_id: str) -> bool:
        # MERGE the region by its unique region_id rather than MATCH: the
        # in-process semvec region may not be mirrored into Neo4j yet when the
        # observer is wired up. MERGE-by-id is idempotent and resolves to the
        # same node once the full Region (with name etc.) is persisted.
        query = """
        MATCH (go:GlobalObserver {observer_id: $observer_id})
        MERGE (r:Region {region_id: $region_id})
        MERGE (go)-[:OBSERVES]->(r)
        RETURN go.observer_id AS oid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, observer_id=observer_id, region_id=region_id)
            return result.single() is not None

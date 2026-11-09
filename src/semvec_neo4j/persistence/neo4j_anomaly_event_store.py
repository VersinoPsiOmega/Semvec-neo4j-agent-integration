"""Neo4j-backed AnomalyEventStore — manages anomaly event nodes.

Mirrors the network anomalies surfaced by the global observer
(``SemvecClient.get_anomalies``) into the graph, linked to the detecting
observer via ``(GlobalObserver)-[:DETECTED]->(AnomalyEvent)`` — the
network-level counterpart to ``(SemanticState)-[:TRIGGERED]->(DriftEvent)``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import Driver

from semvec_neo4j.persistence.base import AnomalyEventStore
from semvec_neo4j.persistence.models import AnomalyEvent, AnomalyType


def _to_anomaly_type(value) -> AnomalyType:
    """Coerce an upstream anomaly-type value into the AnomalyType enum.

    semvec may emit types outside our enum; fall back to SYSTEMIC_DRIFT so
    persistence never fails on an unrecognized value.
    """
    if isinstance(value, AnomalyType):
        return value
    try:
        return AnomalyType(value)
    except ValueError:
        return AnomalyType.SYSTEMIC_DRIFT


class Neo4jAnomalyEventStore(AnomalyEventStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_anomaly_event(
        self, observer_id: str, event: AnomalyEvent
    ) -> AnomalyEvent:
        now = datetime.now(timezone.utc)
        event.timestamp = now
        event.observer_id = observer_id

        query = """
        MATCH (go:GlobalObserver {observer_id: $observer_id})
        CREATE (a:AnomalyEvent {
            event_id: $event_id,
            timestamp: datetime($timestamp),
            anomaly_type: $anomaly_type,
            severity: $severity,
            affected_clusters: $affected_clusters,
            description: $description,
            observer_id: $observer_id
        })
        CREATE (go)-[:DETECTED]->(a)
        RETURN a.event_id AS eid
        """
        params = dict(
            observer_id=observer_id,
            event_id=event.event_id,
            timestamp=now.isoformat(),
            anomaly_type=_to_anomaly_type(event.anomaly_type).value,
            severity=event.severity,
            affected_clusters=event.affected_clusters,
            description=event.description,
        )
        with self._driver.session(database=self._database) as db:
            db.run(query, **params).consume()
        return event

    def get_anomaly_events(
        self, observer_id: str, limit: int = 20
    ) -> list[AnomalyEvent]:
        query = """
        MATCH (a:AnomalyEvent {observer_id: $observer_id})
        RETURN a.event_id AS event_id,
               a.timestamp AS timestamp,
               a.anomaly_type AS anomaly_type,
               a.severity AS severity,
               a.affected_clusters AS affected_clusters,
               a.description AS description,
               a.observer_id AS observer_id
        ORDER BY a.timestamp DESC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, observer_id=observer_id, limit=limit)
            return [self._record_to_event(r) for r in result]

    def get_anomaly_event_count(self, observer_id: str) -> int:
        query = """
        MATCH (a:AnomalyEvent {observer_id: $observer_id})
        RETURN count(a) AS cnt
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, observer_id=observer_id)
            record = result.single()
            return record["cnt"] if record else 0

    @staticmethod
    def _record_to_event(record) -> AnomalyEvent:
        ts = record["timestamp"]
        if hasattr(ts, "to_native"):
            ts = ts.to_native()
        return AnomalyEvent(
            event_id=record["event_id"],
            timestamp=ts,
            anomaly_type=_to_anomaly_type(record["anomaly_type"]),
            severity=record["severity"],
            affected_clusters=list(record["affected_clusters"] or []),
            description=record["description"] or "",
            observer_id=record["observer_id"] or "",
        )

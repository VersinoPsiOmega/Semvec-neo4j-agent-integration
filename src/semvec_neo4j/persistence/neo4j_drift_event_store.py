"""Neo4j-backed DriftEventStore — manages drift event nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import DriftEventStore
from semvec_neo4j.persistence.models import DriftEvent, DriftPhase, DriftSeverity


# Severity ordering for filtering
_SEVERITY_RANK = {
    DriftSeverity.LOW: 0,
    DriftSeverity.MEDIUM: 1,
    DriftSeverity.HIGH: 2,
    DriftSeverity.CRITICAL: 3,
}


class Neo4jDriftEventStore(DriftEventStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_drift_event(
        self, session_id: str, state_id: str, event: DriftEvent
    ) -> DriftEvent:
        now = datetime.now(timezone.utc)
        event.timestamp = now

        query = """
        MATCH (state:SemanticState {state_id: $state_id})
        CREATE (d:DriftEvent {
            event_id: $event_id,
            timestamp: datetime($timestamp),
            drift_score: $drift_score,
            drift_phase: $drift_phase,
            topic_switch: $topic_switch,
            cosine_drop: $cosine_drop,
            mean_sim: $mean_sim,
            variance: $variance,
            severity: $severity,
            session_id: $session_id
        })
        CREATE (state)-[:TRIGGERED]->(d)
        RETURN d.event_id AS eid
        """
        params = dict(
            state_id=state_id,
            session_id=session_id,
            event_id=event.event_id,
            timestamp=now.isoformat(),
            drift_score=event.drift_score,
            drift_phase=event.drift_phase.value,
            topic_switch=event.topic_switch,
            cosine_drop=event.cosine_drop,
            mean_sim=event.mean_sim,
            variance=event.variance,
            severity=event.severity.value,
        )

        with self._driver.session(database=self._database) as db:
            db.run(query, **params).consume()

        return event

    def get_drift_events(
        self,
        session_id: str,
        limit: int = 20,
        min_severity: Optional[DriftSeverity] = None,
        since: Optional[datetime] = None,
    ) -> list[DriftEvent]:
        conditions = ["d.session_id = $session_id"]
        params: dict = {"session_id": session_id, "limit": limit}

        if min_severity is not None:
            rank = _SEVERITY_RANK[min_severity]
            allowed = [s.value for s, r in _SEVERITY_RANK.items() if r >= rank]
            conditions.append("d.severity IN $allowed_severities")
            params["allowed_severities"] = allowed

        if since is not None:
            conditions.append("d.timestamp >= datetime($since)")
            params["since"] = since.isoformat()

        where_clause = " AND ".join(conditions)

        query = f"""
        MATCH (d:DriftEvent)
        WHERE {where_clause}
        RETURN d.event_id AS event_id,
               d.timestamp AS timestamp,
               d.drift_score AS drift_score,
               d.drift_phase AS drift_phase,
               d.topic_switch AS topic_switch,
               d.cosine_drop AS cosine_drop,
               d.mean_sim AS mean_sim,
               d.variance AS variance,
               d.severity AS severity
        ORDER BY d.timestamp DESC
        LIMIT $limit
        """

        with self._driver.session(database=self._database) as db:
            result = db.run(query, **params)
            return [self._record_to_event(r) for r in result]

    def get_drift_event_count(self, session_id: str) -> int:
        query = """
        MATCH (d:DriftEvent {session_id: $session_id})
        RETURN count(d) AS cnt
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            record = result.single()
            return record["cnt"] if record else 0

    @staticmethod
    def _record_to_event(record) -> DriftEvent:
        ts = record["timestamp"]
        if hasattr(ts, "to_native"):
            ts = ts.to_native()

        return DriftEvent(
            event_id=record["event_id"],
            timestamp=ts,
            drift_score=record["drift_score"],
            drift_phase=DriftPhase(record["drift_phase"]) if record["drift_phase"] else DriftPhase.STABLE,
            topic_switch=record["topic_switch"],
            cosine_drop=record["cosine_drop"],
            mean_sim=record["mean_sim"],
            variance=record["variance"],
            severity=DriftSeverity(record["severity"]),
        )

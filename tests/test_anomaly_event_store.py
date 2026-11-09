"""Tests for Neo4j AnomalyEventStore."""

import pytest

from semvec_neo4j.persistence.models import (
    AnomalyEvent, AnomalyType, GlobalObserver,
)
from semvec_neo4j.persistence.neo4j_global_observer_store import (
    Neo4jGlobalObserverStore,
)
from semvec_neo4j.persistence.neo4j_anomaly_event_store import (
    Neo4jAnomalyEventStore,
)


@pytest.fixture
def observer_store(neo4j_driver):
    return Neo4jGlobalObserverStore(neo4j_driver, database="neo4j")


@pytest.fixture
def store(neo4j_driver):
    return Neo4jAnomalyEventStore(neo4j_driver, database="neo4j")


@pytest.fixture
def observer(observer_store):
    obs = GlobalObserver(observer_id="obs-1")
    observer_store.create_observer(obs)
    return obs


class TestAnomalyCreate:
    def test_create(self, store, observer):
        event = AnomalyEvent(
            anomaly_type=AnomalyType.SYSTEMIC_DRIFT,
            severity=0.8,
            affected_clusters=["c1", "c2"],
            description="network-wide drift",
        )
        result = store.create_anomaly_event(observer.observer_id, event)
        assert result.timestamp is not None
        assert result.observer_id == observer.observer_id

    def test_detected_relationship(self, store, observer, db_session):
        event = AnomalyEvent(severity=0.5)
        store.create_anomaly_event(observer.observer_id, event)
        result = db_session.run(
            "MATCH (:GlobalObserver {observer_id: $oid})-[:DETECTED]->"
            "(a:AnomalyEvent {event_id: $eid}) RETURN a.event_id AS eid",
            oid=observer.observer_id, eid=event.event_id,
        )
        assert result.single() is not None

    def test_unknown_anomaly_type_falls_back(self, store, observer):
        event = AnomalyEvent(severity=0.3)
        event.anomaly_type = "some_unknown_type"  # not in the enum
        store.create_anomaly_event(observer.observer_id, event)
        got = store.get_anomaly_events(observer.observer_id)
        assert got[0].anomaly_type == AnomalyType.SYSTEMIC_DRIFT

    def test_roundtrip_preserves_fields(self, store, observer):
        store.create_anomaly_event(observer.observer_id, AnomalyEvent(
            anomaly_type=AnomalyType.CLUSTER_DIVERGENCE,
            severity=0.42,
            affected_clusters=["x"],
            description="divergence",
        ))
        e = store.get_anomaly_events(observer.observer_id)[0]
        assert e.anomaly_type == AnomalyType.CLUSTER_DIVERGENCE
        assert e.severity == 0.42
        assert e.affected_clusters == ["x"]
        assert e.description == "divergence"


class TestAnomalyQuery:
    def test_get_events(self, store, observer):
        for i in range(4):
            store.create_anomaly_event(observer.observer_id, AnomalyEvent(severity=0.1 * i))
        assert len(store.get_anomaly_events(observer.observer_id)) == 4

    def test_count(self, store, observer):
        for _ in range(3):
            store.create_anomaly_event(observer.observer_id, AnomalyEvent())
        assert store.get_anomaly_event_count(observer.observer_id) == 3

    def test_empty(self, store, observer):
        assert store.get_anomaly_events(observer.observer_id) == []
        assert store.get_anomaly_event_count(observer.observer_id) == 0

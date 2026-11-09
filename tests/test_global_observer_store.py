"""Tests for Neo4j GlobalObserverStore."""

import pytest

from semvec_neo4j.persistence.models import GlobalObserver, Region
from semvec_neo4j.persistence.neo4j_global_observer_store import (
    Neo4jGlobalObserverStore,
)
from semvec_neo4j.persistence.neo4j_region_store import Neo4jRegionStore
from tests.conftest import make_random_vector


@pytest.fixture
def store(neo4j_driver):
    return Neo4jGlobalObserverStore(neo4j_driver, database="neo4j")


@pytest.fixture
def region_store(neo4j_driver):
    return Neo4jRegionStore(neo4j_driver, database="neo4j")


class TestObserver:
    def test_create_and_get(self, store):
        obs = GlobalObserver(observer_id="obs-1", pattern_vector=make_random_vector(768))
        result = store.create_observer(obs)
        assert result.last_scan is not None
        got = store.get_observer("obs-1")
        assert got is not None
        assert len(got.pattern_vector) == 768

    def test_create_idempotent_updates_scan(self, store):
        store.create_observer(GlobalObserver(observer_id="obs-1"))
        # MERGE on observer_id + SET: re-creating updates the pattern vector.
        store.create_observer(GlobalObserver(observer_id="obs-1", pattern_vector=[1.0, 2.0]))
        got = store.get_observer("obs-1")
        assert got.pattern_vector == [1.0, 2.0]

    def test_get_missing(self, store):
        assert store.get_observer("nope") is None


class TestObserverRegionLink:
    def test_observe_region(self, store, region_store, db_session):
        store.create_observer(GlobalObserver(observer_id="obs-1"))
        region_store.create_region(Region(region_id="reg-1", name="ward"))
        assert store.observe_region("obs-1", "reg-1") is True

        result = db_session.run(
            "MATCH (:GlobalObserver {observer_id: 'obs-1'})-[:OBSERVES]->(r:Region) "
            "RETURN r.region_id AS rid"
        )
        assert result.single()["rid"] == "reg-1"

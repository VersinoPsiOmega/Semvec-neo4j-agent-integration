"""Tests for ClusterStore and RegionStore — multi-agent topology."""

import pytest

from semvec_neo4j.persistence.models import (
    AgentSession, Cluster, Region, AggregationStrategy,
    ConsensusEvent, ConsensusOutcome, SemanticState,
)
from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_cluster_store import Neo4jClusterStore
from semvec_neo4j.persistence.neo4j_region_store import Neo4jRegionStore
from tests.conftest import make_random_vector


@pytest.fixture
def session_store(neo4j_driver):
    return Neo4jSessionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def state_store(neo4j_driver):
    return Neo4jStateStore(neo4j_driver, database="neo4j")


@pytest.fixture
def cluster_store(neo4j_driver):
    return Neo4jClusterStore(neo4j_driver, database="neo4j")


@pytest.fixture
def region_store(neo4j_driver):
    return Neo4jRegionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def two_sessions(session_store, state_store):
    """Create two sessions each with a semantic state."""
    sessions = []
    for i in range(2):
        s = AgentSession(session_id=f"cluster-sess-{i}", agent_id=f"agent-{i}")
        session_store.create_session(s)
        state = SemanticState(
            vector=make_random_vector(768, seed=i + 1), step=0, beta=0.1,
        )
        state_store.append_state(s.session_id, state, cosine_similarity=1.0)
        sessions.append(s)
    return sessions


class TestClusterStoreBasic:
    def test_create_cluster(self, cluster_store):
        cluster = Cluster(name="finance-cluster", strategy=AggregationStrategy.WEIGHTED_AVG)
        result = cluster_store.create_cluster(cluster)
        assert result.cluster_id == cluster.cluster_id

    def test_get_cluster(self, cluster_store):
        cluster = Cluster(cluster_id="get-c1", name="test-cluster")
        cluster_store.create_cluster(cluster)
        result = cluster_store.get_cluster("get-c1")
        assert result is not None
        assert result.name == "test-cluster"

    def test_get_nonexistent_cluster(self, cluster_store):
        assert cluster_store.get_cluster("nope") is None


class TestClusterMembership:
    def test_add_member(self, cluster_store, two_sessions):
        cluster = Cluster(name="test-cluster")
        cluster_store.create_cluster(cluster)

        result = cluster_store.add_member(cluster.cluster_id, two_sessions[0].session_id, weight=0.8)
        assert result is True

    def test_get_members(self, cluster_store, two_sessions):
        cluster = Cluster(name="test-cluster")
        cluster_store.create_cluster(cluster)

        cluster_store.add_member(cluster.cluster_id, two_sessions[0].session_id, weight=0.8)
        cluster_store.add_member(cluster.cluster_id, two_sessions[1].session_id, weight=0.5)

        members = cluster_store.get_members(cluster.cluster_id)
        assert len(members) == 2
        weights = {s.session_id: w for s, w in members}
        assert abs(weights[two_sessions[0].session_id] - 0.8) < 1e-5
        assert abs(weights[two_sessions[1].session_id] - 0.5) < 1e-5

    def test_remove_member(self, cluster_store, two_sessions):
        cluster = Cluster(name="test-cluster")
        cluster_store.create_cluster(cluster)
        cluster_store.add_member(cluster.cluster_id, two_sessions[0].session_id)
        cluster_store.add_member(cluster.cluster_id, two_sessions[1].session_id)

        cluster_store.remove_member(cluster.cluster_id, two_sessions[0].session_id)
        members = cluster_store.get_members(cluster.cluster_id)
        assert len(members) == 1

    def test_get_member_states(self, cluster_store, two_sessions):
        cluster = Cluster(name="test-cluster")
        cluster_store.create_cluster(cluster)
        cluster_store.add_member(cluster.cluster_id, two_sessions[0].session_id, weight=1.0)
        cluster_store.add_member(cluster.cluster_id, two_sessions[1].session_id, weight=1.0)

        states = cluster_store.get_member_states(cluster.cluster_id)
        assert len(states) == 2
        for state, weight in states:
            assert len(state.vector) == 768

    def test_set_aggregated_state(self, cluster_store):
        cluster = Cluster(name="test-cluster")
        cluster_store.create_cluster(cluster)

        agg_state = SemanticState(vector=make_random_vector(768, seed=99), step=10)
        result = cluster_store.set_aggregated_state(cluster.cluster_id, agg_state)
        assert result is True


class TestRegionStore:
    def test_create_region(self, region_store):
        region = Region(name="us-east", consensus_threshold=0.7)
        result = region_store.create_region(region)
        assert result.region_id == region.region_id

    def test_get_region(self, region_store):
        region = Region(region_id="get-r1", name="eu-west", consensus_threshold=0.6)
        region_store.create_region(region)
        result = region_store.get_region("get-r1")
        assert result is not None
        assert result.name == "eu-west"
        assert abs(result.consensus_threshold - 0.6) < 1e-5

    def test_add_cluster_to_region(self, region_store, cluster_store):
        region = Region(name="us-east")
        region_store.create_region(region)
        cluster = Cluster(name="finance")
        cluster_store.create_cluster(cluster)

        result = region_store.add_cluster_to_region(region.region_id, cluster.cluster_id)
        assert result is True

        clusters = region_store.get_clusters_in_region(region.region_id)
        assert len(clusters) == 1
        assert clusters[0].cluster_id == cluster.cluster_id

    def test_consensus_event(self, region_store):
        region = Region(name="us-east")
        region_store.create_region(region)

        event = ConsensusEvent(
            outcome=ConsensusOutcome.AGREED,
            vote_count=5,
            threshold_used=0.6,
            drift_score_consensus=0.72,
        )
        result = region_store.store_consensus_event(region.region_id, event)
        assert result.event_id == event.event_id

        events = region_store.get_consensus_events(region.region_id)
        assert len(events) == 1
        assert events[0].outcome == ConsensusOutcome.AGREED

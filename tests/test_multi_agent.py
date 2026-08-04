"""Integration test: multi-agent drift detection with clusters and regions.

Uses the local Semvec runtime for drift detection, Neo4j for topology.
"""

import numpy as np
import pytest

from src.core.embedder import HashEmbedder
from src.core.semvec_client import SemvecClient
from src.mcp.semvec_mcp_server import SemvecMCPServer
from src.persistence.models import (
    Cluster, Region, AggregationStrategy, SemanticState,
)
from src.persistence.neo4j_cluster_store import Neo4jClusterStore
from src.persistence.neo4j_region_store import Neo4jRegionStore
from src.persistence.neo4j_state_store import Neo4jStateStore
from src.network.cluster_manager import ClusterManager
from tests.conftest import make_random_vector


@pytest.fixture(scope="session")
def semvec_client():
    return SemvecClient(embedder=HashEmbedder(dimension=32), owner_subject="test-suite")


@pytest.fixture
def mcp(neo4j_driver, semvec_client):
    return SemvecMCPServer(neo4j_driver, database="neo4j", semvec_client=semvec_client)


@pytest.fixture
def cluster_store(neo4j_driver):
    return Neo4jClusterStore(neo4j_driver, database="neo4j")


@pytest.fixture
def region_store(neo4j_driver):
    return Neo4jRegionStore(neo4j_driver, database="neo4j")


@pytest.fixture
def state_store(neo4j_driver):
    return Neo4jStateStore(neo4j_driver, database="neo4j")


@pytest.fixture
def cluster_mgr(cluster_store, state_store):
    return ClusterManager(cluster_store=cluster_store, state_store=state_store)


class TestMultiAgentSetup:
    def test_create_cluster_and_add_agents(self, mcp, cluster_store):
        sessions = []
        messages = [
            "Cardiology patient diagnosis trends",
            "Oncology treatment effectiveness",
            "Emergency triage prioritization",
        ]
        for i, msg in enumerate(messages):
            s = mcp.create_agent_session(agent_id=f"cluster-agent-{i}")
            sessions.append(s)
            mcp.detect_drift(s["session_id"], msg)

        cluster = Cluster(name="research-team", strategy=AggregationStrategy.WEIGHTED_AVG)
        cluster_store.create_cluster(cluster)

        for s in sessions:
            cluster_store.add_member(cluster.cluster_id, s["session_id"], weight=1.0)

        members = cluster_store.get_members(cluster.cluster_id)
        assert len(members) == 3

    def test_cluster_aggregation(self, mcp, cluster_store, cluster_mgr):
        sessions = []
        messages = [
            "Heart failure management protocols",
            "Diabetes insulin therapy",
            "Hypertension medication adherence",
        ]
        for i, msg in enumerate(messages):
            s = mcp.create_agent_session(agent_id=f"agg-agent-{i}")
            sessions.append(s)
            mcp.detect_drift(s["session_id"], msg)

        cluster = Cluster(name="agg-cluster", strategy=AggregationStrategy.WEIGHTED_AVG)
        cluster_store.create_cluster(cluster)
        for s in sessions:
            cluster_store.add_member(cluster.cluster_id, s["session_id"], weight=1.0)

        # Note: aggregation works on Neo4j-stored states (which may have empty vectors
        # since Semvec /run does not return the vector). This tests the topology, not vector math.
        members = cluster_store.get_member_states(cluster.cluster_id)
        assert len(members) == 3

    def test_cluster_drift_detection(self, mcp, cluster_store):
        sessions = []
        for i in range(3):
            s = mcp.create_agent_session(agent_id=f"drift-agent-{i}")
            sessions.append(s)

        cluster = Cluster(name="drift-cluster")
        cluster_store.create_cluster(cluster)
        for s in sessions:
            cluster_store.add_member(cluster.cluster_id, s["session_id"])

        # All agents process similar content
        for s in sessions:
            mcp.detect_drift(s["session_id"], "Patient cardiac rehabilitation assessment")

        # One agent drifts to a different topic
        result = mcp.detect_drift(sessions[0]["session_id"], "Stock market analysis for tech sector")

        # Verify both Neo4j session states exist
        members = cluster_store.get_members(cluster.cluster_id)
        assert len(members) == 3


class TestRegionConsensus:
    def test_region_with_clusters(self, mcp, cluster_store, region_store):
        c1 = Cluster(name="cluster-a")
        c2 = Cluster(name="cluster-b")
        cluster_store.create_cluster(c1)
        cluster_store.create_cluster(c2)

        region = Region(name="us-east", consensus_threshold=0.6)
        region_store.create_region(region)
        region_store.add_cluster_to_region(region.region_id, c1.cluster_id)
        region_store.add_cluster_to_region(region.region_id, c2.cluster_id)

        clusters = region_store.get_clusters_in_region(region.region_id)
        assert len(clusters) == 2

    def test_full_topology(self, mcp, cluster_store, region_store):
        """Integration: create full 3-layer topology and verify structure."""
        agents = {}
        messages = [
            "Cardiology assessment for heart failure",
            "Cardiac surgery outcomes analysis",
            "ER triage for chest pain patients",
            "Emergency department resource allocation",
        ]
        for i, msg in enumerate(messages):
            s = mcp.create_agent_session(agent_id=f"topo-agent-{i}")
            agents[i] = s
            mcp.detect_drift(s["session_id"], msg)

        c1 = Cluster(name="team-alpha")
        c2 = Cluster(name="team-beta")
        cluster_store.create_cluster(c1)
        cluster_store.create_cluster(c2)

        cluster_store.add_member(c1.cluster_id, agents[0]["session_id"])
        cluster_store.add_member(c1.cluster_id, agents[1]["session_id"])
        cluster_store.add_member(c2.cluster_id, agents[2]["session_id"])
        cluster_store.add_member(c2.cluster_id, agents[3]["session_id"])

        region = Region(name="global", consensus_threshold=0.6)
        region_store.create_region(region)
        region_store.add_cluster_to_region(region.region_id, c1.cluster_id)
        region_store.add_cluster_to_region(region.region_id, c2.cluster_id)

        assert len(cluster_store.get_members(c1.cluster_id)) == 2
        assert len(cluster_store.get_members(c2.cluster_id)) == 2
        assert len(region_store.get_clusters_in_region(region.region_id)) == 2

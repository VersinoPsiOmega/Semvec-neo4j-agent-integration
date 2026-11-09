# ABOUTME: Tests for persisting a ConsensusEngine vote round into Neo4j (store and MCP server).
# ABOUTME: Engine, proposal, votes and outcome must form one graph hanging off the region.

import pytest

from semvec_neo4j.core.embedder import HashEmbedder
from semvec_neo4j.core.semvec_client import SemvecClient
from semvec_neo4j.mcp.semvec_mcp_server import SemvecMCPServer
from semvec_neo4j.persistence.models import ConsensusOutcome, ConsensusRound, ConsensusVote
from semvec_neo4j.persistence.neo4j_consensus_store import Neo4jConsensusStore

REGION_ID = "region-ct"
CONSENSUS_LABELS = ["ConsensusEngine", "ConsensusProposal", "ConsensusVote", "ConsensusEvent"]


@pytest.fixture
def region(neo4j_driver):
    neo4j_driver.execute_query(
        """
        CREATE (:Region {region_id: $rid, name: 'hospital-network'})
        CREATE (:AgentSession {session_id: 'ct-s1', agent_id: 'cardio-memorial'})
        CREATE (:AgentSession {session_id: 'ct-s2', agent_id: 'emerg-riverside'})
        """,
        rid=REGION_ID,
        database_="neo4j",
    )
    return REGION_ID


@pytest.fixture
def vote_round():
    return ConsensusRound(
        engine_id="eng-1",
        level="qualified_majority",
        proposal_id="prop-1",
        proposal_type="admin_pivot_alert",
        rationale="Both clusters drifted to facility-admin queries.",
        votes=[
            ConsensusVote(voter="cardio-memorial", approve=True, weight=1.0),
            ConsensusVote(voter="emerg-riverside", approve=False, weight=1.0),
            ConsensusVote(voter="hospital-supervisor", approve=True, weight=1.5),
        ],
        accepted=True,
        ratio=0.714,
    )


def _count(driver, pattern: str, **params) -> int:
    return driver.execute_query(f"MATCH {pattern} RETURN count(*) AS c", database_="neo4j", **params).records[0]["c"]


class TestNeo4jConsensusStore:
    def test_round_hangs_off_the_region(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act
        store.store_round(region, vote_round)

        # Assert
        assert _count(
            neo4j_driver,
            "(:Region {region_id: $rid})-[:HAS_CONSENSUS_ENGINE]->(:ConsensusEngine {engine_id: 'eng-1'})"
            "-[:PROPOSED]->(:ConsensusProposal {proposal_id: 'prop-1', accepted: true})",
            rid=region,
        ) == 1

    def test_every_vote_is_attached_to_the_proposal(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act
        store.store_round(region, vote_round)

        # Assert
        assert _count(neo4j_driver, "(:ConsensusProposal {proposal_id: 'prop-1'})-[:RECEIVED_VOTE]->(:ConsensusVote)") == 3

    def test_votes_of_agent_sessions_are_cast_by_them(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act
        store.store_round(region, vote_round)

        # Assert
        voters = neo4j_driver.execute_query(
            "MATCH (s:AgentSession)-[:CAST]->(v:ConsensusVote) RETURN s.agent_id AS a, v.approve AS ok ORDER BY a",
            database_="neo4j",
        ).records
        assert [(r["a"], r["ok"]) for r in voters] == [("cardio-memorial", True), ("emerg-riverside", False)]

    def test_outcome_is_a_consensus_event_of_the_region(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act
        event = store.store_round(region, vote_round)

        # Assert
        assert event.outcome is ConsensusOutcome.AGREED
        assert event.vote_count == 3
        assert _count(
            neo4j_driver,
            "(:Region {region_id: $rid})-[:CONSENSUS_EVENT]->(:ConsensusEvent {outcome: 'agreed'})"
            "<-[:DECIDED]-(:ConsensusProposal {proposal_id: 'prop-1'})",
            rid=region,
        ) == 1

    def test_rejected_round_is_a_disagreed_event(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")
        vote_round.accepted = False

        # Act
        event = store.store_round(region, vote_round)

        # Assert
        assert event.outcome is ConsensusOutcome.DISAGREED

    def test_unknown_region_raises_instead_of_writing_nothing(self, neo4j_driver, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act / Assert
        with pytest.raises(ValueError, match="Region not found"):
            store.store_round("no-such-region", vote_round)

    def test_no_consensus_node_is_left_unconnected(self, neo4j_driver, region, vote_round):
        # Arrange
        store = Neo4jConsensusStore(neo4j_driver, database="neo4j")

        # Act
        store.store_round(region, vote_round)

        # Assert
        for label in CONSENSUS_LABELS:
            assert _count(neo4j_driver, f"(n:{label}) WHERE NOT (n)--()") == 0, label


class TestMCPConsensusRound:
    def test_runs_the_vote_in_semvec_and_persists_it(self, neo4j_driver, region):
        # Arrange
        server = SemvecMCPServer(
            neo4j_driver, database="neo4j", semvec_client=SemvecClient(embedder=HashEmbedder(dimension=16))
        )

        # Act
        verdict = server.run_consensus_round(
            region_id=region,
            local_id="hospital-orchestrator",
            network_id="hospital-network",
            level="qualified_majority",
            voters={"cardio-memorial": 1.0, "emerg-riverside": 1.0, "hospital-supervisor": 1.5},
            proposal_type="admin_pivot_alert",
            rationale="Both clusters drifted to facility-admin queries.",
            votes={"cardio-memorial": True, "emerg-riverside": False, "hospital-supervisor": True},
        )

        # Assert
        assert verdict["accepted"] is True
        assert verdict["ratio"] == pytest.approx(2.5 / 3.5, abs=1e-3)
        assert _count(neo4j_driver, "(:ConsensusProposal)-[:RECEIVED_VOTE]->(:ConsensusVote)") == 3
        assert _count(neo4j_driver, "(:Region {region_id: $rid})-[:CONSENSUS_EVENT]->(:ConsensusEvent)", rid=region) == 1

# ABOUTME: Neo4j-backed ConsensusStore: mirrors a ConsensusEngine vote round into the graph.
# ABOUTME: Region -> engine -> proposal -> votes (cast by agent sessions) and the outcome as a ConsensusEvent.

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import Driver

from semvec_neo4j.persistence.base import ConsensusStore
from semvec_neo4j.persistence.models import ConsensusEvent, ConsensusOutcome, ConsensusRound

# One transaction: every node of the round is created together with the
# relationship that connects it, so no consensus node is ever left floating.
_STORE_ROUND = """
MATCH (r:Region {region_id: $region_id})
MERGE (e:ConsensusEngine {engine_id: $engine_id})
SET e.level = $level
MERGE (r)-[:HAS_CONSENSUS_ENGINE]->(e)
CREATE (p:ConsensusProposal {
    proposal_id: $proposal_id,
    proposal_type: $proposal_type,
    rationale: $rationale,
    status: $status,
    accepted: $accepted,
    ratio: $ratio,
    created_at: datetime($now)
})
CREATE (e)-[:PROPOSED]->(p)
CREATE (ce:ConsensusEvent {
    event_id: $event_id,
    timestamp: datetime($now),
    outcome: $outcome,
    vote_count: $vote_count,
    drift_score_consensus: 0.0
})
CREATE (r)-[:CONSENSUS_EVENT]->(ce)
CREATE (p)-[:DECIDED]->(ce)
WITH p
UNWIND $votes AS vote
CREATE (v:ConsensusVote {voter: vote.voter, approve: vote.approve, weight: vote.weight,
                         recorded_at: datetime($now)})
CREATE (p)-[:RECEIVED_VOTE]->(v)
WITH v, vote
OPTIONAL MATCH (s:AgentSession {agent_id: vote.voter})
FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END | CREATE (s)-[:CAST]->(v))
"""


class Neo4jConsensusStore(ConsensusStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def store_round(self, region_id: str, vote_round: ConsensusRound) -> ConsensusEvent:
        found = self._driver.execute_query(
            "MATCH (r:Region {region_id: $region_id}) RETURN count(r) AS c",
            region_id=region_id,
            database_=self._database,
        ).records[0]["c"]
        if not found:
            raise ValueError(f"Region not found: {region_id}")
        now = datetime.now(timezone.utc)
        event = ConsensusEvent(
            timestamp=now,
            outcome=ConsensusOutcome.AGREED if vote_round.accepted else ConsensusOutcome.DISAGREED,
            vote_count=len(vote_round.votes),
            threshold_used=None,
        )
        self._driver.execute_query(
            _STORE_ROUND,
            region_id=region_id,
            engine_id=vote_round.engine_id,
            level=vote_round.level,
            proposal_id=vote_round.proposal_id,
            proposal_type=vote_round.proposal_type,
            rationale=vote_round.rationale,
            status="accepted" if vote_round.accepted else "rejected",
            accepted=vote_round.accepted,
            ratio=vote_round.ratio,
            event_id=event.event_id,
            outcome=event.outcome.value,
            vote_count=event.vote_count,
            votes=[{"voter": v.voter, "approve": v.approve, "weight": v.weight} for v in vote_round.votes],
            now=now.isoformat(),
            database_=self._database,
        )
        return event

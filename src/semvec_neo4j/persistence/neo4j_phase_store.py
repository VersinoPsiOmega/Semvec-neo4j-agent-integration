"""Neo4j-backed PhaseStore — manages phase nodes and transitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import PhaseStore
from semvec_neo4j.persistence.models import Phase, PhaseName


class Neo4jPhaseStore(PhaseStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def set_phase(self, session_id: str, phase: Phase) -> Phase:
        now = datetime.now(timezone.utc)
        phase.entered_at = now

        # Try to transition from existing phase
        query_transition = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[old_rel:CURRENT_PHASE]->(prev:Phase)
        SET prev.exited_at = datetime($now)
        CREATE (new:Phase {
            phase_id: $phase_id,
            name: $name,
            entered_at: datetime($now),
            srs_score: $srs_score,
            tc_score: $tc_score,
            fsm_state: $fsm_state,
            be_score: $be_score,
            markov_probability: $markov_probability,
            rule_score: $rule_score
        })
        CREATE (prev)-[:TRANSITIONED_TO {timestamp: datetime($now)}]->(new)
        DELETE old_rel
        CREATE (session)-[:CURRENT_PHASE]->(new)
        RETURN new.phase_id AS pid
        """

        query_first = """
        MATCH (session:AgentSession {session_id: $session_id})
        WHERE NOT EXISTS { (session)-[:CURRENT_PHASE]->() }
        CREATE (new:Phase {
            phase_id: $phase_id,
            name: $name,
            entered_at: datetime($now),
            srs_score: $srs_score,
            tc_score: $tc_score,
            fsm_state: $fsm_state,
            be_score: $be_score,
            markov_probability: $markov_probability,
            rule_score: $rule_score
        })
        CREATE (session)-[:CURRENT_PHASE]->(new)
        RETURN new.phase_id AS pid
        """

        params = dict(
            session_id=session_id,
            phase_id=phase.phase_id,
            name=phase.name.value,
            now=now.isoformat(),
            srs_score=phase.srs_score,
            tc_score=phase.tc_score,
            fsm_state=phase.fsm_state,
            be_score=phase.be_score,
            markov_probability=phase.markov_probability,
            rule_score=phase.rule_score,
        )

        with self._driver.session(database=self._database) as db:
            result = db.run(query_transition, **params)
            record = result.single()
            if record is None:
                db.run(query_first, **params).consume()

        return phase

    def get_current_phase(self, session_id: str) -> Optional[Phase]:
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_PHASE]->(p:Phase)
        RETURN p.phase_id AS phase_id,
               p.name AS name,
               p.entered_at AS entered_at,
               p.exited_at AS exited_at,
               p.srs_score AS srs_score,
               p.tc_score AS tc_score,
               p.fsm_state AS fsm_state,
               p.be_score AS be_score,
               p.markov_probability AS markov_probability,
               p.rule_score AS rule_score
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_phase(record)

    def get_phase_history(self, session_id: str, limit: int = 20) -> list[Phase]:
        # TRANSITIONED_TO goes old->new, so from current we go backwards
        # current has no outgoing TRANSITIONED_TO, oldest has no incoming
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_PHASE]->(current:Phase)
        // Find the full chain ending at current
        MATCH path = (oldest:Phase)-[:TRANSITIONED_TO*0..]->(current)
        WHERE NOT EXISTS { ()-[:TRANSITIONED_TO]->(oldest) }
           OR oldest = current
        WITH path
        ORDER BY length(path) DESC
        LIMIT 1
        WITH nodes(path) AS chain
        // Reverse so newest first
        WITH [i IN range(size(chain)-1, 0, -1) | chain[i]] AS reversed
        UNWIND range(0, size(reversed)-1) AS idx
        WITH reversed[idx] AS p, idx
        ORDER BY idx
        LIMIT $limit
        RETURN p.phase_id AS phase_id,
               p.name AS name,
               p.entered_at AS entered_at,
               p.exited_at AS exited_at,
               p.srs_score AS srs_score,
               p.tc_score AS tc_score,
               p.fsm_state AS fsm_state,
               p.be_score AS be_score,
               p.markov_probability AS markov_probability,
               p.rule_score AS rule_score
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, limit=limit)
            records = list(result)
            if not records or records[0]["phase_id"] is None:
                return []
            return [self._record_to_phase(r) for r in records]

    def count_transitions(
        self, session_id: str, from_phase: PhaseName, to_phase: PhaseName
    ) -> int:
        query = """
        MATCH (p1:Phase {name: $from_phase})-[:TRANSITIONED_TO]->(p2:Phase {name: $to_phase})
        // Ensure these phases belong to the session's chain
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_PHASE]->(current:Phase)
        WHERE (current) = p2
           OR EXISTS { (p2)-[:TRANSITIONED_TO*]->(current) }
        RETURN count(*) AS cnt
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(
                query,
                session_id=session_id,
                from_phase=from_phase.value,
                to_phase=to_phase.value,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_transition_matrix(self, session_id: str) -> dict[str, dict[str, float]]:
        query = """
        MATCH (session:AgentSession {session_id: $session_id})
              -[:CURRENT_PHASE]->(current:Phase)
        // Find all phases in this session's chain
        MATCH (oldest:Phase)-[:TRANSITIONED_TO*0..]->(current)
        WHERE NOT EXISTS { ()-[:TRANSITIONED_TO]->(oldest) }
           OR oldest = current
        WITH current, oldest
        MATCH path = (oldest)-[:TRANSITIONED_TO*0..]->(current)
        WITH nodes(path) AS chain
        ORDER BY size(chain) DESC
        LIMIT 1
        // Extract consecutive pairs
        UNWIND range(0, size(chain)-2) AS idx
        WITH chain[idx].name AS from_phase, chain[idx+1].name AS to_phase
        WITH from_phase, to_phase, count(*) AS transitions
        WITH from_phase, collect({to: to_phase, count: transitions}) AS targets,
             sum(transitions) AS total
        RETURN from_phase,
               [t IN targets | {to_phase: t.to, probability: toFloat(t.count) / total}] AS probs
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            matrix: dict[str, dict[str, float]] = {}
            for record in result:
                from_p = record["from_phase"]
                if from_p is None:
                    continue
                matrix[from_p] = {}
                for entry in record["probs"]:
                    matrix[from_p][entry["to_phase"]] = entry["probability"]
            return matrix

    @staticmethod
    def _record_to_phase(record) -> Phase:
        entered_at = record["entered_at"]
        exited_at = record["exited_at"]
        if hasattr(entered_at, "to_native"):
            entered_at = entered_at.to_native()
        if exited_at is not None and hasattr(exited_at, "to_native"):
            exited_at = exited_at.to_native()

        return Phase(
            phase_id=record["phase_id"],
            name=PhaseName(record["name"]),
            entered_at=entered_at,
            exited_at=exited_at,
            srs_score=record["srs_score"],
            tc_score=record["tc_score"],
            fsm_state=record["fsm_state"],
            be_score=record["be_score"],
            markov_probability=record["markov_probability"],
            rule_score=record["rule_score"],
        )

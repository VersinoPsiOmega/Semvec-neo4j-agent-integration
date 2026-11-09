"""Facade adapter over all Neo4j persistence stores."""

from __future__ import annotations

from neo4j import Driver

from semvec_neo4j.persistence.neo4j_session_store import Neo4jSessionStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore
from semvec_neo4j.persistence.neo4j_phase_store import Neo4jPhaseStore
from semvec_neo4j.persistence.neo4j_drift_event_store import Neo4jDriftEventStore
from semvec_neo4j.persistence.neo4j_memory_store import Neo4jMemoryStore
from semvec_neo4j.persistence.neo4j_agent_store import Neo4jAgentStore
from semvec_neo4j.persistence.neo4j_global_observer_store import (
    Neo4jGlobalObserverStore,
)
from semvec_neo4j.persistence.neo4j_anomaly_event_store import (
    Neo4jAnomalyEventStore,
)
from semvec_neo4j.persistence.neo4j_literal_fact_store import Neo4jLiteralFactStore


class Neo4jSemvecAdapter:
    """Unified access to all Semvec-backed Neo4j stores.

    Single entry point that wires up the session, state, phase, drift
    event, memory, agent, observer, anomaly, and literal-fact stores
    against one driver and database.
    """

    def __init__(self, driver: Driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database

        self.sessions = Neo4jSessionStore(driver, database)
        self.states = Neo4jStateStore(driver, database)
        self.phases = Neo4jPhaseStore(driver, database)
        self.drift_events = Neo4jDriftEventStore(driver, database)
        self.memories = Neo4jMemoryStore(driver, database)
        self.agents = Neo4jAgentStore(driver, database)
        self.observers = Neo4jGlobalObserverStore(driver, database)
        self.anomaly_events = Neo4jAnomalyEventStore(driver, database)
        self.literal_facts = Neo4jLiteralFactStore(driver, database)

    def apply_schema(self, schema_path: str) -> None:
        """Apply the Semvec graph schema from a ``.cypher`` file."""
        with open(schema_path, "r") as f:
            schema_cypher = f.read()

        statements = []
        current: list[str] = []
        for line in schema_cypher.split("\n"):
            stripped = line.strip()
            if stripped.startswith("//") or not stripped:
                continue
            current.append(line)
            if stripped.endswith(";"):
                statements.append("\n".join(current).rstrip(";"))
                current = []

        with self.driver.session(database=self.database) as session:
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    try:
                        session.run(stmt).consume()
                    except Exception:
                        pass  # Constraints/indexes may already exist

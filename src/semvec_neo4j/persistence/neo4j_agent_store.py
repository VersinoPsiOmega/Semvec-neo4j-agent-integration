"""Neo4j-backed AgentStore — manages Agent nodes and their sessions."""

from __future__ import annotations

from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import AgentStore
from semvec_neo4j.persistence.models import Agent, AgentSession, SessionStatus


class Neo4jAgentStore(AgentStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def ensure_agent(self, agent: Agent) -> Agent:
        # MERGE: an agent runs many sessions, so creating it is idempotent.
        query = """
        MERGE (a:Agent {agent_id: $agent_id})
        ON CREATE SET a.name = $name,
                      a.framework = $framework,
                      a.version = $version
        RETURN a.agent_id AS aid
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                agent_id=agent.agent_id,
                name=agent.name,
                framework=agent.framework,
                version=agent.version,
            ).consume()
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        RETURN a.agent_id AS agent_id,
               a.name AS name,
               a.framework AS framework,
               a.version AS version
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, agent_id=agent_id)
            record = result.single()
            if record is None:
                return None
            return Agent(
                agent_id=record["agent_id"],
                name=record["name"] or "",
                framework=record["framework"] or "",
                version=record["version"] or "",
            )

    def link_to_session(self, agent_id: str, session_id: str) -> bool:
        # MERGE the relationship so re-linking the same pair is idempotent.
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        MATCH (s:AgentSession {session_id: $session_id})
        MERGE (a)-[:RAN]->(s)
        RETURN a.agent_id AS aid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, agent_id=agent_id, session_id=session_id)
            return result.single() is not None

    def get_sessions(self, agent_id: str) -> list[AgentSession]:
        query = """
        MATCH (a:Agent {agent_id: $agent_id})-[:RAN]->(s:AgentSession)
        RETURN s.session_id AS session_id,
               s.agent_id AS agent_id,
               s.created_at AS created_at,
               s.last_active AS last_active,
               s.status AS status
        ORDER BY s.created_at DESC
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, agent_id=agent_id)
            sessions = []
            for r in result:
                created_at = r["created_at"]
                last_active = r["last_active"]
                if hasattr(created_at, "to_native"):
                    created_at = created_at.to_native()
                if hasattr(last_active, "to_native"):
                    last_active = last_active.to_native()
                sessions.append(AgentSession(
                    session_id=r["session_id"],
                    agent_id=r["agent_id"] or "",
                    created_at=created_at,
                    last_active=last_active,
                    status=SessionStatus(r["status"]),
                ))
            return sessions

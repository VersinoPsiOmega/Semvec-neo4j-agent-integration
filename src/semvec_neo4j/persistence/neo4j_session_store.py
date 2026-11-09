"""Neo4j-backed SessionStore implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import Driver

from semvec_neo4j.persistence.base import SessionStore
from semvec_neo4j.persistence.models import AgentSession, SessionStatus


class Neo4jSessionStore(SessionStore):
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def create_session(self, session: AgentSession) -> AgentSession:
        now = datetime.now(timezone.utc)
        session.created_at = now
        session.last_active = now

        query = """
        CREATE (s:AgentSession {
            session_id: $session_id,
            agent_id: $agent_id,
            created_at: datetime($created_at),
            last_active: datetime($last_active),
            status: $status
        })
        RETURN s.session_id AS session_id
        """
        with self._driver.session(database=self._database) as db:
            db.run(
                query,
                session_id=session.session_id,
                agent_id=session.agent_id,
                created_at=now.isoformat(),
                last_active=now.isoformat(),
                status=session.status.value,
            ).consume()
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        query = """
        MATCH (s:AgentSession {session_id: $session_id})
        RETURN s.session_id AS session_id,
               s.agent_id AS agent_id,
               s.created_at AS created_at,
               s.last_active AS last_active,
               s.status AS status
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_session(record)

    def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        query = """
        MATCH (s:AgentSession {session_id: $session_id})
        SET s.status = $status,
            s.last_active = datetime($now)
        RETURN s.session_id AS sid
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, status=status.value, now=now)
            return result.single() is not None

    def close_session(self, session_id: str) -> bool:
        return self.update_session_status(session_id, SessionStatus.CLOSED)

    def list_active_sessions(self, agent_id: Optional[str] = None) -> list[AgentSession]:
        if agent_id:
            query = """
            MATCH (s:AgentSession {status: 'active', agent_id: $agent_id})
            RETURN s.session_id AS session_id,
                   s.agent_id AS agent_id,
                   s.created_at AS created_at,
                   s.last_active AS last_active,
                   s.status AS status
            ORDER BY s.created_at DESC
            """
            params = {"agent_id": agent_id}
        else:
            query = """
            MATCH (s:AgentSession {status: 'active'})
            RETURN s.session_id AS session_id,
                   s.agent_id AS agent_id,
                   s.created_at AS created_at,
                   s.last_active AS last_active,
                   s.status AS status
            ORDER BY s.created_at DESC
            """
            params = {}

        with self._driver.session(database=self._database) as db:
            result = db.run(query, **params)
            return [self._record_to_session(r) for r in result]

    @staticmethod
    def _record_to_session(record) -> AgentSession:
        created_at = record["created_at"]
        last_active = record["last_active"]
        # Neo4j returns neo4j.time.DateTime; convert to Python datetime
        if hasattr(created_at, "to_native"):
            created_at = created_at.to_native()
        if hasattr(last_active, "to_native"):
            last_active = last_active.to_native()

        return AgentSession(
            session_id=record["session_id"],
            agent_id=record["agent_id"],
            created_at=created_at,
            last_active=last_active,
            status=SessionStatus(record["status"]),
        )

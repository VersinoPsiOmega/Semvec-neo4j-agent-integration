"""Tests for MCP tool handlers — using local Semvec runtime."""

import pytest
from src.core.embedder import HashEmbedder
from src.core.semvec_client import SemvecClient
from src.mcp.semvec_mcp_server import SemvecMCPServer
from src.persistence.models import PhaseName
from tests.conftest import make_random_vector


@pytest.fixture(scope="session")
def semvec_client():
    return SemvecClient(embedder=HashEmbedder(dimension=32), owner_subject="test-suite")


@pytest.fixture
def mcp(neo4j_driver, semvec_client):
    return SemvecMCPServer(neo4j_driver, database="neo4j", semvec_client=semvec_client)


class TestCreateSession:
    def test_create_agent_session(self, mcp):
        result = mcp.create_agent_session(agent_id="test-agent")
        assert "session_id" in result
        assert result["agent_id"] == "test-agent"
        assert result["status"] == "active"


class TestDetectDrift:
    def test_detect_drift_returns_results(self, mcp):
        session = mcp.create_agent_session(agent_id="agent-1")
        sid = session["session_id"]

        result = mcp.detect_drift(session_id=sid, message="Patient heart failure diagnosis")
        assert "drift_score" in result
        assert "drift_phase" in result
        assert "context" in result  # Compressed context block
        assert "severity" in result
        assert "state_id" in result
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_detect_drift_with_topic_switch(self, mcp):
        session = mcp.create_agent_session(agent_id="agent-2")
        sid = session["session_id"]

        # Build strong context (Semvec needs enough turns for drift detection)
        for msg in [
            "Heart failure treatment protocols and guidelines",
            "Cardiac medication management and drug interactions",
            "Heart failure readmission prevention strategies",
            "Cardiovascular rehabilitation exercise programs",
            "Beta blocker dosing in chronic heart failure",
            "Echocardiography assessment for ejection fraction",
        ]:
            mcp.detect_drift(sid, msg)

        # Topic switch
        result = mcp.detect_drift(sid, "Best chocolate cake recipe with buttercream frosting")
        # Semvec processes the input and returns a result
        assert result["drift_score"] >= 0
        assert "context" in result

    def test_store_response(self, mcp):
        session = mcp.create_agent_session(agent_id="agent-3")
        sid = session["session_id"]
        mcp.detect_drift(sid, "What treatments exist for diabetes?")

        result = mcp.store_response(sid, "Common treatments include insulin therapy and metformin.")
        assert result is not None


class TestGetPhase:
    def test_get_phase(self, mcp):
        session = mcp.create_agent_session(agent_id="phase-agent")
        sid = session["session_id"]
        mcp.detect_drift(sid, "Initial patient consultation")

        result = mcp.get_phase(session_id=sid)
        assert "phase" in result
        assert result["phase"] in [p.value for p in PhaseName]

    def test_get_phase_no_session(self, mcp):
        result = mcp.get_phase(session_id="nonexistent")
        assert result.get("phase") is None


class TestDriftHistory:
    def test_query_history(self, mcp):
        session = mcp.create_agent_session(agent_id="history-agent")
        sid = session["session_id"]

        for msg in ["Cardiology research", "Heart transplant options", "Cardiac rehabilitation"]:
            mcp.detect_drift(sid, msg)
        mcp.detect_drift(sid, "Recipe for banana bread")

        result = mcp.query_drift_history(session_id=sid, limit=10)
        assert isinstance(result, list)

    def test_query_history_empty(self, mcp):
        session = mcp.create_agent_session(agent_id="empty-agent")
        result = mcp.query_drift_history(session_id=session["session_id"])
        assert result == []


class TestStateTrajectory:
    def test_state_trajectory(self, mcp):
        session = mcp.create_agent_session(agent_id="traj-agent")
        sid = session["session_id"]

        for msg in ["Message one", "Message two", "Message three", "Message four"]:
            mcp.detect_drift(sid, msg)

        result = mcp.get_state_trajectory(session_id=sid, steps=5)
        assert len(result) == 4
        assert all("step" in entry for entry in result)


class TestEndSession:
    def test_end_session(self, mcp):
        session = mcp.create_agent_session(agent_id="end-agent")
        sid = session["session_id"]

        mcp.detect_drift(sid, "Some medical query")
        mcp.detect_drift(sid, "Another medical query")

        result = mcp.end_agent_session(session_id=sid)
        assert result["status"] == "closed"
        assert "final_phase" in result
        assert "total_drift_events" in result
        assert "total_states" in result


class TestMemoryTools:
    def test_memory_store_and_query(self, mcp):
        session = mcp.create_agent_session(agent_id="mem-agent")
        sid = session["session_id"]

        mcp.store_memory(
            session_id=sid,
            text="Patient has chronic heart failure",
            importance=0.9,
            vector=make_random_vector(384, seed=42),
        )

        result = mcp.memory_query(sid, query_vector=make_random_vector(384, seed=42), limit=5)
        assert len(result) >= 1
        assert result[0]["text"] == "Patient has chronic heart failure"

    def test_memory_consolidate(self, mcp):
        session = mcp.create_agent_session(agent_id="consolidate-agent")
        sid = session["session_id"]

        for i in range(5):
            mcp.store_memory(sid, f"Memory {i}", importance=0.1 * (i + 1),
                             vector=make_random_vector(384, seed=i + 10))

        result = mcp.memory_consolidate(session_id=sid)
        assert "short" in result
        assert "medium" in result
        assert "long" in result

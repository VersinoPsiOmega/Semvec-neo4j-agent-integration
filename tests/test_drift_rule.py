# ABOUTME: Tests for the shared drift rule: thresholds, cooldown, loading from the environment.
# ABOUTME: Also covers how SemvecClient and SemvecMCPServer apply the rule (no Neo4j needed).

import pytest

from semvec_neo4j.core.drift_rule import DriftGate, DriftRule
from semvec_neo4j.core.embedder import HashEmbedder
from semvec_neo4j.core.semvec_client import SemvecClient
from semvec_neo4j.mcp.semvec_mcp_server import SemvecMCPServer

PERMISSIVE = DriftRule(drift_threshold=0.0, sim_ceiling=1.0)
NEVER = DriftRule(drift_threshold=1.01, sim_ceiling=0.0)


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — these tests need no Neo4j."""
    yield


class TestDriftRuleMatches:
    def test_flags_when_score_reaches_threshold_and_similarity_is_below_ceiling(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.35, sim_ceiling=0.45)

        # Act
        flagged = rule.matches(drift_score=0.40, top_similarity=0.30, drift_detected=False)

        # Assert
        assert flagged is True

    def test_boundaries_are_inclusive(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.35, sim_ceiling=0.45)

        # Act
        flagged = rule.matches(drift_score=0.35, top_similarity=0.45, drift_detected=False)

        # Assert
        assert flagged is True

    def test_does_not_flag_when_similarity_is_above_ceiling(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.35, sim_ceiling=0.45)

        # Act
        flagged = rule.matches(drift_score=0.60, top_similarity=0.50, drift_detected=False)

        # Assert
        assert flagged is False

    def test_does_not_flag_when_score_is_below_threshold(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.35, sim_ceiling=0.45)

        # Act
        flagged = rule.matches(drift_score=0.30, top_similarity=0.10, drift_detected=False)

        # Assert
        assert flagged is False

    def test_semvec_verdict_flags_regardless_of_thresholds(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.35, sim_ceiling=0.45)

        # Act
        flagged = rule.matches(drift_score=0.10, top_similarity=0.90, drift_detected=True)

        # Assert
        assert flagged is True


class TestDriftRuleValidation:
    def test_rejects_negative_cooldown(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="cooldown_turns"):
            DriftRule(drift_threshold=0.35, sim_ceiling=0.45, cooldown_turns=-1)


class TestDriftRuleFromEnv:
    def test_reads_thresholds_and_cooldown(self):
        # Arrange
        environ = {"DRIFT_THRESHOLD": "0.35", "DRIFT_SIM_CEILING": "0.45", "DRIFT_COOLDOWN_TURNS": "3"}

        # Act
        rule = DriftRule.from_env(environ)

        # Assert
        assert rule == DriftRule(drift_threshold=0.35, sim_ceiling=0.45, cooldown_turns=3)

    def test_cooldown_is_off_when_not_set(self):
        # Arrange
        environ = {"DRIFT_THRESHOLD": "0.35", "DRIFT_SIM_CEILING": "0.45"}

        # Act
        rule = DriftRule.from_env(environ)

        # Assert
        assert rule.cooldown_turns == 0

    def test_missing_thresholds_raise_and_name_the_variables(self):
        # Arrange
        environ = {"DRIFT_COOLDOWN_TURNS": "3"}

        # Act / Assert
        with pytest.raises(ValueError, match="DRIFT_THRESHOLD.*DRIFT_SIM_CEILING"):
            DriftRule.from_env(environ)

    def test_unparsable_value_names_the_variable(self):
        # Arrange
        environ = {"DRIFT_THRESHOLD": "0.35", "DRIFT_SIM_CEILING": "0.45", "DRIFT_COOLDOWN_TURNS": "three"}

        # Act / Assert
        with pytest.raises(ValueError, match="DRIFT_COOLDOWN_TURNS"):
            DriftRule.from_env(environ)


class TestDriftGate:
    def test_without_cooldown_every_matching_turn_is_flagged(self):
        # Arrange
        gate = DriftGate(PERMISSIVE)

        # Act
        flags = [gate.check("s1", 0.5, 0.1, False) for _ in range(3)]

        # Assert
        assert flags == [True, True, True]

    def test_cooldown_suppresses_the_following_turns(self):
        # Arrange
        gate = DriftGate(DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=3))

        # Act
        flags = [gate.check("s1", 0.5, 0.1, False) for _ in range(5)]

        # Assert
        assert flags == [True, False, False, False, True]

    def test_non_matching_turns_count_towards_the_cooldown(self):
        # Arrange
        gate = DriftGate(DriftRule(drift_threshold=0.35, sim_ceiling=0.45, cooldown_turns=2))

        # Act
        flags = [
            gate.check("s1", 0.5, 0.1, False),  # drift
            gate.check("s1", 0.1, 0.9, False),  # calm, inside cooldown
            gate.check("s1", 0.1, 0.9, False),  # calm, inside cooldown
            gate.check("s1", 0.5, 0.1, False),  # drift again, cooldown over
        ]

        # Assert
        assert flags == [True, False, False, True]

    def test_forget_ends_the_cooldown_of_a_session(self):
        # Arrange
        gate = DriftGate(DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=3))
        gate.check("s1", 0.5, 0.1, False)

        # Act
        gate.forget("s1")
        flagged = gate.check("s1", 0.5, 0.1, False)

        # Assert
        assert flagged is True

    def test_cooldown_is_tracked_per_session(self):
        # Arrange
        gate = DriftGate(DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=3))
        gate.check("s1", 0.5, 0.1, False)

        # Act
        flagged = gate.check("s2", 0.5, 0.1, False)

        # Assert
        assert flagged is True


class TestSemvecClientDriftVerdict:
    def test_without_rule_drift_equals_semvec_verdict(self):
        # Arrange
        client = SemvecClient(embedder=HashEmbedder(dimension=16))

        # Act
        result = client.run("first message")

        # Assert
        assert result["drift"] == result["drift_detected"]

    def test_rule_decides_the_drift_verdict(self):
        # Arrange
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=PERMISSIVE)

        # Act
        result = client.run("first message")

        # Assert
        assert result["drift"] is True

    def test_cooldown_applies_per_session(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=1)
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=rule)
        sid = client.run("turn one")["session_id"]

        # Act
        flags = [client.run(msg, session_id=sid)["drift"] for msg in ("turn two", "turn three")]

        # Assert
        assert flags == [False, True]

    def test_reset_context_ends_the_cooldown(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=3)
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=rule)
        sid = client.run("turn one")["session_id"]

        # Act
        result = client.run("fresh start", session_id=sid, reset_context=True)

        # Assert
        assert result["drift"] is True

    def test_deleted_session_leaves_no_cooldown_state(self):
        # Arrange
        rule = DriftRule(drift_threshold=0.0, sim_ceiling=1.0, cooldown_turns=3)
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=rule)
        sid = client.run("turn one")["session_id"]

        # Act
        client.delete_session(sid)
        result = client.run("same id again", session_id=sid, reset_context=True)

        # Assert
        assert result["drift"] is True

    def test_exposes_its_rule(self):
        # Arrange
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=PERMISSIVE)

        # Act / Assert
        assert client.drift_rule == PERMISSIVE


class TestRegionDriftPublication:
    def _region_with_two_clusters(self, client: SemvecClient) -> tuple[str, list[str]]:
        cluster_ids = [client.create_cluster(name=f"dept-{i}")["cluster_id"] for i in range(2)]
        region_id = client.create_region(name="network", consensus_threshold=0.5)["region_id"]
        for cid in cluster_ids:
            client.add_region_cluster(region_id, cid)
        return region_id, cluster_ids

    def test_drift_by_the_rule_is_published_to_the_region(self):
        # Arrange
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=PERMISSIVE)
        region_id, cluster_ids = self._region_with_two_clusters(client)

        # Act
        for cid in cluster_ids:
            client.cluster_run(cid, "facility admin question")
        events = client.get_region_events(region_id)

        # Assert
        assert len(events) >= 1

    def test_no_region_event_when_the_rule_does_not_match(self):
        # Arrange
        client = SemvecClient(embedder=HashEmbedder(dimension=16), drift_rule=NEVER)
        region_id, cluster_ids = self._region_with_two_clusters(client)

        # Act
        for cid in cluster_ids:
            client.cluster_run(cid, "facility admin question")
        events = client.get_region_events(region_id)

        # Assert
        assert events == []


class TestMCPServerDriftRule:
    def test_without_client_the_rule_comes_from_the_environment(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("DRIFT_THRESHOLD", "0.30")
        monkeypatch.setenv("DRIFT_SIM_CEILING", "0.40")
        monkeypatch.setenv("DRIFT_COOLDOWN_TURNS", "2")

        # Act
        server = SemvecMCPServer(driver=object())

        # Assert
        assert server.semvec_client.drift_rule == DriftRule(0.30, 0.40, 2)

    def test_without_client_missing_thresholds_raise(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("DRIFT_THRESHOLD", raising=False)
        monkeypatch.delenv("DRIFT_SIM_CEILING", raising=False)

        # Act / Assert
        with pytest.raises(ValueError, match="DRIFT_THRESHOLD"):
            SemvecMCPServer(driver=object())

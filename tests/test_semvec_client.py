"""Contract tests for the local SemvecClient.

Asserts that the wrapper around the bundled ``semvec`` package returns
the dictionary shapes the rest of the codebase depends on.

A 16-dimensional :class:`HashEmbedder` keeps the tests offline and fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.embedder import HashEmbedder
from src.core.semvec_client import SemvecClient


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — these tests need no Neo4j."""
    yield


@pytest.fixture
def client():
    return SemvecClient(embedder=HashEmbedder(dimension=16), owner_subject="test-suite")


# ---------------------------------------------------------------------------
# Health


class TestHealth:
    def test_health_returns_ok(self, client):
        h = client.health()
        assert h["status"] == "ok"

    def test_health_includes_active_sessions(self, client):
        h = client.health()
        assert "active_sessions" in h
        assert isinstance(h["active_sessions"], int)


# ---------------------------------------------------------------------------
# Layer 1 — Sessions / Run / Store


class TestLayer1Run:
    def test_run_creates_session_when_none_given(self, client):
        r = client.run("Patient with chest pain")
        assert "session_id" in r
        assert isinstance(r["session_id"], str)

    def test_run_returns_required_keys(self, client):
        r = client.run("Heart failure follow-up")
        for key in (
            "session_id",
            "context",
            "top_similarity",
            "short_circuit",
            "drift_score",
            "drift_detected",
            "drift_phase",
        ):
            assert key in r, f"missing key {key!r} in run response"
        assert isinstance(r["context"], str)
        assert isinstance(r["top_similarity"], float)
        assert isinstance(r["drift_score"], float)
        assert isinstance(r["drift_detected"], bool)
        assert r["drift_phase"] in {"stable", "shifting", "drifted"}

    def test_run_reuses_session(self, client):
        first = client.run("Initial visit")
        sid = first["session_id"]
        second = client.run("Follow-up visit", session_id=sid)
        assert second["session_id"] == sid

    def test_run_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.run("Hi", session_id="does-not-exist")

    def test_run_reset_context_creates_session_with_id(self, client):
        r = client.run("first", session_id="custom-sid", reset_context=True)
        assert r["session_id"] == "custom-sid"

    def test_run_inline_store_response(self, client):
        first = client.run("Question one")
        sid = first["session_id"]
        # Pass response so it gets stored before the next turn
        second = client.run("Question two", session_id=sid, response="Answer one")
        assert second["session_id"] == sid

    def test_store_persists_response(self, client):
        first = client.run("Patient labs")
        sid = first["session_id"]
        result = client.store(sid, "Labs are within normal range.")
        assert result is not None


class TestLayer1bSession:
    def test_create_session_returns_session_id(self, client):
        result = client.create_session()
        assert "session_id" in result
        assert isinstance(result["session_id"], str)

    def test_export_session_round_trip(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        export = client.export_session(sid)
        assert "session_id" in export
        assert "state_dict" in export
        assert "checksum" in export

    def test_import_session_returns_dict(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        export = client.export_session(sid)
        # Create a target session and import into it
        target = client.create_session()
        result = client.import_session(target["session_id"], export["state_dict"])
        assert isinstance(result, dict)
        assert "session_id" in result

    def test_add_anchor_returns_dict(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("oncology").astype(float))
        result = client.add_anchor(sid, emb)
        assert isinstance(result, dict)
        assert "anchor_count" in result or "anchor_index" in result

    def test_get_anchor_score_returns_float(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("oncology").astype(float))
        client.add_anchor(sid, emb)
        score = client.get_anchor_score(sid)
        assert "anchor_score" in score
        assert isinstance(score["anchor_score"], float)

    def test_inject_memory(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = list(HashEmbedder(dimension=16).get_embedding("interaction").astype(float))
        result = client.inject_memory(
            sid, embedding=emb, text="Metformin contraindicated with renal failure",
            tier="short_term", importance=0.9,
        )
        assert isinstance(result, dict)

    def test_set_isolation(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        result = client.set_isolation(sid, level="QUARANTINE", similarity_threshold=0.5)
        assert isinstance(result, dict)
        assert result.get("level") == "QUARANTINE"

    def test_add_trigger(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        result = client.add_trigger(sid, "CRITICAL")
        assert isinstance(result, dict)


class TestSessionLifecycle:
    """delete_session, reset_session — wraps SessionManager.delete_session/reset_session."""

    def test_delete_session_removes_it(self, client):
        first = client.run("init turn")
        sid = first["session_id"]
        result = client.delete_session(sid)
        assert isinstance(result, dict)
        assert result.get("deleted") is True
        # Subsequent run on a deleted session must surface as a clear error.
        with pytest.raises(ValueError):
            client.run("follow-up", session_id=sid)

    def test_delete_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.delete_session("does-not-exist")

    def test_delete_cluster_backed_session_is_refused(self, client):
        """Deleting a cluster's backing session would orphan the cluster."""
        cluster = client.create_cluster(name="ward-test")
        cid = cluster["cluster_id"]
        with pytest.raises(ValueError, match="cluster"):
            client.delete_session(cid)
        # Cluster still exists and is usable
        assert client.get_cluster(cid)["cluster_id"] == cid

    def test_reset_session_keeps_session_alive(self, client):
        # Build up state across several turns
        first = client.run("Patient one — diabetes")
        sid = first["session_id"]
        for msg in (
            "Patient one — Metformin",
            "Patient one — Lisinopril",
            "Patient one — Atorvastatin",
        ):
            client.run(msg, session_id=sid)
        before_metrics = client.get_metrics(sid) if hasattr(client, "get_metrics") else None
        result = client.reset_session(sid)
        assert isinstance(result, dict)
        assert result.get("reset") is True
        # Session still works and starts a fresh similarity history
        post = client.run("Patient one — fresh start", session_id=sid)
        assert post["session_id"] == sid
        # First turn after reset has top_similarity 0.0 because semantic_state is reset
        assert post["top_similarity"] == 0.0

    def test_reset_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.reset_session("does-not-exist")


class TestTriggerLifecycle:
    """clear_triggers, release_quarantine — inverse of add_trigger / set_isolation."""

    def test_clear_triggers_drops_all(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        client.add_trigger(sid, "CRITICAL")
        client.add_trigger(sid, "contraindication")

        result = client.clear_triggers(sid)
        assert isinstance(result, dict)
        assert result.get("trigger_count") == 0

    def test_clear_triggers_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.clear_triggers("does-not-exist")

    def test_release_quarantine_returns_count(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        client.set_isolation(sid, level="QUARANTINE", similarity_threshold=0.5)
        result = client.release_quarantine(sid)
        assert isinstance(result, dict)
        assert "released_count" in result
        assert isinstance(result["released_count"], int)

    def test_release_quarantine_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.release_quarantine("does-not-exist")


class TestSessionMetrics:
    """get_session_metrics — wraps SessionManager.get_metrics."""

    def test_metrics_returns_required_keys(self, client):
        first = client.run("Patient diabetes intake")
        sid = first["session_id"]
        for q in ("Metformin contraindications", "Lisinopril interactions"):
            client.run(q, session_id=sid, response="(stub)")
        m = client.get_session_metrics(sid)
        for key in (
            "session_id", "phase", "interaction_count", "total_memories",
            "beta_history", "similarity_history",
            "fsm_history", "norm_history", "phase_history",
        ):
            assert key in m, f"missing key {key!r}"
        assert m["session_id"] == sid

    def test_metrics_grow_with_turns(self, client):
        first = client.run("turn 0")
        sid = first["session_id"]
        m0 = client.get_session_metrics(sid)
        for i in range(3):
            client.run(f"turn {i + 1}", session_id=sid, response="(stub)")
        m1 = client.get_session_metrics(sid)
        assert m1["interaction_count"] >= m0["interaction_count"]
        assert len(m1["similarity_history"]) >= len(m0["similarity_history"])

    def test_phase_history_is_string_list(self, client):
        first = client.run("seed")
        sid = first["session_id"]
        m = client.get_session_metrics(sid)
        assert isinstance(m["phase_history"], list)
        for entry in m["phase_history"]:
            assert isinstance(entry, str)

    def test_metrics_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.get_session_metrics("does-not-exist")


class TestMemoryRecall:
    """get_relevant_memories + get_memory_by_hash — wraps
    SessionManager.get_context / get_memory_by_hash."""

    @pytest.fixture
    def warmed_session(self, client):
        first = client.run("Patient with chest pain — initial workup")
        sid = first["session_id"]
        for q, a in [
            ("Metformin dosage check",   "Metformin 500mg BID, hold if GFR < 30."),
            ("Lisinopril titration",     "Start Lisinopril 5mg, titrate to 10mg."),
            ("Atorvastatin LDL goal",    "Atorvastatin 40mg, target LDL < 70."),
            ("Insulin regimen review",   "Basal insulin 10 units, adjust per CGM."),
        ]:
            r = client.run(q, session_id=sid, response=a)
            sid = r["session_id"]
        return sid

    def test_returns_list_of_entries(self, client, warmed_session):
        out = client.get_relevant_memories(warmed_session, top_k=3)
        assert isinstance(out, list)
        assert 1 <= len(out) <= 3

    def test_entry_has_hash_relevance_text(self, client, warmed_session):
        out = client.get_relevant_memories(warmed_session, top_k=3)
        assert out, "expected at least one memory"
        for entry in out:
            assert {"text", "relevance", "memory_hash", "truncated"} <= entry.keys()
            assert isinstance(entry["text"], str)
            assert isinstance(entry["relevance"], float)
            assert isinstance(entry["memory_hash"], str)
            assert isinstance(entry["truncated"], bool)

    def test_full_first_keeps_top_untruncated(self, client, warmed_session):
        out = client.get_relevant_memories(
            warmed_session, top_k=3, max_text_chars=10, full_first=True,
        )
        assert out, "expected at least one memory"
        # First entry must not be flagged as truncated
        assert out[0]["truncated"] is False

    def test_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.get_relevant_memories("does-not-exist")

    def test_get_memory_by_hash_round_trip(self, client, warmed_session):
        listed = client.get_relevant_memories(warmed_session, top_k=3)
        # Find first entry with a non-empty hash; older memories may
        # surface without a stable hash, so be tolerant.
        candidate = next((e for e in listed if e["memory_hash"]), None)
        if candidate is None:
            pytest.skip("no memory with stable semantic_hash on this turn")
        full = client.get_memory_by_hash(warmed_session, candidate["memory_hash"])
        assert full is not None
        assert full["memory_hash"] == candidate["memory_hash"]
        assert isinstance(full["text"], str) and full["text"]

    def test_get_memory_by_hash_unknown_returns_none(self, client, warmed_session):
        result = client.get_memory_by_hash(warmed_session, "definitely-not-a-real-hash")
        assert result is None

    def test_get_memory_by_hash_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.get_memory_by_hash("does-not-exist", "any-hash")


class TestVerifyConsistency:
    """verify_consistency — wraps SessionManager.verify_consistency.

    The upstream method runs each test embedding through the session and
    a reference session and returns True iff cosine similarities match
    within ``tolerance``. We test both the self-consistent case and
    the unknown-session error path.
    """

    def test_returns_bool_for_known_session(self, client):
        first = client.run("Init")
        sid = first["session_id"]
        emb = HashEmbedder(dimension=16)
        probes = [
            list(emb.get_embedding("Patient one").astype(float)),
            list(emb.get_embedding("Patient two").astype(float)),
            list(emb.get_embedding("Patient three").astype(float)),
        ]
        result = client.verify_consistency(sid, probes)
        assert isinstance(result, bool)

    def test_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.verify_consistency("does-not-exist", [[0.0] * 16])


class TestConsensusEngine:
    """Wraps semvec.cortex.ConsensusEngine.

    Lifecycle: create_consensus_engine -> register_consensus_voter ->
    submit_consensus_proposal -> vote_on_consensus -> evaluate_consensus.
    """

    def test_create_engine_returns_engine_id(self, client):
        eng = client.create_consensus_engine(
            local_id="orchestrator", network_id="ward-net",
        )
        assert "engine_id" in eng
        assert eng["local"] == "orchestrator"
        assert eng["network"] == "ward-net"
        assert eng["level"] == "qualified_majority"   # default

    def test_register_voters_and_submit_proposal(self, client):
        eng = client.create_consensus_engine("orch", "net")
        eid = eng["engine_id"]
        for inst, w in [("dr-chen", 1.0), ("dr-volkov", 1.0), ("dr-tanaka", 1.5)]:
            r = client.register_consensus_voter(eid, inst, weight=w)
            assert r.get("registered") is True
            assert r["instance_id"] == inst

        proposal = client.submit_consensus_proposal(
            eid,
            proposal_type="treatment_alignment",
            proposed_state=[0.0] * 8,
            rationale="Park requires LVH workup",
        )
        assert "proposal_id" in proposal
        assert proposal["status"] == "pending"

    def test_vote_and_evaluate_simple_majority_passes(self, client):
        eng = client.create_consensus_engine("orch", "net", level="simple_majority")
        eid = eng["engine_id"]
        for inst in ("a", "b", "c"):
            client.register_consensus_voter(eid, inst)
        prop = client.submit_consensus_proposal(
            eid, proposal_type="x", proposed_state=[0.0] * 4, rationale="r",
        )
        pid = prop["proposal_id"]
        client.vote_on_consensus(eid, pid, True, voting_instance="a")
        client.vote_on_consensus(eid, pid, True, voting_instance="b")
        client.vote_on_consensus(eid, pid, False, voting_instance="c")
        result = client.evaluate_consensus(eid, pid)
        assert result["accepted"] is True
        assert result["status"] == "accepted"
        assert "ratio" in result

    def test_vote_and_evaluate_unanimous_fails_on_dissent(self, client):
        eng = client.create_consensus_engine("orch", "net", level="unanimous")
        eid = eng["engine_id"]
        for inst in ("a", "b", "c"):
            client.register_consensus_voter(eid, inst)
        prop = client.submit_consensus_proposal(
            eid, proposal_type="x", proposed_state=[0.0] * 4, rationale="r",
        )
        pid = prop["proposal_id"]
        client.vote_on_consensus(eid, pid, True, voting_instance="a")
        client.vote_on_consensus(eid, pid, True, voting_instance="b")
        client.vote_on_consensus(eid, pid, False, voting_instance="c")
        result = client.evaluate_consensus(eid, pid)
        assert result["accepted"] is False
        assert result["status"] == "pending"

    def test_unknown_engine_raises(self, client):
        with pytest.raises(ValueError):
            client.register_consensus_voter("does-not-exist", "voter-1")
        with pytest.raises(ValueError):
            client.submit_consensus_proposal(
                "does-not-exist", proposal_type="x",
                proposed_state=[0.0] * 4, rationale="r",
            )
        with pytest.raises(ValueError):
            client.evaluate_consensus("does-not-exist", "proposal-1")

    def test_unknown_proposal_raises(self, client):
        eng = client.create_consensus_engine("orch", "net")
        with pytest.raises(ValueError):
            client.evaluate_consensus(eng["engine_id"], "no-such-proposal")

    def test_invalid_level_rejected(self, client):
        with pytest.raises(ValueError):
            client.create_consensus_engine("o", "n", level="banana_majority")

    def test_get_consensus_statistics(self, client):
        eng = client.create_consensus_engine("orch", "net")
        eid = eng["engine_id"]
        client.register_consensus_voter(eid, "a")
        client.register_consensus_voter(eid, "b")
        stats = client.get_consensus_statistics(eid)
        assert stats["known_instances"] == 2
        assert stats["network_id"] == "net"


class TestFactExtraction:
    """Wraps semvec.compliance.extractors.extract_facts.

    Healthcare context: chemotherapy schedule strings carry dosages,
    schedule dates, and identifiers that must survive embedding
    compression. The extractor recognises ISO/DE dates, EUR/USD/kg/%
    numerics with units, plus UUID/IBAN/DE-VAT identifiers (medical
    domain extensions are not in upstream — store them via
    inject_memory or a domain-specific helper).
    """

    def test_extract_iso_date(self, client):
        facts = client.extract_facts(
            "First infusion 2026-05-15. Repeat every 21 days."
        )
        date_facts = [f for f in facts if f["kind"] == "date"]
        assert len(date_facts) >= 1
        assert "2026-05-15" in date_facts[0]["raw"]

    def test_extract_de_date(self, client):
        facts = client.extract_facts(
            "Aufnahme am 08.05.2026 zur Routinekontrolle."
        )
        date_facts = [f for f in facts if f["kind"] == "date"]
        assert len(date_facts) >= 1

    def test_extract_currency_amount(self, client):
        facts = client.extract_facts(
            "Treatment plan estimate 1.250,00 EUR per cycle."
        )
        num_facts = [f for f in facts if f["kind"] == "numeric"]
        assert len(num_facts) >= 1
        assert num_facts[0]["unit"] == "EUR"

    def test_extract_iban_identifier(self, client):
        # Valid IBAN (mod-97 == 1)
        facts = client.extract_facts(
            "Reimbursement account DE89 3704 0044 0532 0130 00."
        )
        id_facts = [f for f in facts if f["kind"] == "identifier"]
        assert any(f.get("id_type") == "iban" for f in id_facts)

    def test_no_facts_returns_empty(self, client):
        assert client.extract_facts("Patient asks about side effects.") == []

    def test_store_facts_as_entities_returns_count(self, client):
        first = client.run("init")
        sid = first["session_id"]
        text = "Carlos starts therapy on 2026-05-15. Cost 4.500,00 EUR."
        result = client.store_facts_as_entities(sid, text)
        assert "stored" in result
        assert result["stored"] >= 2  # 1 date + 1 currency
        # Entities are queryable on the session
        count = client._sessions.get_entity_count(sid)
        assert count >= result["stored"]

    def test_store_facts_unknown_session_raises(self, client):
        with pytest.raises(ValueError):
            client.store_facts_as_entities("does-not-exist", "anything")


class TestChatProxy:
    """Wraps semvec.token_reduction.SemvecChatProxy.

    The proxy compresses conversation history into a fixed-size context
    block while transparently calling a user-supplied LLM. We verify
    the handle returned by :meth:`SemvecClient.create_chat_proxy`
    surfaces the per-turn shape (response, pss_input_tokens,
    baseline_input_tokens, pss_prompt, phase, turn_number) and the
    aggregate :meth:`summary`.
    """

    def _fake_llm(self):
        responses: list[str] = []

        def llm(messages: list) -> str:
            responses.append(messages)
            return f"answer-{len(responses)}"

        return llm, responses

    def test_create_chat_proxy_returns_handle(self, client):
        llm, _ = self._fake_llm()
        handle = client.create_chat_proxy(
            llm_call=llm, system_prompt="You are a healthcare assistant.",
        )
        assert hasattr(handle, "turn")
        assert hasattr(handle, "summary")

    def test_turn_returns_required_keys(self, client):
        llm, _ = self._fake_llm()
        handle = client.create_chat_proxy(
            llm_call=llm, system_prompt="You are a healthcare assistant.",
        )
        result = handle.turn("What treatment for James Morrison's diabetes?")
        for key in (
            "response", "pss_input_tokens", "baseline_input_tokens",
            "pss_prompt", "phase", "turn_number",
        ):
            assert key in result, f"missing key {key!r}"
        assert result["response"] == "answer-1"
        assert result["turn_number"] == 1
        assert isinstance(result["pss_prompt"], str)
        assert isinstance(result["phase"], str)

    def test_turn_increments_turn_number(self, client):
        llm, _ = self._fake_llm()
        handle = client.create_chat_proxy(llm_call=llm, system_prompt="…")
        for i, q in enumerate([
            "Question one", "Question two", "Question three",
        ], start=1):
            r = handle.turn(q)
            assert r["turn_number"] == i

    def test_summary_aggregates_turns(self, client):
        llm, log = self._fake_llm()
        handle = client.create_chat_proxy(llm_call=llm, system_prompt="…")
        for q in ("a", "b", "c", "d"):
            handle.turn(q)
        summary = handle.summary()
        assert summary["total_turns"] == 4
        assert "per_turn" in summary
        assert len(summary["per_turn"]) == 4
        # The fake LLM does not expose `last_usage`, so token counts
        # come back as None — that's the documented contract.
        assert summary["total_pss_tokens"] is None or isinstance(
            summary["total_pss_tokens"], int
        )

    def test_turn_passes_messages_to_llm_call(self, client):
        llm, log = self._fake_llm()
        handle = client.create_chat_proxy(llm_call=llm, system_prompt="HC system")
        handle.turn("Hello")
        assert log, "llm_call was not invoked"
        messages = log[-1]
        # Always at least system + user
        assert any(m.role == "system" for m in messages)
        assert any(m.role == "user" and m.content == "Hello" for m in messages)


# ---------------------------------------------------------------------------
# Layer 2 — Cluster


class TestLayer2Cluster:
    def test_create_cluster(self, client):
        c = client.create_cluster(name="ward-A")
        assert "cluster_id" in c
        assert c["name"] == "ward-A"

    def test_list_clusters_includes_created(self, client):
        c = client.create_cluster(name="ward-B")
        all_clusters = client.list_clusters()
        ids = [x["cluster_id"] for x in all_clusters]
        assert c["cluster_id"] in ids

    def test_get_cluster(self, client):
        c = client.create_cluster(name="ward-C")
        got = client.get_cluster(c["cluster_id"])
        assert got["name"] == "ward-C"

    def test_delete_cluster(self, client):
        c = client.create_cluster(name="ephemeral")
        ok = client.delete_cluster(c["cluster_id"])
        assert ok.get("deleted") is True or ok.get("status") in {"deleted", "ok"}

    def test_cluster_run_returns_run_shape(self, client):
        c = client.create_cluster(name="ward-D")
        r = client.cluster_run(c["cluster_id"], "Ward round assessment")
        for key in ("session_id", "context", "top_similarity", "drift_score", "drift_detected", "drift_phase"):
            assert key in r

    def test_cluster_store(self, client):
        c = client.create_cluster(name="ward-E")
        r = client.cluster_store(c["cluster_id"], "Q?", "A!")
        assert isinstance(r, dict)

    def test_cluster_feedback(self, client):
        c = client.create_cluster(name="ward-F", coupling_factor=0.2)
        r = client.cluster_feedback(c["cluster_id"])
        assert "sessions_updated" in r

    def test_add_remove_cluster_member(self, client):
        c = client.create_cluster(name="ward-G")
        first = client.run("Hello")
        sid = first["session_id"]
        added = client.add_cluster_member(c["cluster_id"], sid)
        assert added.get("added") is True or added.get("status") in {"ok", "added"}
        removed = client.remove_cluster_member(c["cluster_id"], sid)
        assert removed.get("removed") is True or removed.get("status") in {"ok", "removed"}


# ---------------------------------------------------------------------------
# Layer 3 — Region


class TestLayer3Region:
    def test_create_region(self, client):
        r = client.create_region(name="hospital")
        assert "region_id" in r
        assert r["name"] == "hospital"

    def test_list_regions(self, client):
        client.create_region(name="hospital-x")
        regions = client.list_regions()
        names = [x["name"] for x in regions]
        assert "hospital-x" in names

    def test_add_region_cluster_then_events(self, client):
        c = client.create_cluster(name="ward-H")
        r = client.create_region(name="hospital-y")
        added = client.add_region_cluster(r["region_id"], c["cluster_id"])
        assert added.get("added") is True or added.get("status") in {"ok", "added"}

        # Cluster activity should not error and should give an event list
        client.cluster_run(c["cluster_id"], "Initial discussion")
        events = client.get_region_events(r["region_id"], limit=10)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Layer 4 — Observer


class TestLayer4Observer:
    def test_create_observer_then_summary(self, client):
        obs = client.create_observer(sample_interval_seconds=5.0)
        assert isinstance(obs, dict)
        summary = client.get_observer_summary()
        assert isinstance(summary, dict)

    def test_observer_sample(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        sample = client.observer_sample()
        assert isinstance(sample, dict)

    def test_get_anomalies_returns_list(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        anomalies = client.get_anomalies(limit=5)
        assert isinstance(anomalies, list)

    def test_clear_anomalies_returns_dict(self, client):
        client.create_observer(sample_interval_seconds=5.0)
        result = client.clear_anomalies()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Layer 5 — Network


class TestLayer5Network:
    def test_switch_user(self, client):
        result = client.switch_user("alice")
        assert isinstance(result, dict)

    def test_get_active_user(self, client):
        client.switch_user("bob")
        active = client.get_active_user()
        assert isinstance(active, dict)
        assert active.get("active_user") == "bob"

    def test_transfer_delta(self, client):
        a = client.run("source")
        b = client.run("target")
        result = client.transfer_delta(a["session_id"], b["session_id"], max_weight=0.1)
        assert isinstance(result, dict)

    def test_get_trust_scores(self, client):
        scores = client.get_trust_scores()
        assert isinstance(scores, dict)

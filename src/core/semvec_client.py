"""Local Semvec client.

A thin, in-process façade over the bundled :mod:`semvec` package that
exposes the full Layer 1–5 surface (sessions, clusters, regions, the
global observer, and the network transfer layer) as a single ergonomic
client returning plain Python dictionaries.

No network, no API key. The runtime lives in this Python process.
"""

from __future__ import annotations

from typing import Any, Optional

# ``semvec.api.__init__`` pulls in FastAPI/slowapi at import time, which
# we don't need for in-process orchestration. We side-step it by
# registering a minimal stub so the manager submodules import directly.
import sys
import types
import os
import importlib.util

if "semvec.api" not in sys.modules:
    import semvec  # noqa: F401 — ensure package is importable first

    _semvec_origin = importlib.util.find_spec("semvec").origin
    if _semvec_origin is None:
        raise ImportError("Cannot locate the installed `semvec` package")
    _api_path = os.path.join(os.path.dirname(_semvec_origin), "api")
    _stub = types.ModuleType("semvec.api")
    _stub.__path__ = [_api_path]
    sys.modules["semvec.api"] = _stub

from semvec.api.cluster_manager import ClusterManager  # noqa: E402
from semvec.api.event_bus import DriftEventBus  # noqa: E402
from semvec.api.global_observer import GlobalObserver  # noqa: E402
from semvec.api.network_manager import NetworkManager  # noqa: E402
from semvec.api.regional_manager import RegionalManager  # noqa: E402
from semvec.api.session_manager import SessionManager  # noqa: E402

_VALID_DRIFT_PHASES = {"stable", "shifting", "drifted"}


class _DimensionedSessionManager(SessionManager):
    """SessionManager whose default dimension and model match the embedder.

    Cluster-backed sessions are created indirectly by the cluster manager
    without an explicit dimension argument; this subclass supplies the
    defaults derived from the embedder so every session lines up.
    """

    def __init__(
        self,
        default_dimension: int = 768,
        default_model: str = "all-mpnet-base-v2",
        default_owner_subject: str | None = None,
    ) -> None:
        super().__init__()
        self._default_dim = int(default_dimension)
        self._default_model = default_model
        self._default_owner_subject = default_owner_subject

    def create_session(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("dimension", self._default_dim)
        kwargs.setdefault("model_name", self._default_model)
        kwargs.setdefault("owner_subject", self._default_owner_subject)
        return super().create_session(*args, **kwargs)


class SemvecClient:
    """In-process client wrapping the bundled Semvec managers.

    Provides one ergonomic surface that orchestrates session, cluster,
    region, observer, and network managers while returning plain
    dictionaries that are easy to mirror into Neo4j.

    Parameters
    ----------
    embedder:
        Optional duck-typed embedder with ``get_embedding(str) -> ndarray``
        and ``get_dimension() -> int``. When omitted, Semvec lazily loads
        ``sentence-transformers`` with the default model.
    dimension:
        Default embedding dimension used when creating sessions without a
        provided embedder. Ignored if ``embedder`` is supplied.
    model_name:
        Sentence-transformer model name advertised in session metadata.
    owner_subject:
        License subject that owns every session, observer, and network
        partition created through this client (Semvec >= 0.8.7 checks
        session ownership against it). Keyword-only and deliberately
        without a default, mirroring Semvec's own API: forgetting it is a
        ``TypeError`` at construction instead of a silently permissive
        anonymous client. Pass ``None`` explicitly to opt into the
        anonymous/unowned mode.
    """

    def __init__(
        self,
        embedder: Any = None,
        dimension: int = 768,
        model_name: str = "all-mpnet-base-v2",
        *,
        owner_subject: str | None,
    ) -> None:
        if embedder is not None:
            self._dimension = int(embedder.get_dimension())
        else:
            self._dimension = int(dimension)
        self._model_name = model_name
        self._owner_subject = owner_subject

        self._sessions = _DimensionedSessionManager(
            default_dimension=self._dimension,
            default_model=self._model_name,
            default_owner_subject=owner_subject,
        )
        if embedder is not None:
            self._sessions.inject_embedder(embedder)

        self._clusters = ClusterManager(self._sessions)
        self._event_bus = DriftEventBus()
        self._regions = RegionalManager(
            self._sessions, self._clusters, self._event_bus
        )
        self._observer: Optional[GlobalObserver] = None

        self._network = NetworkManager(self._sessions)

    # ------------------------------------------------------------------
    # Health

    def health(self) -> dict:
        return {
            "status": "ok",
            "active_sessions": self._sessions.session_count(),
            "version": "semvec-local",
        }

    # ------------------------------------------------------------------
    # Layer 1 — /run, /store

    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        response: Optional[str] = None,
        short_circuit_threshold: float = 0.85,
        reset_context: bool = False,
    ) -> dict:
        sid = session_id
        if sid is None:
            sid = self._sessions.create_session(
                dimension=self._dimension, model_name=self._model_name
            )
        elif reset_context:
            if self._sessions.get_session(sid) is None:
                sid = self._sessions.create_session(
                    session_id=sid,
                    dimension=self._dimension,
                    model_name=self._model_name,
                )
            else:
                self._sessions.reset_session(sid)
        elif self._sessions.get_session(sid) is None:
            raise ValueError(f"Session not found: {sid}")

        if response:
            self._sessions.store_qa(sid, response)

        top_sim, short_circuit = self._sessions.compute_short_circuit(
            sid, message, threshold=short_circuit_threshold
        )
        drift_score, drift_detected, drift_phase = self._sessions.compute_drift(
            sid, message
        )

        self._sessions.buffer_message(sid, message)
        context = self._sessions.context_block(sid, message, top_k=5)

        if drift_detected:
            self._publish_drift_event(sid, float(drift_score), drift_phase)

        if drift_phase not in _VALID_DRIFT_PHASES:
            drift_phase = "stable"

        return {
            "session_id": sid,
            "context": context,
            "top_similarity": float(top_sim),
            "short_circuit": bool(short_circuit),
            "drift_score": float(drift_score),
            "drift_detected": bool(drift_detected),
            "drift_phase": drift_phase,
        }

    def store(self, session_id: str, response: str) -> dict:
        self._sessions.store_qa(session_id, response)
        return {"session_id": session_id}

    # ------------------------------------------------------------------
    # Layer 1b — Session control

    def create_session(
        self,
        dimension: Optional[int] = None,
        model_name: Optional[str] = None,
        use_cortex: bool = False,
        enable_topic_switch: bool = True,
    ) -> dict:
        sid = self._sessions.create_session(
            dimension=dimension or self._dimension,
            model_name=model_name or self._model_name,
            use_cortex=use_cortex,
            enable_topic_switch=enable_topic_switch,
        )
        return {"session_id": sid, "created": True}

    def export_session(self, session_id: str) -> dict:
        payload = self._sessions.export_state(session_id)
        if payload is None:
            raise ValueError(f"Session not found: {session_id}")
        return payload

    def import_session(self, session_id: str, state_dict: dict) -> dict:
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        ok = self._sessions.import_state(session_id, state_dict)
        if not ok:
            raise ValueError("Malformed or tampered state_dict")
        return {"session_id": session_id, "imported": True}

    def add_anchor(self, session_id: str, embedding: list[float]) -> dict:
        count = self._sessions.add_anchor(session_id, embedding)
        if count is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "anchor_count": count}

    def get_anchor_score(self, session_id: str) -> dict:
        score = self._sessions.get_anchor_score(session_id)
        if score is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, **score}

    def inject_memory(
        self,
        session_id: str,
        embedding: list[float],
        text: str,
        tier: str = "short_term",
        importance: float = 0.5,
        access_count: int = 0,
    ) -> dict:
        total = self._sessions.inject_memory(
            session_id,
            embedding=embedding,
            text=text,
            tier=tier,
            importance=importance,
            access_count=access_count,
        )
        if total is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "tier": tier, "total_memories": total}

    def set_isolation(
        self,
        session_id: str,
        level: str = "OPEN",
        similarity_threshold: float = 0.7,
        exclusion_embeddings: Optional[list[list[float]]] = None,
        allowlist_embeddings: Optional[list[list[float]]] = None,
    ) -> dict:
        ok = self._sessions.set_isolation(
            session_id,
            level=level,
            exclusion_embeddings=exclusion_embeddings,
            allowlist_embeddings=allowlist_embeddings,
            similarity_threshold=similarity_threshold,
        )
        if not ok:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "level": level}

    def add_trigger(
        self,
        session_id: str,
        keyword: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        threshold: float = 0.8,
    ) -> dict:
        count = self._sessions.add_trigger(
            session_id, keyword=keyword, embedding=embedding, threshold=threshold
        )
        if count is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "trigger_count": count}

    def clear_triggers(self, session_id: str) -> dict:
        count = self._sessions.clear_triggers(session_id)
        if count is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "trigger_count": count}

    def release_quarantine(self, session_id: str) -> dict:
        count = self._sessions.release_quarantine(session_id)
        if count is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, "released_count": count}

    # ------------------------------------------------------------------
    # Layer 1c — Session lifecycle (delete / reset)

    def delete_session(self, session_id: str) -> dict:
        """Drop a session from the in-process pool.

        Refuses to delete a cluster's backing session (would orphan the
        cluster). Use :meth:`delete_cluster` for that case.
        """
        if self._clusters.get_cluster(session_id) is not None:
            raise ValueError(
                f"Session {session_id} is the backing session of a cluster — "
                "use delete_cluster() instead"
            )
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        ok = self._sessions.delete_session(session_id)
        return {"session_id": session_id, "deleted": bool(ok)}

    def reset_session(self, session_id: str) -> dict:
        """Wipe accumulated context but keep the session id and config."""
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        ok = self._sessions.reset_session(session_id)
        return {"session_id": session_id, "reset": bool(ok)}

    # ------------------------------------------------------------------
    # Layer 1d — Metrics, memory recall, consistency probe

    def get_session_metrics(self, session_id: str) -> dict:
        """Return phase, FSM and history arrays for the session.

        Mirrors :meth:`SessionManager.get_metrics` and includes
        ``session_id`` for caller convenience. Raises ``ValueError``
        when the session is not in the pool.
        """
        metrics = self._sessions.get_metrics(session_id)
        if metrics is None:
            raise ValueError(f"Session not found: {session_id}")
        return {"session_id": session_id, **metrics}

    def get_relevant_memories(
        self,
        session_id: str,
        top_k: int = 5,
        max_text_chars: int = 500,
        full_first: bool = False,
    ) -> list[dict]:
        """Top-K memories ranked against the session's current state.

        Each entry carries ``text``, ``relevance``, ``memory_hash`` and
        ``truncated``. Pass ``full_first=True`` to keep the highest
        ranked memory un-truncated regardless of ``max_text_chars``.
        """
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        result = self._sessions.get_context(
            session_id,
            top_k=top_k,
            full_first=full_first,
            max_text_chars=max_text_chars,
        )
        return list(result or [])

    def get_memory_by_hash(self, session_id: str, memory_hash: str) -> Optional[dict]:
        """Expand a single memory previously seen via
        :meth:`get_relevant_memories`. Returns ``None`` if the hash is
        unknown to the session; raises ``ValueError`` if the session
        itself is not in the pool.
        """
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        return self._sessions.get_memory_by_hash(session_id, memory_hash)

    # ------------------------------------------------------------------
    # Layer 1e — Compliance helpers (verbatim fact extraction)

    def extract_facts(self, text: str) -> list[dict]:
        """Pull verbatim numeric/date/identifier facts out of *text*.

        Wraps :func:`semvec.compliance.extractors.extract_facts`. The
        upstream extractor is intentionally narrow: ISO/DE/US dates,
        EUR/USD plus a small unit whitelist (``kg``, ``kWh``, ``%`` …),
        and UUID/IBAN/DE-VAT identifiers. Medical-specific units like
        ``mg/m²`` or MRNs are not in the whitelist — store those via
        :meth:`inject_memory` instead.
        """
        from semvec.compliance.extractors import extract_facts as _extract
        out: list[dict] = []
        for fact in _extract(text):
            entry: dict = {
                "kind": fact.kind,
                "raw": fact.raw,
                "start": int(fact.start),
                "end": int(fact.end),
            }
            if fact.kind == "numeric":
                entry["value"] = str(fact.value)   # Decimal → str (json-safe)
                entry["unit"] = fact.unit
            elif fact.kind == "date":
                entry["value"] = fact.value.isoformat()
            elif fact.kind == "identifier":
                entry["value"] = fact.value
                entry["id_type"] = fact.id_type
            out.append(entry)
        return out

    def store_facts_as_entities(self, session_id: str, text: str) -> dict:
        """Extract facts from *text* and store each one as a literal-cache
        entity on the session. Returns ``{"stored": int}``.

        Fact kinds (``numeric``/``date``/``identifier``) are mapped to
        the upstream literal-cache kind ``"constant"`` because that is
        the closest semantic match in
        :class:`semvec._core.EntityKind` — and ``store_entity`` rejects
        anything outside its whitelist. The original fact ``kind`` is
        preserved in the entity ``context`` (``kind=numeric; …``) so
        downstream queries can still tell facts apart.
        """
        if self._sessions.get_session(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")
        from semvec.compliance.extractors import extract_facts as _extract
        stored = 0
        for fact in _extract(text):
            entity = fact.to_code_entity()
            context = f"kind={fact.kind}; {entity.context}"
            ok = self._sessions.store_entity(
                session_id,
                key=entity.key,
                kind="constant",
                value=entity.value,
                context=context,
                importance=1.0,
            )
            if ok is not None:
                stored += 1
        return {"session_id": session_id, "stored": stored}

    # ------------------------------------------------------------------
    # Layer 2c — Cortex consensus engine

    _VALID_CONSENSUS_LEVELS = (
        "simple_majority",
        "qualified_majority",
        "unanimous",
        "weighted_vote",
        "adaptive_threshold",
    )

    def _resolve_consensus_level(self, level: str):
        from semvec.cortex import ConsensusLevel
        if level not in self._VALID_CONSENSUS_LEVELS:
            raise ValueError(
                f"level must be one of {self._VALID_CONSENSUS_LEVELS!r}, got {level!r}"
            )
        return ConsensusLevel(level)

    def _ensure_consensus_engines(self):
        if not hasattr(self, "_consensus_engines"):
            self._consensus_engines: dict[str, dict] = {}

    def create_consensus_engine(
        self,
        local_id: str,
        network_id: str,
        level: str = "qualified_majority",
    ) -> dict:
        from semvec.cortex import ConsensusEngine
        # Validate level eagerly so callers get a clean error.
        default_level = self._resolve_consensus_level(level)
        engine = ConsensusEngine(local_id, network_id)
        self._ensure_consensus_engines()
        self._consensus_engines[engine.consensus_id] = {
            "engine": engine,
            "default_level": default_level,
            # The Rust-backed engine doesn't expose its proposal store
            # via Python attributes, so we track the proposals returned
            # by create_proposal here for later evaluate_consensus calls.
            "proposals": {},
        }
        return {
            "engine_id": engine.consensus_id,
            "local": local_id,
            "network": network_id,
            "level": level,
        }

    def _get_engine(self, engine_id: str):
        self._ensure_consensus_engines()
        slot = self._consensus_engines.get(engine_id)
        if slot is None:
            raise ValueError(f"Consensus engine not found: {engine_id}")
        return slot

    def register_consensus_voter(
        self, engine_id: str, instance_id: str, weight: float = 1.0,
    ) -> dict:
        slot = self._get_engine(engine_id)
        slot["engine"].register_instance(instance_id, weight=weight)
        return {
            "engine_id": engine_id,
            "instance_id": instance_id,
            "weight": float(weight),
            "registered": True,
        }

    def submit_consensus_proposal(
        self,
        engine_id: str,
        proposal_type: str,
        proposed_state: list[float],
        rationale: str,
        level: Optional[str] = None,
        voting_timeout: float = 300.0,
    ) -> dict:
        import numpy as np
        slot = self._get_engine(engine_id)
        cl = (
            self._resolve_consensus_level(level)
            if level is not None
            else slot["default_level"]
        )
        prop = slot["engine"].create_proposal(
            proposal_type,
            proposed_state=np.asarray(proposed_state, dtype=np.float64),
            rationale=rationale,
            consensus_level=cl,
            voting_timeout=voting_timeout,
        )
        slot["proposals"][prop.proposal_id] = prop
        return {
            "engine_id": engine_id,
            "proposal_id": prop.proposal_id,
            "status": prop.status,
            "voting_deadline": prop.voting_deadline,
        }

    def vote_on_consensus(
        self,
        engine_id: str,
        proposal_id: str,
        vote: bool,
        voting_instance: Optional[str] = None,
    ) -> dict:
        slot = self._get_engine(engine_id)
        recorded = slot["engine"].vote_on_proposal(
            proposal_id, bool(vote), voting_instance=voting_instance,
        )
        return {
            "engine_id": engine_id,
            "proposal_id": proposal_id,
            "voting_instance": voting_instance,
            "recorded": bool(recorded),
        }

    def evaluate_consensus(self, engine_id: str, proposal_id: str) -> dict:
        slot = self._get_engine(engine_id)
        proposal = slot["proposals"].get(proposal_id)
        if proposal is None:
            raise ValueError(f"Proposal not found: {proposal_id}")
        accepted, ratio = proposal.calculate_consensus()
        return {
            "engine_id": engine_id,
            "proposal_id": proposal_id,
            "accepted": bool(accepted),
            "ratio": float(ratio),
            "status": proposal.status,
            "votes_for": int(sum(1 for v in proposal.votes.values() if v)),
            "votes_against": int(sum(1 for v in proposal.votes.values() if not v)),
        }

    def get_consensus_statistics(self, engine_id: str) -> dict:
        slot = self._get_engine(engine_id)
        return slot["engine"].get_statistics()

    # ------------------------------------------------------------------
    # Layer 6 — Token reduction (chat proxy with measured savings)

    def create_chat_proxy(
        self,
        llm_call,
        system_prompt: str = "You are a helpful assistant.",
    ) -> "ChatProxyHandle":
        """Open a Semvec-compressed chat session over a user-supplied LLM.

        The proxy keeps its own private :class:`SemvecState` (separate
        from the manager session pool) and compresses prior turns into
        a fixed-size context block before each ``llm_call``. To get
        meaningful ``pss_input_tokens`` your ``llm_call`` callable
        should expose its prompt-token count via a ``last_usage`` dict
        attribute (matches OpenAI's response shape).
        """
        from semvec.token_reduction import SemvecChatProxy
        proxy = SemvecChatProxy(llm_call=llm_call, system_prompt=system_prompt)
        return ChatProxyHandle(_proxy=proxy)

    # ------------------------------------------------------------------
    # Layer 1f — Behavioural-consistency probe

    def verify_consistency(
        self,
        session_id: str,
        test_embeddings: list[list[float]],
        reference_session_id: Optional[str] = None,
        tolerance: float = 1e-3,
    ) -> bool:
        """Behavioural-consistency probe.

        Runs each ``test_embedding`` through the session (and an
        optional reference session) and returns True iff the cosine
        similarities match within ``tolerance``. Useful after
        ``import_session`` to confirm semantic equivalence beyond a
        bare checksum match.
        """
        result = self._sessions.verify_consistency(
            session_id,
            test_embeddings=test_embeddings,
            reference_session_id=reference_session_id,
            tolerance=tolerance,
        )
        if result is None:
            raise ValueError(f"Session not found: {session_id}")
        return bool(result)

    # ------------------------------------------------------------------
    # Layer 2 — Cluster

    def create_cluster(
        self,
        name: str,
        aggregation_mode: str = "weighted_average",
        coupling_factor: float = 0.0,
    ) -> dict:
        rec = self._clusters.create_cluster(
            name=name,
            aggregation_mode=aggregation_mode,
            coupling_factor=coupling_factor,
        )
        return {
            "cluster_id": rec.cluster_id,
            "name": rec.name,
            "aggregation_mode": rec.aggregation_mode,
            "coupling_factor": rec.coupling_factor,
        }

    def list_clusters(self) -> list[dict]:
        return [
            {
                "cluster_id": r.cluster_id,
                "name": r.name,
                "member_count": len(r.member_session_ids),
            }
            for r in self._clusters.list_clusters()
        ]

    def get_cluster(self, cluster_id: str) -> dict:
        state = self._clusters.get_cluster_state(cluster_id)
        if state is None:
            raise ValueError(f"Cluster not found: {cluster_id}")
        return state

    def delete_cluster(self, cluster_id: str) -> dict:
        ok = self._clusters.delete_cluster(cluster_id)
        if not ok:
            raise ValueError(f"Cluster not found: {cluster_id}")
        return {"deleted": True}

    def cluster_run(
        self,
        cluster_id: str,
        message: str,
        response: Optional[str] = None,
        short_circuit_threshold: Optional[float] = None,
    ) -> dict:
        if self._clusters.get_cluster(cluster_id) is None:
            raise ValueError(f"Cluster not found: {cluster_id}")
        threshold = 0.85 if short_circuit_threshold is None else short_circuit_threshold
        return self.run(
            message=message,
            session_id=cluster_id,
            response=response,
            short_circuit_threshold=threshold,
        )

    def cluster_store(self, cluster_id: str, message: str, response: str) -> dict:
        result = self._clusters.store_in_cluster(cluster_id, message, response)
        if result is None:
            raise ValueError(f"Cluster not found: {cluster_id}")
        return result

    def cluster_feedback(self, cluster_id: str) -> dict:
        updated = self._clusters.apply_coupling_feedback(cluster_id)
        if updated < 0:
            raise ValueError(f"Cluster not found: {cluster_id}")
        return {"cluster_id": cluster_id, "sessions_updated": updated}

    def add_cluster_member(self, cluster_id: str, session_id: str) -> dict:
        added = self._clusters.add_member(cluster_id, session_id)
        return {"cluster_id": cluster_id, "session_id": session_id, "added": bool(added)}

    def remove_cluster_member(self, cluster_id: str, session_id: str) -> dict:
        removed = self._clusters.remove_member(cluster_id, session_id)
        return {
            "cluster_id": cluster_id,
            "session_id": session_id,
            "removed": bool(removed),
        }

    # ------------------------------------------------------------------
    # Layer 3 — Region

    def create_region(
        self,
        name: str,
        consensus_threshold: float = 0.5,
        vote_window_seconds: float = 60.0,
    ) -> dict:
        rec = self._regions.create_region(
            name=name,
            consensus_threshold=consensus_threshold,
            vote_window_seconds=vote_window_seconds,
        )
        return {
            "region_id": rec.region_id,
            "name": rec.name,
            "meta_session_id": rec.meta_session_id,
        }

    def list_regions(self) -> list[dict]:
        return [
            {"region_id": r.region_id, "name": r.name, "cluster_count": len(r.cluster_ids)}
            for r in self._regions.list_regions()
        ]

    def get_region(self, region_id: str) -> dict:
        state = self._regions.get_region_state(region_id)
        if state is None:
            raise ValueError(f"Region not found: {region_id}")
        return state

    def delete_region(self, region_id: str) -> dict:
        ok = self._regions.delete_region(region_id)
        if not ok:
            raise ValueError(f"Region not found: {region_id}")
        return {"deleted": True}

    def add_region_cluster(self, region_id: str, cluster_id: str) -> dict:
        added = self._regions.add_cluster(region_id, cluster_id)
        if not added:
            raise ValueError(f"Region not found: {region_id}")
        return {"added": True}

    def remove_region_cluster(self, region_id: str, cluster_id: str) -> dict:
        removed = self._regions.remove_cluster(region_id, cluster_id)
        if not removed:
            raise ValueError(f"Region or cluster not found")
        return {"removed": True}

    def get_region_events(self, region_id: str, limit: int = 20) -> list[dict]:
        if self._regions.get_region(region_id) is None:
            raise ValueError(f"Region not found: {region_id}")
        return [
            {
                "cluster_id": ev.cluster_id,
                "region_id": ev.region_id,
                "drift_score": ev.drift_score,
                "drift_phase": ev.drift_phase,
                "timestamp": ev.timestamp,
            }
            for ev in self._regions.get_recent_events(region_id, limit=limit)
        ]

    # ------------------------------------------------------------------
    # Layer 4 — Global Observer

    def create_observer(
        self,
        sample_interval_seconds: float = 30.0,
        region_ids: Optional[list[str]] = None,
    ) -> dict:
        if self._observer is None:
            self._observer = GlobalObserver(
                cluster_manager=self._clusters,
                regional_manager=self._regions,
                owner_subject=self._owner_subject,
                session_manager=self._sessions,
                sample_interval_seconds=sample_interval_seconds,
            )
        for rid in region_ids or []:
            self._observer.register_region(rid)
        return {
            "observer_id": self._observer.observer_id,
            "registered_regions": len(self._observer.get_registered_regions()),
            "meta_session_id": self._observer.meta_session_id,
        }

    def get_observer_summary(self) -> dict:
        if self._observer is None:
            raise ValueError("Observer not created — call create_observer() first")
        return self._observer.get_summary()

    def observer_sample(self) -> dict:
        if self._observer is None:
            raise ValueError("Observer not created — call create_observer() first")
        return self._observer.sample()

    def get_anomalies(self, limit: int = 20) -> list[dict]:
        if self._observer is None:
            return []
        return [
            {
                "anomaly_id": a.anomaly_id,
                "timestamp": a.timestamp,
                "anomaly_type": a.anomaly_type,
                "affected_cluster_ids": a.affected_cluster_ids,
                "description": a.description,
                "severity": a.severity,
            }
            for a in self._observer.get_anomalies(limit=limit)
        ]

    def clear_anomalies(self) -> dict:
        if self._observer is None:
            return {"cleared": 0}
        return {"cleared": self._observer.clear_anomalies()}

    # ------------------------------------------------------------------
    # Layer 5 — Network

    def transfer_delta(
        self,
        source_session_id: str,
        target_session_id: str,
        max_weight: float = 0.15,
    ) -> dict:
        result = self._network.transfer_delta(
            source_session_id,
            target_session_id,
            self._owner_subject,
            max_weight=max_weight,
        )
        if result is None:
            raise ValueError("Source or target session not found")
        return result

    def switch_user(self, user_id: str) -> dict:
        result = self._network.switch_user(user_id, self._owner_subject)
        if result is None:
            raise ValueError(
                f"User partition for {user_id!r} exists but is owned by another subject"
            )
        return result

    def get_active_user(self) -> dict:
        return {
            "active_user": self._network.get_active_user(
                owner_subject=self._owner_subject
            )
        }

    def propose_consensus(
        self, proposer_session_id: str, target_embedding: list[float]
    ) -> dict:
        result = self._network.propose_consensus(
            proposer_session_id, target_embedding, self._owner_subject
        )
        if result is None:
            raise ValueError(f"Proposer session not found: {proposer_session_id}")
        return result

    def get_trust_scores(self) -> dict:
        return {"trust_scores": self._network.get_trust_scores(self._owner_subject)}

    # ------------------------------------------------------------------
    # Internal — drift event publication mirrors REST orchestration.

    def _publish_drift_event(
        self, session_id: str, drift_score: float, drift_phase: str
    ) -> None:
        cluster_id = self._clusters.get_cluster_for_session(session_id)
        # A session IS a cluster's backing session when ``session_id == cluster_id``.
        if cluster_id is None and self._clusters.get_cluster(session_id) is not None:
            cluster_id = session_id
        if cluster_id is None:
            return
        region_id = self._regions.get_region_for_cluster(cluster_id)
        if region_id is None:
            return
        self._regions.publish_drift(cluster_id, region_id, drift_score, drift_phase)


class ChatProxyHandle:
    """Lightweight wrapper around :class:`SemvecChatProxy`.

    Constructed via :meth:`SemvecClient.create_chat_proxy`. Each
    :meth:`turn` returns a plain dict so callers do not have to import
    :class:`semvec.token_reduction.TurnResult`.
    """

    __slots__ = ("_proxy",)

    def __init__(self, *, _proxy):
        self._proxy = _proxy

    def turn(self, user_message: str) -> dict:
        result = self._proxy.chat(user_message)
        return {
            "response": result.response,
            "pss_input_tokens": result.pss_input_tokens,
            "baseline_input_tokens": result.baseline_input_tokens,
            "pss_prompt": result.pss_prompt,
            "phase": result.phase,
            "turn_number": result.turn_number,
        }

    def summary(self) -> dict:
        return self._proxy.get_summary()

    def print_report(self) -> None:
        self._proxy.print_report()

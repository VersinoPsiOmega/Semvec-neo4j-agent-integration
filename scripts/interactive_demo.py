#!/usr/bin/env python3
"""Interactive Semvec + Neo4j Demo — shows the interplay between both systems.

Scenarios:
  1. Live Drift Detection    — Type messages, watch drift score + Neo4j state chain grow
  2. Topic Switch Detection  — Guided scenario: stay on topic, then switch, observe drift
  3. Multi-Agent Comparison  — Run parallel agents, compare trajectories in Neo4j
  4. Memory & Recall         — Store memories, query by similarity, consolidate tiers
  5. Cross-Session Analytics  — Influence scoring, similarity matrix, drift points
  6. Session Export/Transfer  — Semvec Layer 5: transfer knowledge between agents
  7. Neo4j Graph Explorer     — Run Cypher queries directly, see the graph grow

Usage:
    python3 scripts/interactive_demo.py              # real LLM responses (default)
    python3 scripts/interactive_demo.py --no-llm     # skip LLM calls (shows N/A for responses)
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.core.embedder import SentenceTransformerEmbedder
from src.core.semvec_client import SemvecClient
from src.mcp.semvec_mcp_server import SemvecMCPServer

_SHARED_SEMVEC: SemvecClient | None = None


def _build_semvec_client() -> SemvecClient:
    """Single factory so all scenarios share one embedder model load."""
    global _SHARED_SEMVEC
    if _SHARED_SEMVEC is None:
        _SHARED_SEMVEC = SemvecClient(embedder=SentenceTransformerEmbedder(), owner_subject="local-demo")
    return _SHARED_SEMVEC
from src.persistence.adapter import Neo4jSemvecAdapter
from scripts.demo_helpers import format_observer_sample, sim_bar


def _print_observer_sample(label: str, sample) -> None:
    """Thin wrapper so the call sites stay one-liners."""
    print(f"  {format_observer_sample(label, sample)}")

# ── Config ──────────────────────────────────────────────────────────────


def _require_env(*names: str) -> dict[str, str]:
    """Read required env vars; abort with a single clear error if any are missing."""
    values = {n: os.environ.get(n) for n in names}
    missing = [n for n, v in values.items() if not v]
    if missing:
        sys.stderr.write(
            "Missing environment variable(s): " + ", ".join(missing) + "\n"
            "Run `cp .env.example .env` and fill in values.\n"
        )
        sys.exit(1)
    return values  # type: ignore[return-value]


_env = _require_env("NEO4J_TEST_URI", "NEO4J_TEST_USER", "NEO4J_TEST_PASSWORD", "NEO4J_TEST_DATABASE")
NEO4J_URI = _env["NEO4J_TEST_URI"]
NEO4J_USER = _env["NEO4J_TEST_USER"]
NEO4J_PASSWORD = _env["NEO4J_TEST_PASSWORD"]
NEO4J_DATABASE = _env["NEO4J_TEST_DATABASE"]
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema", "semvec_schema.cypher")

USE_LLM = "--llm" in sys.argv or "--no-llm" not in sys.argv  # LLM on by default, --no-llm to disable

# Drift-detection thresholds. A DRIFT is flagged when BOTH the drift score
# crosses ``DRIFT_THRESHOLD`` and the cosine similarity to the accumulated
# context falls below ``DRIFT_SIM_CEILING``. The combined check filters out
# in-topic sub-shifts (high drift but still high similarity) and isolates
# real topic switches. Configure both knobs via .env.
_drift_env = _require_env("DRIFT_THRESHOLD", "DRIFT_SIM_CEILING")
DRIFT_THRESHOLD = float(_drift_env["DRIFT_THRESHOLD"])
DRIFT_SIM_CEILING = float(_drift_env["DRIFT_SIM_CEILING"])


def _is_drift(result: dict) -> bool:
    """Return True iff the turn looks like a real topic switch.

    Combines Semvec's own ``drift_detected`` flag with a sim/drift gate so
    sub-topic shifts inside the same domain (high drift but still high
    similarity to accumulated context) are not flagged as drift events.
    """
    if result.get("drift_detected", False):
        return True
    drift_score = float(result.get("drift_score", 0.0))
    similarity = float(result.get("top_similarity", 1.0))
    return drift_score >= DRIFT_THRESHOLD and similarity <= DRIFT_SIM_CEILING

# ── LLM Client ──────────────────────────────────────────────────────────

class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self):
        import openai
        env = _require_env("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
        self._client = openai.OpenAI(base_url=env["OPENAI_BASE_URL"], api_key=env["OPENAI_API_KEY"])
        self._model = env["OPENAI_MODEL"]
        # Reasoning off by default. Set OPENAI_ENABLE_THINKING=1 in .env
        # to let reasoning models emit chain-of-thought.
        self._default_enable_thinking = (
            os.environ.get("OPENAI_ENABLE_THINKING", "0") == "1"
        )

    def respond(
        self,
        user_message: str,
        system_prompt: str = "",
        context: str = "",
        max_tokens: int = 400,
        enable_thinking: bool | None = None,
    ) -> str:
        """Generate a response. Returns content (or reasoning_content as fallback).

        enable_thinking:
            None  (default) — use class default (env OPENAI_ENABLE_THINKING, default False).
            False          — force reasoning off (sends enable_thinking=False via
                             chat_template_kwargs; Qwen3 / DeepSeek-R1 / o1-like models
                             will then answer directly without a chain-of-thought prefix).
            True           — force reasoning on; budget is bumped to 2048 tokens min.
        """
        if enable_thinking is None:
            enable_thinking = self._default_enable_thinking

        messages: list[dict] = []
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if context:
            system_parts.append(f"Conversation context so far:\n{context}")
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": user_message})

        # Reasoning models (Qwen3 etc.) can burn the whole budget on chain-of-thought
        # before emitting any visible content. Bump the budget when thinking is on.
        effective_max = max(max_tokens, 2048) if enable_thinking else max_tokens

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max,
        }
        if not enable_thinking:
            # Tell reasoning-capable models to skip chain-of-thought.
            # Unknown keys are ignored by non-reasoning models.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            r = self._client.chat.completions.create(**kwargs)
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            if content:
                return content
            if reasoning:
                return reasoning
            return "(empty response — model returned no content)"
        except Exception as e:
            return f"(LLM error: {e})"


_llm: LLMClient | None = None

def get_llm() -> LLMClient:
    """Lazy-init singleton."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def generate_response(
    user_message: str,
    agent_role: str = "a helpful medical research assistant",
    semvec_context: str = "",
) -> str:
    """Generate an agent response via LLM. Returns 'N/A' if LLM is disabled."""
    if not USE_LLM:
        return "(LLM disabled — use --llm or remove --no-llm)"
    llm = get_llm()
    system = (
        f"You are {agent_role}. "
        "Answer concisely in 2-3 sentences. Stay factual."
    )
    return llm.respond(user_message, system_prompt=system, context=semvec_context)

# ── ANSI Colors ─────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

def color_drift(score: float) -> str:
    if score < 0.1:
        return f"{GREEN}{score:.3f}{RESET}"
    elif score < 0.3:
        return f"{YELLOW}{score:.3f}{RESET}"
    elif score < 0.5:
        return f"{RED}{score:.3f}{RESET}"
    return f"{BG_RED}{WHITE}{score:.3f}{RESET}"

def color_phase(phase: str) -> str:
    colors = {
        "initialization": DIM,
        "exploration": BLUE,
        "convergence": CYAN,
        "resonance": GREEN,
        "stability": f"{BOLD}{GREEN}",
        "instability": f"{BOLD}{RED}",
    }
    return f"{colors.get(phase, '')}{phase}{RESET}"

def bar(value: float, width: int = 30, fill="█", empty="░") -> str:
    filled = int(value * width)
    return f"{fill * filled}{empty * (width - filled)}"

def header(text: str):
    w = 70
    print(f"\n{CYAN}{'═' * w}{RESET}")
    print(f"{CYAN}  {BOLD}{text}{RESET}")
    print(f"{CYAN}{'═' * w}{RESET}\n")

def subheader(text: str):
    print(f"\n{YELLOW}── {text} {'─' * max(0, 60 - len(text))}{RESET}\n")

def info(label: str, value: str):
    print(f"  {DIM}{label:>22}{RESET}  {value}")

# Real embedding via sentence-transformers (GPU → CPU)
_embedder = None

def embed(text: str) -> list[float]:
    """Embed text using all-MiniLM-L6-v2 (GPU if available, else CPU).
    Raises ImportError if sentence-transformers is not installed."""
    global _embedder
    if _embedder is None:
        import warnings, logging, io, contextlib
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        for logger_name in ["sentence_transformers", "huggingface_hub", "transformers"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        with contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print(f"  {DIM}Embedder: all-MiniLM-L6-v2 on {device}{RESET}")
    return _embedder.encode(text).tolist()


# ── Reset: clean Semvec artifacts, keep healthcare template ────────────────

_SEMVEC_LABELS = [
    "AgentSession", "SemanticState", "DriftEvent", "Phase",
    "Memory", "MemoryCluster", "Cluster", "Region",
    "GlobalObserver", "ConsensusEvent", "AnomalyEvent",
]
# Healthcare labels that must NEVER be deleted
_HEALTHCARE_LABELS = {
    "Patient", "Provider", "Diagnosis", "Treatment", "Encounter",
    "Facility", "Medication", "Person", "Organization", "Location",
    "Event", "Object", "Document", "DecisionTrace", "TraceStep",
}


def _reset_semvec_artifacts(driver, semvec=None):
    """Delete all Semvec demo artifacts. Healthcare template stays untouched.

    When the in-process SemvecClient is also supplied, every non-cluster
    session in its pool is dropped via :meth:`SemvecClient.delete_session`
    so re-running scenarios in the same process starts from a clean slate
    on both sides (Neo4j + in-memory Semvec).
    """
    with driver.session(database=NEO4J_DATABASE) as db:
        # Delete INVESTIGATED relationships (demo-created links to healthcare)
        db.run("MATCH ()-[r:INVESTIGATED]->() DELETE r").consume()
        # Delete all Semvec node types + their relationships
        for label in _SEMVEC_LABELS:
            db.run(f"MATCH (n:{label}) DETACH DELETE n").consume()

    dropped = 0
    if semvec is not None:
        # Walk the in-process session pool. Skip cluster-backed sessions
        # (they go away with delete_cluster); drop the rest so the next
        # scenario does not inherit context. Refusing-on-cluster is built
        # into delete_session, but we filter eagerly to keep the loop quiet.
        try:
            sm = semvec._sessions       # private but stable across 0.5.x
            cm = semvec._clusters
            session_ids = list(getattr(sm, "_sessions", {}).keys())
            for sid in session_ids:
                if cm.get_cluster(sid) is not None:
                    continue
                try:
                    semvec.delete_session(sid)
                    dropped += 1
                except Exception:
                    pass
        except Exception:
            pass
    suffix = f", dropped {dropped} in-process sessions" if dropped else ""
    print(f"  {DIM}Reset: Semvec artifacts cleared, healthcare template intact{suffix}{RESET}")


# ── Audit-Trail: rich INVESTIGATED edges ────────────────────────────────
# Centralised so every edge carries the same metadata (timestamps, duration,
# query/response previews, top-K similarity, Semvec drift phase, short-circuit,
# llm_call). Scenario 7's graph queries surface this as a complete AI audit.

def _shorten(text: str | None, maxlen: int = 160) -> str:
    if not text:
        return ""
    t = " ".join(str(text).split())
    return t if len(t) <= maxlen else t[: maxlen - 1] + "…"


def record_investigated(
    driver,
    session_id: str,
    entity_label: str,
    entity_name: str,
    *,
    step: int,
    phase: str,
    drift_score: float = 0.0,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    query: str | None = None,
    response: str | None = None,
    top_k_sim: float | None = None,
    semvec_drift_phase: str | None = None,
    short_circuit: bool | None = None,
    extra: dict | None = None,
) -> None:
    """MERGE an INVESTIGATED edge enriched with timing + drift metadata.

    All keyword args are optional. Empty fields are omitted (no clutter in
    the graph). Timestamps are written as ISO-8601 UTC strings; duration_ms
    is computed automatically when both timestamps are provided.
    """
    if started_at is None:
        started_at = datetime.now(timezone.utc)
    if ended_at is None:
        ended_at = started_at

    params: dict = {
        "sid": session_id,
        "name": entity_name,
        "step": step,
        "phase": phase,
        "drift_score": float(drift_score),
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "ended_at": ended_at.astimezone(timezone.utc).isoformat(),
        "duration_ms": int(max(0.0, (ended_at - started_at).total_seconds() * 1000)),
    }

    if query is not None:
        params["query_preview"] = _shorten(query, 200)
    if response is not None:
        params["response_preview"] = _shorten(response, 240)
    if top_k_sim is not None:
        params["top_k_similarity"] = float(top_k_sim)
    if semvec_drift_phase is not None:
        params["semvec_drift_phase"] = str(semvec_drift_phase)
    if short_circuit is not None:
        params["short_circuit"] = bool(short_circuit)
        params["llm_call"] = not bool(short_circuit)
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            params[k] = v

    static_keys = ("step", "phase", "drift_score",
                   "started_at", "ended_at", "duration_ms")
    optional_keys = [k for k in params.keys()
                     if k not in ("sid", "name") and k not in static_keys]
    set_pairs = ", ".join(f"{k}: ${k}" for k in (*static_keys, *optional_keys))
    cypher = (
        f"MATCH (s:AgentSession {{session_id: $sid}}) "
        f"MATCH (e:{entity_label} {{name: $name}}) "
        f"MERGE (s)-[inv:INVESTIGATED {{step: $step, phase: $phase}}]->(e) "
        f"SET inv += {{{set_pairs}}}"
    )
    with driver.session(database=NEO4J_DATABASE) as db:
        db.run(cypher, **params).consume()


# ── Scenario 1: Live Drift + Token Savings ─────────────────────────────

def scenario_live_drift(mcp: SemvecMCPServer, driver):
    header("Scenario 1: Live Drift + Token Savings")
    print("  Each turn shows two things side by side:")
    print(f"    - {BOLD}Drift{RESET}: similarity, drift score, Semvec phase, Neo4j state chain")
    print(f"    - {BOLD}Token savings{RESET}: PSS-compressed prompt vs naïve full-history baseline")
    print(f"  This scenario {BOLD}requires{RESET} a working LLM endpoint (OPENAI_BASE_URL + OPENAI_API_KEY).")
    print(f"  Type {BOLD}'quit'{RESET} to return to menu.\n")

    if not USE_LLM:
        print(f"  {RED}This scenario needs the LLM. Restart without --no-llm.{RESET}")
        return

    # Hard pre-flight: surface a clear error before the user types anything
    # if the LLM endpoint is down. No silent fallback to placeholder text —
    # the whole point of Scenario 1 is the live LLM round-trip.
    print(f"  {DIM}…{RESET} probing LLM endpoint at {os.environ['OPENAI_BASE_URL']} …", flush=True)
    try:
        probe = get_llm().respond("Say OK", max_tokens=8)
        if not probe:
            raise RuntimeError("LLM returned an empty response")
    except Exception as e:
        print()
        print(f"  {RED}{BOLD}LLM endpoint unreachable.{RESET}")
        print(f"  {RED}Reason:{RESET} {e}")
        print(f"  {DIM}OPENAI_BASE_URL = {os.environ['OPENAI_BASE_URL']}{RESET}")
        print(f"  {DIM}OPENAI_MODEL    = {os.environ['OPENAI_MODEL']}{RESET}")
        print(f"  {DIM}Check the value in .env and that the endpoint is reachable.{RESET}")
        print(f"  {YELLOW}Scenario 1 aborted. Pick another scenario or fix .env and restart.{RESET}")
        return
    print(f"  {GREEN}LLM ready.{RESET} ({os.environ['OPENAI_MODEL']})\n")

    session = mcp.create_agent_session(agent_id=f"interactive-{int(time.time())}")
    sid = session["session_id"]
    info("Session", sid)
    info("Agent", session["agent_id"])
    info("LLM", f"{GREEN}enabled{RESET} ({os.environ['OPENAI_MODEL']})")
    print()

    # Live token-savings panel: compare the prompt size Semvec would
    # send (system + compressed context + user message) against the
    # naïve baseline (full chat history replayed each turn).
    # estimate_tokens raises RuntimeError when tiktoken is missing AND
    # ImportError when the token_reduction module itself is missing —
    # we tolerate both and just skip the panel.
    try:
        from semvec.token_reduction import estimate_tokens
        _token_panel = True
    except Exception as _e:
        estimate_tokens = None  # type: ignore[assignment]
        _token_panel = False
        print(f"  {DIM}Token-savings panel disabled: {_e}{RESET}")
    chat_history: list[tuple[str, str]] = []   # (role, content)
    pss_token_total = 0
    baseline_token_total = 0
    SYS_PROMPT = "You are a helpful assistant. Answer using the conversation context."

    step = 0
    while True:
        try:
            msg = input(f"  {CYAN}[step {step}]{RESET} You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() == "quit":
            break

        t0 = time.time()
        result = mcp.detect_drift(sid, msg)
        semvec_latency = (time.time() - t0) * 1000

        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        phase = result["drift_phase"]
        detected = _is_drift(result)

        # Generate agent response — real LLM, no fallback. If the call
        # fails mid-loop we surface the error immediately and break so
        # the user sees the failure, not a placeholder string.
        t1 = time.time()
        try:
            response = get_llm().respond(
                msg,
                system_prompt="You are a helpful assistant. Answer concisely in 2-3 sentences.",
                context=result.get("context", ""),
            )
        except Exception as e:
            print()
            print(f"  {RED}{BOLD}LLM call failed at step {step}.{RESET}")
            print(f"  {RED}Reason:{RESET} {e}")
            print(f"  {YELLOW}Scenario 1 aborted; the session up to here is in Neo4j.{RESET}")
            break
        llm_latency = (time.time() - t1) * 1000

        # Teach Semvec the response
        mcp.store_response(sid, response)

        print()
        print(f"  {GREEN}Agent:{RESET} {textwrap.fill(response, 72, subsequent_indent='         ')}")
        print()
        info("Similarity", f"{sim:.3f}  {bar(sim)}")
        info("Drift Score", f"{drift_score:.3f}")
        info("Phase", color_phase(phase))
        info("Drift Detected", f"{RED}YES{RESET}" if detected else f"{GREEN}no{RESET}")
        info("Context (Semvec)", textwrap.shorten(result["context"], 80) if result["context"] else "(empty)")
        info("Neo4j State", f"#{result['step']}  (id: {result['state_id'][:12]}...)")
        info("Latency", f"Semvec {semvec_latency:.0f}ms + LLM {llm_latency:.0f}ms")

        # Token-savings accounting for this turn. The PSS context block
        # carries a small constant header — early turns can cost slightly
        # MORE than the naïve baseline. The compression pays off as soon
        # as the conversation has enough history that replaying it costs
        # more than the compressed context. The cumulative panel below
        # shows the cross-over.
        pss_prompt = (
            f"{SYS_PROMPT}\n\nContext:\n{result.get('context', '')}\n\n"
            f"User: {msg}"
        )
        baseline_history = [SYS_PROMPT]
        baseline_history.extend(f"{role.capitalize()}: {content}" for role, content in chat_history)
        baseline_history.append(f"User: {msg}")
        baseline_prompt = "\n".join(baseline_history)
        try:
            if not _token_panel:
                raise RuntimeError("token panel disabled at startup")
            pss_t = estimate_tokens(pss_prompt)
            base_t = estimate_tokens(baseline_prompt)
            pss_token_total += pss_t
            baseline_token_total += base_t
            delta = pss_t - base_t
            arrow = f"{RED}+{delta}{RESET}" if delta > 0 else f"{GREEN}{delta}{RESET}"
            info(
                "Tokens (this turn)",
                f"PSS {pss_t}  baseline {base_t}  (Δ {arrow})",
            )
            # Memory tier metric — context stays linear until the
            # short-term tier (default 15) starts evicting older
            # entries, after which PSS plateaus while baseline keeps
            # growing. The cross-over typically happens around turn
            # 15-25 depending on message length.
            try:
                semvec_client = _build_semvec_client()
                # mcp.detect_drift uses its own internal session id;
                # we have to ask the detector for the matched semvec
                # session.
                semvec_sid = mcp._detector.get_semvec_session_id(sid)
                if semvec_sid:
                    m = semvec_client.get_session_metrics(semvec_sid).get(
                        "total_memories", "?"
                    )
                    info("Memories", f"{m}  (cross-over near tier limit ~15)")
            except Exception:
                pass
        except Exception:
            # tiktoken missing or token panel disabled — skip silently
            pass
        chat_history.append(("user", msg))
        chat_history.append(("assistant", response))

        # Show Neo4j state chain length
        trajectory = mcp.get_state_trajectory(sid, steps=50)
        current_phase = mcp.get_phase(sid)
        info("Chain Length", f"{len(trajectory)} states in Neo4j")
        info("Neo4j Phase", color_phase(current_phase.get("phase", "N/A")))
        print()
        step += 1

    # End session summary
    summary = mcp.end_agent_session(sid)
    subheader("Session Summary")
    info("Final Phase", summary.get("final_phase", "N/A"))
    info("Total States", str(summary.get("total_states", 0)))
    info("Drift Events", str(summary.get("total_drift_events", 0)))
    if baseline_token_total:
        saving_total = (baseline_token_total - pss_token_total) / baseline_token_total * 100
        if saving_total > 0:
            label = f"{GREEN}{saving_total:.1f}% saved{RESET}"
        else:
            label = f"{YELLOW}PSS used {abs(saving_total):.1f}% MORE (overhead phase){RESET}"
        info(
            "Tokens (cumulative)",
            f"PSS {pss_token_total}  baseline {baseline_token_total}  ({label})",
        )
        if saving_total <= 0:
            print(
                f"  {DIM}Note: {step} turn(s) is below the short-term tier limit (default 15). "
                f"PSS context grows until the tier fills, then plateaus while baseline keeps "
                f"growing — keep the conversation going to see the cross-over.{RESET}"
            )


# ── Scenario 2: Topic Switch Detection ──────────────────────────────────

def scenario_topic_switch(mcp: SemvecMCPServer, driver):
    header("Scenario 2: Drift Detection + Short-Circuit")
    print("  Four phases demonstrating drift AND short-circuit in one narrative:")
    print(f"    {GREEN}Phase 1{RESET}  Morrison diabetes workup — deep clinical context")
    print(f"    {RED}Phase 2{RESET}  Patel psychiatry (MDD/Sertraline/CBT) — different specialty")
    print(f"    {CYAN}Phase 3{RESET}  Rodriguez cardiology — return to clinical domain")
    print(f"    {MAGENTA}Phase 4{RESET}  Paraphrased Phase 1 queries — short-circuit, LLM skipped")
    print(f"  {BOLD}30 steps total.{RESET}\n")

    semvec = _build_semvec_client()

    # We use the raw Semvec client so we can leverage the inline-store pattern
    # (response= on the next /run call) and read short_circuit directly.
    # Neo4j mirroring happens via MCP in parallel.
    neo4j_session = mcp.create_agent_session(agent_id="drift-shortcircuit-demo")
    neo4j_sid = neo4j_session["session_id"]

    # Phase 1: 17 clinical investigation queries — deep diabetes patient workup
    # (References: Patient James Morrison, Type 2 Diabetes + COPD + HTN,
    #  treated by Dr. Volkov and Dr. Tanaka, Metformin/Lisinopril/Atorvastatin)
    on_topic_1 = [
        "James Morrison presents with Type 2 Diabetes Mellitus — what is his current treatment protocol?",
        "He is on Metformin 500mg — are there contraindications given his COPD diagnosis?",
        "Metformin is contraindicated with renal impairment GFR below 30 — check his lab results",
        "Dr. Elena Volkov and Dr. Yuki Tanaka both treat Morrison — who should lead the diabetes management?",
        "Lisinopril 10mg is also indicated for his Essential Hypertension — is the combination safe?",
        "What Atorvastatin 40mg interactions should we monitor alongside Metformin?",
        "His encounter at Emergency Room Visit showed chest pain — should we adjust his diabetes meds?",
        "Review the Cardiac Catheterization treatment — does it use Metformin or Atorvastatin?",
        "What is the recommended insulin regimen if Metformin fails to achieve target A1C?",
        "Are there drug interactions between Metformin and the contrast dye used in catheterization?",
        "What dietary recommendations should Morrison follow alongside his diabetes medications?",
        "How does continuous glucose monitoring improve glycemic control in Type 2 Diabetes?",
        "What role does bariatric surgery play in diabetes remission for patients like Morrison?",
        "How do we manage diabetic nephropathy progression in stage 3 CKD?",
        "What are the cardiovascular risk factors we should monitor given his comorbidities?",
        "Should we consider adding an SGLT2 inhibitor like Empagliflozin to Morrisons regimen?",
        "What is the long-term prognosis for Morrison with optimal diabetes management?",
    ]
    # Phase 2: Different patient, different specialty — psychiatry
    # (References: Aisha Patel, Major Depressive Disorder, Sertraline 50mg, CBT)
    # Realistic: same clinician sees next patient in a different domain.
    off_topic = [
        "Aisha Patel presents with Major Depressive Disorder — what is her current treatment plan?",
        "She is on Sertraline 50mg — what are the common side effects and monitoring requirements?",
        "Is Cognitive Behavioral Therapy recommended alongside Sertraline for treatment-resistant depression?",
        "What PHQ-9 score threshold should trigger a medication adjustment for Patel?",
        "Are there contraindications between Sertraline and any medications Patel might be taking?",
    ]
    # Phase 3: Return to clinical — different patient, cardiology focus
    # (References: Maria Rodriguez, AMI + Hypertension, Dr. O'Brien → Dr. Okonkwo)
    on_topic_2 = [
        "Maria Rodriguez was diagnosed with Acute Myocardial Infarction — what is her treatment plan?",
        "Dr. O'Brien referred her to Dr. Okonkwo — is the referral for cardiology follow-up?",
        "Her encounter at Emergency Room Visit resulted in which diagnoses?",
    ]
    # Phase 4: Paraphrased Phase 1 queries — Semvec should short-circuit (context from Phase 1)
    short_circuit_queries = [
        ("What treatment protocol does James Morrison follow for his diabetes?",           "≈ Phase 1 Q1"),
        ("Are there Metformin contraindications for a patient with COPD?",                 "≈ Phase 1 Q2"),
        ("Check renal function thresholds for Metformin prescribing",                      "≈ Phase 1 Q3"),
        ("Which providers are responsible for Morrison's diabetes care?",                   "≈ Phase 1 Q4"),
        ("Is Lisinopril safe to combine with Metformin for hypertension and diabetes?",    "≈ Phase 1 Q5"),
    ]

    all_msgs = on_topic_1 + off_topic + on_topic_2
    all_labels = (
        ["ON-TOPIC"] * len(on_topic_1)
        + ["OFF-TOPIC"] * len(off_topic)
        + ["NEW-TOPIC"] * len(on_topic_2)
    )
    all_roles = (
        ["a clinical pharmacist reviewing patient James Morrison's diabetes treatment"] * len(on_topic_1)
        + ["a psychiatrist managing Aisha Patel's depression treatment"] * len(off_topic)
        + ["a cardiologist reviewing Maria Rodriguez's post-MI care"] * len(on_topic_2)
    )

    # ── Audit-only top-K rolling window ──
    # We keep a small rolling window of recent embeddings purely to enrich
    # the INVESTIGATED edges with a top-K similarity signal (useful for
    # post-hoc analysis). Drift detection itself stays on the verified
    # Semvec-side combination ``drift_score >= DRIFT_THRESHOLD AND
    # top_similarity <= DRIFT_SIM_CEILING`` (see ``_is_drift``).
    import numpy as np
    RECENT_WINDOW_K = 10
    recent_embeddings: deque = deque(maxlen=RECENT_WINDOW_K)

    def _top_k_similarity(qv: np.ndarray) -> float:
        if not recent_embeddings:
            return 1.0
        qn = float(np.linalg.norm(qv))
        if qn == 0:
            return 0.0
        best = 0.0
        for e in recent_embeddings:
            en = float(np.linalg.norm(e))
            if en == 0:
                continue
            best = max(best, float(np.dot(qv, e) / (qn * en)))
        return best

    # Legend
    subheader("How to read the output")
    print(f"  Each step shows the {BOLD}Semvec similarity{RESET} — how well the query matches accumulated context.")
    print(f"  All signals come directly from the Semvec runtime, no client-side heuristics.\n")
    print(f"  {GREEN}{'█' * 5}{RESET} sim > 0.4 (on-topic)     "
          f"{YELLOW}{'█' * 5}{RESET} sim 0.2–0.4 (shifting)     "
          f"{RED}{'█' * 5}{RESET} sim < 0.2 (off-topic)")
    print(f"  {BOLD}DRIFT{RESET} = Semvec drift_detected=True OR (drift_score >= {DRIFT_THRESHOLD} AND sim <= {DRIFT_SIM_CEILING})")
    print(f"  {DIM}phase{RESET} = Semvec drift_phase: stable / shifting / drifted")
    print()

    # Map queries to healthcare entities they reference (for INVESTIGATED relationships)
    _entity_refs: list[list[tuple[str, str]]] = [
        # Phase 1: Morrison diabetes workup (17 queries)
        [("Patient", "James Morrison"), ("Diagnosis", "Type 2 Diabetes Mellitus")],
        [("Medication", "Metformin 500mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
        [("Medication", "Metformin 500mg")],
        [("Provider", "Dr. Elena Volkov"), ("Provider", "Dr. Yuki Tanaka")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Essential Hypertension")],
        [("Medication", "Atorvastatin 40mg"), ("Medication", "Metformin 500mg")],
        [("Encounter", "Emergency Room Visit"), ("Patient", "James Morrison")],
        [("Treatment", "Cardiac Catheterization"), ("Medication", "Metformin 500mg")],
        [("Medication", "Metformin 500mg")],          # insulin regimen
        [("Medication", "Metformin 500mg"), ("Treatment", "Cardiac Catheterization")],  # contrast
        [("Patient", "James Morrison")],               # dietary
        [("Patient", "James Morrison")],               # CGM
        [("Patient", "James Morrison")],               # bariatric
        [("Patient", "James Morrison"), ("Medication", "Lisinopril 10mg")],  # nephropathy
        [("Patient", "James Morrison"), ("Medication", "Atorvastatin 40mg")],  # CV risk
        [("Patient", "James Morrison")],               # SGLT2i
        [("Patient", "James Morrison")],               # prognosis
        # Phase 2: Psychiatry — Aisha Patel
        [("Patient", "Aisha Patel"), ("Diagnosis", "Major Depressive Disorder")],
        [("Medication", "Sertraline 50mg")],
        [("Treatment", "Cognitive Behavioral Therapy")],
        [("Patient", "Aisha Patel")],
        [("Medication", "Sertraline 50mg")],
        # Phase 3: Rodriguez cardiology
        [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")],
        [("Provider", "Dr. Michael O'Brien"), ("Provider", "Dr. Rachel Okonkwo")],
        [("Encounter", "Emergency Room Visit")],
    ]

    semvec_sid = None
    prev_response = None
    sc_threshold = 0.65
    drift_events_count = 0
    BAR_W = 20  # bar width

    prev_label = None
    for i, (msg, label, role) in enumerate(zip(all_msgs, all_labels, all_roles)):
        # Phase separator
        if label != prev_label:
            phase_names = {"ON-TOPIC": "Phase 1: Morrison Diabetes", "OFF-TOPIC": "Phase 2: Patel Psychiatry", "NEW-TOPIC": "Phase 3: Rodriguez Cardiology"}
            phase_colors = {"ON-TOPIC": GREEN, "OFF-TOPIC": RED, "NEW-TOPIC": CYAN}
            c = phase_colors.get(label, "")
            print(f"\n  {c}{'─' * 60}")
            print(f"  {BOLD}{phase_names.get(label, label)}{RESET}")
            print(f"  {c}{'─' * 60}{RESET}\n")
            prev_label = label

        # Client-side top-K rolling similarity (audit signal, not used for
        # drift verdict — see scenario-specific calibration discussion).
        query_vec = np.array(embed(msg), dtype=np.float32)
        top_k_sim = _top_k_similarity(query_vec)

        # Timestamp before the LLM call
        t_start = datetime.now(timezone.utc)

        # Mirror to Neo4j via MCP (drift detection + persistence)
        result = mcp.detect_drift(neo4j_sid, msg)

        # Also run on raw Semvec session for short-circuit tracking (inline-store)
        semvec_data = semvec.run(msg, session_id=semvec_sid, response=prev_response,
                           short_circuit_threshold=sc_threshold)
        semvec_sid = semvec_data["session_id"]

        response = generate_response(msg, agent_role=role, semvec_context=result.get("context", ""))
        mcp.store_response(neo4j_sid, response)
        prev_response = response
        t_end = datetime.now(timezone.utc)

        # Update rolling window after the turn so the next step's top_k_sim
        # is measured against the prior context only.
        recent_embeddings.append(query_vec)

        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        semvec_phase = result.get("drift_phase", "stable")
        sc_val = bool(semvec_data.get("short_circuit", False))
        detected = _is_drift(result)

        # Persist enriched INVESTIGATED edges to healthcare entities
        if i < len(_entity_refs):
            for entity_label, entity_name in _entity_refs[i]:
                record_investigated(
                    driver, neo4j_sid, entity_label, entity_name,
                    step=i, phase=label.lower(),
                    drift_score=drift_score,
                    started_at=t_start, ended_at=t_end,
                    query=msg, response=response,
                    top_k_sim=top_k_sim,
                    semvec_drift_phase=semvec_phase,
                    short_circuit=sc_val,
                    extra={
                        "agent_role": role,
                        "semvec_top_similarity": float(sim),
                    },
                )

        if detected:
            drift_events_count += 1

        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(semvec_phase, DIM)
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET}" if detected else ""

        print(f"  {i:>2}  {sim_bar(sim)} sim={sim:.2f}  drift={drift_score:.2f}  "
              f"{phase_c}{semvec_phase:>8}{RESET}  {textwrap.shorten(msg, 40)}{drift_flag}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 68)}{RESET}")

    # ── Phase 4: Short-circuit ──
    n_phases123 = len(all_msgs)
    print(f"\n  {MAGENTA}{'─' * 60}")
    print(f"  {BOLD}Phase 4: Short-Circuit — paraphrased Phase 1 queries{RESET}")
    print(f"  {MAGENTA}{'─' * 60}{RESET}")
    print(f"\n  Semvec remembers the {len(on_topic_1)} diabetes Q&A pairs from Phase 1.")
    print(f"  Paraphrases with sim >= {sc_threshold} → {GREEN}HIT{RESET} (LLM skipped, cached context returned)\n")

    sc_hits = 0
    # Phase 4 paraphrases reference the same Phase-1 healthcare entities so
    # the audit trail keeps a complete picture of what was asked even when
    # the LLM was short-circuited.
    _phase4_entity_refs = _entity_refs[: len(short_circuit_queries)]
    for j, (q, ref_label) in enumerate(short_circuit_queries):
        step = n_phases123 + j
        t_start = datetime.now(timezone.utc)
        semvec_data = semvec.run(q, session_id=semvec_sid, response=prev_response,
                           short_circuit_threshold=sc_threshold)
        sim = semvec_data["top_similarity"]
        sc = semvec_data["short_circuit"]
        mcp.detect_drift(neo4j_sid, q)

        # Phase 4 forces the bar colour by HIT/MISS state rather than the
        # similarity threshold (the imported ``sim_bar`` colours by sim
        # band). Use a local variable name that doesn't shadow the helper.
        if sc:
            sc_hits += 1
            response = ""
            forced_bar = f"{GREEN}{'█' * int(sim * BAR_W)}{'░' * (BAR_W - int(sim * BAR_W))}{RESET}"
            print(f"  {step:>2}  {forced_bar} sim={sim:.2f}  {GREEN}HIT{RESET}  LLM skipped  {DIM}{ref_label}{RESET}")
            prev_response = None
        else:
            forced_bar = f"{RED}{'█' * int(sim * BAR_W)}{'░' * (BAR_W - int(sim * BAR_W))}{RESET}"
            response = generate_response(q, agent_role="a clinical pharmacist reviewing diabetes treatment",
                                         semvec_context=semvec_data.get("context", ""))
            mcp.store_response(neo4j_sid, response)
            prev_response = response
            print(f"  {step:>2}  {forced_bar} sim={sim:.2f}  {RED}MISS{RESET} LLM called   {DIM}{ref_label}{RESET}")
        t_end = datetime.now(timezone.utc)

        edge_phase = "phase-4-cache" if sc else "phase-4-miss"
        for entity_label, entity_name in _phase4_entity_refs[j]:
            record_investigated(
                driver, neo4j_sid, entity_label, entity_name,
                step=step, phase=edge_phase,
                drift_score=0.0,
                started_at=t_start, ended_at=t_end,
                query=q,
                response=response or None,
                top_k_sim=sim,
                short_circuit=sc,
                extra={"paraphrase_of": ref_label},
            )

    # ── Summary ──
    subheader("Summary")
    total_steps = n_phases123 + len(short_circuit_queries)
    n1, n2, n3 = len(on_topic_1), len(off_topic), len(on_topic_2)

    print(f"  Steps:  {total_steps} total  ({n1} diabetes + {n2} psychiatry + {n3} cardiology + {len(short_circuit_queries)} paraphrase)")
    print(f"  Drift:  {drift_events_count} events stored in Neo4j")
    print(f"  HITs:   {sc_hits}/{len(short_circuit_queries)} paraphrases recognized  (threshold: sim >= {sc_threshold})")
    print(f"  LLM:    {total_steps - sc_hits} calls made, {sc_hits} skipped\n")

    print(f"  {BOLD}All signals from Semvec:{RESET}")
    print(f"    drift_detected   Semvec flags a topic switch (drift_score >= threshold)")
    print(f"    drift_score      semantic drift intensity (0.0–1.0)")
    print(f"    top_similarity   cosine match between query and accumulated context")
    print(f"    short_circuit    paraphrase recognized, LLM skippable")

    phase_info = mcp.get_phase(neo4j_sid)
    print(f"\n  Final phase: {color_phase(phase_info.get('phase', 'N/A') or 'N/A')}")

    mcp.end_agent_session(neo4j_sid)


# ── Scenario 3: Multi-Specialist Ward Round (Semvec Layer 2 — Clusters) ─────

def scenario_ward_round(mcp: SemvecMCPServer, driver):
    header("Scenario 3: Multi-Specialist Ward Round (Semvec Layer 2 — Clusters)")
    print("  Three specialists examine David Park (Essential Hypertension + COPD).")
    print("  They share a Semvec cluster so findings propagate between agents.")
    print(f"  {GREEN}Dr. Chen{RESET} (internist) seeds baseline findings.")
    print(f"  {CYAN}Dr. Volkov{RESET} (pulmonologist) queries — gets HITs from Chen's work.")
    print(f"  {MAGENTA}Dr. Tanaka{RESET} (cardiologist) adds ECG finding — Volkov re-queries.\n")

    semvec = _build_semvec_client()

    BAR_W = 20

    # ── Step 1: Create cluster ──
    subheader("Step 1: Create Semvec cluster 'park-ward-round'")
    try:
        cluster = semvec.create_cluster(
            name="park-ward-round",
            aggregation_mode="weighted_average",
            coupling_factor=0.25,
        )
        cid = cluster["cluster_id"]
        info("Cluster ID", cid[:20] + "...")
        info("Coupling factor", "0.25")
    except Exception as e:
        print(f"  {RED}Cluster creation failed: {e}{RESET}")
        return

    # ── Step 2: Dr. Chen seeds baseline (threshold=0.99 → always MISS) ──
    subheader("Step 2: Dr. Chen seeds baseline — 4 Q&A pairs about David Park")
    info("Role", "Dr. Sarah Chen — internist performing baseline workup for David Park")
    print()

    chen_role = "an internist performing baseline workup for David Park"
    chen_queries = [
        "David Park presents with Essential Hypertension — current BP readings and medication?",
        "Park also has Chronic Obstructive Pulmonary Disease — what is his FEV1 and current inhaler regimen?",
        "Are there contraindications between Lisinopril for hypertension and his COPD medications?",
        "What is Park's encounter history — when was his last Annual Physical Exam?",
    ]
    chen_entity_refs = [
        [("Patient", "David Park"), ("Diagnosis", "Essential Hypertension"), ("Medication", "Lisinopril 10mg")],
        [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "David Park")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
        [("Patient", "David Park"), ("Encounter", "Annual Physical Exam")],
    ]

    # Create Neo4j AgentSessions for each doctor
    neo4j_chen = mcp.create_agent_session(agent_id="chen-baseline")
    neo4j_chen_sid = neo4j_chen["session_id"]

    # Threshold strategy for the ward round:
    #   0.99 = "impossible" — the cluster's cosine similarity to a paraphrased
    #          query is realistically capped well below 1.0, so 0.99 forces
    #          a deterministic MISS. Used during the seeding phase (Chen) and
    #          on each agent's *member* session where we want every turn to
    #          extend conversation context, never short-circuit.
    #   0.52 = below the expected cluster-to-paraphrase similarity for the
    #          shared Park topic, so paraphrases from Volkov and Tanaka can
    #          HIT against Chen's seeded entries. Tuned for MPNet-768 and
    #          the specific seed/paraphrase pairs in this demo.
    chen_sid = None
    for i, msg in enumerate(chen_queries):
        t0 = datetime.now(timezone.utc)
        cdata = semvec.cluster_run(cid, msg, short_circuit_threshold=0.99)
        response = generate_response(msg, agent_role=chen_role, semvec_context=cdata.get("context", ""))
        semvec.cluster_store(cid, msg, response)
        chen_data = semvec.run(msg, session_id=chen_sid, short_circuit_threshold=0.99)
        chen_sid = chen_data["session_id"]
        mcp.detect_drift(neo4j_chen_sid, msg)
        mcp.store_response(neo4j_chen_sid, response)
        t1 = datetime.now(timezone.utc)
        sim = cdata.get("top_similarity", 0.0)
        # Threshold deliberately at 0.99 so every Chen turn MISSes; this
        # is the seeding phase, the cluster cache must be primed first.
        print(f"  {DIM}seed MISS{RESET}  {sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score', 0.0):.3f}  {DIM}{textwrap.shorten(msg, 50)}{RESET}")
        if USE_LLM:
            print(f"        {DIM}A: {textwrap.shorten(response, 68)}{RESET}")
        for entity_label, entity_name in chen_entity_refs[i]:
            record_investigated(
                driver, neo4j_chen_sid, entity_label, entity_name,
                step=i, phase="chen-baseline",
                drift_score=cdata.get("drift_score", 0.0),
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=cdata.get("drift_phase"),
                short_circuit=bool(cdata.get("short_circuit", False)),
                extra={"agent_role": chen_role, "cluster_id": cid},
            )

    semvec.add_cluster_member(cid, chen_sid)
    print()
    info("Chen session", f"{chen_sid[:16]}... added to cluster")

    # ── Step 3: Dr. Volkov queries (threshold=0.52) ──
    subheader("Step 3: Dr. Volkov queries (threshold=0.52 — expect HITs from Chen)")
    info("Role", "Dr. Elena Volkov — pulmonologist reviewing David Park's respiratory status")
    print()

    volkov_role = "a pulmonologist reviewing David Park's respiratory status"
    volkov_queries = [
        "What is David Park's blood pressure status and antihypertensive medication?",
        "Does Park have any pulmonary comorbidities affecting his treatment plan?",
        "What interactions should we monitor between his cardiac and pulmonary medications?",
    ]
    volkov_entity_refs = [
        [("Patient", "David Park"), ("Diagnosis", "Essential Hypertension")],
        [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "David Park")],
        [("Medication", "Lisinopril 10mg"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease")],
    ]

    neo4j_volkov = mcp.create_agent_session(agent_id="volkov-pulm")
    neo4j_volkov_sid = neo4j_volkov["session_id"]
    volkov_sid = None
    volkov_hits = 0
    for i, msg in enumerate(volkov_queries):
        t0 = datetime.now(timezone.utc)
        cdata = semvec.cluster_run(cid, msg, short_circuit_threshold=0.52)
        volkov_data = semvec.run(msg, session_id=volkov_sid, short_circuit_threshold=0.99)
        volkov_sid = volkov_data["session_id"]
        mcp.detect_drift(neo4j_volkov_sid, msg)
        sim = cdata.get("top_similarity", 0.0)
        sc = bool(cdata.get("short_circuit", False))
        response = ""
        if sc:
            volkov_hits += 1
            print(f"  {GREEN}HIT{RESET}   {sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score',0.0):.3f}  sc={sc}  {DIM}{textwrap.shorten(msg, 46)}{RESET}")
        else:
            response = generate_response(msg, agent_role=volkov_role, semvec_context=cdata.get("context", ""))
            mcp.store_response(neo4j_volkov_sid, response)
            print(f"  {RED}MISS{RESET}  {sim_bar(sim)} sim={sim:.3f}  drift={cdata.get('drift_score',0.0):.3f}  sc={sc}  {DIM}{textwrap.shorten(msg, 46)}{RESET}")
            if USE_LLM:
                print(f"        {DIM}A: {textwrap.shorten(response, 68)}{RESET}")
        t1 = datetime.now(timezone.utc)
        for entity_label, entity_name in volkov_entity_refs[i]:
            record_investigated(
                driver, neo4j_volkov_sid, entity_label, entity_name,
                step=i, phase="volkov-queries",
                drift_score=cdata.get("drift_score", 0.0),
                started_at=t0, ended_at=t1,
                query=msg, response=response or None,
                top_k_sim=sim,
                semvec_drift_phase=cdata.get("drift_phase"),
                short_circuit=sc,
                extra={"agent_role": volkov_role, "cluster_id": cid},
            )

    # ── Step 4: Dr. Tanaka — novel ECG finding ──
    subheader("Step 4: Dr. Tanaka — queries + stores new ECG finding")
    info("Role", "Dr. Yuki Tanaka — cardiologist evaluating David Park's cardiac function")
    print()

    neo4j_tanaka = mcp.create_agent_session(agent_id="tanaka-cardio")
    neo4j_tanaka_sid = neo4j_tanaka["session_id"]
    tanaka_role = "a cardiologist evaluating David Park's cardiac function"
    tanaka_q1 = "What is the hypertension management plan for David Park?"
    cdata_t1 = semvec.cluster_run(cid, tanaka_q1, short_circuit_threshold=0.52)
    sim_t1 = cdata_t1.get("top_similarity", 0.0)
    sc_t1 = cdata_t1.get("short_circuit", False)
    # Same explicit-session pattern as Chen/Volkov: start with None so Semvec
    # mints a fresh session, then thread the returned id through any follow-up
    # turns. Tanaka has only one semvec.run call today, but the pattern keeps
    # the three doctors symmetric and future-proof.
    tanaka_sid = None
    tanaka_data = semvec.run(tanaka_q1, session_id=tanaka_sid, short_circuit_threshold=0.99)
    tanaka_sid = tanaka_data["session_id"]
    mcp.detect_drift(neo4j_tanaka_sid, tanaka_q1)

    if sc_t1:
        print(f"  {GREEN}HIT{RESET}   {sim_bar(sim_t1)} sim={sim_t1:.3f}  drift={cdata_t1.get('drift_score',0.0):.3f}  (≈ Chen Q1)")
    else:
        resp_t1 = generate_response(tanaka_q1, agent_role=tanaka_role)
        mcp.store_response(neo4j_tanaka_sid, resp_t1)
        print(f"  {RED}MISS{RESET}  {sim_bar(sim_t1)} sim={sim_t1:.3f}  drift={cdata_t1.get('drift_score',0.0):.3f}  (≈ Chen Q1)")

    # Tanaka stores novel ECG finding
    ecg_msg = "Park's latest ECG shows left ventricular hypertrophy — should we adjust treatment?"
    t0 = datetime.now(timezone.utc)
    ecg_resp = generate_response(ecg_msg, agent_role=tanaka_role)
    semvec.cluster_store(cid, ecg_msg, "LVH confirmed on ECG. Consider adding Amlodipine 5mg. Echo scheduled.")
    mcp.detect_drift(neo4j_tanaka_sid, ecg_msg)
    mcp.store_response(neo4j_tanaka_sid, ecg_resp)
    t1 = datetime.now(timezone.utc)
    print()
    info("Tanaka ECG finding stored", "LVH + Amlodipine recommendation")
    record_investigated(
        driver, neo4j_tanaka_sid, "Patient", "David Park",
        step=0, phase="tanaka-ecg",
        drift_score=cdata_t1.get("drift_score", 0.0),
        started_at=t0, ended_at=t1,
        query=ecg_msg, response=ecg_resp,
        extra={"agent_role": tanaka_role, "cluster_id": cid,
               "ecg_finding": "LVH"},
    )

    semvec.add_cluster_member(cid, tanaka_sid)

    # ── Step 5: Dr. Volkov re-queries after Tanaka's contribution ──
    subheader("Step 5: Dr. Volkov re-queries — should HIT Tanaka's ECG finding")
    print()
    ecg_query = "Any new cardiac findings for David Park?"
    cdata_v2 = semvec.cluster_run(cid, ecg_query, short_circuit_threshold=0.52)
    sim_v2 = cdata_v2.get("top_similarity", 0.0)
    sc_v2 = cdata_v2.get("short_circuit", False)
    drift_v2 = cdata_v2.get("drift_score", 0.0)
    phase_v2 = cdata_v2.get("drift_phase", "stable")
    if sc_v2:
        print(f"  {GREEN}HIT{RESET}   {sim_bar(sim_v2)} sim={sim_v2:.3f}  drift={drift_v2:.3f}  phase={phase_v2}  (Tanaka's ECG finding)")
    else:
        print(f"  {RED}MISS{RESET}  {sim_bar(sim_v2)} sim={sim_v2:.3f}  drift={drift_v2:.3f}  phase={phase_v2}  (searching for ECG finding)")

    # ── Step 6: Apply G4 coupling feedback ──
    subheader("Step 6: Apply G4 coupling feedback")
    try:
        feedback = semvec.cluster_feedback(cid)
        sessions_updated = feedback.get("sessions_updated", 0)
        info("Sessions coupled", str(sessions_updated))
        info("Coupling factor", "α=0.25 (cluster vector blended into member sessions)")
    except Exception as e:
        print(f"  {YELLOW}Feedback: {e}{RESET}")

    # ── Step 7: GET cluster state ──
    subheader("Step 7: Cluster state summary")
    try:
        cs = semvec.get_cluster(cid)
        info("Member count", str(cs.get("member_count", cs.get("members", "N/A"))))
        info("Interaction count", str(cs.get("interaction_count", "N/A")))
        info("Aggregate vector dims", str(cs.get("aggregate_vector_dims", cs.get("dimension", "N/A"))))
    except Exception as e:
        print(f"  {DIM}Cluster state: {e}{RESET}")

    # ── Persist cluster topology to Neo4j ──
    subheader("Neo4j — Cluster topology")
    with driver.session(database=NEO4J_DATABASE) as db:
        # Create Cluster node
        db.run("""
            MERGE (c:Cluster {cluster_id: $cid})
            SET c.name = 'park-ward-round', c.coupling_factor = 0.25,
                c.scenario = 'ward-round'
        """, cid=cid).consume()
        # MEMBER_OF relationships
        for sid, role in [(neo4j_chen_sid, "internist"), (neo4j_volkov_sid, "pulmonologist"), (neo4j_tanaka_sid, "cardiologist")]:
            if sid:
                db.run("""
                    MATCH (s:AgentSession {session_id: $sid})
                    MATCH (c:Cluster {cluster_id: $cid})
                    MERGE (s)-[:MEMBER_OF {role: $role}]->(c)
                """, sid=sid, cid=cid, role=role).consume()

        # Show what we created
        result = list(db.run("""
            MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster {cluster_id: $cid})
            RETURN s.agent_id AS agent, m.role AS role
            ORDER BY s.agent_id
        """, cid=cid))
        for r in result:
            print(f"  {GREEN}{r['agent']}{RESET} ({r['role']}) → Cluster park-ward-round")

        result2 = list(db.run("""
            MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
            WHERE s.session_id IN [$chen_sid, $volkov_sid, $tanaka_sid]
            RETURN s.agent_id AS agent, labels(e)[0] AS type, e.name AS entity,
                   inv.step AS step
            ORDER BY s.agent_id, inv.step
        """, chen_sid=neo4j_chen_sid, volkov_sid=neo4j_volkov_sid,
             tanaka_sid=neo4j_tanaka_sid))
        if result2:
            print()
            for r in result2:
                print(f"  {DIM}{r['agent']} step {r['step']} → {r['type']}:{r['entity']}{RESET}")

    # ── Summary ──
    subheader("Ward Round Summary")
    info("Patient", "David Park — Essential Hypertension + COPD")
    info("Cluster", f"{cid[:20]}...")
    info("Dr. Chen HITs", "0/4 (seeding phase — always MISS)")
    info("Dr. Volkov HITs", f"{volkov_hits}/3 (from Chen's baseline)")
    info("Dr. Tanaka", "1 novel ECG finding stored")
    info("Volkov post-Tanaka", f"{'HIT' if sc_v2 else 'MISS'} (sim={sim_v2:.3f})")
    info("Neo4j", "INVESTIGATED + MEMBER_OF relationships created")

    # Cleanup Semvec cluster
    try:
        semvec.delete_cluster(cid)
        info("Cleanup", "Semvec cluster deleted")
    except Exception:
        pass

    # End Neo4j sessions — must pass the Neo4j AgentSession ids, not
    # the raw Semvec session ids that semvec.run() returned.
    for nsid in [neo4j_chen_sid, neo4j_volkov_sid, neo4j_tanaka_sid]:
        if nsid:
            try:
                mcp.end_agent_session(nsid)
            except Exception:
                pass


# ── Scenario 4: Medication Safety Guard (Semvec Layer 1b) ───────────────────

def scenario_medication_safety(mcp: SemvecMCPServer, driver):
    header("Scenario 4: Medication Safety Guard (Semvec Layer 1b — Anchors, Triggers, Isolation, Memory)")
    print("  An agent monitors Carlos Gutierrez starting Chemotherapy Cycle 1.")
    print("  Safety guardrails are set up using Semvec Layer 1b features:")
    print(f"    {GREEN}Drift Anchor{RESET}  locks session to oncology domain")
    print(f"    {YELLOW}Resonance Trigger{RESET}  'CRITICAL' / 'contraindication' keywords → high importance")
    print(f"    {CYAN}Synthetic Memory{RESET}  injects known drug interactions from Neo4j graph")
    print(f"    {RED}Input Isolation{RESET}   quarantines off-topic inputs\n")

    semvec = _build_semvec_client()
    BAR_W = 20

    # ── Step 1: Create session ──
    subheader("Step 1: Create Semvec session with enable_topic_switch")
    sess = semvec.create_session(enable_topic_switch=True)
    sid = sess["session_id"]
    info("Session ID", sid[:20] + "...")
    info("Layer", "1b — session control enabled")

    # Also mirror to Neo4j
    neo4j_sess = mcp.create_agent_session(agent_id="oncology-safety-guard")
    neo4j_sid = neo4j_sess["session_id"]

    # ── Step 2: Add drift anchor (oncology domain) ──
    subheader("Step 2: Set drift anchor — oncology domain")
    anchor_text = "oncology chemotherapy cancer treatment cytotoxic drugs Carlos Gutierrez"
    print(f"  {DIM}Computing oncology domain embedding...{RESET}")
    oncology_embedding = embed(anchor_text)
    try:
        result = semvec.add_anchor(sid, oncology_embedding)
        info("Anchor set", "oncology domain (384-dim embedding)")
    except Exception as e:
        print(f"  {YELLOW}anchor endpoint: {e}{RESET}")

    # ── Step 3: Add resonance triggers ──
    # Single source of truth: the same list drives the registration
    # call AND the per-turn display flag.
    TRIGGER_KEYWORDS = ("CRITICAL", "contraindication")
    subheader("Step 3: Add resonance triggers")
    for keyword in TRIGGER_KEYWORDS:
        try:
            semvec.add_trigger(sid, keyword)
            info(f"Trigger '{keyword}'", "added — will flag high-importance turns")
        except Exception as e:
            print(f"  {YELLOW}trigger '{keyword}': {e}{RESET}")

    # ── Step 4: Inject synthetic memories from Neo4j contraindication graph ──
    subheader("Step 4: Inject synthetic memories (contraindications from Neo4j)")
    contraindication_memories = [
        ("Amoxicillin 250mg is contraindicated with Metformin 500mg",
         [("Medication", "Amoxicillin 250mg"), ("Medication", "Metformin 500mg")]),
        ("Atorvastatin 40mg is contraindicated with Sertraline 50mg",
         [("Medication", "Atorvastatin 40mg"), ("Medication", "Sertraline 50mg")]),
        ("Carlos Gutierrez Chemotherapy Cycle 1 uses Atorvastatin 40mg and Metformin 500mg",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1"),
          ("Medication", "Atorvastatin 40mg"), ("Medication", "Metformin 500mg")]),
    ]
    for mem_text, entity_refs in contraindication_memories:
        embedding = embed(mem_text)
        try:
            semvec.inject_memory(sid, embedding=embedding, text=mem_text,
                              tier="short_term", importance=1.0)
            info("Memory injected", textwrap.shorten(mem_text, 60))
        except Exception as e:
            print(f"  {YELLOW}memory inject: {e}{RESET}")
        # Persist to Neo4j: INVESTIGATED from safety session → referenced entities
        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_sid, entity_label, entity_name,
                step=0, phase="memory-injection",
                drift_score=0.0,
                query=mem_text,
                extra={"memory_text": mem_text, "tier": "short_term"},
            )

    # ── Step 5: Set isolation to QUARANTINE ──
    # Single source of truth — display, flag and summary all read from
    # this constant so they cannot drift apart.
    QUARANTINE_THRESHOLD = 0.5
    subheader("Step 5: Set input isolation to QUARANTINE")
    try:
        semvec.set_isolation(sid, level="QUARANTINE",
                              similarity_threshold=QUARANTINE_THRESHOLD)
        info("Isolation level",
             f"{RED}QUARANTINE{RESET} (similarity_threshold={QUARANTINE_THRESHOLD:.2f})")
        info("Effect", "Off-topic inputs will be filtered")
    except Exception as e:
        print(f"  {YELLOW}isolation endpoint: {e}{RESET}")

    # ── Step 6: Run oncology queries ──
    subheader("Step 6: Oncology queries — should process normally")
    info("Agent role", "an oncology pharmacist monitoring Carlos Gutierrez's chemotherapy safety")
    print()

    agent_role = "an oncology pharmacist monitoring Carlos Gutierrez's chemotherapy safety"
    # Some queries deliberately carry verbatim facts the upstream
    # extractor recognises (ISO/DE dates, EUR currency, validated
    # IBAN, DE-VAT) so the "Verbatim facts cached" counter goes up
    # in a healthcare-realistic way: schedule, billing, identifiers.
    oncology_queries = [
        ("Carlos Gutierrez is starting Chemotherapy Cycle 1 on 2026-05-15 — what pre-treatment labs are required?",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("Repeat infusion every 21 days; next dates 05.06.2026 and 26.06.2026. What antiemetic protocol for his cisplatin-based regimen?",
         [("Treatment", "Chemotherapy Cycle 1")]),
        ("CRITICAL: Monitor for neutropenic fever — what ANC thresholds for dose delay?",
         [("Patient", "Carlos Gutierrez"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("Does Chemotherapy Cycle 1 use Atorvastatin 40mg — any interactions?",
         [("Medication", "Atorvastatin 40mg"), ("Treatment", "Chemotherapy Cycle 1")]),
        ("Estimated treatment cost 4.500,00 EUR per cycle, billed to IBAN DE89 3704 0044 0532 0130 00. What is the contraindication profile between chemo drugs and his existing medications?",
         [("Medication", "Metformin 500mg"), ("Medication", "Atorvastatin 40mg")]),
    ]

    # 0.75 is the operational sweet spot for in-domain Q&A on MPNet-768:
    # high enough that paraphrases of *the same question* HIT, low enough
    # that genuinely new on-topic queries still drive the LLM. The off-topic
    # branch below reuses the same threshold so we can compare apples-to-
    # apples — only the QUARANTINE_THRESHOLD (0.5) below decides isolation.
    prev_resp = None
    facts_total = 0
    for i, (msg, entity_refs) in enumerate(oncology_queries):
        t0 = datetime.now(timezone.utc)
        result = semvec.run(msg, session_id=sid, response=prev_resp,
                         short_circuit_threshold=0.75)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        sc = bool(result.get("short_circuit", False))

        # Compliance: pull verbatim facts (dates, currency amounts,
        # validated IBANs, etc.) into the literal cache before the LLM
        # runs. The fact extractor is regex-based and bypasses
        # embedding compression so safety-critical values survive.
        facts_this_turn = 0
        try:
            facts_extracted = semvec.store_facts_as_entities(sid, msg)
            facts_this_turn = facts_extracted.get("stored", 0)
            facts_total += facts_this_turn
        except Exception:
            pass

        # Mirror to Neo4j
        mcp.detect_drift(neo4j_sid, msg)
        response = generate_response(msg, agent_role=agent_role,
                                     semvec_context=result.get("context", ""))
        mcp.store_response(neo4j_sid, response)
        prev_resp = response
        t1 = datetime.now(timezone.utc)

        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(drift_phase, DIM)

        msg_lower = msg.lower()
        triggered = next((k for k in TRIGGER_KEYWORDS if k.lower() in msg_lower), None)
        crit_flag = f"  {RED}{BOLD}TRIGGER {triggered}{RESET}" if triggered else ""
        facts_flag = f"  {CYAN}+{facts_this_turn} facts{RESET}" if facts_this_turn else ""
        print(f"  {i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{crit_flag}{facts_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()

        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_sid, entity_label, entity_name,
                step=i, phase="oncology",
                drift_score=drift_score,
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=drift_phase,
                short_circuit=sc,
                extra={
                    "agent_role": agent_role,
                    "trigger_keyword": triggered,
                },
            )

    # ── Step 7: Off-topic queries → should be quarantined ──
    subheader("Step 7: Off-topic queries — should be quarantined/filtered")
    print()
    off_topic = [
        "When is the next Pharmacy and Therapeutics Committee meeting?",
        "What is the bed count at Cedar Grove Clinic?",
    ]
    for i, msg in enumerate(off_topic):
        result = semvec.run(msg, session_id=sid, response=prev_resp,
                         short_circuit_threshold=0.75)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        phase_c = {
            "stable": GREEN, "shifting": YELLOW, "drifted": RED,
        }.get(drift_phase, DIM)
        quarantine_flag = f"  {RED}OFF-TOPIC{RESET}" if sim < QUARANTINE_THRESHOLD else ""
        print(f"  {5+i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{quarantine_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        print()

    # ── Step 8: Get anchor score ──
    subheader("Step 8: Anchor score — how far has session drifted from oncology?")
    try:
        anchor = semvec.get_anchor_score(sid)
        score = anchor.get("anchor_score", 0.0)
        threshold = anchor.get("drift_threshold", 0.5)
        if score >= threshold:
            colour, verdict = RED, "DRIFTED"
        elif score >= threshold / 2:
            colour, verdict = YELLOW, "shifting"
        else:
            colour, verdict = GREEN, "on-anchor"
        info(
            "Anchor score",
            f"{colour}{score:.3f}{RESET} {verdict}  "
            f"(threshold {threshold:.2f}, anchors {anchor.get('anchor_count', 0)}, "
            f"realignment_remaining {anchor.get('realignment_remaining', 0)})",
        )
    except Exception as e:
        print(f"  {YELLOW}anchor_score: {e}{RESET}")

    # ── Summary ──
    subheader("Safety Guard Summary")
    info("Patient", "Carlos Gutierrez — Chemotherapy Cycle 1")
    info("Session", sid[:20] + "...")
    info("Contraindications injected", "3 synthetic memories")
    info("Oncology queries", f"{len(oncology_queries)} processed")
    info("Off-topic queries",
         f"{len(off_topic)} (quarantined if sim < {QUARANTINE_THRESHOLD:.2f})")
    info("Verbatim facts cached", f"{facts_total} (dates / amounts / identifiers)")
    info("Neo4j", "INVESTIGATED relationships to Gutierrez + Chemo + Medications")

    mcp.end_agent_session(neo4j_sid)


# ── Scenario 5: Hospital Network Consensus (Semvec Layers 3+4) ──────────────

def scenario_hospital_consensus(mcp: SemvecMCPServer, driver):
    header("Scenario 5: Hospital Network Consensus (Semvec Layers 3+4 — Regions, Observer)")
    print("  Two department clusters across two facilities:")
    print(f"    {GREEN}Cluster A{RESET}  'cardiology-memorial' at Memorial General  → Maria Rodriguez (AMI)")
    print(f"    {CYAN}Cluster B{RESET}  'emergency-riverside' at Riverside Medical  → James Morrison (ER)")
    print(f"  Both in Region 'hospital-network'. When cardiology drifts,")
    print(f"  the Observer checks for cross-cluster anomalies.\n")

    semvec = _build_semvec_client()
    BAR_W = 20

    # Clean observer state from previous scenario runs (in-process, may carry over)
    try:
        cleared = semvec.clear_anomalies()
        if cleared.get("cleared", 0) > 0:
            print(f"  {DIM}Observer: cleared {cleared['cleared']} old anomalies{RESET}")
    except Exception:
        pass  # observer may not exist yet

    cid_a = cid_b = rid = None

    # ── Step 1: Create two clusters ──
    subheader("Step 1: Create cardiology + emergency clusters")
    try:
        ca = semvec.create_cluster(name="cardiology-memorial", aggregation_mode="weighted_average",
                                coupling_factor=0.2)
        cid_a = ca["cluster_id"]
        info("Cluster A (cardiology)", cid_a[:20] + "...")
    except Exception as e:
        print(f"  {RED}cardiology cluster failed: {e}{RESET}")
        return

    try:
        cb = semvec.create_cluster(name="emergency-riverside", aggregation_mode="weighted_average",
                                coupling_factor=0.2)
        cid_b = cb["cluster_id"]
        info("Cluster B (emergency)", cid_b[:20] + "...")
    except Exception as e:
        print(f"  {RED}emergency cluster failed: {e}{RESET}")
        semvec.delete_cluster(cid_a)
        return

    # ── Step 2: Seed clusters ──
    subheader("Step 2: Seed clusters with domain Q&A pairs")

    cardiology_seed = [
        ("Maria Rodriguez Acute Myocardial Infarction — current troponin levels?",
         "Troponin I peaked at 4.2 ng/mL at 6h post-admission.",
         [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")]),
        ("Post-MI antiplatelet therapy — aspirin plus clopidogrel or ticagrelor?",
         "Dual antiplatelet: Aspirin 81mg + Ticagrelor 90mg BID per AHA 2023.",
         [("Diagnosis", "Acute Myocardial Infarction")]),
        ("Dr. O'Brien referred Rodriguez to Dr. Okonkwo — for cardiac rehab?",
         "Yes — referral to Dr. Okonkwo for outpatient cardiac rehabilitation.",
         [("Provider", "Dr. Michael O'Brien"), ("Provider", "Dr. Rachel Okonkwo")]),
        ("Rodriguez ER encounter — which interventions were performed?",
         "Emergency PCI with drug-eluting stent to LAD. Aspirin + heparin loading.",
         [("Encounter", "Emergency Room Visit"), ("Patient", "Maria Rodriguez")]),
    ]
    emergency_seed = [
        ("James Morrison Emergency Room Visit — chief complaint and triage priority?",
         "Chief complaint: chest pain 7/10. Triage ESI-2 (emergent).",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison has Type 2 Diabetes and COPD — medication reconciliation needed?",
         "Hold Metformin pre-procedure, review inhalers for COPD.",
         [("Diagnosis", "Type 2 Diabetes Mellitus"), ("Diagnosis", "Chronic Obstructive Pulmonary Disease"),
          ("Medication", "Metformin 500mg")]),
        ("Dr. Volkov attending Morrison's ER encounter — any cardiac biomarker results?",
         "Troponin negative x2, BNP 210 pg/mL — non-ischemic.",
         [("Provider", "Dr. Elena Volkov"), ("Patient", "James Morrison")]),
        ("Morrison's chest pain workup — should we consult cardiology?",
         "Yes — cardiology consultation requested given new LBBB on ECG.",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
    ]

    # Seed cluster sessions AND build member session context (with inline-store)
    info("Cardiology", f"seeding {len(cardiology_seed)} Q&A pairs + member context")
    cardio_sess_id = None
    cardio_prev = None
    for msg, resp, _ in cardiology_seed:
        semvec.cluster_store(cid_a, msg, resp)
        r = semvec.run(msg, session_id=cardio_sess_id, response=cardio_prev)
        cardio_sess_id = r["session_id"]
        cardio_prev = resp
        semvec.store(cardio_sess_id, resp)

    info("Emergency", f"seeding {len(emergency_seed)} Q&A pairs + member context")
    emerg_sess_id = None
    emerg_prev = None
    for msg, resp, _ in emergency_seed:
        semvec.cluster_store(cid_b, msg, resp)
        r = semvec.run(msg, session_id=emerg_sess_id, response=emerg_prev)
        emerg_sess_id = r["session_id"]
        emerg_prev = resp
        semvec.store(emerg_sess_id, resp)

    # Register sessions as cluster members
    if cardio_sess_id:
        semvec.add_cluster_member(cid_a, cardio_sess_id)
    if emerg_sess_id:
        semvec.add_cluster_member(cid_b, emerg_sess_id)

    # ── Step 3: Create Region + add clusters ──
    # IMPORTANT: Region + Observer must exist BEFORE drift events fire,
    # otherwise the region won't receive them.
    #   consensus_threshold=0.5  — half the clusters in the region must
    #     fire a drift event in the vote window for a Region-level
    #     consensus alert to trigger. Two clusters → both must drift.
    #   vote_window_seconds=60.0 — drift events older than this are
    #     evicted from the consensus window. 60s comfortably covers a
    #     full demo run; in production this is typically minutes.
    subheader("Step 3: Create Region 'hospital-network' (consensus_threshold=0.5)")
    try:
        region = semvec.create_region(name="hospital-network", consensus_threshold=0.5,
                                   vote_window_seconds=60.0)
        rid = region["region_id"]
        info("Region ID", rid[:20] + "...")
        semvec.add_region_cluster(rid, cid_a)
        info("Added", "cardiology-memorial to region")
        semvec.add_region_cluster(rid, cid_b)
        info("Added", "emergency-riverside to region")
    except Exception as e:
        print(f"  {YELLOW}Region: {e}{RESET}")

    # ── Step 4: Create Observer ──
    # sample_interval_seconds=30.0 is the *automatic* sampling cadence for
    # a long-running observer. The demo calls observer_sample() manually
    # at strategic moments (Steps 5b/5d), so this value only matters if
    # you let the demo idle — which we don't. Set well above demo runtime.
    subheader("Step 4: Create Global Observer")
    try:
        observer = semvec.create_observer(
            sample_interval_seconds=30.0,
            region_ids=[rid] if rid else [],
        )
        info("Observer", "created and registered to hospital-network region")
    except Exception as e:
        print(f"  {YELLOW}Observer: {e}{RESET}")

    # ── Step 5: Run cluster queries ──
    subheader("Step 5: Cardiology cluster — on-topic queries + drift pivot")
    info("Role", "a cardiologist at Memorial General Hospital treating Maria Rodriguez")
    print()

    cardio_role = "a cardiologist at Memorial General Hospital treating Maria Rodriguez"
    neo4j_cardio = mcp.create_agent_session(agent_id="cardio-memorial")
    neo4j_cardio_sid = neo4j_cardio["session_id"]

    cardio_run_msgs = [
        ("Maria Rodriguez post-MI — troponin trend over 24h?",
         [("Patient", "Maria Rodriguez"), ("Diagnosis", "Acute Myocardial Infarction")]),
        ("Post-MI antiplatelet: is Ticagrelor preferred over Clopidogrel for Rodriguez?",
         [("Diagnosis", "Acute Myocardial Infarction")]),
        ("Dr. Okonkwo's cardiac rehab plan for Rodriguez — when to start?",
         [("Provider", "Dr. Rachel Okonkwo"), ("Patient", "Maria Rodriguez")]),
        ("Rodriguez ER intervention — stent type and anti-thrombotic protocol?",
         [("Encounter", "Emergency Room Visit"), ("Patient", "Maria Rodriguez")]),
        # PIVOT — drift-inducing
        ("What is the latest infection control audit status at Memorial General?",
         [("Facility", "Memorial General Hospital")]),
    ]
    # 0.60 is calibrated for the consensus scenario: cluster-to-paraphrase
    # similarity inside one well-seeded department typically lands in
    # 0.55-0.70 with MPNet-768, so 0.60 yields a HIT for the four on-topic
    # turns and a clear MISS on the deliberate pivot at step 5.
    cardio_prev = None
    for i, (msg, entity_refs) in enumerate(cardio_run_msgs):
        t0 = datetime.now(timezone.utc)
        cdata = semvec.cluster_run(cid_a, msg, short_circuit_threshold=0.60)
        # Also run on member session so drift propagates to Region
        member_r = semvec.run(msg, session_id=cardio_sess_id, response=cardio_prev)
        mcp.detect_drift(neo4j_cardio_sid, msg)
        response = generate_response(msg, agent_role=cardio_role,
                                     semvec_context=cdata.get("context", ""))
        cardio_prev = response
        semvec.store(cardio_sess_id, response)
        mcp.store_response(neo4j_cardio_sid, response)
        t1 = datetime.now(timezone.utc)
        sim = cdata.get("top_similarity", 0.0)
        sc = bool(cdata.get("short_circuit", False))
        drift_score = cdata.get("drift_score", 0.0)
        drift_phase = cdata.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        member_drift_score = member_r.get("drift_score", 0.0)
        member_drift = _is_drift(member_r)
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET} (member: {member_drift_score:.2f})" if member_drift else ""
        hit_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        print(f"  {i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{hit_flag}{drift_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_cardio_sid, entity_label, entity_name,
                step=i, phase="cardio",
                drift_score=drift_score,
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=drift_phase,
                short_circuit=sc,
                extra={"agent_role": cardio_role, "cluster_id": cid_a,
                       "member_drift_score": float(member_drift_score)},
            )

    # ── Observer sample #1: after cardiology drift ──
    subheader("Step 5b: Observer sample after cardiology drift")
    try:
        sample1 = semvec.observer_sample()
        _print_observer_sample("Sample #1", sample1)
    except Exception as e:
        print(f"  {YELLOW}observer sample: {e}{RESET}")

    try:
        events1 = semvec.get_region_events(rid, limit=10) if rid else []
        info("Region events", f"{len(events1)} after cardiology drift")
        for ev in events1:
            print(f"    drift={ev.get('drift_score', 0):.2f}  phase={ev.get('drift_phase', '?')}  "
                  f"cluster={str(ev.get('cluster_id', ''))[:16]}...")
    except Exception as e:
        print(f"  {YELLOW}region events: {e}{RESET}")

    # ── Step 5c: Emergency cluster — on-topic + drift pivot ──
    subheader("Step 5c: Emergency cluster — on-topic + drift pivot")
    info("Role", "an emergency physician at Riverside Medical Center treating James Morrison")
    print()

    emerg_role = "an emergency physician at Riverside Medical Center treating James Morrison"
    neo4j_emerg = mcp.create_agent_session(agent_id="emerg-riverside")
    neo4j_emerg_sid = neo4j_emerg["session_id"]

    emerg_run_msgs = [
        ("James Morrison ER chief complaint — triage classification?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison diabetes medication reconciliation — Metformin hold indication?",
         [("Diagnosis", "Type 2 Diabetes Mellitus"), ("Medication", "Metformin 500mg")]),
        ("Dr. Volkov managing Morrison — cardiac biomarkers at 2h and 6h?",
         [("Provider", "Dr. Elena Volkov"), ("Patient", "James Morrison")]),
        ("Should Morrison have cardiology consult given LBBB?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        # PIVOT — same domain shift as cardiology (admin topic)
        ("What is the current bed availability at Riverside Medical Center?",
         [("Facility", "Riverside Medical Center")]),
    ]
    emerg_prev = None
    for i, (msg, entity_refs) in enumerate(emerg_run_msgs):
        t0 = datetime.now(timezone.utc)
        cdata = semvec.cluster_run(cid_b, msg, short_circuit_threshold=0.60)
        # Also run on member session so drift propagates to Region
        member_r = semvec.run(msg, session_id=emerg_sess_id, response=emerg_prev)
        mcp.detect_drift(neo4j_emerg_sid, msg)
        response = generate_response(msg, agent_role=emerg_role,
                                     semvec_context=cdata.get("context", ""))
        emerg_prev = response
        semvec.store(emerg_sess_id, response)
        mcp.store_response(neo4j_emerg_sid, response)
        t1 = datetime.now(timezone.utc)
        sim = cdata.get("top_similarity", 0.0)
        sc = bool(cdata.get("short_circuit", False))
        drift_score = cdata.get("drift_score", 0.0)
        drift_phase = cdata.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        member_drift_score = member_r.get("drift_score", 0.0)
        member_drift = _is_drift(member_r)
        drift_flag = f"  {RED}{BOLD}DRIFT{RESET} (member: {member_drift_score:.2f})" if member_drift else ""
        hit_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        print(f"  {i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{hit_flag}{drift_flag}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_emerg_sid, entity_label, entity_name,
                step=i, phase="emergency",
                drift_score=drift_score,
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=drift_phase,
                short_circuit=sc,
                extra={"agent_role": emerg_role, "cluster_id": cid_b,
                       "member_drift_score": float(member_drift_score)},
            )

    # ── Observer sample #2: after BOTH clusters drifted ──
    subheader("Step 5d: Observer sample after both clusters drifted")
    try:
        sample2 = semvec.observer_sample()
        _print_observer_sample("Sample #2", sample2)
    except Exception as e:
        print(f"  {YELLOW}observer sample: {e}{RESET}")

    # ── Step 6: Region events + Observer results ──
    subheader("Step 6: Region Consensus + Observer Anomaly Detection (Layer 3+4)")

    # Region events
    region_event_count = 0
    if rid:
        try:
            events = semvec.get_region_events(rid, limit=20)
            region_event_count = len(events)
            info("Region events", f"{BOLD}{len(events)}{RESET} drift events in hospital-network")
            for ev in events:
                drift = ev.get("drift_score", 0)
                phase = ev.get("drift_phase", "?")
                cid_ev = str(ev.get("cluster_id", ""))[:16]
                ts = ev.get("timestamp", "")
                color = RED if phase == "drifted" else YELLOW
                print(f"    {color}drift={drift:.2f}  phase={phase:>8}  cluster={cid_ev}...{RESET}")
            if not events:
                print(f"    {DIM}(Region consensus requires drift in multiple clusters within vote window){RESET}")
        except Exception as e:
            print(f"  {YELLOW}Region events: {e}{RESET}")

    # Observer anomalies
    print()
    anomaly_count = 0
    try:
        anomalies = semvec.get_anomalies(limit=20)
        anomaly_count = len(anomalies)
        info("Observer anomalies", f"{BOLD}{len(anomalies)}{RESET} detected")
        for a in anomalies:
            atype = a.get("anomaly_type", "unknown")
            sev = a.get("severity", 0)
            desc = a.get("description", "")
            affected = a.get("affected_cluster_ids", [])
            print(f"    {RED}{BOLD}{atype}{RESET}  severity={sev:.1f}")
            if desc:
                print(f"    {DIM}{desc}{RESET}")
            if affected:
                print(f"    affected clusters: {', '.join(str(c)[:16] + '...' for c in affected)}")
        if not anomalies:
            print(f"    {DIM}(Observer detects anomalies when multiple clusters drift simultaneously){RESET}")
    except Exception as e:
        print(f"  {YELLOW}Observer anomalies: {e}{RESET}")

    # Observer summary
    print()
    try:
        summary = semvec.get_observer_summary()
        info("Observer summary", "")
        info("  Registered regions", str(summary.get("registered_regions", 0)))
        info("  Clusters observed", str(summary.get("total_clusters_observed", "N/A")))
        info("  Last sample at", str(summary.get("last_sample_at", "N/A")))
        info("  Total anomalies", str(summary.get("anomaly_count", 0)))
    except Exception as e:
        print(f"  {YELLOW}Observer summary: {e}{RESET}")

    # ── Neo4j: persist cluster/region topology ──
    subheader("Neo4j — Region/Cluster topology")
    with driver.session(database=NEO4J_DATABASE) as db:
        # Region node
        if rid:
            db.run("""
                MERGE (r:Region {region_id: $rid})
                SET r.name = 'hospital-network', r.consensus_threshold = 0.5
            """, rid=rid).consume()
        # Cluster nodes + CONTAINS_CLUSTER. Use ``c_id`` (not ``cid``) for the
        # loop variable so we don't shadow ``cid_a``/``cid_b`` from the outer
        # scope — the same name was rebinding the outer references after the
        # loop completed.
        for c_id, cname in [(cid_a, "cardiology-memorial"), (cid_b, "emergency-riverside")]:
            db.run("""
                MERGE (c:Cluster {cluster_id: $cid})
                SET c.name = $name, c.scenario = 'consensus'
            """, cid=c_id, name=cname).consume()
            if rid:
                db.run("""
                    MATCH (r:Region {region_id: $rid})
                    MATCH (c:Cluster {cluster_id: $cid})
                    MERGE (r)-[:CONTAINS_CLUSTER]->(c)
                """, rid=rid, cid=c_id).consume()
        # MEMBER_OF
        for sess_id, c_id, fac_name in [
            (neo4j_cardio_sid, cid_a, "Memorial General Hospital"),
            (neo4j_emerg_sid, cid_b, "Riverside Medical Center"),
        ]:
            db.run("""
                MATCH (s:AgentSession {session_id: $sid})
                MATCH (c:Cluster {cluster_id: $cid})
                MERGE (s)-[:MEMBER_OF {facility: $fac}]->(c)
            """, sid=sess_id, cid=c_id, fac=fac_name).consume()

    # Audit-trail: link each agent session to its facility (outside the with-block
    # so record_investigated can open its own session safely).
    for sess_id, c_id, fac_name in [
        (neo4j_cardio_sid, cid_a, "Memorial General Hospital"),
        (neo4j_emerg_sid, cid_b, "Riverside Medical Center"),
    ]:
        record_investigated(
            driver, sess_id, "Facility", fac_name,
            step=0, phase="cluster-facility",
            drift_score=0.0,
            extra={"cluster_id": c_id},
        )

    with driver.session(database=NEO4J_DATABASE) as db:

        # Show topology
        result = list(db.run("""
            MATCH (r:Region)-[:CONTAINS_CLUSTER]->(c:Cluster)
            OPTIONAL MATCH (s:AgentSession)-[m:MEMBER_OF]->(c)
            RETURN r.name AS region, c.name AS cluster, s.agent_id AS agent, m.facility AS facility
            ORDER BY c.name, s.agent_id
        """))
        for row in result:
            print(f"  {GREEN}{row['region']}{RESET} → {CYAN}{row['cluster']}{RESET} "
                  f"← {row['agent'] or '?'} ({row['facility'] or '?'})")

        result2 = list(db.run("""
            MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
            WHERE s.agent_id IN ['cardio-memorial', 'emerg-riverside']
            RETURN s.agent_id AS agent, labels(e)[0] AS type, e.name AS entity, inv.step AS step
            ORDER BY s.agent_id, inv.step
        """))
        if result2:
            print()
            for r in result2:
                print(f"  {DIM}{r['agent']} step {r['step']} → {r['type']}:{r['entity']}{RESET}")

    # ── Step 7: Cortex ConsensusEngine — qualified majority vote ──
    subheader("Step 7: Cortex ConsensusEngine — qualified-majority vote on admin-pivot drift")
    try:
        eng = semvec.create_consensus_engine(
            local_id="hospital-orchestrator",
            network_id="hospital-network",
            level="qualified_majority",
        )
        eid = eng["engine_id"]
        for inst, weight in [
            ("cardio-memorial", 1.0),
            ("emerg-riverside", 1.0),
            ("hospital-supervisor", 1.5),
        ]:
            semvec.register_consensus_voter(eid, inst, weight=weight)

        proposal = semvec.submit_consensus_proposal(
            eid,
            proposal_type="admin_pivot_alert",
            proposed_state=[0.0] * 8,
            rationale=(
                "Both cardiology and emergency clusters drifted to "
                "facility-admin queries within the vote window — promote "
                "to a hospital-wide IT incident?"
            ),
        )
        pid = proposal["proposal_id"]
        info("Proposal", f"{pid[:24]}...  status={proposal['status']}")

        # Cardiology + supervisor say yes (admin signal is real),
        # emergency abstains via "no" (we want a non-trivial split).
        semvec.vote_on_consensus(eid, pid, True, voting_instance="cardio-memorial")
        semvec.vote_on_consensus(eid, pid, False, voting_instance="emerg-riverside")
        semvec.vote_on_consensus(eid, pid, True, voting_instance="hospital-supervisor")
        verdict = semvec.evaluate_consensus(eid, pid)
        colour = GREEN if verdict["accepted"] else YELLOW
        info(
            "Verdict",
            f"{colour}{'ACCEPTED' if verdict['accepted'] else 'REJECTED'}{RESET}  "
            f"ratio={verdict['ratio']:.2f}  "
            f"({verdict['votes_for']} for / {verdict['votes_against']} against)",
        )
    except Exception as e:
        print(f"  {YELLOW}consensus engine: {e}{RESET}")

    # ── Summary ──
    subheader("Hospital Network Consensus Summary")
    info("Region", "hospital-network (consensus_threshold=0.5)")
    info("Cluster A", f"cardiology-memorial — {len(cardiology_seed)} seeded, {len(cardio_run_msgs)} run")
    info("Cluster B", f"emergency-riverside — {len(emergency_seed)} seeded, {len(emerg_run_msgs)} run")
    info("Pivot query", "Both clusters drift on admin-topic pivot (step 5)")
    info("Cortex consensus", "qualified-majority vote on the pivot alert")
    info("Neo4j", "Region → CONTAINS_CLUSTER → Clusters → MEMBER_OF ← Sessions")

    # Cleanup Semvec — use ``c_id`` so we don't shadow ``cid_a``/``cid_b``.
    for c_id in [cid_a, cid_b]:
        if c_id is None:
            continue
        try:
            semvec.delete_cluster(c_id)
        except Exception:
            pass
    if rid:
        try:
            semvec.delete_region(rid)
        except Exception:
            pass

    mcp.end_agent_session(neo4j_cardio_sid)
    mcp.end_agent_session(neo4j_emerg_sid)


# ── Scenario 6: Shift Handoff (Semvec Layer 5 — Export/Import, Transfer) ───

def scenario_shift_handoff(mcp: SemvecMCPServer, driver):
    header("Scenario 6: Shift Handoff (Semvec Layer 5 — Export/Import, Network Transfer)")
    print("  Night shift Dr. Volkov finishes Morrison's overnight monitoring.")
    print("  She exports her Semvec session state.")
    print("  Day shift Dr. Tanaka imports it — continues with full context.")
    print(f"  Compare: Tanaka's sim WITH vs WITHOUT import.\n")

    semvec = _build_semvec_client()
    BAR_W = 20

    # ── Step 1: Volkov night shift — build deep context ──
    subheader("Step 1: Dr. Volkov night shift — 6 overnight queries about James Morrison")
    info("Role", "Dr. Elena Volkov, night shift internist monitoring James Morrison overnight")
    print()

    volkov_role = "Dr. Elena Volkov, night shift internist monitoring James Morrison overnight"
    night_queries = [
        ("James Morrison overnight vitals — any concerning trends in blood glucose?",
         [("Patient", "James Morrison"), ("Diagnosis", "Type 2 Diabetes Mellitus")]),
        ("Morrison's Metformin was held for catheterization — when to resume?",
         [("Medication", "Metformin 500mg"), ("Treatment", "Cardiac Catheterization")]),
        # Troponin is checked to rule OUT AMI given Morrison's chest pain;
        # Morrison's confirmed diagnoses are T2DM + COPD + HTN, not AMI.
        # Link to the encounter that triggered the workup, not to AMI.
        ("Overnight troponin trend for Morrison — any elevation?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison's COPD — overnight SpO2 readings and oxygen requirements?",
         [("Diagnosis", "Chronic Obstructive Pulmonary Disease"), ("Patient", "James Morrison")]),
        ("Lab results from Morrison's midnight blood draw — CBC and CMP?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison's morning insulin dose — calculate based on fasting glucose",
         [("Patient", "James Morrison"), ("Medication", "Metformin 500mg")]),
    ]

    neo4j_volkov = mcp.create_agent_session(agent_id="volkov-night-shift")
    neo4j_volkov_sid = neo4j_volkov["session_id"]
    volkov_semvec_sid = None
    volkov_prev_response = None

    for i, (msg, entity_refs) in enumerate(night_queries):
        t0 = datetime.now(timezone.utc)
        # Inline-store: send previous response with this /run call.
        # Threshold 0.99 = always MISS; we want Volkov to BUILD context
        # turn by turn, not short-circuit on similar prior memories.
        result = semvec.run(msg, session_id=volkov_semvec_sid,
                         response=volkov_prev_response, short_circuit_threshold=0.99)
        volkov_semvec_sid = result["session_id"]
        mcp.detect_drift(neo4j_volkov_sid, msg)
        response = generate_response(msg, agent_role=volkov_role,
                                     semvec_context=result.get("context", ""))
        volkov_prev_response = response  # buffer for next inline-store
        mcp.store_response(neo4j_volkov_sid, response)
        # Also store explicitly so Semvec has the response even if no next /run follows
        semvec.store(volkov_semvec_sid, response)
        t1 = datetime.now(timezone.utc)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        print(f"  {i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}  {DIM}{textwrap.shorten(msg, 45)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_volkov_sid, entity_label, entity_name,
                step=i, phase="volkov-night",
                drift_score=drift_score,
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=drift_phase,
                extra={"agent_role": volkov_role, "shift": "night"},
            )

    print()
    info(
        "Volkov Semvec session",
        f"{volkov_semvec_sid[:16]}... ({len(night_queries)} overnight turns)",
    )

    # ── Step 2: Baseline — fresh Tanaka session (no context) ──
    # 0.65 = realistic operational threshold for the handoff scenario.
    # Without an import, Tanaka's brand-new session has no prior context,
    # so the baseline call is expected to MISS — that's the contrast we
    # show vs. the post-import call later, which uses the same threshold
    # but HITs because the imported state primed Tanaka's session.
    subheader("Step 2: Baseline — Tanaka without import (fresh session)")
    tanaka_fresh = semvec.run(
        "What happened overnight with James Morrison — any changes in condition?",
        short_circuit_threshold=0.65,
    )
    baseline_sim = tanaka_fresh["top_similarity"]
    baseline_sc = tanaka_fresh.get("short_circuit", False)
    print(f"  baseline (no import): {sim_bar(baseline_sim)} sim={baseline_sim:.3f}  "
          f"sc={baseline_sc}  drift={tanaka_fresh['drift_score']:.3f}")

    # ── Step 3: Export Volkov's session ──
    subheader("Step 3: Export Volkov's session state (SHA-256 checksum)")
    volkov_state_dict = None
    export_ok = False
    try:
        exported = semvec.export_session(volkov_semvec_sid)
        volkov_state_dict = exported.get("state_dict", exported)
        checksum = exported.get("checksum", "(no checksum field)")
        info("Exported", "state_dict obtained")
        info("Checksum", str(checksum)[:40] + ("..." if len(str(checksum)) > 40 else ""))
        export_ok = True
    except Exception as e:
        print(f"  {YELLOW}export_session: {e} — will proceed without import{RESET}")

    # ── Step 4: Tanaka imports Volkov's state ──
    subheader("Step 4: Tanaka day shift — create session + import Volkov's state")
    info("Role", "Dr. Yuki Tanaka, day shift internist taking over Morrison's care")

    # Create Tanaka's session via /run
    tanaka_init = semvec.run(
        "Dr. Tanaka day shift starting morning rounds for Morrison",
        short_circuit_threshold=0.99,
    )
    tanaka_semvec_sid = tanaka_init["session_id"]
    neo4j_tanaka = mcp.create_agent_session(agent_id="tanaka-day-shift")
    neo4j_tanaka_sid = neo4j_tanaka["session_id"]

    imported = False
    if export_ok and volkov_state_dict:
        try:
            import_result = semvec.import_session(tanaka_semvec_sid, volkov_state_dict)
            info("Import", f"Volkov state restored into Tanaka's session")
            imported = True
        except Exception as e:
            print(f"  {YELLOW}import_session: {e}{RESET}")

        # Behavioural-consistency probe — beats a bare checksum match
        if imported:
            try:
                probe_text = [
                    "Morrison overnight glucose trend",
                    "Metformin held for catheterization",
                    "Cardiac biomarkers at 6h",
                ]
                # Reuse the module-level embed() helper so the MPNet
                # model is loaded once per process, not per scenario.
                probes = [list(map(float, embed(t))) for t in probe_text]
                consistent = semvec.verify_consistency(
                    tanaka_semvec_sid, probes,
                    reference_session_id=volkov_semvec_sid,
                    tolerance=1e-3,
                )
                colour = GREEN if consistent else RED
                info(
                    "Consistency probe",
                    f"{colour}{'PASSED' if consistent else 'FAILED'}{RESET}  "
                    f"({len(probes)} probes, tolerance 1e-3)",
                )
            except Exception as e:
                print(f"  {YELLOW}verify_consistency: {e}{RESET}")

    # ── Step 5: Network delta transfer (Layer 5) ──
    subheader("Step 5: Network delta transfer (Layer 5 — may 404)")
    try:
        transfer = semvec.transfer_delta(
            source_session_id=volkov_semvec_sid,
            target_session_id=tanaka_semvec_sid,
            max_weight=0.15,
        )
        info("Transfer", f"delta applied (max_weight=0.15): {str(transfer)[:60]}")
    except Exception as e:
        print(f"  {YELLOW}network/transfer: {e}{RESET}")
        info("Fallback", "Tanaka proceeds with import-only context")

    # ── Step 6: Tanaka continues — 3 morning queries ──
    subheader("Step 6: Tanaka day rounds — 3 queries (with Volkov's context)")
    print()

    tanaka_role = "Dr. Yuki Tanaka, day shift internist taking over Morrison's care"
    day_queries = [
        ("What happened overnight with James Morrison — any changes in condition?",
         [("Patient", "James Morrison"), ("Encounter", "Emergency Room Visit")]),
        ("Morrison's morning labs — should we restart Metformin today?",
         [("Medication", "Metformin 500mg"), ("Patient", "James Morrison")]),
        ("Plan for Morrison's discharge — medication reconciliation and follow-up schedule?",
         [("Patient", "James Morrison"), ("Medication", "Metformin 500mg"),
          ("Medication", "Lisinopril 10mg")]),
    ]

    tanaka_prev_response = None
    for i, (msg, entity_refs) in enumerate(day_queries):
        t0 = datetime.now(timezone.utc)
        result = semvec.run(msg, session_id=tanaka_semvec_sid,
                         response=tanaka_prev_response, short_circuit_threshold=0.65)
        mcp.detect_drift(neo4j_tanaka_sid, msg)
        response = generate_response(msg, agent_role=tanaka_role,
                                     semvec_context=result.get("context", ""))
        tanaka_prev_response = response
        mcp.store_response(neo4j_tanaka_sid, response)
        semvec.store(tanaka_semvec_sid, response)
        t1 = datetime.now(timezone.utc)
        sim = result["top_similarity"]
        drift_score = result["drift_score"]
        drift_phase = result.get("drift_phase", "stable")
        sc = bool(result.get("short_circuit", False))
        phase_c = {"stable": GREEN, "shifting": YELLOW, "drifted": RED}.get(drift_phase, DIM)
        context_flag = f"  {GREEN}HIT{RESET}" if sc else ""
        improvement = ""
        if i == 0:
            delta = sim - baseline_sim
            improvement = f"  {GREEN}+{delta:.3f} vs baseline{RESET}" if delta > 0 else f"  {YELLOW}{delta:.3f} vs baseline{RESET}"
        print(f"  {i+1:>2}  {sim_bar(sim)} sim={sim:.3f}  drift={drift_score:.3f}  "
              f"{phase_c}{drift_phase:>8}{RESET}{context_flag}{improvement}")
        print(f"      {DIM}{textwrap.shorten(msg, 65)}{RESET}")
        if USE_LLM:
            print(f"      {DIM}A: {textwrap.shorten(response, 66)}{RESET}")
        print()
        for entity_label, entity_name in entity_refs:
            record_investigated(
                driver, neo4j_tanaka_sid, entity_label, entity_name,
                step=i, phase="tanaka-day",
                drift_score=drift_score,
                started_at=t0, ended_at=t1,
                query=msg, response=response,
                top_k_sim=sim,
                semvec_drift_phase=drift_phase,
                short_circuit=sc,
                extra={"agent_role": tanaka_role, "shift": "day",
                       "imported_from": volkov_semvec_sid if i == 0 else None,
                       "baseline_delta": float(sim - baseline_sim) if i == 0 else None},
            )

    # ── Summary ──
    subheader("Shift Handoff Summary")
    info("Patient", "James Morrison — overnight monitoring → day rounds")
    info("Dr. Volkov", f"6 overnight queries  (Semvec: {volkov_semvec_sid[:16]}...)")
    info("Dr. Tanaka", f"3 day queries  (Semvec: {tanaka_semvec_sid[:16]}...)")
    info("Export/Import", f"{'SUCCESS' if imported else 'SKIPPED (endpoint unavailable)'}")
    info("Baseline sim", f"{baseline_sim:.3f} (fresh Tanaka session)")
    info("Neo4j", "INVESTIGATED relationships: Volkov night + Tanaka day → Morrison")

    mcp.end_agent_session(neo4j_volkov_sid)
    mcp.end_agent_session(neo4j_tanaka_sid)




# ── Scenario 7: Neo4j Graph Explorer ────────────────────────────────────

def scenario_explorer(mcp: SemvecMCPServer, driver):
    header("Scenario 7: Neo4j Cypher Explorer")
    print("  Run Cypher queries directly against the Neo4j graph.")
    print(f"  Type a {BOLD}number{RESET} to run a preset, or type {BOLD}Cypher{RESET} directly.")
    print(f"  Type {BOLD}'quit'{RESET} to return.\n")

    examples = [
        # ════════════════════════════════════════════════════════════════
        #  TABLE QUERIES  (copy to Neo4j Browser → Table view)
        # ════════════════════════════════════════════════════════════════

        ("TABLE  Database overview — node types and counts",
         "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC"),

        ("TABLE  Agent dashboard — all sessions with phase + drift stats",
         """MATCH (s:AgentSession)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(p:Phase)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (d:DriftEvent {session_id: s.session_id})
WITH s, p, st, count(d) AS drift_events,
     round(coalesce(avg(d.drift_score), 0) * 1000) / 1000 AS avg_drift
RETURN s.agent_id AS agent, s.status AS status,
       coalesce(p.name, 'N/A') AS phase, coalesce(st.step, 0) AS steps,
       drift_events, avg_drift
ORDER BY drift_events DESC"""),

        ("TABLE  Ward round (Sc.3) — who investigated David Park?",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
WHERE s.agent_id IN ['chen-baseline', 'volkov-pulm', 'tanaka-cardio']
RETURN s.agent_id AS doctor, inv.phase AS role, labels(e)[0] AS entity_type,
       e.name AS entity, inv.step AS step
ORDER BY s.agent_id, inv.step"""),

        ("TABLE  Drift+Short-Circuit (Sc.2) — full step-by-step trajectory",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})-[inv:INVESTIGATED]->(e)
WITH inv, e
ORDER BY inv.step, labels(e)[0]
RETURN inv.step AS step,
       inv.phase AS phase,
       inv.semvec_drift_phase AS drift_phase,
       round(inv.drift_score * 100) / 100 AS drift,
       round(inv.top_k_similarity * 100) / 100 AS topk_sim,
       inv.short_circuit AS sc,
       inv.duration_ms AS dur_ms,
       collect(DISTINCT labels(e)[0] + ':' + e.name) AS entities,
       inv.query_preview AS query
ORDER BY step"""),

        ("TABLE  Drift+Short-Circuit (Sc.2) — Phase 4 cache hits vs misses",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})-[inv:INVESTIGATED]->(e)
WHERE inv.phase IN ['phase-4-cache', 'phase-4-miss']
WITH inv, collect(DISTINCT labels(e)[0] + ':' + e.name) AS entities
RETURN inv.step AS step,
       CASE WHEN inv.short_circuit THEN 'HIT (LLM skipped)' ELSE 'MISS (LLM called)' END AS verdict,
       round(inv.top_k_similarity * 100) / 100 AS sim,
       inv.paraphrase_of AS paraphrases,
       inv.duration_ms AS dur_ms,
       entities,
       inv.query_preview AS paraphrase_query
ORDER BY step"""),

        ("TABLE  Drift+Short-Circuit (Sc.2) — LLM cost dashboard per phase",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})-[inv:INVESTIGATED]->()
WITH inv.phase AS phase,
     count(DISTINCT inv.step) AS steps,
     sum(CASE WHEN inv.llm_call = true OR inv.llm_call IS NULL THEN 1 ELSE 0 END) AS llm_calls,
     sum(CASE WHEN inv.short_circuit = true THEN 1 ELSE 0 END) AS llm_skipped,
     round(avg(inv.duration_ms)) AS avg_ms,
     round(avg(inv.drift_score) * 100) / 100 AS avg_drift
RETURN phase, steps, llm_calls, llm_skipped, avg_ms, avg_drift
ORDER BY phase"""),

        ("TABLE  Medication safety (Sc.4) — what did the oncology guard investigate?",
         """MATCH (s:AgentSession {agent_id: 'oncology-safety-guard'})-[inv:INVESTIGATED]->(e)
RETURN inv.phase AS phase, labels(e)[0] AS entity_type, e.name AS entity,
       inv.step AS step, inv.memory_text AS memory
ORDER BY phase, step"""),

        ("TABLE  Medication safety (Sc.4) — drift signals around chemotherapy",
         """MATCH (s:AgentSession {agent_id: 'oncology-safety-guard'})-[inv:INVESTIGATED]->(e)
WITH inv, e
ORDER BY inv.step
RETURN inv.step AS step,
       inv.phase AS phase,
       round(inv.drift_score * 100) / 100 AS drift,
       round(coalesce(inv.top_k_similarity, 0) * 100) / 100 AS topk_sim,
       inv.short_circuit AS sc,
       collect(DISTINCT labels(e)[0] + ':' + e.name) AS entities,
       inv.query_preview AS query
ORDER BY step"""),

        ("TABLE  Shift handoff (Sc.6) — Volkov night vs Tanaka day, side-by-side",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
WHERE s.agent_id IN ['volkov-night-shift', 'tanaka-day-shift']
RETURN s.agent_id AS doctor,
       inv.phase AS phase,
       coalesce(inv.shift, '?') AS shift,
       inv.step AS step,
       labels(e)[0] AS entity_type,
       e.name AS entity,
       round(inv.drift_score * 100) / 100 AS drift
ORDER BY doctor, step"""),

        ("TABLE  Shift handoff (Sc.6) — drift trajectory across night → day",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->()
WHERE s.agent_id IN ['volkov-night-shift', 'tanaka-day-shift']
WITH s.agent_id AS doctor, inv.step AS step,
     round(inv.drift_score * 100) / 100 AS drift,
     inv.semvec_drift_phase AS phase,
     round(coalesce(inv.top_k_similarity, 0) * 100) / 100 AS topk_sim
RETURN doctor, step, drift, phase, topk_sim
ORDER BY doctor, step"""),

        ("TABLE  Cluster members — which agents belong to which cluster?",
         """MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster)
RETURN c.name AS cluster, s.agent_id AS agent, m.role AS role, m.facility AS facility
ORDER BY c.name, s.agent_id"""),

        ("TABLE  Drift events — all events with severity + trigger step",
         """MATCH (st:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN d.severity AS severity, round(d.drift_score * 100) / 100 AS score,
       d.drift_phase AS phase, st.step AS at_step,
       round(st.mean_similarity * 100) / 100 AS sim_at_trigger
ORDER BY d.timestamp DESC"""),

        ("TABLE  Phase transitions — Markov matrix",
         """MATCH (p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
RETURN p1.name AS from_phase, p2.name AS to_phase, count(*) AS count
ORDER BY count DESC"""),

        ("TABLE  Investigation summary — which patients were investigated by which agents?",
         """MATCH (s:AgentSession)-[:INVESTIGATED]->(p:Patient)
WITH s, p, count(*) AS touches
RETURN s.agent_id AS agent, p.name AS patient, touches
ORDER BY patient, agent"""),

        ("TABLE  Medication safety — all contraindications in the graph",
         """MATCH (m1:Medication)-[:CONTRAINDICATED_WITH]->(m2:Medication)
OPTIONAL MATCH (prov:Provider)-[:PRESCRIBED]->(m1)
RETURN m1.name AS drug_1, m2.name AS drug_2, collect(DISTINCT prov.name) AS prescribed_by
ORDER BY drug_1"""),

        ("TABLE  Semvec topology — clusters, regions, members (run Sc.3/5 first)",
         """MATCH (s:AgentSession)
OPTIONAL MATCH (s)-[m:MEMBER_OF]->(c:Cluster)
OPTIONAL MATCH (r:Region)-[:CONTAINS_CLUSTER]->(c)
RETURN s.agent_id AS agent, c.name AS cluster, m.role AS role,
       m.facility AS facility, r.name AS region
ORDER BY cluster, agent"""),

        ("TABLE  Region/cluster hierarchy — full hospital network topology",
         """MATCH (c:Cluster)
OPTIONAL MATCH (r:Region)-[:CONTAINS_CLUSTER]->(c)
OPTIONAL MATCH (s:AgentSession)-[m:MEMBER_OF]->(c)
OPTIONAL MATCH (s)-[:INVESTIGATED]->(e)
WITH r, c, s, m, count(e) AS investigations
RETURN coalesce(r.name, '(no region)') AS region,
       c.name AS cluster,
       c.coupling_factor AS coupling,
       s.agent_id AS agent,
       coalesce(m.role, m.facility, '') AS role_or_facility,
       investigations
ORDER BY region, cluster, agent"""),

        # ════════════════════════════════════════════════════════════════
        #  GRAPH QUERIES  (copy to Neo4j Browser → Graph view)
        # ════════════════════════════════════════════════════════════════

        ("GRAPH  Schema visualization — all node types and their relationships",
         "CALL db.schema.visualization()"),

        ("GRAPH  Ward round (Sc.3) — 3 doctors → Cluster → David Park + diagnoses",
         """MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster {name: 'park-ward-round'})
MATCH (s)-[inv:INVESTIGATED]->(e)
OPTIONAL MATCH (e)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED','CONTRAINDICATED_WITH']
RETURN s, m, c, inv, e, r1, connected"""),

        ("GRAPH  Drift+Short-Circuit (Sc.2) — Morrison/Patel/Rodriguez arc with drift events",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})-[inv:INVESTIGATED]->(e)
OPTIONAL MATCH (e)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED','HAD_ENCOUNTER','CONTRAINDICATED_WITH']
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(d:DriftEvent)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(ph:Phase)
RETURN s, inv, e, r1, connected, st, tr, d, ph"""),

        ("GRAPH  Drift+Short-Circuit (Sc.2) — Phase 4 paraphrase HITs back to Phase 1 entities",
         """MATCH (s:AgentSession {agent_id: 'drift-shortcircuit-demo'})-[inv:INVESTIGATED]->(e)
WHERE inv.phase IN ['phase-4-cache', 'phase-4-miss', 'on-topic']
RETURN s, inv, e"""),

        ("GRAPH  Medication safety (Sc.4) — oncology guard's view of Gutierrez chemo",
         """MATCH (s:AgentSession {agent_id: 'oncology-safety-guard'})-[inv:INVESTIGATED]->(e)
OPTIONAL MATCH (e)-[ci:CONTRAINDICATED_WITH]->(other:Medication)
OPTIONAL MATCH (treat:Treatment)-[u:USES]->(e)
OPTIONAL MATCH (pat:Patient)-[hasT:HAS_TREATMENT]->(treat)
RETURN s, inv, e, ci, other, treat, u, pat, hasT"""),

        ("GRAPH  Shift handoff (Sc.6) — Volkov night ⇄ Tanaka day around Morrison",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
WHERE s.agent_id IN ['volkov-night-shift', 'tanaka-day-shift']
OPTIONAL MATCH (e)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED','HAD_ENCOUNTER','CONTRAINDICATED_WITH']
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(ph:Phase)
RETURN s, inv, e, r1, connected, ph"""),

        ("GRAPH  Hospital network — Region → Clusters → Agents → Facilities",
         """MATCH (c:Cluster)
OPTIONAL MATCH (r:Region)-[rc:CONTAINS_CLUSTER]->(c)
OPTIONAL MATCH (s:AgentSession)-[m:MEMBER_OF]->(c)
OPTIONAL MATCH (s)-[inv:INVESTIGATED]->(e)
WHERE labels(e)[0] IN ['Patient', 'Facility', 'Diagnosis']
RETURN r, rc, c, m, s, inv, e"""),

        ("GRAPH  Drift cascade — states that triggered drift events",
         """MATCH (st:SemanticState)-[t:TRIGGERED]->(d:DriftEvent)
OPTIONAL MATCH (s:AgentSession)-[:CURRENT_STATE]->(current:SemanticState)
OPTIONAL MATCH path = (current)-[:STATE_HISTORY*0..20]->(st)
RETURN s, st, t, d"""),

        ("GRAPH  Agent × Patient × Drift — Semvec investigation + clinical graph + drift events",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat:Patient)
MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (treat:Treatment)-[treats:TREATS]->(diag)
OPTIONAL MATCH (treat)-[uses:USES]->(med:Medication)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(drift:DriftEvent)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(phase:Phase)
OPTIONAL MATCH (s)-[:MEMBER_OF]->(cluster:Cluster)
RETURN s, inv, pat, dx, diag, tb, prov, treats, treat, uses, med,
       st, tr, drift, phase, cluster"""),

        ("GRAPH  Clinical network — patients → diagnoses ← treatments → medications",
         """MATCH (pat:Patient)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (treat:Treatment)-[tr:TREATS]->(diag)
OPTIONAL MATCH (treat)-[u:USES]->(med:Medication)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
RETURN pat, dx, diag, tr, treat, u, med, tb, prov"""),

        ("GRAPH  Medication safety × Semvec — contraindications, who investigated them?",
         """MATCH (m1:Medication)-[ci:CONTRAINDICATED_WITH]->(m2:Medication)
OPTIONAL MATCH (prov:Provider)-[rx:PRESCRIBED]->(m1)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(m1)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(phase:Phase)
RETURN m1, ci, m2, prov, rx, s, inv, phase"""),

        ("GRAPH  Provider referral × Semvec — referrals + which agents investigated each provider",
         """MATCH (p1:Provider)-[ref:REFERRED_TO]->(p2:Provider)
OPTIONAL MATCH (p1)-[a1:AFFILIATED_WITH]->(f1:Facility)
OPTIONAL MATCH (p2)-[a2:AFFILIATED_WITH]->(f2:Facility)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(p1)
OPTIONAL MATCH (s)-[:MEMBER_OF]->(cluster:Cluster)
RETURN p1, ref, p2, a1, f1, a2, f2, s, inv, cluster"""),

        ("GRAPH  Patient journey × Semvec — encounters + which agents tracked each step",
         """MATCH (pat:Patient)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (enc)-[ri:RESULTED_IN]->(diag:Diagnosis)
OPTIONAL MATCH (enc)-[oa:OCCURRED_AT]->(fac:Facility)
OPTIONAL MATCH (prov:Provider)-[att:ATTENDED]->(enc)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(drift:DriftEvent)
RETURN pat, he, enc, ri, diag, oa, fac, prov, att, s, inv, st, tr, drift"""),

        ("GRAPH  James Morrison — full clinical picture + agent investigations",
         """MATCH (pat:Patient {name: 'James Morrison'})
OPTIONAL MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (prov)-[rx:PRESCRIBED]->(med:Medication)
OPTIONAL MATCH (pat)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat)
RETURN pat, dx, diag, tb, prov, rx, med, he, enc, s, inv"""),

        ("GRAPH  Full story — all agents + all healthcare entities + drift events",
         """MATCH (s:AgentSession)-[inv:INVESTIGATED]->(entity)
OPTIONAL MATCH (entity)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED',
                    'HAD_ENCOUNTER','AFFILIATED_WITH','CONTRAINDICATED_WITH']
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(d:DriftEvent)
RETURN s, inv, entity, r1, connected, st, tr, d"""),

        ("GRAPH  Everything — complete graph (limit 300)",
         """MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 300"""),
    ]

    printed_graph_header = False
    for i, (name, _) in enumerate(examples, 1):
        if not printed_graph_header and name.startswith("GRAPH"):
            print(f"\n  {YELLOW}{'─' * 60}{RESET}")
            printed_graph_header = True
        tag = name[:5]  # TABLE or GRAPH
        rest = name[7:]  # strip "TABLE  " or "GRAPH  "
        tag_color = DIM if tag == "TABLE" else GREEN
        print(f"  {CYAN}{i:>2}{RESET}  {tag_color}[{tag}]{RESET}  {rest}")
    print()

    while True:
        try:
            raw = input(f"  {MAGENTA}cypher>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() == "quit":
            break

        # Allow selecting preset by number
        if raw.isdigit() and 1 <= int(raw) <= len(examples):
            idx = int(raw) - 1
            name, query = examples[idx]
            print(f"\n  {CYAN}{BOLD}{name}{RESET}")
            for line in query.strip().splitlines():
                print(f"  {DIM}{line}{RESET}")
            print()
        elif raw.lower() == "list":
            ph = False
            for i, (name, _) in enumerate(examples, 1):
                if not ph and name.startswith("GRAPH"):
                    print(f"\n  {YELLOW}{'─' * 60}{RESET}")
                    ph = True
                tag = name[:5]
                rest = name[7:]
                tc = DIM if tag == "TABLE" else GREEN
                print(f"  {CYAN}{i:>2}{RESET}  {tc}[{tag}]{RESET}  {rest}")
            print()
            continue
        else:
            query = raw

        try:
            with driver.session(database=NEO4J_DATABASE) as db:
                t0 = time.time()
                result = db.run(query)
                records = [dict(r) for r in result]
                elapsed = (time.time() - t0) * 1000

            if not records:
                print(f"  {DIM}(no results){RESET}  {elapsed:.0f}ms\n")
                continue

            # Pretty-print as table
            keys = list(records[0].keys())
            # Cap column widths for readability
            col_widths = {}
            for k in keys:
                val_width = max(len(str(r.get(k, "")))
                                for r in records[:20])
                col_widths[k] = min(max(len(str(k)), val_width), 40)

            header_line = "  " + "  ".join(f"{k:>{col_widths[k]}}" for k in keys)
            print(header_line)
            print("  " + "  ".join("─" * col_widths[k] for k in keys))
            for r in records[:30]:
                cells = []
                for k in keys:
                    val = str(r.get(k, ""))
                    if len(val) > col_widths[k]:
                        val = val[:col_widths[k] - 1] + "…"
                    cells.append(f"{val:>{col_widths[k]}}")
                print("  " + "  ".join(cells))
            if len(records) > 30:
                print(f"  {DIM}... and {len(records) - 30} more rows{RESET}")
            print(f"  {DIM}{len(records)} rows, {elapsed:.0f}ms{RESET}\n")

        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}\n")


# ── Main Menu ────────────────────────────────────────────────────────────

def main():
    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   {BOLD}Semvec × Neo4j Interactive Demo{RESET}{CYAN}                                       ║
║                                                                      ║
║   Semvec computes drift, phases, memory.                                ║
║   Neo4j persists, queries, and analyzes the results.                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
""")

    # Connect (verbose so the user can see which step is slow)
    def _step(msg: str) -> None:
        print(f"  {DIM}…{RESET} {msg}", flush=True)

    _step(f"connecting to Neo4j at {NEO4J_URI} …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"  {RED}Neo4j not reachable at {NEO4J_URI}: {e}{RESET}")
        print(f"  Start it: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5.26")
        sys.exit(1)

    _step("loading embedder + Semvec runtime (first run downloads the MPNet model, ~420 MB) …")
    semvec = _build_semvec_client()
    try:
        h = semvec.health()
        semvec_status = f"{GREEN}OK{RESET} v{h.get('version','?')} ({h.get('active_sessions','?')} sessions)"
    except Exception as e:
        print(f"  {RED}Semvec runtime unavailable: {e}{RESET}")
        sys.exit(1)

    _step("applying Neo4j schema (constraints + vector indexes) …")
    adapter = Neo4jSemvecAdapter(driver, database=NEO4J_DATABASE)
    adapter.apply_schema(SCHEMA_PATH)

    _step("clearing previous Semvec artifacts (healthcare template stays) …")
    _reset_semvec_artifacts(driver, semvec=semvec)

    mcp = SemvecMCPServer(driver, database=NEO4J_DATABASE, semvec_client=semvec)

    if USE_LLM:
        llm_probe_timeout = float(os.environ.get("LLM_PROBE_TIMEOUT_SEC", "8"))
        _step(f"probing LLM endpoint (timeout {llm_probe_timeout:.0f}s, env LLM_PROBE_TIMEOUT_SEC) …")
        import threading

        # Box pattern: keep the list reference distinct from the final str.
        # The probe thread may still be running after t.join() times out,
        # so we never rebind the box itself.
        _probe_box: list[str] = [f"{RED}probe timeout — LLM may be slow or unreachable{RESET}"]

        def _probe():
            try:
                _ = get_llm().respond("Say OK", max_tokens=20)
                _probe_box[0] = f"{GREEN}enabled{RESET} ({os.environ['OPENAI_MODEL']})"
            except Exception as exc:
                _probe_box[0] = f"{RED}error: {exc}{RESET}"

        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join(timeout=llm_probe_timeout)
        llm_status = _probe_box[0]
    else:
        llm_status = f"{YELLOW}disabled{RESET} (--no-llm flag, responses will show N/A)"

    print()
    info("Neo4j", f"{GREEN}Connected{RESET} ({NEO4J_URI})")
    info("Semvec", semvec_status)
    info("LLM", llm_status)
    info("Neo4j Browser", f"http://localhost:7474")

    scenarios = {
        "1": ("Live Drift + Token Savings — type messages, watch drift AND PSS-vs-baseline tokens per turn (LLM required)", scenario_live_drift),
        "2": ("Drift + Short-Circuit — Morrison diabetes → Patel psychiatry → Rodriguez cardiology → paraphrase HITs", scenario_topic_switch),
        "3": ("Ward Round Cluster — Dr. Chen/Volkov/Tanaka on David Park (Semvec Layer 2 Clusters)", scenario_ward_round),
        "4": ("Medication Safety Guard — Gutierrez chemo, anchors/triggers/isolation (Semvec Layer 1b)", scenario_medication_safety),
        "5": ("Hospital Network Consensus — Rodriguez/Morrison, Region + Observer (Semvec Layers 3+4)", scenario_hospital_consensus),
        "6": ("Shift Handoff — Volkov night → Tanaka day, export/import/transfer (Semvec Layer 5)", scenario_shift_handoff),
        "7": ("Neo4j Cypher Explorer — run queries directly against the graph", scenario_explorer),
    }

    while True:
        print(f"\n{CYAN}{'─' * 70}{RESET}")
        print(f"  {BOLD}Choose a scenario:{RESET}\n")
        for key, (desc, _) in scenarios.items():
            print(f"    {CYAN}{key}{RESET}  {desc}")
        print(f"    {CYAN}r{RESET}  Reset Neo4j (clear Semvec artifacts, keep healthcare template)")
        print(f"    {CYAN}q{RESET}  Quit\n")

        try:
            choice = input(f"  {CYAN}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q" or choice == "quit":
            break

        if choice == "r":
            _reset_semvec_artifacts(driver, semvec=semvec)
            print(f"  {GREEN}Done — ready for a fresh run.{RESET}")
            continue

        if choice in scenarios:
            try:
                scenarios[choice][1](mcp, driver)
            except KeyboardInterrupt:
                print(f"\n  {YELLOW}Interrupted{RESET}")
            except Exception as e:
                print(f"\n  {RED}Error: {e}{RESET}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  {RED}Unknown choice: {choice}{RESET}")

    print(f"\n  {DIM}Closing connections...{RESET}")
    driver.close()
    print(f"  {GREEN}Done!{RESET}\n")


if __name__ == "__main__":
    main()

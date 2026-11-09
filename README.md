# Semvec + Neo4j: Semantic Memory Meets Knowledge Graphs

> **Persistent, constant-cost semantic memory for multi-agent LLM systems** — combining [Semvec](https://pypi.org/project/semvec/) drift detection with a Neo4j knowledge graph. Agent memory, multi-agent clusters, semantic drift detection, and verbatim-fact persistence, all queryable in one graph.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.11%2B-008CC1.svg)
![semvec 0.8.9+](https://img.shields.io/badge/semvec-0.8.9%2B-7b53a8.svg)
![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen.svg)

**What happens when you give an AI agent infinite memory AND a knowledge graph?**

[Semvec](https://pypi.org/project/semvec/) compresses any conversation into a fixed-size context block — O(1) memory per turn, no matter how long the conversation. It runs entirely in-process: no API key, no network round-trips. Neo4j stores the structured world the agent operates in: patients, medications, providers, diagnoses, treatments, facilities, and all the relationships between them.

Alone, each is powerful. Together, they unlock capabilities that neither can achieve independently.

---

## Why Semvec + Neo4j?

### Semvec alone knows *what was said* — but not *what exists*

Semvec tracks semantic drift, detects topic switches, caches repeated queries, and manages multi-agent coordination. But it operates on unstructured text. It doesn't know that Metformin is contraindicated with renal failure, or that Dr. Volkov works at Memorial General.

### Neo4j alone knows *what exists* — but not *what was discussed*

Neo4j holds the complete healthcare knowledge graph: 5 patients, 5 providers, 5 diagnoses, 5 medications, their relationships (DIAGNOSED_WITH, PRESCRIBED, CONTRAINDICATED_WITH, REFERRED_TO). But it has no memory of which agent asked what, when they drifted off-topic, or whether a question was already answered.

### Together: agents that *remember* AND *reason over structure*

| Capability | Semvec | Neo4j | Semvec + Neo4j |
|---|:---:|:---:|:---:|
| Detect when agent switches from diabetes to psychiatry | yes | — | yes + graph shows which entities were investigated before/after drift |
| Know that Metformin is contraindicated with Amoxicillin | — | yes | yes + Semvec injects this as synthetic memory, agent recalls it in conversation |
| Skip LLM call when a paraphrased question was already answered | yes | — | yes + Neo4j shows which healthcare entity the cached answer relates to |
| Three specialists share findings on the same patient | partial | — | yes: Semvec cluster propagates HITs, Neo4j shows who investigated what |
| Detect that two hospital departments drifted simultaneously | — | — | yes: Semvec regions detect consensus drift, Neo4j stores the topology |
| Transfer night-shift knowledge to day-shift doctor | partial | — | yes: Semvec export/import + delta transfer, Neo4j traces which provider investigated which patient at which shift |

---

## How it compares

Agent-memory frameworks like [Mem0](https://github.com/mem0ai/mem0), [Zep / Graphiti](https://github.com/getzep/graphiti), [Letta (MemGPT)](https://github.com/letta-ai/letta), and LangMem focus on the *memory layer* on its own. This project is different in two ways: Semvec computes memory **in-process with no generative-LLM calls at ingest** (no API key, no per-turn network egress), and Neo4j stores that memory **in the same graph as your domain knowledge** — so a single Cypher query joins *what the agent discussed* with *what actually exists*.

| | **Semvec + Neo4j** | Mem0 | Zep / Graphiti | Letta |
|---|---|---|---|---|
| Memory substrate | Neo4j (graph + vector) | vector + graph + KV | temporal knowledge graph | hierarchical tiers |
| Generative-LLM calls at ingest | none (in-process) | LLM extraction | LLM extraction | LLM-managed |
| Domain graph in the **same** store | yes — the `INVESTIGATED` bridge | no | partial | no |
| Semantic drift detection | yes (first-class signal) | no | no | no |
| Multi-agent clusters / regions / observer | yes | no | no | partial |
| Verbatim-fact persistence (byte-exact) | yes (`LiteralFact`) | no | no | no |

When to reach for this: you already run Neo4j (or want one operational system for both memory and domain), you need per-turn drift/topic-switch signals, or you coordinate **multiple agents** over shared state. Semvec itself is a proprietary engine ([`pip install semvec`](https://pypi.org/project/semvec/)); this repository is the MIT-licensed Neo4j integration on top of it.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Your LLM Agent               │
                    └─────────┬───────────────┬───────────────┘
                              │               │
                    ┌─────────▼─────────┐   ┌─▼─────────────────┐
                    │  Semvec runtime   │   │   Neo4j            │
                    │  (in-process)     │   │                    │
                    │                   │   │ • Healthcare KG    │
                    │ • run → context   │   │   (patients,       │
                    │ • drift_detected  │   │    medications,    │
                    │ • short_circuit   │   │    diagnoses...)   │
                    │ • cluster HITs    │   │                    │
                    │ • region events   │   │ • Semvec state     │
                    │ • observer alerts │   │   chain (drift     │
                    │ • export/import   │   │   events, phases,  │
                    │ • anchors/triggers│   │   INVESTIGATED)    │
                    └───────────────────┘   └────────────────────┘
```

Semvec does all computation (drift scoring, context compression, short-circuit detection, multi-agent coordination) inside your Python process via `pip install semvec`. Neo4j persists the results and connects them to the structured knowledge graph. The `INVESTIGATED` relationship bridges both worlds: it links an agent session to the healthcare entities it queried, with the drift score at the time of investigation.

---

## What the Demo Shows

### Scenario 1: Live Drift + Token Savings (Layer 1 — Chat Proxy)

Free-form chat against a live LLM endpoint. Each turn shows two things side by side: the drift panel (similarity, drift score, Semvec phase, Neo4j state chain) and the token-savings panel — `SemvecChatProxy` reports the PSS-compressed prompt token count next to what a naïve full-history replay would have cost. The headline ~8× token reduction reproduces live against your own questions.

**Semvec signals**: `create_chat_proxy`, `run`, `drift_score`, `drift_phase`
**Neo4j graph**: AgentSession → SemanticState chain growing turn by turn

### Scenario 2: Drift Detection + Short-Circuit

An agent investigates **James Morrison** (Type 2 Diabetes, Metformin, Lisinopril, Atorvastatin) across 17 queries. Then switches to **Aisha Patel** (Major Depressive Disorder, Sertraline, CBT). Semvec detects the specialty switch (`drift_score=0.44, phase=drifted`). Returns to **Maria Rodriguez** (AMI, cardiology). Finally, paraphrases of Phase 1 queries trigger short-circuit — LLM skipped, cached context returned.

**Semvec signals**: `drift_detected`, `drift_score`, `drift_phase`, `top_similarity`, `short_circuit`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Diagnosis/Medication + DriftEvent nodes at pivot points

### Scenario 3: Multi-Specialist Ward Round (Layer 2 — Clusters)

Three doctors examine **David Park** (Essential Hypertension + COPD):
- **Dr. Chen** (internist) seeds a shared Semvec cluster with 4 baseline Q&A pairs
- **Dr. Volkov** (pulmonologist) queries the cluster → gets HITs from Chen's findings (LLM skipped)
- **Dr. Tanaka** (cardiologist) stores a novel ECG finding → Volkov re-queries and gets it

**Semvec signals**: `cluster_run`, `short_circuit`, `cluster_store`, `cluster_feedback`
**Neo4j graph**: 3 Agents → MEMBER_OF → Cluster + INVESTIGATED → Patient/Diagnosis/Medication

### Scenario 4: Medication Safety Guard (Layer 1b — Anchors, Triggers, Isolation)

An oncology pharmacist monitors **Carlos Gutierrez** (Chemotherapy Cycle 1):
- **Drift Anchor** locks session to oncology domain (real 768-dim embedding via sentence-transformers)
- **Resonance Triggers** on "CRITICAL" and "contraindication" keywords
- **Synthetic Memory Injection**: contraindications from Neo4j graph (Amoxicillin↔Metformin, Atorvastatin↔Sertraline) injected into Semvec memory tiers
- **Input Isolation**: QUARANTINE mode filters off-topic queries
- **Anchor Score**: measures how far the session drifted from the oncology domain

**Semvec signals**: `create_session`, `add_anchor`, `get_anchor_score`, `add_trigger`, `set_isolation`, `inject_memory`
**Neo4j graph**: Agent → INVESTIGATED → Patient/Treatment/Medication + Memory nodes with contraindication knowledge

### Scenario 5: Hospital Network Consensus (Layers 3+4 — Regions, Observer)

Two department clusters across two facilities:
- **Cardiology** at Memorial General → Maria Rodriguez (AMI)
- **Emergency** at Riverside Medical → James Morrison (ER visit)

Both clusters are in a Semvec **Region** (`consensus_threshold=0.5`). A **Global Observer** monitors for cross-cluster anomalies. When both departments drift to admin topics simultaneously, the drift rule flags the pivot in both clusters and the Region records their drift events (two per department: the cluster run and its member session each publish), while the Observer samples for cross-cluster anomalies. On top of that signal, a **ConsensusEngine** runs an explicit qualified-majority vote (supervisor weighted 1.5) on whether to escalate to a network-wide incident.

**Semvec signals**: `create_region`, `get_region_events`, `create_observer`, `observer_sample`, `get_anomalies`, `create_consensus_engine`, `submit_consensus_proposal`, `vote_on_consensus`, `evaluate_consensus`
**Neo4j graph**: Region → CONTAINS_CLUSTER → Clusters → MEMBER_OF ← Agents → INVESTIGATED → Patient/Facility; the vote as Region → HAS_CONSENSUS_ENGINE → ConsensusEngine → PROPOSED → ConsensusProposal → RECEIVED_VOTE → ConsensusVote ← CAST ← Agents, and ConsensusProposal → DECIDED → ConsensusEvent ← CONSENSUS_EVENT ← Region

### Scenario 6: Shift Handoff (Layer 5 — Export/Import, Network Transfer)

Night shift **Dr. Volkov** monitors Morrison overnight (6 queries, builds deep Semvec context). She exports her session state (SHA-256 checksum). Day shift **Dr. Tanaka** imports the state + receives a semantic delta transfer. Tanaka's first query immediately shows `sim=0.578` (vs `0.000` baseline without import) — full context continuity across shift change. After import, a **behavioural-consistency probe** (`verify_consistency`) runs reference embeddings through both sessions: the SHA-256 checksum catches bit-level corruption, the probe catches algorithmic drift before the imported state is used.

**Semvec signals**: `export_session`, `import_session`, `transfer_delta`, `verify_consistency`
**Neo4j graph**: Volkov → INVESTIGATED → Morrison (night) + Tanaka → INVESTIGATED → Morrison (day)

### Scenario 7: Neo4j Cypher Explorer

Run Cypher directly against the graph the previous scenarios produced — including the audit-trail queries from `docs/demo_queries.md` (who investigated whom, which turns short-circuited, where drift fired).

---

## The INVESTIGATED Relationship

This is where Semvec and Neo4j truly merge. Every time an agent queries about a healthcare entity, an `INVESTIGATED` relationship is created, carrying a full audit-trail payload:

```cypher
(agent:AgentSession)-[:INVESTIGATED {
  step: 5, phase: "drifted", drift_score: 0.44,
  started_at: "2026-06-07T06:12:03Z", ended_at: "2026-06-07T06:12:04Z", duration_ms: 840,
  query_preview: "What is Morrison's overnight glucose trend?",
  response_preview: "Glucose stable 110-140 mg/dL overnight ...",
  top_k_similarity: 0.81, semvec_drift_phase: "stable",
  short_circuit: false, llm_call: true
  // per-scenario extras: cluster_id, shift, imported_from,
  // baseline_delta, paraphrase_of, trigger_keyword, agent_role
}]->(entity:Patient)
```

Conversational state becomes queryable like structured data — and structured data becomes injectable as synthetic memory.

This enables queries that neither system could answer alone:

```cypher
// Which agents investigated Morrison, and did any drift during the investigation?
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(p:Patient {name: 'James Morrison'})
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN s.agent_id, inv.step, inv.drift_score, d.severity

// Full picture: agent → patient → diagnoses → medications + drift events + cluster membership
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat:Patient)
MATCH (pat)-[:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (treat:Treatment)-[:TREATS]->(diag)
OPTIONAL MATCH (treat)-[:USES]->(med:Medication)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[:TRIGGERED]->(drift:DriftEvent)
OPTIONAL MATCH (s)-[:MEMBER_OF]->(cluster:Cluster)
RETURN s, inv, pat, diag, treat, med, drift, cluster
```

---

## Quick Start

### Prerequisites

- **Docker with Compose** — Neo4j 5.26 always runs as the `neo4j` service of
  `docker-compose.yml`; seed script, demo and tests run in the same image
- **OpenAI-compatible LLM endpoint** for agent responses *(optional — only needed for the demo with LLM enabled)*
- *Optional:* **Python 3.10+** to run the demo or the tests outside a container
  (GPU recommended for embeddings, CPU works too)

### Setup

```bash
git clone https://github.com/VersinoPsiOmega/Semvec-neo4j-agent-integration.git
cd Semvec-neo4j-agent-integration

# 1. Configure environment — every variable below is required.
#    The demo and tests refuse to start with missing values rather
#    than falling back to silent defaults.
cp .env.example .env
$EDITOR .env                   # fill in NEO4J_TEST_PASSWORD, SEMVEC_LICENSE_KEY + OPENAI_*
                               # (without a licence key semvec >= 0.8.9 throttles
                               #  to 5 QPS, which breaks seeding and the demo)

# 2. Start Neo4j (browser :7474, bolt :7687) and the query guide (:7477).
#    NEO4J_TEST_PASSWORD also sets the container's Neo4j password.
docker compose up -d

# 3. Seed the healthcare graph (uses the fixtures bundled under
#    healthcare-fixtures/ — same patients, providers, diagnoses and
#    medications the demo scenarios reference by name).
docker compose run --rm seed
```

> Want to regenerate the fixtures from scratch? Run
> `uvx create-context-graph healthcare-test-data --domain healthcare
> --framework pydanticai --demo-data` and point `FIXTURES_PATH` and
> `HEALTHCARE_SCHEMA_PATH` in your `.env` at the new scaffold.

### Run the Demo

```bash
docker compose run --rm demo                        # interactive demo, LLM enabled (default)
docker compose run --rm demo python scripts/interactive_demo.py --no-llm   # skip LLM calls
docker compose run --rm test                        # pytest against the container Neo4j
docker compose down                                 # stop; add -v to drop the Neo4j data and model cache
```

Inside the containers `NEO4J_TEST_URI` is pointed at the `neo4j` service
automatically. The first demo or seed run downloads the embedding model
(~420 MB) into the `hf-cache` volume. If `OPENAI_BASE_URL` points at an LLM on
your machine (`localhost`), use `host.docker.internal` instead — `localhost`
inside a container is the container itself.

**Without a container** (local Python, same Neo4j): set
`NEO4J_TEST_URI=bolt://localhost:7687` in `.env`, then

```bash
python3 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"                          # or: pip install -e ".[dev]"
python scripts/interactive_demo.py                  # add --no-llm to skip LLM calls
python -m pytest
```

### Neo4j Browser

Open http://localhost:7474 and log in as `neo4j` with `NEO4J_TEST_PASSWORD`
(connect URL `localhost:7687`). After login the browser opens the guide
"Semvec × Neo4j demo queries": the 22 queries of `docs/demo_queries.md`, one
per step, each with a button to run it. The `guide` service renders it from
that file on every request.

### Tuning Constants

The interactive demo uses three `.env`-driven knobs and a handful of inline,
scenario-specific thresholds. The latter are commented at their first use,
but for orientation:

| Knob | Where | Default | What it controls |
|---|---|---|---|
| `DRIFT_THRESHOLD` | `.env` (required) | `0.35` in `.env.example` | Drift score from which a turn counts as a real topic switch. |
| `DRIFT_SIM_CEILING` | `.env` (required) | `0.45` in `.env.example` | Cosine-similarity ceiling for the same gate — drift_score crosses *and* similarity drops. Together these filter in-domain sub-shifts. |
| `DRIFT_COOLDOWN_TURNS` | `.env` (optional) | `0` (off) | Turns skipped after a flagged drift, so the turns settling into the new topic are not flagged again. |

The three knobs form one `DriftRule` (`semvec_neo4j.core.drift_rule`), which
`SemvecClient(drift_rule=...)` applies per session. Its verdict is the `drift`
key of every `run()` result; the demo display, the persisted `DriftEvent`
nodes and the drift events published to a session's region all follow it.
The thresholds are calibrated per LLM (0.35 / 0.45 were tuned for Qwen3.6 and
remain the best trade-off for Qwen3.8), so `.env` has no silent fallback.
| `short_circuit_threshold=0.99` | Sc 3 / Sc 6 | — | "Impossible" threshold — forces deterministic MISS during seeding so context accumulates. |
| `short_circuit_threshold=0.52` | Sc 3 (Volkov, Tanaka) | — | Calibrated below typical paraphrase similarity on the Park topic (MPNet-768) so cluster HITs fire. |
| `short_circuit_threshold=0.60` | Sc 5 (cardiology, ER) | — | Mid-band threshold for the consensus scenario — HIT on on-topic, MISS on the deliberate admin pivot. |
| `short_circuit_threshold=0.65` | Sc 6 (Tanaka) | — | Realistic operational threshold; baseline (no import) MISSes, post-import HITs. |
| `short_circuit_threshold=0.75` | Sc 4 (oncology) | — | Operational sweet spot for in-domain Q&A: paraphrases HIT, new on-topic queries MISS. |
| `QUARANTINE_THRESHOLD=0.5` | Sc 4 | — | Below this similarity, the isolation layer quarantines the turn. |
| `consensus_threshold=0.5` | Sc 5 | — | Fraction of clusters in the region that must drift within `vote_window_seconds` for region consensus. |
| `vote_window_seconds=60.0` | Sc 5 | — | Sliding window for region consensus voting. |
| `sample_interval_seconds=30.0` | Sc 5 | — | Automatic observer cadence. The demo calls `observer_sample()` manually, so this only matters if you let it idle. |
| `coupling_factor=0.20` (Sc 5) / `0.25` (Sc 3) | per cluster | — | Strength with which the cluster's aggregate vector is blended back into member sessions on `cluster_feedback`. |

These defaults are tuned for `all-mpnet-base-v2` (768-dim, the demo's
default embedder). If you swap to `all-MiniLM-L6-v2` (384-dim) or another
model, expect the numeric thresholds to drift — re-tune by watching the
`sim_bar` colour bands in the demo output.

### Run Tests

```bash
docker compose run --rm test       # in the container, against the neo4j service
python -m pytest                   # locally, against bolt://localhost:7687 (see "Without a container")
```

The unit tests for `SemvecClient`, the drift rule, the browser guide and the
embedder run fully offline; the Neo4j-backed integration tests need the `neo4j`
service running (`docker compose up -d`) and skip otherwise.

---

## Semvec Coverage

| Layer | Capability | `SemvecClient` methods | Scenario |
|---|---|---|---|
| **1** | run, store, drift detection, short-circuit, live token-savings chat proxy | `run`, `store`, `create_chat_proxy` | 1, 2, 3, 5, 6 |
| **1b** | session control, anchors, triggers, isolation, memory injection, verbatim facts | `create_session`, `add_anchor`, `get_anchor_score`, `add_trigger`, `set_isolation`, `inject_memory`, `extract_facts`, `store_facts_as_entities` | 4 |
| **2** | cluster CRUD + run/store/feedback/members | `create_cluster`, `cluster_run`, `cluster_store`, `cluster_feedback`, `add_cluster_member`, `remove_cluster_member` | 3, 5 |
| **3** | region CRUD, attached clusters, drift events | `create_region`, `add_region_cluster`, `get_region_events` | 5 |
| **4** | global observer (sample, anomalies, summary) | `create_observer`, `observer_sample`, `get_anomalies`, `get_observer_summary` | 5 |
| **5** | shift handover: export/import + network delta transfer + trust | `export_session`, `import_session`, `transfer_delta`, `propose_consensus`, `get_trust_scores` | 6 |
| **6** | behavioural-consistency probe after import | `verify_consistency` | 6 |
| **7** | explicit consensus voting (5 levels: simple/qualified majority, unanimous, weighted, adaptive) | `create_consensus_engine`, `register_consensus_voter`, `submit_consensus_proposal`, `vote_on_consensus`, `evaluate_consensus`, `get_consensus_statistics` | 5 |

---

## Neo4j Schema

```
Semvec Layer: AgentSession → SemanticState → DriftEvent
              AgentSession → Phase (TRANSITIONED_TO chain)
              AgentSession → Memory (tiered: short/medium/long)
              AgentSession → MEMBER_OF → Cluster
              Region → CONTAINS_CLUSTER → Cluster
              Region → HAS_CONSENSUS_ENGINE → ConsensusEngine → PROPOSED → ConsensusProposal
              ConsensusProposal → RECEIVED_VOTE → ConsensusVote ← CAST ← AgentSession
              ConsensusProposal → DECIDED → ConsensusEvent ← CONSENSUS_EVENT ← Region

Healthcare:   Patient → DIAGNOSED_WITH → Diagnosis
              Patient → TREATED_BY → Provider
              Provider → PRESCRIBED → Medication
              Treatment → TREATS → Diagnosis
              Treatment → USES → Medication
              Medication → CONTRAINDICATED_WITH → Medication
              Provider → REFERRED_TO → Provider
              Provider → AFFILIATED_WITH → Facility
              Patient → HAD_ENCOUNTER → Encounter
              (only the domain ontology is imported: the generic POLE+O base types
               Person / Organization / Location / Event / Object that create-context-graph
               adds to every domain are left out, see scripts/healthcare_fixtures.py)

Bridge:       AgentSession → INVESTIGATED {step, phase, drift_score, started_at, ended_at,
              duration_ms, query_preview, response_preview, top_k_similarity,
              semvec_drift_phase, short_circuit, llm_call, …} → (any healthcare entity)
```

---

## Project Structure

```
semvec_neo4j/                  # repo root (PyPI package: semvec_neo4j)
  src/
    semvec_neo4j/              # the installable package (src layout)
      core/
        semvec_client.py       # In-process Semvec client (Layers 1-7)
        embedder.py            # HashEmbedder (offline) + SentenceTransformerEmbedder
        drift_detector.py      # Thin bridge: Semvec → Neo4j (no heuristics)
      mcp/
        semvec_mcp_server.py   # MCP tools for agent frameworks
      persistence/             # Neo4j stores: session, state, phase, drift, memory,
                               # cluster, region, agent, observer, anomaly, literal-fact
      network/                 # Cluster/network coordination
      analytics/               # Similarity, influence, trajectories
  scripts/
    interactive_demo.py        # 7 scenarios showcasing Semvec × Neo4j
    seed_test_data.py          # Bulk seed healthcare + Semvec data
    demo_helpers.py            # Pure-Python render helpers (sim_bar, observer sample)
  schema/
    semvec_schema.cypher       # Neo4j constraints + vector indexes
  healthcare-fixtures/         # Bundled healthcare graph (5 patients,
                               # 5 providers, diagnoses, medications,
                               # encounters) — pinned for reproducibility
  tests/                       # pytest suite — unit (offline) + integration (Neo4j)
  docs/
    API.md                     # Public API surface documentation
    demo_queries.md            # Public Cypher / demo query reference
    # docs/blog/, docs/plans/  → gitignored (drafts + internal notes)
```

---

## License

The `semvec_neo4j` integration code in this repository is released under the
[MIT License](LICENSE) — free to use, modify, and distribute.

It depends on [`semvec`](https://pypi.org/project/semvec/), a separate
**proprietary** package distributed on PyPI under its own license terms.
`semvec_neo4j` builds on top of semvec but does not redistribute or relicense
it — you obtain semvec directly from PyPI. In short: this integration layer is
open source; the underlying semvec engine is not.

Copyright (c) 2026 Michael Neuberger.

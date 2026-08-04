#!/usr/bin/env python3
"""Seed Neo4j with healthcare test data from create-context-graph
and run local Semvec drift-detection sessions on top of it.

Semvec does all computation. Neo4j stores results for graph queries.

Usage:
    python3 scripts/seed_test_data.py [--fixtures /path/to/fixtures.json]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.core.embedder import SentenceTransformerEmbedder
from src.core.semvec_client import SemvecClient
from src.mcp.semvec_mcp_server import SemvecMCPServer
from src.persistence.models import Cluster, Region, AggregationStrategy
from src.persistence.neo4j_cluster_store import Neo4jClusterStore
from src.persistence.neo4j_region_store import Neo4jRegionStore
from src.persistence.adapter import Neo4jSemvecAdapter


_required = ("NEO4J_TEST_URI", "NEO4J_TEST_USER", "NEO4J_TEST_PASSWORD", "NEO4J_TEST_DATABASE")
_missing = [n for n in _required if not os.environ.get(n)]
if _missing:
    sys.stderr.write(
        "Missing environment variable(s): " + ", ".join(_missing) + "\n"
        "Run `cp .env.example .env` and fill in values.\n"
    )
    sys.exit(1)

NEO4J_URI = os.environ["NEO4J_TEST_URI"]
NEO4J_USER = os.environ["NEO4J_TEST_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_TEST_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_TEST_DATABASE"]

# Default to the fixtures bundled with this repo so a fresh clone is
# self-contained and produces the same patients/providers as the demo
# scenarios reference by name. Override via .env to point elsewhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_BUNDLED_FIXTURES = os.path.join(_PROJECT_ROOT, "healthcare-fixtures", "data", "fixtures.json")
_BUNDLED_SCHEMA = os.path.join(_PROJECT_ROOT, "healthcare-fixtures", "cypher", "schema.cypher")

FIXTURES_PATH = os.environ.get("FIXTURES_PATH") or _BUNDLED_FIXTURES
HEALTHCARE_SCHEMA_PATH = os.environ.get("HEALTHCARE_SCHEMA_PATH") or _BUNDLED_SCHEMA
SCHEMA_PATH = os.path.join(_PROJECT_ROOT, "schema", "semvec_schema.cypher")


def load_fixtures(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def seed_healthcare_graph(driver, fixtures: dict):
    """Load healthcare entities and relationships into Neo4j."""
    print("\n=== Seeding Healthcare Knowledge Graph ===")

    if os.path.exists(HEALTHCARE_SCHEMA_PATH):
        with open(HEALTHCARE_SCHEMA_PATH) as f:
            schema = f.read()
        with driver.session(database=NEO4J_DATABASE) as session:
            for stmt in schema.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("//"):
                    try:
                        session.run(stmt).consume()
                    except Exception:
                        pass
        print("  Healthcare schema applied")

    entities = fixtures.get("entities", {})
    total_entities = 0

    with driver.session(database=NEO4J_DATABASE) as session:
        for label, items in entities.items():
            for item in items:
                props = {k: v for k, v in item.items() if v is not None}
                prop_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"MERGE (n:{label} {{{prop_str}}}) RETURN n"
                session.run(query, **props).consume()
                total_entities += 1
            print(f"  {label}: {len(items)} entities")

    relationships = fixtures.get("relationships", [])
    rels_loaded = 0
    rels_skipped = 0
    with driver.session(database=NEO4J_DATABASE) as session:
        for rel in relationships:
            source_label = rel.get("source_label", "")
            target_label = rel.get("target_label", "")
            rel_type = rel.get("type", "RELATED_TO")
            # The fixtures use ``source_name`` / ``target_name``. The
            # earlier ``rel.get("source") / rel.get("target")`` lookups
            # always returned "" and silently dropped every edge.
            source_name = rel.get("source_name") or rel.get("source", "")
            target_name = rel.get("target_name") or rel.get("target", "")
            props = rel.get("properties", {})

            if not (source_name and target_name and source_label and target_label):
                rels_skipped += 1
                continue

            prop_str = ""
            if props:
                prop_str = " {" + ", ".join(f"{k}: ${k}" for k in props) + "}"
            query = (
                f"MATCH (a:{source_label} {{name: $source_name}}) "
                f"MATCH (b:{target_label} {{name: $target_name}}) "
                f"MERGE (a)-[:{rel_type}{prop_str}]->(b)"
            )
            try:
                session.run(query, source_name=source_name,
                            target_name=target_name, **props).consume()
                rels_loaded += 1
            except Exception as exc:
                rels_skipped += 1
                # Surface mismatched node labels rather than silently
                # dropping; the previous bug went undetected for months.
                print(f"  {rel_type} {source_label}:{source_name!r} -> "
                      f"{target_label}:{target_name!r}  skipped ({exc})")

    print(f"  {rels_loaded} relationships loaded "
          f"({rels_skipped} skipped of {len(relationships)})")
    print(f"  Total: {total_entities} entities, {rels_loaded} relationships")


def seed_agent_sessions(driver):
    """Create agent sessions using the local Semvec runtime for drift detection."""
    print("\n=== Seeding Agent Sessions (local Semvec runtime) ===")

    semvec = SemvecClient(embedder=SentenceTransformerEmbedder(), owner_subject="local-demo")
    health = semvec.health()
    print(f"  Semvec runtime: {health.get('status', 'unknown')} "
          f"(v{health.get('version', '?')}, {health.get('active_sessions', '?')} active sessions)")

    mcp = SemvecMCPServer(driver, database=NEO4J_DATABASE, semvec_client=semvec)
    cluster_store = Neo4jClusterStore(driver, database=NEO4J_DATABASE)
    region_store = Neo4jRegionStore(driver, database=NEO4J_DATABASE)

    # --- Agent 1: Cardiology researcher — stable topic ---
    s1 = mcp.create_agent_session(agent_id="cardiology-researcher")
    cardiology_queries = [
        "Patient heart failure diagnosis history and clinical presentation",
        "Heart failure treatment outcomes comparison across facilities",
        "Cardiac medication contraindications and drug interactions",
        "Heart failure readmission rates by facility and provider",
        "Cardiology referral network analysis between providers",
        "Heart failure patient demographics and risk factors",
        "Cardiac treatment efficacy comparison by medication class",
        "Heart failure comorbidity patterns and multi-morbidity analysis",
    ]
    for q in cardiology_queries:
        result = mcp.detect_drift(s1["session_id"], q)
        mcp.store_response(s1["session_id"], f"Analysis of: {q}")
    print(f"  Agent 1 (cardiology-researcher): {len(cardiology_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']}")

    # --- Agent 2: Oncology researcher — drifts to pharmacology ---
    s2 = mcp.create_agent_session(agent_id="oncology-researcher")
    onc_queries = [
        "Cancer diagnosis patterns across hospital facilities",
        "Oncology treatment success rates and survival analysis",
        "Chemotherapy side effects tracking and adverse events",
        "Cancer patient survival rates by treatment type and stage",
        "Tumor marker trending and prognostic value assessment",
        # DRIFT: topic switch to pharmacology
        "Medication interaction database analysis across all drug classes",
        "Drug contraindication patterns across all patient populations",
        "Pharmacy inventory optimization and supply chain management",
    ]
    for q in onc_queries:
        result = mcp.detect_drift(s2["session_id"], q)
        mcp.store_response(s2["session_id"], f"Research findings: {q}")
    print(f"  Agent 2 (oncology-researcher): {len(onc_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']} (topic switch at step 5)")

    # --- Agent 3: ER triage — high variance, rapid topic changes ---
    s3 = mcp.create_agent_session(agent_id="er-triage-agent")
    er_queries = [
        "Emergency patient assessment protocol for chest pain",
        "Bed availability across all hospital facilities now",
        "Critical diagnosis triage priority for incoming patient",
        "Provider availability for emergency surgery tonight",
        "Patient allergy check for penicillin before administration",
        "Referral to cardiology specialist for acute myocardial infarction",
        "Discharge planning for ER patient with stable vitals",
        "Infection control protocol activation for suspected COVID case",
        "Urgent lab result interpretation for troponin levels",
        "Resource allocation during patient surge event",
    ]
    for q in er_queries:
        result = mcp.detect_drift(s3["session_id"], q)
    print(f"  Agent 3 (er-triage-agent): {len(er_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']} (high variance)")

    # --- Agent 4: Care coordinator — moderate drift ---
    s4 = mcp.create_agent_session(agent_id="care-coordinator")
    care_queries = [
        "Multi-provider care coordination for chronic heart failure patient",
        "Provider referral network mapping for specialist access",
        "Treatment plan alignment across cardiology and pulmonology",
        "Patient follow-up scheduling optimization for post-discharge",
        "Care gap identification for chronic disease management",
        "Insurance coverage verification for cardiac rehabilitation",
    ]
    for q in care_queries:
        result = mcp.detect_drift(s4["session_id"], q)
    print(f"  Agent 4 (care-coordinator): {len(care_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']}")

    # --- Agent 5: Lab analyst — very stable ---
    s5 = mcp.create_agent_session(agent_id="lab-analyst")
    lab_queries = [
        "Lab test results for cardiac patient cohort analysis",
        "Lab result trends over time for troponin and BNP levels",
        "Lab test ordering patterns by provider and department",
        "Lab result interpretation for differential diagnosis",
        "Lab equipment utilization by facility and test volume",
    ]
    for q in lab_queries:
        result = mcp.detect_drift(s5["session_id"], q)
    print(f"  Agent 5 (lab-analyst): {len(lab_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']} (stable)")

    # --- Agent 6: Admin analyst — broad topics ---
    s6 = mcp.create_agent_session(agent_id="admin-analyst")
    admin_queries = [
        "Facility utilization rates comparison across network",
        "Provider workload distribution and burnout risk assessment",
        "Patient satisfaction survey analysis by department",
        "Cost analysis by treatment type and DRG category",
        "Readmission rate benchmarking against national averages",
        "Staff scheduling optimization for night shift coverage",
        "Supply chain medication tracking and shortage alerts",
    ]
    for q in admin_queries:
        result = mcp.detect_drift(s6["session_id"], q)
    print(f"  Agent 6 (admin-analyst): {len(admin_queries)} turns — "
          f"drift={result['drift_score']:.3f} phase={result['drift_phase']}")

    sessions = [s1, s2, s3, s4, s5, s6]

    # --- Store Memories (with pseudo-embeddings for Neo4j vector search) ---
    print("\n=== Storing Agent Memories ===")
    import math

    def _pseudo_embed(text: str, dim: int = 384) -> list[float]:
        vec = [0.0] * dim
        for i, ch in enumerate(text.encode("utf-8")):
            idx = (ch * (i + 1)) % dim
            vec[idx] += math.sin(ch * 0.1 + i * 0.01)
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    for s in sessions:
        queries = {
            "cardiology-researcher": cardiology_queries,
            "oncology-researcher": onc_queries,
            "er-triage-agent": er_queries,
            "care-coordinator": care_queries,
            "lab-analyst": lab_queries,
            "admin-analyst": admin_queries,
        }.get(s["agent_id"], [])

        for i, q in enumerate(queries[:5]):  # Top 5 as memories
            mcp.store_memory(
                s["session_id"], q,
                importance=round(0.9 - i * 0.1, 1),
                vector=_pseudo_embed(q),
            )
    print(f"  Stored memories for {len(sessions)} agents")

    # --- Create Clusters ---
    print("\n=== Creating Multi-Agent Topology ===")

    c_clinical = Cluster(name="clinical-research", strategy=AggregationStrategy.WEIGHTED_AVG, coupling_strength=0.3)
    c_operations = Cluster(name="operations", strategy=AggregationStrategy.WEIGHTED_AVG, coupling_strength=0.2)

    cluster_store.create_cluster(c_clinical)
    cluster_store.create_cluster(c_operations)

    cluster_store.add_member(c_clinical.cluster_id, s1["session_id"], weight=1.0)
    cluster_store.add_member(c_clinical.cluster_id, s2["session_id"], weight=0.9)
    cluster_store.add_member(c_clinical.cluster_id, s5["session_id"], weight=0.7)
    print(f"  Cluster 'clinical-research': cardiology + oncology + lab")

    cluster_store.add_member(c_operations.cluster_id, s3["session_id"], weight=1.0)
    cluster_store.add_member(c_operations.cluster_id, s4["session_id"], weight=0.8)
    cluster_store.add_member(c_operations.cluster_id, s6["session_id"], weight=0.6)
    print(f"  Cluster 'operations': ER + care-coord + admin")

    # --- Create Region ---
    region = Region(name="hospital-network", consensus_threshold=0.6)
    region_store.create_region(region)
    region_store.add_cluster_to_region(region.region_id, c_clinical.cluster_id)
    region_store.add_cluster_to_region(region.region_id, c_operations.cluster_id)
    print(f"  Region 'hospital-network': 2 clusters")

    # --- Print Summary ---
    print("\n=== Seed Summary ===")
    print(f"  {'Agent':25s} | {'Phase':15s} | {'Drift':>6s} | {'Events':>6s} | {'States':>6s}")
    print(f"  {'-'*25} | {'-'*15} | {'-'*6} | {'-'*6} | {'-'*6}")
    for s in sessions:
        sid = s["session_id"]
        phase = mcp.get_phase(sid)
        score = mcp.get_drift_score(sid)
        events = mcp.query_drift_history(sid)
        trajectory = mcp.get_state_trajectory(sid, steps=20)
        print(f"  {s['agent_id']:25s} | {phase.get('phase', 'N/A'):15s} | "
              f"{score['drift_score']:6.3f} | {len(events):6d} | {len(trajectory):6d}")

    print(f"\nDone! All data seeded via the local Semvec runtime.")
    print(f"  Neo4j:  {NEO4J_URI} / {NEO4J_DATABASE}")
    print(f"  Semvec: in-process (semvec=={semvec.health().get('version', '?')})")


def main():
    fixtures_path = FIXTURES_PATH
    if len(sys.argv) > 2 and sys.argv[1] == "--fixtures":
        fixtures_path = sys.argv[2]

    if not os.path.exists(fixtures_path):
        print(f"Fixtures not found at {fixtures_path}")
        print(
            "Repo ships bundled fixtures at "
            "healthcare-fixtures/data/fixtures.json — "
            "if those are missing, restore them with `git checkout HEAD -- "
            "healthcare-fixtures/`."
        )
        print(
            "Or generate fresh fixtures via "
            "`uvx create-context-graph healthcare-test-data --domain healthcare "
            "--framework pydanticai --demo-data` and override FIXTURES_PATH in .env."
        )
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"Cannot connect to Neo4j at {NEO4J_URI}: {e}")
        sys.exit(1)

    # Apply Semvec schema
    adapter = Neo4jSemvecAdapter(driver, database=NEO4J_DATABASE)
    adapter.apply_schema(SCHEMA_PATH)
    print("Semvec schema applied")

    # Load and seed healthcare data
    fixtures = load_fixtures(fixtures_path)
    seed_healthcare_graph(driver, fixtures)

    # Seed agent sessions via the local Semvec runtime
    seed_agent_sessions(driver)

    driver.close()


if __name__ == "__main__":
    main()

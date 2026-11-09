"""Shared test fixtures for the Semvec drift-detection suite."""

from __future__ import annotations

import os
import pytest
from neo4j import GraphDatabase

# Load .env (copy .env.example first if missing) — single source of truth
# for connection strings and credentials.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

NEO4J_URI = os.environ.get("NEO4J_TEST_URI")
NEO4J_USER = os.environ.get("NEO4J_TEST_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_TEST_DATABASE")

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "schema", "semvec_schema.cypher"
)


@pytest.fixture(scope="session")
def neo4j_driver():
    """Create a Neo4j driver for the test session."""
    missing = [k for k, v in {
        "NEO4J_TEST_URI": NEO4J_URI,
        "NEO4J_TEST_USER": NEO4J_USER,
        "NEO4J_TEST_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_TEST_DATABASE": NEO4J_DATABASE,
    }.items() if not v]
    if missing:
        pytest.skip(
            f"Missing env vars: {', '.join(missing)}. "
            f"Run `cp .env.example .env` and fill in values."
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        pytest.skip(f"Neo4j not available at {NEO4J_URI}: {e}")
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def neo4j_schema(neo4j_driver):
    """Apply the Semvec graph schema to the test database (once per session)."""
    with open(SCHEMA_PATH, "r") as f:
        schema_cypher = f.read()

    # Execute each statement separately (skip comments and empty lines)
    statements = []
    current = []
    for line in schema_cypher.split("\n"):
        stripped = line.strip()
        if stripped.startswith("//") or not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip(";"))
            current = []

    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        for stmt in statements:
            stmt = stmt.strip()
            if stmt:
                try:
                    session.run(stmt)
                except Exception:
                    pass  # Constraints/indexes may already exist

    return True


@pytest.fixture(autouse=True)
def clean_test_data(neo4j_driver, neo4j_schema):
    """Wipe Semvec-related Neo4j nodes before each test."""
    labels = [
        "AgentSession", "Agent", "SemanticState", "DriftEvent",
        "Phase", "Memory", "Cluster", "Region",
        "GlobalObserver", "ConsensusEvent", "AnomalyEvent", "LiteralFact",
        "ConsensusEngine", "ConsensusProposal", "ConsensusVote",
    ]
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        for label in labels:
            session.run(f"MATCH (n:{label}) DETACH DELETE n")
    yield


@pytest.fixture
def db_session(neo4j_driver):
    """Provide a Neo4j session for individual tests."""
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        yield session


def make_vector(dim: int = 768, value: float = 0.0) -> list[float]:
    """Create a test vector of given dimension."""
    return [value] * dim


def make_random_vector(dim: int = 768, seed: int = 42) -> list[float]:
    """Create a reproducible random-ish test vector with nonzero L2 norm."""
    import math
    # Use seed+1 offset to avoid zero vector when seed=0
    return [math.sin((i + 1) * (seed + 1) * 0.1) for i in range(dim)]

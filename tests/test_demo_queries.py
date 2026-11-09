# ABOUTME: Runs every GRAPH query of docs/demo_queries.md against a small marked fixture graph in Neo4j.
# ABOUTME: A graph query must not return nodes without one of its returned relationships (no floating nodes).

from pathlib import Path

import pytest
from neo4j.graph import Node, Relationship
from neo4j.graph import Path as GraphPath

from scripts.browser_guide import parse_queries

DEMO_QUERIES = Path(__file__).parent.parent / "docs" / "demo_queries.md"
SCHEMA_QUERY = "11."
MARK = "demo-query-test"

# Every node carries `_test: MARK`, so cleanup never touches seeded or demo data in the same database.
FIXTURE = """
CREATE (s:AgentSession {session_id: 'qt-s1', agent_id: 'qt-agent', status: 'active', _test: $mark})
CREATE (a:Agent {agent_id: 'qt-agent', name: 'qt-agent', _test: $mark})-[:RAN]->(s)
CREATE (s)-[:CURRENT_PHASE]->(:Phase {name: 'exploration', _test: $mark})
CREATE (c:Cluster {cluster_id: 'qt-c1', name: 'qt-cluster', _test: $mark})
CREATE (s)-[:MEMBER_OF {role: 'member'}]->(c)
CREATE (:Region {region_id: 'qt-r1', name: 'qt-region', _test: $mark})-[:CONTAINS_CLUSTER]->(c)
CREATE (cur:SemanticState {state_id: 'qt-st2', step: 1, mean_similarity: 0.5, _test: $mark})
CREATE (old:SemanticState {state_id: 'qt-st1', step: 0, mean_similarity: 0.2, _test: $mark})
CREATE (s)-[:CURRENT_STATE]->(cur)-[:STATE_HISTORY]->(old)
CREATE (old)-[:TRIGGERED]->(:DriftEvent {event_id: 'qt-d1', session_id: 'qt-s1', drift_score: 0.5,
        severity: 'MEDIUM', drift_phase: 'drifted', timestamp: datetime(), _test: $mark})
CREATE (p:Patient {name: 'QT Patient', _test: $mark})
CREATE (d:Diagnosis {name: 'QT Diagnosis', _test: $mark})
CREATE (t:Treatment {name: 'QT Treatment', _test: $mark})
CREATE (m1:Medication {name: 'QT Med A', _test: $mark})
CREATE (m2:Medication {name: 'QT Med B', _test: $mark})
CREATE (pr1:Provider {name: 'QT Provider A', _test: $mark})
CREATE (pr2:Provider {name: 'QT Provider B', _test: $mark})
CREATE (f:Facility {name: 'QT Facility', _test: $mark})
CREATE (e:Encounter {name: 'QT Encounter', _test: $mark})
CREATE (p)-[:DIAGNOSED_WITH]->(d), (t)-[:TREATS]->(d), (t)-[:USES]->(m1)
CREATE (m1)-[:CONTRAINDICATED_WITH]->(m2), (pr1)-[:PRESCRIBED]->(m1), (p)-[:TREATED_BY]->(pr1)
CREATE (pr1)-[:REFERRED_TO]->(pr2), (pr1)-[:AFFILIATED_WITH]->(f), (pr2)-[:AFFILIATED_WITH]->(f)
CREATE (p)-[:HAD_ENCOUNTER]->(e), (e)-[:OCCURRED_AT]->(f)
CREATE (s)-[:INVESTIGATED {step: 0}]->(p), (s)-[:INVESTIGATED {step: 1}]->(m1)
CREATE (s)-[:INVESTIGATED {step: 2}]->(pr1), (s)-[:INVESTIGATED {step: 3}]->(d)
"""

GRAPH_QUERIES = [q for q in parse_queries(DEMO_QUERIES.read_text(encoding="utf-8"))
                 if q.section.startswith("GRAPH") and not q.title.startswith(SCHEMA_QUERY)]


@pytest.fixture
def fixture_graph(neo4j_driver):
    cleanup = "MATCH (n {_test: $mark}) DETACH DELETE n"
    neo4j_driver.execute_query(cleanup, mark=MARK, database_="neo4j")
    neo4j_driver.execute_query(FIXTURE, mark=MARK, database_="neo4j")
    try:
        yield neo4j_driver
    finally:
        neo4j_driver.execute_query(cleanup, mark=MARK, database_="neo4j")


def _collect(value, nodes: dict, rels: dict) -> None:
    if isinstance(value, Node):
        nodes[value.element_id] = value
    elif isinstance(value, Relationship):
        rels[value.element_id] = value
    elif isinstance(value, GraphPath):
        nodes.update({n.element_id: n for n in value.nodes})
        rels.update({r.element_id: r for r in value.relationships})
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, nodes, rels)


def _graph_result(driver, cypher: str) -> tuple[dict, dict]:
    nodes: dict = {}
    rels: dict = {}
    for record in driver.execute_query(cypher, database_="neo4j").records:
        for value in record.values():
            _collect(value, nodes, rels)
    return nodes, rels


@pytest.mark.parametrize("query", GRAPH_QUERIES, ids=lambda q: q.title.split(" ")[0].rstrip("."))
def test_graph_query_returns_no_floating_nodes(fixture_graph, query):
    # Arrange / Act
    nodes, rels = _graph_result(fixture_graph, query.cypher)

    # Assert
    touched = {r.start_node.element_id for r in rels.values()} | {r.end_node.element_id for r in rels.values()}
    floating = sorted(next(iter(n.labels), "?") for nid, n in nodes.items() if nid not in touched)
    assert floating == [], f"{query.title}: nodes without a returned relationship: {floating}"


@pytest.mark.parametrize("number", ["14.", "15.", "17.", "18."])
def test_fixed_graph_queries_reach_the_fixture(fixture_graph, number):
    # Arrange
    query = next(q for q in GRAPH_QUERIES if q.title.startswith(number))

    # Act
    nodes, _ = _graph_result(fixture_graph, query.cypher)

    # Assert
    assert any(n.get("_test") == MARK for n in nodes.values()), f"{query.title} returned none of the fixture"


def test_drift_cascade_links_each_event_to_its_own_session(fixture_graph):
    # Arrange
    query = next(q for q in GRAPH_QUERIES if q.title.startswith("14."))

    # Act
    records = fixture_graph.execute_query(query.cypher, database_="neo4j").records
    paths = [v for r in records for v in r.values() if isinstance(v, GraphPath)]

    # Assert
    assert paths, "drift cascade must return paths"
    assert all(p.start_node["session_id"] == p.end_node["session_id"] for p in paths)

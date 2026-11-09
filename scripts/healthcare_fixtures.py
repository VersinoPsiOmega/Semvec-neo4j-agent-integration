# ABOUTME: Filters the generated healthcare fixtures and schema down to the domain ontology.
# ABOUTME: Drops the generic POLE+O base types and unused Document / DecisionTrace, also from existing DBs.

from __future__ import annotations

import re

from neo4j import Driver

# create-context-graph adds the POLE+O base types of healthcare-fixtures/data/_base.yaml to every
# domain. The demo, its queries and its scenarios never use them, and their sample data contradicts
# the domain data (e.g. "Nurse James Morrison" next to the patient James Morrison), so they are not
# imported. Document / DecisionTrace only get indexes in the generated schema, never any data.
EXCLUDED_LABELS = frozenset(
    {"Person", "Organization", "Location", "Event", "Object", "Document", "DecisionTrace"}
)

_STATEMENT_LABEL = re.compile(r"\(\s*\w*\s*:\s*(\w+)\s*\)")


def domain_entities(entities: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The fixture entities without the excluded labels, in their original order."""
    return {label: items for label, items in entities.items() if label not in EXCLUDED_LABELS}


def domain_relationships(relationships: list[dict]) -> list[dict]:
    """The fixture relationships whose source and target are both domain labels."""
    return [
        rel
        for rel in relationships
        if rel.get("source_label") not in EXCLUDED_LABELS and rel.get("target_label") not in EXCLUDED_LABELS
    ]


_SCHEMA_OBJECT = re.compile(r"^CREATE\s+(INDEX|CONSTRAINT)\s+(\w+)", re.IGNORECASE)


def _schema_statements(schema: str) -> list[tuple[str, bool]]:
    """Statements of a schema file without comment lines, each flagged whether it targets an excluded label."""
    statements = []
    for chunk in schema.split(";"):
        statement = "\n".join(line for line in chunk.splitlines() if not line.strip().startswith("//")).strip()
        if statement:
            excluded = bool(set(_STATEMENT_LABEL.findall(statement)) & EXCLUDED_LABELS)
            statements.append((statement, excluded))
    return statements


def domain_schema_statements(schema: str) -> list[str]:
    """Split a schema file into statements, drop comment lines and statements for excluded labels."""
    return [statement for statement, excluded in _schema_statements(schema) if not excluded]


def remove_excluded(driver: Driver, database: str, fixtures: dict, schema: str) -> None:
    """Remove what earlier seeds created for the excluded labels, and nothing else.

    Deletes only excluded-label nodes whose name appears in the fixtures, and drops only the
    indexes / constraints the generated schema declares for them, so unrelated data in the same
    database stays untouched. Constraints go first: a constraint's backing index cannot be
    dropped on its own.
    """
    for label, items in fixtures.get("entities", {}).items():
        if label in EXCLUDED_LABELS:
            names = [item["name"] for item in items if item.get("name")]
            driver.execute_query(
                f"MATCH (n:`{label}`) WHERE n.name IN $names DETACH DELETE n", names=names, database_=database
            )
    declared = [_SCHEMA_OBJECT.match(s) for s, excluded in _schema_statements(schema) if excluded]
    for kind in ("CONSTRAINT", "INDEX"):
        for match in declared:
            if match and match.group(1).upper() == kind:
                driver.execute_query(f"DROP {kind} `{match.group(2)}` IF EXISTS", database_=database)

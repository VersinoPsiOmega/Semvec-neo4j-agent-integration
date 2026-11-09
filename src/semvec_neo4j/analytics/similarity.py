"""Cross-session semantic state similarity via Neo4j vector index.

Finds sessions with similar current states, enabling "experience transfer"
between agents that are in similar conversation contexts.
"""

from __future__ import annotations

import numpy as np
from neo4j import Driver


class SimilarityAnalyzer:
    def __init__(self, driver: Driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def find_similar_sessions(
        self, session_id: str, top_k: int = 5
    ) -> list[dict]:
        """Find sessions with similar current semantic states.

        Uses Neo4j's native vector index for O(log n) similarity search.
        """
        query = """
        MATCH (target:AgentSession {session_id: $session_id})
              -[:CURRENT_STATE]->(target_state:SemanticState)
        CALL db.index.vector.queryNodes('semantic_state_vector', $top_k, target_state.vector)
        YIELD node AS similar_state, score
        MATCH (other:AgentSession)-[:CURRENT_STATE]->(similar_state)
        WHERE other.session_id <> $session_id
        RETURN other.session_id AS session_id,
               other.agent_id AS agent_id,
               score AS similarity,
               similar_state.step AS step
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_id=session_id, top_k=top_k + 1)
            return [
                {
                    "session_id": r["session_id"],
                    "agent_id": r["agent_id"],
                    "similarity": r["similarity"],
                    "step": r["step"],
                }
                for r in result
            ]

    def compute_similarity_matrix(
        self, session_ids: list[str]
    ) -> list[list[float]]:
        """Compute pairwise cosine similarity matrix for given sessions.

        Returns N×N matrix where matrix[i][j] is cosine similarity between
        session i and session j's current states.
        """
        # Fetch all current state vectors
        query = """
        UNWIND $session_ids AS sid
        MATCH (s:AgentSession {session_id: sid})-[:CURRENT_STATE]->(state:SemanticState)
        RETURN sid AS session_id, state.vector AS vector
        ORDER BY sid
        """
        with self._driver.session(database=self._database) as db:
            result = db.run(query, session_ids=session_ids)
            records = list(result)

        # Build ID-to-vector mapping
        vectors = {}
        for r in records:
            vectors[r["session_id"]] = np.array(r["vector"])

        n = len(session_ids)
        matrix = [[0.0] * n for _ in range(n)]

        for i, sid_i in enumerate(session_ids):
            for j, sid_j in enumerate(session_ids):
                if sid_i not in vectors or sid_j not in vectors:
                    matrix[i][j] = 0.0
                elif i == j:
                    matrix[i][j] = 1.0
                else:
                    v1 = vectors[sid_i]
                    v2 = vectors[sid_j]
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 == 0 or norm2 == 0:
                        matrix[i][j] = 0.0
                    else:
                        matrix[i][j] = float(np.dot(v1, v2) / (norm1 * norm2))

        return matrix

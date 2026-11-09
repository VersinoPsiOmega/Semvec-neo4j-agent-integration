"""ClusterManager — Semvec Layer 2 cluster aggregation and management.

Computes weighted-average or attention-based aggregation of member states
and persists the result back to Neo4j.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from semvec_neo4j.persistence.models import AggregationStrategy, Cluster, SemanticState
from semvec_neo4j.persistence.neo4j_cluster_store import Neo4jClusterStore
from semvec_neo4j.persistence.neo4j_state_store import Neo4jStateStore


class ClusterManager:
    def __init__(
        self,
        cluster_store: Neo4jClusterStore,
        state_store: Neo4jStateStore,
    ):
        self._cluster_store = cluster_store
        self._state_store = state_store

    def aggregate(self, cluster_id: str) -> Optional[SemanticState]:
        """Compute aggregated state from all cluster members.

        Uses the cluster's configured strategy (weighted_avg or attention).
        Persists the result as the cluster's AGGREGATED_STATE.
        """
        cluster = self._cluster_store.get_cluster(cluster_id)
        if cluster is None:
            return None

        member_states = self._cluster_store.get_member_states(cluster_id)
        if not member_states:
            return None

        if cluster.strategy == AggregationStrategy.WEIGHTED_AVG:
            agg_vector = self._weighted_average(member_states)
        else:
            agg_vector = self._attention_aggregate(member_states)

        # Compute aggregate metrics
        betas = [s.beta for s, _ in member_states]
        mean_sims = [s.mean_similarity for s, _ in member_states]
        variances = [s.variance for s, _ in member_states]

        agg_state = SemanticState(
            vector=agg_vector.tolist(),
            step=max(s.step for s, _ in member_states),
            beta=float(np.mean(betas)),
            mean_similarity=float(np.mean(mean_sims)),
            variance=float(np.mean(variances)),
        )

        self._cluster_store.set_aggregated_state(cluster_id, agg_state)
        return agg_state

    def _weighted_average(
        self, member_states: list[tuple[SemanticState, float]]
    ) -> np.ndarray:
        """G2a: Weighted average aggregation."""
        vectors = np.array([s.vector for s, _ in member_states])
        weights = np.array([w for _, w in member_states])

        if np.sum(weights) == 0:
            weights = np.ones(len(member_states))

        weights = weights / np.sum(weights)
        return np.average(vectors, axis=0, weights=weights)

    def _attention_aggregate(
        self, member_states: list[tuple[SemanticState, float]]
    ) -> np.ndarray:
        """G2b: Attention-based aggregation (simplified Q/K/V)."""
        vectors = np.array([s.vector for s, _ in member_states])
        dim = vectors.shape[1]

        # Use mean as query
        query = np.mean(vectors, axis=0)

        # Simple dot-product attention
        scores = vectors @ query
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        return np.sum(attention_weights[:, np.newaxis] * vectors, axis=0)

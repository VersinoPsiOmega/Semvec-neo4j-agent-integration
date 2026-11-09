"""Memory deduplication via vector similarity.

Identifies near-duplicate memories across a session's memory tiers
and merges them, keeping the higher-importance version.
"""

from __future__ import annotations

import numpy as np

from semvec_neo4j.persistence.models import MemoryTier
from semvec_neo4j.persistence.neo4j_memory_store import Neo4jMemoryStore


class MemoryDeduplicator:
    def __init__(self, memory_store: Neo4jMemoryStore):
        self._memory_store = memory_store

    def find_duplicates(
        self,
        session_id: str,
        similarity_threshold: float = 0.95,
        tier: MemoryTier | None = None,
    ) -> list[tuple[str, str, float]]:
        """Find near-duplicate memory pairs above the similarity threshold.

        Returns list of (memory_id_1, memory_id_2, similarity) tuples.
        """
        tiers = [tier] if tier else [MemoryTier.SHORT, MemoryTier.MEDIUM, MemoryTier.LONG]
        all_memories = []
        for t in tiers:
            all_memories.extend(
                self._memory_store.get_memories_by_tier(session_id, t, limit=200)
            )

        if len(all_memories) < 2:
            return []

        # Compute pairwise similarities
        duplicates = []
        seen = set()

        for i, mem_i in enumerate(all_memories):
            if not mem_i.content_vector:
                continue
            vec_i = np.array(mem_i.content_vector)
            norm_i = np.linalg.norm(vec_i)
            if norm_i == 0:
                continue

            for j in range(i + 1, len(all_memories)):
                mem_j = all_memories[j]
                if not mem_j.content_vector:
                    continue

                pair_key = (min(mem_i.memory_id, mem_j.memory_id),
                            max(mem_i.memory_id, mem_j.memory_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                vec_j = np.array(mem_j.content_vector)
                norm_j = np.linalg.norm(vec_j)
                if norm_j == 0:
                    continue

                sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                if sim >= similarity_threshold:
                    duplicates.append((mem_i.memory_id, mem_j.memory_id, sim))

        return duplicates

    def deduplicate(
        self,
        session_id: str,
        similarity_threshold: float = 0.95,
    ) -> int:
        """Find and merge duplicate memories.

        For each duplicate pair, keeps the one with higher importance
        and deletes the other. Returns number of memories merged (deleted).
        """
        duplicates = self.find_duplicates(session_id, similarity_threshold)
        if not duplicates:
            return 0

        # Build a map of memory_id -> importance for quick lookup
        all_tiers = [MemoryTier.SHORT, MemoryTier.MEDIUM, MemoryTier.LONG]
        importance_map = {}
        for t in all_tiers:
            for mem in self._memory_store.get_memories_by_tier(session_id, t, limit=200):
                importance_map[mem.memory_id] = mem.importance

        deleted = set()
        merged_count = 0

        for m1_id, m2_id, sim in duplicates:
            if m1_id in deleted or m2_id in deleted:
                continue

            imp1 = importance_map.get(m1_id, 0)
            imp2 = importance_map.get(m2_id, 0)

            # Delete the lower-importance one
            to_delete = m2_id if imp1 >= imp2 else m1_id
            to_keep = m1_id if imp1 >= imp2 else m2_id

            # Increment access count on the kept memory
            self._memory_store.update_memory_access(to_keep)
            self._memory_store.delete_memory(to_delete)

            deleted.add(to_delete)
            merged_count += 1

        return merged_count

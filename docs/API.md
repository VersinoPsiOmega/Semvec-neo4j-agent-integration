# SemvecClient — Local API Reference

`SemvecClient` is the in-process client this project uses to drive the
bundled [`semvec`](https://pypi.org/project/semvec/) runtime. Every call
returns plain Python dictionaries; no network, no API key.

For the full Semvec runtime documentation see
<https://semvec-docs.pages.dev/>.

```python
from src.core.embedder import SentenceTransformerEmbedder
from src.core.semvec_client import SemvecClient

semvec = SemvecClient(
    embedder=SentenceTransformerEmbedder(),
    owner_subject="my-license-subject",
)
```

When `embedder` is omitted, Semvec lazily loads
`sentence-transformers` with the default 384-dimensional MiniLM model.
For tests and offline runs, swap in `HashEmbedder(dimension=…)` from
`src.core.embedder`.

`owner_subject` is required since semvec 0.8.7: it is the license
subject that owns every session, observer, and network partition the
client creates, and session ownership is checked against it on
observer and network operations. It is keyword-only and has no
default — a forgotten argument fails with a `TypeError` instead of
silently creating a permissive anonymous client. Pass `None`
explicitly to opt into anonymous/unowned mode (user partitions such
as `switch_user` refuse `None` at call time).

---

## Health

| Method | Returns |
|---|---|
| `health()` | `{status, active_sessions, version}` |

## Layer 1 — Run / Store

| Method | Notes |
|---|---|
| `run(message, session_id=None, response=None, short_circuit_threshold=0.85, reset_context=False)` | Runs one turn. Returns `session_id, context, top_similarity, short_circuit, drift_score, drift_detected, drift_phase`. Pass `response=` to inline-store the previous LLM answer. |
| `store(session_id, response)` | Persist the latest LLM answer for the session. |

## Layer 1b — Session control

| Method | Returns |
|---|---|
| `create_session(dimension, model_name, use_cortex=False, enable_topic_switch=True)` | `{session_id, created}` |
| `export_session(session_id)` | `{session_id, state_dict, checksum}` |
| `import_session(session_id, state_dict)` | `{session_id, imported}` |
| `add_anchor(session_id, embedding)` | `{session_id, anchor_count}` |
| `get_anchor_score(session_id)` | `{session_id, anchor_score, …}` |
| `inject_memory(session_id, embedding, text, tier="short_term", importance=0.5)` | `{session_id, tier, total_memories}` |
| `set_isolation(session_id, level, similarity_threshold=0.7)` | `{session_id, level}` (`OPEN` / `FILTER` / `QUARANTINE` / `LOCKDOWN`) |
| `add_trigger(session_id, keyword=None, embedding=None, threshold=0.8)` | `{session_id, trigger_count}` |

## Layer 2 — Cluster

| Method | Returns |
|---|---|
| `create_cluster(name, aggregation_mode="weighted_average", coupling_factor=0.0)` | `{cluster_id, name, aggregation_mode, coupling_factor}` |
| `list_clusters()` | `[{cluster_id, name, member_count}, …]` |
| `get_cluster(cluster_id)` | full cluster state dict |
| `delete_cluster(cluster_id)` | `{deleted: True}` |
| `cluster_run(cluster_id, message, response=None, short_circuit_threshold=None)` | Same shape as `run`. |
| `cluster_store(cluster_id, message, response)` | Persists a Q&A pair into the cluster. |
| `cluster_feedback(cluster_id)` | `{cluster_id, sessions_updated}` |
| `add_cluster_member(cluster_id, session_id)` | `{cluster_id, session_id, added}` |
| `remove_cluster_member(cluster_id, session_id)` | `{cluster_id, session_id, removed}` |

## Layer 3 — Region

| Method | Returns |
|---|---|
| `create_region(name, consensus_threshold=0.5, vote_window_seconds=60.0)` | `{region_id, name, meta_session_id}` |
| `list_regions()` | `[{region_id, name, cluster_count}, …]` |
| `get_region(region_id)` | full region state dict |
| `delete_region(region_id)` | `{deleted: True}` |
| `add_region_cluster(region_id, cluster_id)` | `{added: True}` |
| `remove_region_cluster(region_id, cluster_id)` | `{removed: True}` |
| `get_region_events(region_id, limit=20)` | `[{cluster_id, region_id, drift_score, drift_phase, timestamp}, …]` |

## Layer 4 — Global Observer

| Method | Returns |
|---|---|
| `create_observer(sample_interval_seconds=30.0, region_ids=None)` | `{observer_id, registered_regions, meta_session_id}` |
| `get_observer_summary()` | summary dict |
| `observer_sample()` | `{sampled_at, clusters_sampled, regions_sampled, sample_duration_ms, new_anomalies}` |
| `get_anomalies(limit=20)` | list of anomaly dicts |
| `clear_anomalies()` | `{cleared: int}` |

## Layer 5 — Network

| Method | Returns |
|---|---|
| `transfer_delta(source_session_id, target_session_id, max_weight=0.15)` | `{delta_norm, …}` |
| `switch_user(user_id)` | `{active_user, previous_user}` |
| `get_active_user()` | `{active_user}` |
| `propose_consensus(proposer_session_id, target_embedding)` | `{accepted, trust_scores}` |
| `get_trust_scores()` | `{trust_scores: {user_id: float}}` |

---

## End-to-end example

```python
from src.core.embedder import SentenceTransformerEmbedder
from src.core.semvec_client import SemvecClient

semvec = SemvecClient(
    embedder=SentenceTransformerEmbedder(),
    owner_subject="my-license-subject",
)

turn1 = semvec.run("Patient with chest pain, ECG shows ST elevation")
sid = turn1["session_id"]

# … your LLM answers …
turn2 = semvec.run(
    "Confirm STEMI and start antiplatelet therapy",
    session_id=sid,
    response="Aspirin 325mg PO + Ticagrelor 180mg loading dose.",
)
print(turn2["drift_phase"], turn2["drift_score"])
```

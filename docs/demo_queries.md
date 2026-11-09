# Neo4j Browser Demo Queries

Open http://localhost:7474 (`docker compose up -d`) and log in as `neo4j` with
`NEO4J_TEST_PASSWORD` from `.env`.

These queries open automatically after login as the guide "Semvec × Neo4j demo queries"
(one query per step, with buttons to run or edit it). The guide is rendered from this file by
`scripts/browser_guide.py` (the `guide` service), so edits here show up on the next login.
To reopen it, enter `:play http://localhost:7477/semvec-demo.html`. Or copy each query into
the top bar and hit **Play** (or Ctrl+Enter).
Switch between **Graph** / **Table** view with the icons on the left of the result.

These queries match **Scenario 7 (Neo4j Cypher Explorer)** in `interactive_demo.py`.
Run scenarios 1–6 first to populate the graph with Semvec + healthcare data.

---

## TABLE Queries

### 1. Database overview — node types and counts

```cypher
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS count
ORDER BY count DESC
```

### 2. Agent dashboard — all sessions with phase + drift stats

```cypher
MATCH (s:AgentSession)
OPTIONAL MATCH (s)-[:CURRENT_PHASE]->(p:Phase)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)
OPTIONAL MATCH (d:DriftEvent {session_id: s.session_id})
WITH s, p, st, count(d) AS drift_events,
     round(coalesce(avg(d.drift_score), 0) * 1000) / 1000 AS avg_drift
RETURN s.agent_id AS agent, s.status AS status,
       coalesce(p.name, 'N/A') AS phase, coalesce(st.step, 0) AS steps,
       drift_events, avg_drift
ORDER BY drift_events DESC
```

### 3. Ward round (Sc.3) — who investigated David Park?

```cypher
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(e)
WHERE s.agent_id IN ['chen-baseline', 'volkov-pulm', 'tanaka-cardio']
RETURN s.agent_id AS doctor, inv.phase AS role, labels(e)[0] AS entity_type,
       e.name AS entity, inv.step AS step
ORDER BY s.agent_id, inv.step
```

### 4. Cluster members — which agents belong to which cluster?

```cypher
MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster)
RETURN c.name AS cluster, s.agent_id AS agent, m.role AS role, m.facility AS facility
ORDER BY c.name, s.agent_id
```

### 5. Drift events — all events with severity + trigger step

```cypher
MATCH (st:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN d.severity AS severity, round(d.drift_score * 100) / 100 AS score,
       d.drift_phase AS phase, st.step AS at_step,
       round(st.mean_similarity * 100) / 100 AS sim_at_trigger
ORDER BY d.timestamp DESC
```

### 6. Phase transitions — Markov matrix

```cypher
MATCH (p1:Phase)-[:TRANSITIONED_TO]->(p2:Phase)
RETURN p1.name AS from_phase, p2.name AS to_phase, count(*) AS count
ORDER BY count DESC
```

### 7. Investigation summary — which patients were investigated by which agents?

```cypher
MATCH (s:AgentSession)-[:INVESTIGATED]->(p:Patient)
WITH s, p, count(*) AS touches
RETURN s.agent_id AS agent, p.name AS patient, touches
ORDER BY patient, agent
```

### 8. Medication safety — all contraindications in the graph

```cypher
MATCH (m1:Medication)-[:CONTRAINDICATED_WITH]->(m2:Medication)
OPTIONAL MATCH (prov:Provider)-[:PRESCRIBED]->(m1)
RETURN m1.name AS drug_1, m2.name AS drug_2, collect(DISTINCT prov.name) AS prescribed_by
ORDER BY drug_1
```

### 9. Semvec topology — clusters, regions, members (run Sc.3/5 first)

```cypher
MATCH (s:AgentSession)
OPTIONAL MATCH (s)-[m:MEMBER_OF]->(c:Cluster)
OPTIONAL MATCH (r:Region)-[:CONTAINS_CLUSTER]->(c)
RETURN s.agent_id AS agent, c.name AS cluster, m.role AS role,
       m.facility AS facility, r.name AS region
ORDER BY cluster, agent
```

### 10. Region/cluster hierarchy — full hospital network topology

```cypher
MATCH (c:Cluster)
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
ORDER BY region, cluster, agent
```

---

## GRAPH Queries

### 11. Schema visualization — all node types and their relationships

```cypher
CALL db.schema.visualization()
```

### 12. Ward round (Sc.3) — 3 doctors -> Cluster -> David Park + diagnoses

```cypher
MATCH (s:AgentSession)-[m:MEMBER_OF]->(c:Cluster {name: 'park-ward-round'})
MATCH (s)-[inv:INVESTIGATED]->(e)
OPTIONAL MATCH (e)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED','CONTRAINDICATED_WITH']
RETURN s, m, c, inv, e, r1, connected
```

### 13. Hospital network — Region -> Clusters -> Agents -> Facilities

```cypher
MATCH (c:Cluster)
OPTIONAL MATCH (r:Region)-[rc:CONTAINS_CLUSTER]->(c)
OPTIONAL MATCH (s:AgentSession)-[m:MEMBER_OF]->(c)
OPTIONAL MATCH (s)-[inv:INVESTIGATED]->(e)
WHERE labels(e)[0] IN ['Patient', 'Facility', 'Diagnosis']
RETURN r, rc, c, m, s, inv, e
```

### 14. Drift cascade — states that triggered drift events

```cypher
MATCH path = (s:AgentSession)-[:CURRENT_STATE]->(:SemanticState)
             -[:STATE_HISTORY*0..]->(:SemanticState)-[:TRIGGERED]->(d:DriftEvent)
RETURN path
```

### 15. Agent x Patient x Drift — Semvec investigation + clinical graph + drift events

```cypher
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat:Patient)
MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (treat:Treatment)-[treats:TREATS]->(diag)
OPTIONAL MATCH (treat)-[uses:USES]->(med:Medication)
OPTIONAL MATCH drift = (s)-[:CURRENT_STATE]->(:SemanticState)
                       -[:STATE_HISTORY*0..]->(:SemanticState)-[:TRIGGERED]->(:DriftEvent)
OPTIONAL MATCH (s)-[cp:CURRENT_PHASE]->(phase:Phase)
OPTIONAL MATCH (s)-[mo:MEMBER_OF]->(cluster:Cluster)
RETURN s, inv, pat, dx, diag, tb, prov, treats, treat, uses, med,
       drift, cp, phase, mo, cluster
```

### 16. Clinical network — patients -> diagnoses <- treatments -> medications

```cypher
MATCH (pat:Patient)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (treat:Treatment)-[tr:TREATS]->(diag)
OPTIONAL MATCH (treat)-[u:USES]->(med:Medication)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
RETURN pat, dx, diag, tr, treat, u, med, tb, prov
```

### 17. Medication safety x Semvec — contraindications, who investigated them?

```cypher
MATCH (m1:Medication)-[ci:CONTRAINDICATED_WITH]->(m2:Medication)
OPTIONAL MATCH (prov:Provider)-[rx:PRESCRIBED]->(m1)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(m1)
OPTIONAL MATCH (s)-[cp:CURRENT_PHASE]->(phase:Phase)
RETURN m1, ci, m2, prov, rx, s, inv, cp, phase
```

### 18. Provider referral x Semvec — referrals + which agents investigated each provider

```cypher
MATCH (p1:Provider)-[ref:REFERRED_TO]->(p2:Provider)
OPTIONAL MATCH (p1)-[a1:AFFILIATED_WITH]->(f1:Facility)
OPTIONAL MATCH (p2)-[a2:AFFILIATED_WITH]->(f2:Facility)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(p1)
OPTIONAL MATCH (s)-[mo:MEMBER_OF]->(cluster:Cluster)
RETURN p1, ref, p2, a1, f1, a2, f2, s, inv, mo, cluster
```

### 19. Patient journey x Semvec — encounters + which agents tracked each step

```cypher
MATCH (pat:Patient)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (enc)-[ri:RESULTED_IN]->(diag:Diagnosis)
OPTIONAL MATCH (enc)-[oa:OCCURRED_AT]->(fac:Facility)
OPTIONAL MATCH (prov:Provider)-[att:ATTENDED]->(enc)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat)
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(drift:DriftEvent)
RETURN pat, he, enc, ri, diag, oa, fac, prov, att, s, inv, st, tr, drift
```

### 20. James Morrison — full clinical picture + agent investigations

```cypher
MATCH (pat:Patient {name: 'James Morrison'})
OPTIONAL MATCH (pat)-[dx:DIAGNOSED_WITH]->(diag:Diagnosis)
OPTIONAL MATCH (pat)-[tb:TREATED_BY]->(prov:Provider)
OPTIONAL MATCH (prov)-[rx:PRESCRIBED]->(med:Medication)
OPTIONAL MATCH (pat)-[he:HAD_ENCOUNTER]->(enc:Encounter)
OPTIONAL MATCH (s:AgentSession)-[inv:INVESTIGATED]->(pat)
RETURN pat, dx, diag, tb, prov, rx, med, he, enc, s, inv
```

### 21. Full story — all agents + all healthcare entities + drift events

```cypher
MATCH (s:AgentSession)-[inv:INVESTIGATED]->(entity)
OPTIONAL MATCH (entity)-[r1]->(connected)
WHERE type(r1) IN ['DIAGNOSED_WITH','TREATED_BY','PRESCRIBED',
                    'HAD_ENCOUNTER','AFFILIATED_WITH','CONTRAINDICATED_WITH']
OPTIONAL MATCH (s)-[:CURRENT_STATE]->(st:SemanticState)-[tr:TRIGGERED]->(d:DriftEvent)
RETURN s, inv, entity, r1, connected, st, tr, d
```

### 22. Everything — complete graph (limit 300)

```cypher
MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 300
```

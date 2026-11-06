# Scaling concept: Google Cloud (GKE) for 10,000 users

> **Status: concept only — not implemented, not deployed, not load-tested.** Recorded 2026-11-05 at the user's
> request as the target design (ADR-021, Proposed). The running system is the local Docker setup in
> `docker-compose.yml`. Numbers below are planning estimates from local measurements; every sizing and cost figure
> must be confirmed with a load test and the GCP pricing calculator before any decision.

## 1. Goals and assumptions

| Item | Value | Source |
|---|---|---|
| Registered users | 10,000 | user requirement |
| Design load ("Busy") | ~20 questions/s at peak (30% of users active, 20 questions/hour) | assumption, to confirm |
| Stress test load | ~170 questions/s (all users ask within one minute) | assumption |
| Answers in flight | ≈ QPS × 6 s → ~120 at design load, ~1,000 at stress | measured answer time ~6 s |
| Proposed SLOs (pending, ADR-007) | first text P95 < 1.5 s · full answer P95 < 8 s · retrieval P95 < 300 ms | proposal |
| Scope | public documents only (DailyMed, PMC, LiverTox, FDA); **no patient data**; informational, not medical advice | CLAUDE.md |

Measured on the dev laptop (4 cores, no GPU): question embedding 105–185 ms, BM25 + vector search 40–100 ms,
follow-up rewrite ~3.2 s, answer generation 2.6–4.0 s (Sonnet 5.5), ClickHouse/Redis < 20 ms. **The language model is
~70% of latency**; retrieval only becomes a bottleneck through per-worker CPU embedding at high concurrency.

## 2. Target architecture on GCP

```
 Users ──HTTPS──► Cloud DNS ─► Global external Application Load Balancer ─► Cloud Armor (WAF, rate limits)
                                   │  /assets, /           │  /api, /info, /chunks (SSE, timeout ≥ 120 s)
                                   ▼                       ▼
                      Cloud Storage + Cloud CDN     GKE regional cluster (3 zones, private nodes)
                      (built React app)             ┌───────────────────────────────────────────────┐
                                                    │ ns medrag-app                                  │
                                                    │  api            Deployment + HPA (stateless)  │
                                                    │  embedder       Deployment + HPA (ONNX / GPU) │
                                                    │  llm-gateway    concurrency limit + queue     │
                                                    │ ns medrag-data                                 │
                                                    │  opensearch     StatefulSet ×3 (operator)     │
                                                    │  clickhouse     2 replicas + Keeper ×3        │
                                                    │ ns observability                               │
                                                    │  langfuse web/worker, OTel collector          │
                                                    │ ns medrag-jobs                                 │
                                                    │  ingest / chunk / embed / index  (Jobs)       │
                                                    └───────────────────────────────────────────────┘
                                                       │            │             │
                              Memorystore for Redis (HA)  Cloud SQL Postgres   Cloud Storage buckets
                              short-term memory,          (Langfuse metadata)  raw / processed data,
                              rate limits, answer cache                        Langfuse blobs, backups
                                                       │
                              Cloud NAT (static egress IP) ─► Claude API   — or — Claude on Vertex AI
                              Secret Manager (API keys) · Artifact Registry (images) · Cloud Logging / Monitoring
```

### Component mapping (local → GCP)

| Local today | GCP concept | Why |
|---|---|---|
| FastAPI single process | **GKE Deployment `api`**, async workers, HPA | stateless; scale horizontally |
| bge model inside each API worker | **`embedder` Deployment** (ONNX int8 on CPU, or 1 NVIDIA L4 GPU node pool), request batching | removes ~1 GB RAM + CPU per API pod; 10–50 ms per question |
| OpenSearch 3.8.0 single node | **OpenSearch StatefulSet ×3** via the OpenSearch Kubernetes operator, pd-ssd volumes, 1 replica per shard | HA; full corpus (~2–3 M chunks, ~9 GB vectors) fits 3 × 32 GB nodes; consider byte/fp16 vectors (ADR-012 open) |
| Redis container | **Memorystore for Redis** (Standard tier, HA) | managed failover; keys are tiny (~10 KB per active chat) |
| ClickHouse container | **ClickHouse on GKE** (Altinity operator): 2 replicas + ClickHouse Keeper ×3, or ClickHouse Cloud on GCP | chat history; use `async_insert=1`; session list cached in Redis |
| Claude API from the laptop | **`llm-gateway`** (in-process at first) → Claude API via Cloud NAT, *or* **Claude on Vertex AI** | concurrency cap, queue, retries, per-user quotas; Vertex keeps billing/IAM in GCP (check model availability per region) |
| `.env` secrets | **Secret Manager** + Workload Identity (no keys in pods or images) | CLAUDE.md rule 6 |
| scripts/phase1–3 | **Kubernetes Jobs / CronJobs** reading/writing **Cloud Storage**; new index + alias swap (already how `build_index.py` works) | zero-downtime re-index |
| logs | **Cloud Logging + Managed Service for Prometheus**, **Langfuse** for traces/cost | P50/P95 per stage; SLO alerts |

### Request path and latency budget (design load)

| Step | Target P95 | Notes |
|---|---|---|
| LB + API | < 30 ms | keep SSE connections open; no sticky sessions needed |
| Redis memory + rate limit | < 10 ms | Memorystore in the same region |
| Follow-up rewrite | < 800 ms | skip for standalone questions; Haiku-class model for the rest |
| Embedding service | < 50 ms | batched ONNX/GPU |
| OpenSearch BM25 + kNN | < 150 ms | HNSW, filters, 1 replica |
| **First answer text** | **< 1.5 s** | requires streaming plain text (today the answer is returned in one piece) |
| Full answer | < 8 s | Sonnet 5.5, k ≈ 6–8 passages |
| ClickHouse write | async | does not block the response |

## 3. Sizing (planning estimates for ~20 questions/s; confirm by load test)

| Workload | Nodes / pods | Scaling rule |
|---|---|---|
| `api` | 4–12 pods × (2 vCPU, 2 GB) on an n2-standard node pool | HPA on CPU 60% **and** in-flight streams per pod (custom metric, ~100 streams/pod) |
| `embedder` | 2–4 pods × 4 vCPU (ONNX) **or** 1–2 pods on an L4 GPU pool | HPA on request latency / queue length |
| `opensearch` | 3 × n2-highmem-4 (32 GB), 200 GB pd-ssd each | manual; add data nodes as the corpus grows |
| `clickhouse` | 2 × n2-standard-4 + 3 small Keeper pods | manual |
| Memorystore | 5 GB Standard (HA) | — |
| Langfuse | web 2 pods, worker 2 pods; Cloud SQL (small), its own ClickHouse database | HPA on CPU |
| Claude API | ~120 concurrent answers × ~4.5 k input tokens → ~6 M input tokens/min at peak | needs a higher Anthropic rate-limit tier or Vertex quota |

Indicative cost drivers: the **LLM dominates** (≈ $0.013 per answer with Sonnet 5.5 at today's prompt size, plus ≈
$0.003 per follow-up rewrite); infrastructure is a fixed monthly base. No infrastructure price is quoted here
because it depends on region, commitments and the final sizing.

## 4. Reliability and operations

- **Regional GKE cluster** across 3 zones; PodDisruptionBudgets; topology spread for `api`, `embedder`, OpenSearch.
- **Readiness** on `/health` (503 when OpenSearch is not usable); the embedding model is loaded before a pod is ready.
- **Graceful shutdown for streaming:** `terminationGracePeriodSeconds: 120` and a preStop delay so open answers finish.
- **Load balancer:** backend timeout ≥ 120 s for SSE; connection draining 120 s.
- **Index updates:** build a new versioned index in a Job, validate counts, then move the `medrag-chunks` alias (rollback = move it back).
- **Backups:** OpenSearch snapshots and ClickHouse backups to Cloud Storage; Redis holds only rebuildable data.
- **Releases:** GitHub Actions → Artifact Registry (images pinned by digest) → rolling update or canary (Cloud Deploy / Argo CD).

## 5. Security and data

- Private GKE nodes, Cloud NAT for egress, NetworkPolicies between namespaces, Private Service Connect to Memorystore and Cloud SQL.
- Cloud Armor: WAF rules and per-IP rate limits in front of the per-user Redis limits.
- Identity: anonymous cookie today; Identity Platform or IAP for sign-in later, filling the same `user_id`.
- Secrets in Secret Manager via Workload Identity; no secrets in images, manifests or Git.
- Chat history retention by ClickHouse TTL (90 days default); question text not written to logs by default (CLAUDE.md rule 9). Public documents only — no patient data.
- Region choice (e.g. asia-south1 for users in India) is a pending decision, together with data-residency needs.

## 6. Illustrative manifests (not applied)

```yaml
# api Deployment + HPA — illustration of the concept only
apiVersion: apps/v1
kind: Deployment
metadata: { name: api, namespace: medrag-app }
spec:
  replicas: 4
  selector: { matchLabels: { app: api } }
  template:
    metadata: { labels: { app: api } }
    spec:
      serviceAccountName: medrag-api          # Workload Identity → Secret Manager
      terminationGracePeriodSeconds: 120      # let open SSE answers finish
      containers:
        - name: api
          image: REGION-docker.pkg.dev/PROJECT/medrag/api@sha256:<digest>
          args: ["uvicorn", "medical_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
          ports: [{ containerPort: 8000 }]
          env:
            - { name: MEDRAG_OPENSEARCH_URL, value: "http://opensearch.medrag-data:9200" }
            - { name: MEDRAG_REDIS_URL, value: "redis://MEMORYSTORE_IP:6379/0" }
          resources: { requests: { cpu: "1", memory: 1Gi }, limits: { cpu: "2", memory: 2Gi } }
          readinessProbe: { httpGet: { path: /health, port: 8000 }, periodSeconds: 5 }
          lifecycle: { preStop: { exec: { command: ["sleep", "15"] } } }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: api, namespace: medrag-app }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: api }
  minReplicas: 4
  maxReplicas: 12
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 60 } }
```

## 7. What must change in the code first (independent of GCP)

| # | Change | Why |
|---|---|---|
| C1 | Stream answer text (plain text with `[S1]` markers, citations validated at the end) | first text in ~1 s |
| C2 | Async endpoints and async clients (OpenSearch, Redis, ClickHouse, Anthropic) | hundreds of concurrent streams per pod |
| C3 | Embedding behind an HTTP interface (`Embedder` already abstracts it) | separate `embedder` service |
| C4 | LLM gateway: global concurrency limit, queue, retry with backoff | protect rate limits under bursts |
| C5 | Cheaper follow-up rewrite (skip when standalone; small model) | −2 to −3 s on follow-ups |
| C6 | Configuration only through `MEDRAG_*` env vars and Secret Manager (already the pattern) | 12-factor deployment |
| C7 | Langfuse tracing per stage | measure the SLOs |

## 8. Validation plan

1. **Infrastructure load test** (k6/Locust) with a mocked LLM up to ~200 questions/s: retrieval P95, Redis/ClickHouse behaviour, SSE connection limits, HPA reaction.
2. **Real-LLM test** at 5–10 questions/s for 10 minutes: first-text and full-answer P95, error and rate-limit behaviour, cost per answer.
3. Record results and the chosen SLOs in `DECISION_LOG.md` (ADR-007, ADR-021).

## 9. Open decisions

GCP region and data residency · Claude API vs Claude on Vertex AI · GKE Autopilot vs Standard · in-cluster vs
managed ClickHouse · OpenSearch on GKE vs a managed service · GPU vs CPU embedding · sign-in provider · SLO values ·
load target (design ~20 questions/s assumed).

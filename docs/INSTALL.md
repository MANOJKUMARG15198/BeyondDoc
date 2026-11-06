# Installing the Medical RAG demo

Three ways to run it, from smallest to largest:

1. [Local machine (Linux)](#1-local-machine-linux): development and offline evaluation.
2. [Single Linux server](#2-single-linux-server-behind-nginx): a password-protected demo under a path of an
   existing website (`https://<your-domain>/medrag/`). This is how the current demo runs.
3. [Google Cloud](#3-google-cloud-view): a single VM now, GKE for heavy traffic later (concept only).

Optional for all three: [Langfuse tracing](#observability-langfuse-tracing-optional).

Informational demo over public documents; not medical advice, diagnosis or treatment. No patient data.

**Secrets rule for every option:** API keys and passwords go only into an env file that is never committed
(`.env` locally, `/opt/medrag/medrag.env` on a server, Secret Manager on Google Cloud). The repository holds only
`*.example` templates with empty values.

Versions below are the ones tested (2026-11-06). Placeholders are written `<like-this>`.

---

## 1. Local machine (Linux)

### Requirements

| Item | Tested version | Notes |
|---|---|---|
| Linux, x86-64 | Ubuntu 22.04+/Debian 12/RHEL 9 class | macOS works the same way; Windows paths differ (`.venv/Scripts/`) |
| Python | **3.11** (`requires-python >=3.11,<3.12`) | |
| Docker Engine + Compose v2 | 26.x / 2.27+ | OpenSearch 3.8.0, Redis 7.4.11, ClickHouse 25.3 |
| Node.js | 22 LTS (22.20.0) | only to build the chat UI (React 19, Vite 8) |
| RAM / disk | 8 GB+ free RAM / ~10 GB disk | the raw corpus is ~2.1 GB, Python packages ~2 GB |
| Accounts | Anthropic API key (answers); optional Fireworks key (Ragas judge) | your own keys |

OpenSearch needs `vm.max_map_count >= 262144` on the host for the default `docker-compose.yml`
(`sudo sysctl -w vm.max_map_count=262144`). If you cannot change it, add `node.store.allow_mmap=false` to the
OpenSearch environment, as `deploy/server/docker-compose.yml` does.

### Steps

```bash
# 1. Code and Python packages (CPU build of PyTorch)
git clone <repository-url> RagMedical && cd RagMedical
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple -r requirements-dev.txt -e .

# 2. Settings: copy the template, then fill in your own values in .env
cp .env.example .env
#   ANTHROPIC_API_KEY=<your key>            (answers)
#   MEDRAG_CLICKHOUSE_PASSWORD=<any long random string>
#   MEDRAG_RETRIEVAL_PLAN=v12_t3            (medicine/source/section-aware retrieval; "none" = plain hybrid)
#   MEDRAG_NCBI_EMAIL=<your email>          (optional, only for the corpus download)
chmod 600 .env

# 3. Local services (bound to 127.0.0.1 only)
docker compose up -d && docker compose ps     # wait until all three show "healthy"

# 4. Corpus: download, chunk, embed, index (the corpus is not stored in Git)
.venv/bin/python -I scripts/phase1/openfda_bulk_download.py       # openFDA bulk exports (~1.8 GB)
.venv/bin/python -I scripts/phase1/select_dailymed_candidates.py
.venv/bin/python -I scripts/phase1/plan_batch_01.py
.venv/bin/python -I scripts/phase1/download_batch.py --batch batch_01
.venv/bin/python -X utf8 scripts/phase2/chunk_sample.py --set batch01          # chunking experiments E1-E4
.venv/bin/python -X utf8 scripts/phase3/embed_chunks.py --experiment E3_fill   # 5,815 chunks, CPU: ~40-80 min
.venv/bin/python -X utf8 scripts/phase3/build_index.py --experiment E3_fill    # index + alias medrag-chunks
.venv/bin/python -X utf8 scripts/phase5/build_medicine_aliases.py              # alias table for v12_t3

# 5. Chat UI (built once into src/medical_rag/api/static/app)
(cd frontend && npm ci && npm run build)

# 6. Run
.venv/bin/uvicorn medical_rag.api.main:app --host 127.0.0.1 --port 8000
#   http://127.0.0.1:8000/          chat UI
#   http://127.0.0.1:8000/explorer  retrieval explorer
#   http://127.0.0.1:8000/docs      API (Swagger)
```

### Checks

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest                    # unit tests (no network, LLM calls mocked)
.venv/bin/pytest -m integration     # needs the Docker services
curl -s http://127.0.0.1:8000/health
```

Offline evaluation (optional, costs API credits): `scripts/phase5/eval_v12.py` (retrieval only, no LLM),
`scripts/phase3/eval_ragas.py --plan v12_t3 --prompt v4.1` (answers + judge), `scripts/phase5/bench_retrieval.py`
(latency). Reports land in `data/processed/eval/` (gitignored).

---

## 2. Single Linux server, behind nginx

Target: `https://<your-domain>/medrag/` → nginx (HTTPS, password) → `127.0.0.1:8400` → `medrag.service`.
Everything runs on one server next to an existing website; nothing new is reachable from outside except the
`/medrag/` path. Files are in `deploy/server/`.

### Requirements

- A Linux server with nginx already serving `<your-domain>` over HTTPS, Docker + Compose, Python 3.11,
  `openssl`, and a user with sudo for the marked steps.
- About 4 CPU cores and 8 GB RAM for the demo (2 workers ≈ 0.9 GB, OpenSearch 2 GB heap), 5 GB disk.
- Outbound HTTPS to `api.anthropic.com` and `huggingface.co` (the embedding model is downloaded once).
- The ports 8400, 9201, 6380 and 8124 free on 127.0.0.1 (change them in the files if not).

### Steps (✱ = needs sudo)

**1. Folder and service user ✱**

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin medrag      # or use an existing account
sudo mkdir -p /opt/medrag && sudo chown medrag:medrag /opt/medrag
```

If you use another account, replace `medrag` in `deploy/server/medrag.service` (`User=`, `Group=`).

**2. Code, UI build and index data (from a machine that has run section 1)**

The server needs no Node.js and does not re-embed: copy the committed code, the built UI and the index inputs.

```bash
# on the build machine
git archive --format=tar.gz -o code.tar.gz HEAD
tar -czf data.tar.gz src/medical_rag/api/static/app \
    data/processed/chunking/experiments/batch01/E3_fill \
    data/processed/embeddings/batch01/E3_fill \
    data/processed/medicine/aliases_*.json
scp code.tar.gz data.tar.gz <user>@<server>:/opt/medrag/
# on the server
cd /opt/medrag && mkdir -p app && tar -xzf code.tar.gz -C app && tar -xzf data.tar.gz -C app
```

Text files must have LF line endings on Linux; `.gitattributes` makes `git archive` keep them for `deploy/`.

**3. Python environment and embedding model**

```bash
cd /opt/medrag
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple -r app/requirements.txt
venv/bin/pip install --no-deps -e app
cd app && HF_HOME=/opt/medrag/hf-cache ../venv/bin/python deploy/server/fetch_model.py   # prints MODEL_OK
```

**4. Settings file (keys added by hand on the server)**

```bash
cd /opt/medrag
umask 077
sed "s|<generated on the server>|$(openssl rand -hex 24)|" app/deploy/server/medrag.env.example > medrag.env
chmod 600 medrag.env
vi medrag.env        # fill in ANTHROPIC_API_KEY=… yourself; save with Esc then :wq
```

The password for the demo's ClickHouse is generated here and never printed. Defaults in the template:
`MEDRAG_RETRIEVAL_PLAN=v12_t3`, `MEDRAG_ANSWER_PROMPT=v4.1`, query text not logged.

**5. Containers ✱, then the index**

```bash
cd /opt/medrag/app/deploy/server
sudo docker compose -p medrag --env-file /opt/medrag/medrag.env up -d
sudo docker compose -p medrag ps                 # all healthy
cd /opt/medrag/app && set -a && . ../medrag.env && set +a
MEDRAG_OPENSEARCH_TIMEOUT_S=120 HF_HOME=/opt/medrag/hf-cache HF_HUB_OFFLINE=1 \
  ../venv/bin/python -X utf8 scripts/phase3/build_index.py --experiment E3_fill   # 5,815 chunks, ~35 s
```

**6. Service ✱**

```bash
sudo cp /opt/medrag/app/deploy/server/medrag.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now medrag
curl -s http://127.0.0.1:8400/health             # "status":"ok", opensearch green
```

Runs as the service user on 127.0.0.1:8400 with 2 workers, `CPUQuota=800%`, `MemoryMax=6G`, restart on failure.

**7. nginx location and demo password ✱**

```bash
# password file (no extra package needed; openssl prompts for the password twice)
printf 'demo:%s\n' "$(openssl passwd -apr1)" | sudo tee /etc/nginx/medrag.htpasswd > /dev/null
sudo chown root:nginx /etc/nginx/medrag.htpasswd && sudo chmod 640 /etc/nginx/medrag.htpasswd
# back up the site config, then insert the location block above the site's `location / {`
sudo cp /etc/nginx/conf.d/<site>.conf /etc/nginx/conf.d/<site>.conf.bak-medrag
sudo vi /etc/nginx/conf.d/<site>.conf      # paste deploy/server/nginx-medrag-location.conf in the 443 server block
sudo nginx -t && sudo systemctl reload nginx
```

Then, from any machine (asks for the demo password, never prints it):

```bash
python deploy/server/check_deploy.py https://<your-domain>/medrag
```

It checks 401 without and with a wrong password, health, the UI assets, a search, two cited answers, and that
the rest of the site still answers.

### Updating and rolling back

```bash
cd /opt/medrag && tar -czf backup-app-$(date +%Y%m%d).tar.gz --exclude=app/data app   # keep the old version
# copy the new code.tar.gz / UI build as in step 2, then:
sudo systemctl restart medrag
```

Undo everything: restore the `.bak-medrag` nginx file, `nginx -t`, reload; `sudo systemctl disable --now medrag`;
`sudo docker compose -p medrag down` (add `-v` to delete the index and chat history); remove `/opt/medrag`.

---

## Observability: Langfuse tracing (optional)

Off by default. One trace per `/search`, `/ask` and chat turn, with the stages `understand` (follow-up rewrite),
`retrieve` (timings per sub-step, chunk and document ids, plan reasons) and `compose` (model, token usage, coverage,
abstention, cited ids). **No question, answer or passage text is sent** unless `MEDRAG_TRACE_CONTENT=true`, which is
meant for synthetic demo questions only; credential-shaped strings are masked either way, and the user id is a hash
of the anonymous cookie. If Langfuse is unreachable, requests still work (errors are logged and skipped).

**Langfuse Cloud** (nothing to run): create a project at https://cloud.langfuse.com (EU) or
https://us.cloud.langfuse.com (US), then put its keys in `.env` (or `/opt/medrag/medrag.env` on a server):

```bash
MEDRAG_TRACING=langfuse
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...      # your project's keys; never commit them
LANGFUSE_SECRET_KEY=sk-lf-...
```

**Self-hosted** (traces stay on your machine; ~4 GB RAM, Langfuse server 4.69.0):

```bash
bash deploy/langfuse/make_env.sh <admin-email> http://127.0.0.1:3000   # random secrets → deploy/langfuse/langfuse.env (600)
docker compose -p langfuse -f deploy/langfuse/docker-compose.yml --env-file deploy/langfuse/langfuse.env up -d
# UI: http://127.0.0.1:3000 (admin email from above, password in langfuse.env). Then in the app's .env:
#   MEDRAG_TRACING=langfuse, LANGFUSE_HOST=http://127.0.0.1:3000, and the two LANGFUSE_INIT_PROJECT_* keys
```

Only the web UI is published, on 127.0.0.1; reach it through nginx/HTTPS or an SSH tunnel. Restart the app after
changing these settings. Set trace retention in the Langfuse project settings (30 days suggested), and add the
Claude model prices there if you want cost per answer.

---

## 3. Google Cloud view

Two stages. Stage A is the single-server setup above on Google Cloud; stage B is the target design for heavy
traffic. **Stage B is a concept: it is not built and not load-tested.** Details, sizing and open choices:
[SCALING_GCP_GKE.md](SCALING_GCP_GKE.md).

### Stage A: one Compute Engine VM (demo, today's design)

| Piece | Google Cloud service |
|---|---|
| Server | Compute Engine VM, e.g. `e2-standard-4` (4 vCPU, 16 GB), Ubuntu 22.04 or Rocky 9, 50 GB balanced disk |
| HTTPS and access | nginx on the VM with a certificate for `<your-domain>`, **or** an HTTPS load balancer with **IAP** (Google sign-in) instead of a shared password |
| Firewall | allow 443 (and 22 from your IP only); nothing else |
| Secrets | `medrag.env` on the VM (mode 600), or fetched at boot from **Secret Manager** |
| Backups | disk snapshots; the index can always be rebuilt from Cloud Storage copies of the embeddings |

```bash
# illustrative commands (project, region and names are yours to choose)
gcloud compute instances create medrag-demo --zone=<zone> --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=50GB --tags=https-server
gcloud compute firewall-rules create allow-https --allow=tcp:443 --target-tags=https-server
gcloud secrets create anthropic-api-key --data-file=-      # paste the key on stdin, never in a file in Git
# then follow section 2 on the VM
```

### Stage B: GKE for heavy traffic (target design)

```
Users ─► Cloud Armor + HTTPS Load Balancer (IAP or sign-in) ─► GKE
          ├─ api          stateless FastAPI pods, autoscaled on CPU and open streams
          ├─ embedder     batched embedding service (ONNX on CPU or an L4 GPU node pool)
          ├─ llm-gateway  concurrency cap, queue, retries → Claude API (or Claude on Vertex AI)
          ├─ OpenSearch   3 nodes (operator), replicas, new index + alias swap for updates
          └─ ClickHouse   2 replicas + Keeper (chat history, async writes)
        Memorystore for Redis (memory, rate limits, quotas) · Secret Manager + Workload Identity
        Cloud Storage (corpus, chunks, embeddings) · Kubernetes Jobs (ingest, chunk, embed, index)
        Artifact Registry + Cloud Build (images) · Cloud Logging, Managed Prometheus, Langfuse (tracing)
```

How it would be installed, in order:

1. **Containerise.** Add a `Dockerfile` for the API (Python 3.11 slim, CPU PyTorch, the built UI copied in) and
   one for the embedder; build with Cloud Build into Artifact Registry. *Not in the repository yet.*
2. **Network and cluster.** A VPC with private subnets, Cloud NAT for outbound calls to the Claude API, a GKE
   cluster (Autopilot is simplest; Standard if a GPU pool is wanted).
3. **Data services.** Memorystore for Redis; OpenSearch and ClickHouse on GKE through their Kubernetes operators
   with SSD persistent volumes (or managed offerings).
4. **Secrets.** Anthropic key, ClickHouse password and demo credentials in Secret Manager, read by pods through
   Workload Identity; no keys in images, manifests or Git.
5. **Indexing as Jobs.** Upload the corpus and embeddings to Cloud Storage; a Kubernetes Job builds a new index and
   switches the `medrag-chunks` alias (the same zero-downtime method `build_index.py` already uses).
6. **Deploy.** `api`, `embedder` and `llm-gateway` Deployments with Horizontal Pod Autoscalers; a GKE Ingress
   behind the HTTPS load balancer; Cloud Armor rules for rate limiting and bad traffic.
7. **Control and observe.** Per-user quotas in Redis, global LLM concurrency cap with a "busy, try again" (503)
   response, answer cache for repeated questions, dashboards and alerts on P50/P95 per stage and cost per answer.
8. **Load-test before opening up.** Ramp to the design load (~20 questions/s) and a burst test; check the
   proposed targets (first text P95 < 1.5 s, full answer P95 < 8 s, retrieval P95 < 300 ms).

Code changes needed first (independent of Google Cloud): stream answer text, async clients, embedding behind an
HTTP interface, the LLM gateway, a cheaper follow-up rewrite, and per-stage tracing
([SCALING_GCP_GKE.md](SCALING_GCP_GKE.md), section 7).

"""Download the pinned embedding model into HF_HOME, then load it the way the app does and embed one query.

HF_HOME=/opt/medrag/hf-cache /opt/medrag/venv/bin/python deploy/server/fetch_model.py
Run once per server (needs outbound HTTPS to huggingface.co); the service itself runs with HF_HUB_OFFLINE=1.
"""

import time

from huggingface_hub import snapshot_download

from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig
from medical_rag.embedding.st_embedder import MODEL_FILES, SentenceTransformerEmbedder

cfg = EmbeddingConfig(query_instruction=BGE_QUERY_INSTRUCTION)
snapshot_download(cfg.model_name, revision=cfg.revision, allow_patterns=list(MODEL_FILES))
start = time.perf_counter()
vector = SentenceTransformerEmbedder(cfg).embed_query("gabapentin dose in renal impairment")  # verifies sha256
print("MODEL_OK", cfg.model_name, cfg.revision[:8], vector.shape, f"{time.perf_counter() - start:.1f}s")

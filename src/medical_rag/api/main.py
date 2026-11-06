"""FastAPI application. Run: uvicorn medical_rag.api.main:app --reload"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from opensearchpy import OpenSearch
from pydantic import BaseModel

from medical_rag import __version__
from medical_rag.core.config import Settings, get_settings
from medical_rag.observability.logging import configure_logging
from medical_rag.search.client import cluster_health, make_client

STATIC = Path(__file__).parent / "static"
DISCLAIMER = "Informational demo over public documents. Not medical advice; not a diagnosis or treatment system."


class DependencyHealth(BaseModel):
    status: str
    version: str | None = None
    nodes: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    env: str
    opensearch: DependencyHealth
    disclaimer: str = DISCLAIMER


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_search_client(settings: SettingsDep) -> OpenSearch:
    return make_client(settings)


SearchClientDep = Annotated[OpenSearch, Depends(get_search_client)]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Load the embedding model once at startup so the first search is not a 20 s cold start."""
        from medical_rag.api.rag import _embedder

        try:
            _embedder()
        except Exception as exc:  # noqa: BLE001 - /health and BM25 must still work without the model
            logging.getLogger("medical_rag.api").warning("embedder_not_loaded", extra={"error": str(exc)})
        yield
        from medical_rag.observability.tracing import get_tracer

        get_tracer().flush()  # send buffered traces before the process exits

    app = FastAPI(title="Medical RAG", version=__version__, description=DISCLAIMER, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response, settings: SettingsDep, client: SearchClientDep) -> HealthResponse:
        """Liveness plus dependency status. Returns 503 when OpenSearch is not usable."""
        os_health = DependencyHealth(**cluster_health(client))
        ok = os_health.status in ("green", "yellow")
        if not ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if ok else "degraded", version=__version__, env=settings.env, opensearch=os_health
        )

    from medical_rag.api.chat import COOKIE, assign_user
    from medical_rag.api.chat import router as chat_router
    from medical_rag.api.rag import router

    app.include_router(router)
    app.include_router(chat_router)

    @app.middleware("http")
    async def user_cookie(request: Request, call_next):
        """Anonymous per-browser user id (HttpOnly cookie); chat history is scoped to it."""
        new_uid = assign_user(request)
        response = await call_next(request)
        if new_uid:
            response.set_cookie(COOKIE, new_uid, max_age=365 * 86400, httponly=True, samesite="lax")
        return response

    @app.get("/explorer", include_in_schema=False)
    def explorer() -> FileResponse:
        """Retrieval explorer: compare retrievers, inspect chunks, judge relevance."""
        return FileResponse(STATIC / "index.html")

    app_dir = STATIC / "app"
    if (app_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=app_dir / "assets"), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon() -> FileResponse:
        icon = app_dir / "favicon.svg"  # copied from frontend/public by the Vite build
        if not icon.exists():
            raise HTTPException(status_code=404, detail="favicon not built")
        return FileResponse(icon, media_type="image/svg+xml")

    @app.get("/", include_in_schema=False)
    def chat_ui() -> FileResponse:
        """Beyond Doc chat UI (React build in static/app); falls back to the explorer if it is not built."""
        index = app_dir / "index.html"
        return FileResponse(index if index.exists() else STATIC / "index.html")

    return app


app = create_app()

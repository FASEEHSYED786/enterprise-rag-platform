"""FastAPI service: tenant-scoped ingestion + querying, SSE streaming, Prometheus metrics."""
from __future__ import annotations

import base64
import binascii
import json
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..llm.providers import LLM, build_llm
from ..observability.metrics import Metrics, log
from ..pipeline import RagPipeline
from ..retrieval.embeddings import Embedder, HashEmbedder
from ..security.ratelimit import RateLimiter
from ..store import Store


class DocumentIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content: str | None = None
    content_b64: str | None = None


class QueryIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class TenantIn(BaseModel):
    tenant_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,40}$")
    name: str


def create_app(settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None,
               store: Store | None = None) -> FastAPI:
    s = settings or Settings()
    store = store or Store(s.db_path)
    pipeline = RagPipeline(store, embedder or HashEmbedder(s.embed_dim), llm or build_llm(s), s)
    limiter = RateLimiter(s.rate_limit_per_min)
    metrics = Metrics()
    app = FastAPI(title="Enterprise RAG Platform", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.inc("http_requests_total", method=request.method, path=path, status=str(response.status_code))
        metrics.observe("http_request_duration_seconds", dt, path=path)
        response.headers["x-request-id"] = rid
        log.info(json.dumps({"rid": rid, "method": request.method, "path": path, "status": response.status_code, "ms": round(dt * 1000, 1)}))
        return response

    def auth(authorization: str = Header(default="")) -> tuple[str, str]:
        key = authorization.removeprefix("Bearer ").strip()
        ident = store.authenticate(key) if key else None
        if not ident:
            raise HTTPException(401, "invalid or missing API key")
        ok, retry = limiter.allow(key)
        if not ok:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry)})
        return ident

    def admin(authorization: str = Header(default="")) -> None:
        key = authorization.removeprefix("Bearer ").strip()
        if not s.admin_key or not secrets.compare_digest(key, s.admin_key):
            raise HTTPException(403, "admin key required")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        return metrics.render()

    @app.post("/v1/admin/tenants", status_code=201, dependencies=[Depends(admin)])
    def create_tenant(body: TenantIn):
        try:
            key = store.create_tenant(body.tenant_id, body.name)
        except Exception:
            raise HTTPException(409, "tenant already exists") from None
        return {"tenant_id": body.tenant_id, "api_key": key, "note": "store this key now; it is not shown again"}

    @app.post("/v1/documents", status_code=201)
    def ingest(body: DocumentIn, ident=Depends(auth)):
        if body.content is not None:
            raw = body.content.encode()
        elif body.content_b64:
            try:
                raw = base64.b64decode(body.content_b64, validate=True)
            except binascii.Error:
                raise HTTPException(422, "content_b64 is not valid base64") from None
        else:
            raise HTTPException(422, "provide content or content_b64")
        try:
            result = pipeline.ingest(ident[0], body.name, raw)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        metrics.inc("documents_ingested_total", tenant=ident[0])
        return result

    @app.get("/v1/documents")
    def list_docs(ident=Depends(auth)):
        return {"documents": store.list_documents(ident[0])}

    @app.delete("/v1/documents/{doc_id}", status_code=204)
    def delete_doc(doc_id: int, ident=Depends(auth)):
        if not store.delete_document(ident[0], doc_id):
            raise HTTPException(404, "document not found")

    @app.post("/v1/query")
    def query(body: QueryIn, ident=Depends(auth)):
        t0 = time.perf_counter()
        ans = pipeline.ask(ident[0], body.question, body.k)
        metrics.observe("rag_query_seconds", time.perf_counter() - t0)
        metrics.inc("rag_queries_total", grounded=str(ans.grounded).lower())
        return {"answer": ans.answer, "grounded": ans.grounded, "sources": ans.sources, "flags": ans.flags}

    @app.post("/v1/query/stream")
    def query_stream(body: QueryIn, ident=Depends(auth)):
        def events():
            for ev in pipeline.ask_stream(ident[0], body.question, body.k):
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app

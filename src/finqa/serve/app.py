"""FastAPI service: hybrid retrieval (BM25 + fine-tuned bi-encoder), optional rerank, off by default."""
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from finqa.serve.pipeline import Retriever
from finqa.serve.store import load_artifacts

log = logging.getLogger("finqa.serve")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=100)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    alpha: float | None = Field(None, ge=0.0, le=1.0, description="weight on the dense score; default from the manifest")
    rerank: bool = Field(False, description="rerank the top candidates with a cross-encoder (off by default: it did not help on dev)")
    rerank_depth: int = Field(20, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class Hit(BaseModel):
    doc_id: str
    score: float
    rerank_score: float | None = None
    snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    k: int
    hits: list[Hit]
    timings_ms: dict[str, float]


def create_app(artifacts_dir: str | None = None, threads: int | None = None) -> FastAPI:
    directory = artifacts_dir or os.environ.get("FINQA_ARTIFACTS", "serving_artifacts")
    n_threads = threads or int(os.environ.get("FINQA_ORT_THREADS", "4"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = Retriever(load_artifacts(directory, n_threads))
        log.info("artifacts loaded from %s (%d docs)", directory, len(app.state.retriever.art.doc_ids))
        yield

    app = FastAPI(title="Financial QA retrieval", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        r = getattr(app.state, "retriever", None)
        if r is None:
            raise HTTPException(status_code=503, detail="artifacts not loaded")
        m = r.art.manifest
        return {"status": "ready", "docs": len(r.art.doc_ids), "alpha": r.default_alpha,
                "reranker": r.art.reranker is not None, "model": m.get("bi_encoder_sha256_10"), "built_at": m.get("built_at")}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        r: Retriever = app.state.retriever
        if req.rerank and r.art.reranker is None:
            raise HTTPException(status_code=400, detail="this deployment has no reranker")
        out = r.search(req.query, req.k, req.mode, req.alpha, req.rerank, req.rerank_depth)
        return SearchResponse(query=req.query, mode=req.mode, k=req.k, **out)

    return app


app = create_app()  # uvicorn finqa.serve.app:app; artifacts load at startup

"""FastAPI application exposing the RAG pipeline over HTTP."""

import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import MAX_QUERY_LENGTH, MAX_REQUESTS_PER_MINUTE
from app.schemas import AskRequest, AskResponse, Metrics, SourceChunk
from app.service import RAGService

service = RAGService()

_rate_windows: defaultdict[str, deque] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> None:
    """Simple in-memory sliding-window limiter: MAX_REQUESTS_PER_MINUTE per IP."""
    now = time.monotonic()
    window = _rate_windows[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= MAX_REQUESTS_PER_MINUTE:
        retry_after = int(60 - (now - window[0])) + 1
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {retry_after}s.")
    window.append(now)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.initialize()
    yield


app = FastAPI(
    title="Enterprise RAG Agent",
    description="Hybrid retrieval (Qdrant + BM25), FlashRank reranking, and grounded "
    "financial Q&A generation on the Walmart 2025 annual report.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Enterprise RAG Agent API is running.", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "ready": service.is_ready, "chunks_indexed": service.num_chunks}


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask(request: Request, payload: AskRequest):
    _check_rate_limit(_client_ip(request))
    try:
        result = service.answer(payload.query, k=payload.k, top_n=payload.top_n)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        query=result["query"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        sources=[SourceChunk(**src) for src in result["sources"]],
        metrics=Metrics(**result["metrics"]),
    )

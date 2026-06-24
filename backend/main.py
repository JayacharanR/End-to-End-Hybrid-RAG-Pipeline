"""WikiMind FastAPI Backend Application.

Entry point for the WikiMind RAG API server. Configures the FastAPI application
with lifespan-managed resource initialization, CORS middleware, Prometheus
metrics instrumentation, and cache-first query routing.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.cache import cache_lookup, close_redis, get_redis_client
from backend.config import get_settings
from backend.llmops import get_langfuse_client, init_observability
from backend.models import ChatRequest, ChatResponse, HealthResponse, RetrievalMetadata, ServiceStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup: validates Langfuse connection, initializes Redis client,
    and logs readiness status.
    On shutdown: closes Redis connection and flushes any pending state.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("WikiMind Backend starting up...")
    init_observability()

    # Pre-warm Redis connection
    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("Redis connection established at %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis connection failed during startup: %s", exc)

    # Pre-initialize Guardrails
    from backend.llmops import get_guardrails
    get_guardrails()
    
    # TODO: Initialize Qdrant collection once qdrant_client.py is implemented

    logger.info("WikiMind Backend ready on %s:%d", settings.app_host, settings.app_port)

    yield

    # Shutdown
    logger.info("WikiMind Backend shutting down...")
    await close_redis()

    langfuse = get_langfuse_client()
    if langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass

    logger.info("WikiMind Backend shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WikiMind RAG API",
    description="Production-grade Tri-Brid Hybrid Agentic RAG Pipeline backed by Wikipedia",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    logger.info("Prometheus metrics instrumentation enabled.")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics disabled.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with per-component status.

    Pings Qdrant, Redis, and Langfuse to report connectivity and latency
    for each infrastructure dependency.
    """
    components = []

    # Redis health
    try:
        redis_client = await get_redis_client()
        start = time.monotonic()
        await redis_client.ping()
        latency = (time.monotonic() - start) * 1000
        components.append(ServiceStatus(name="redis", healthy=True, latency_ms=round(latency, 2)))
    except Exception as exc:
        components.append(ServiceStatus(name="redis", healthy=False, detail=str(exc)))

    # Qdrant health
    settings = get_settings()
    try:
        import httpx
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
            latency = (time.monotonic() - start) * 1000
            healthy = resp.status_code == 200
            components.append(ServiceStatus(name="qdrant", healthy=healthy, latency_ms=round(latency, 2)))
    except Exception as exc:
        components.append(ServiceStatus(name="qdrant", healthy=False, detail=str(exc)))

    # Langfuse health
    langfuse = get_langfuse_client()
    if langfuse is not None:
        try:
            start = time.monotonic()
            auth_ok = langfuse.auth_check()
            latency = (time.monotonic() - start) * 1000
            components.append(ServiceStatus(name="langfuse", healthy=auth_ok, latency_ms=round(latency, 2)))
        except Exception as exc:
            components.append(ServiceStatus(name="langfuse", healthy=False, detail=str(exc)))
    else:
        components.append(ServiceStatus(name="langfuse", healthy=False, detail="Not configured"))

    overall = "healthy" if all(c.healthy for c in components) else "degraded"
    return HealthResponse(status=overall, components=components)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Primary RAG chat endpoint with cache-first routing.

    Checks the dual-layer cache (L1 exact-match, then L2 semantic) before
    invoking the LangGraph agent pipeline. Cache hits are returned immediately
    with appropriate metadata.

    In the full implementation, cache misses will invoke the LangGraph
    CRAG/Self-RAG state machine and stream the response via SSE.
    """
    query = request.query

    # Cache-first: check L1 and L2 before running the agent
    cached_response, cache_level = await cache_lookup(query)
    if cached_response is not None:
        logger.info("Serving cached response (level=%s) for: %s", cache_level, query[:60])
        return JSONResponse(content={
            "answer": cached_response.get("answer", ""),
            "sources": cached_response.get("sources", []),
            "metadata": {
                "cache_hit": True,
                "cache_level": cache_level,
                "strategies_used": [],
                "agent_steps": 0,
            },
        })

    # Cache miss: placeholder for LangGraph agent invocation
    # TODO: Invoke the LangGraph CRAG/Self-RAG pipeline here
    logger.info("Cache miss. Agent pipeline invocation pending for: %s", query[:60])

    return JSONResponse(content={
        "answer": "WikiMind agent pipeline not yet connected. Cache miss for your query.",
        "sources": [],
        "metadata": {
            "cache_hit": False,
            "strategies_used": list(
                k for k, v in request.strategies.model_dump().items() if v
            ),
            "agent_steps": 0,
        },
    })

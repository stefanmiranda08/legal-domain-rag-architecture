"""FastAPI application for the Legal RAG service."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from qdrant_client import QdrantClient
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings as _get_settings
from src.database import (
    get_qdrant_client,
    get_postgres_engine,
    check_qdrant_health,
    check_postgres_health,
    init_postgres_schema,
    create_qdrant_collection,
)
from src.generation import generate_answer
from src.models import (
    ChunkingStrategy,
    QueryRequest,
    QueryResponse,
    QueryFilters,
    Citation,
    IngestRequest,
    IngestResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationMetrics,
    HealthResponse,
)
from src.retrieval import search_chunks


# Global state for dependency injection
_qdrant_client: QdrantClient | None = None
_postgres_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global _qdrant_client, _postgres_engine

    # Startup
    settings = _get_settings()
    _qdrant_client = get_qdrant_client(settings=settings)
    _postgres_engine = get_postgres_engine(settings=settings)

    # Initialize database schema
    init_postgres_schema(_postgres_engine)

    # Create collections for all strategies
    for strategy in ChunkingStrategy:
        create_qdrant_collection(
            _qdrant_client,
            strategy,
            vector_size=settings.embedding_dimensions,
        )

    yield

    # Shutdown
    _qdrant_client = None
    _postgres_engine = None


app = FastAPI(
    title="Legal RAG API",
    description="Case law research assistant using RAG over Australian legal corpus",
    version="0.1.0",
    lifespan=lifespan,
)


# Dependency injection functions
def get_settings() -> Settings:
    """Get application settings."""
    return _get_settings()


def get_qdrant() -> QdrantClient:
    """Get Qdrant client."""
    if _qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    return _qdrant_client


def get_postgres() -> Engine:
    """Get PostgreSQL engine."""
    if _postgres_engine is None:
        raise HTTPException(status_code=503, detail="PostgreSQL not initialized")
    return _postgres_engine


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant)]
PostgresDep = Annotated[Engine, Depends(get_postgres)]


@app.get("/health", response_model=HealthResponse)
def health_check(
    qdrant: QdrantDep,
    postgres: PostgresDep,
) -> HealthResponse:
    """Check health of all services."""
    qdrant_healthy = check_qdrant_health(qdrant)
    postgres_healthy = check_postgres_health(postgres)

    return HealthResponse(
        status="healthy" if (qdrant_healthy and postgres_healthy) else "degraded",
        qdrant="connected" if qdrant_healthy else "disconnected",
        postgres="connected" if postgres_healthy else "disconnected",
        timestamp=datetime.utcnow(),
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    qdrant: QdrantDep,
    postgres: PostgresDep,
    settings: SettingsDep,
) -> QueryResponse:
    """
    Submit a natural language query.

    Retrieves relevant document chunks and generates an answer with citations.
    """
    start_time = time.time()
    query_id = f"q_{uuid4().hex[:8]}"

    # Search for relevant chunks
    chunks = search_chunks(
        query=request.query,
        qdrant_client=qdrant,
        strategy=request.chunking_strategy,
        openai_api_key=settings.openai_api_key,
        filters=request.filters,
        top_k=request.top_k,
        postgres_engine=postgres,
        settings=settings,
    )

    # Generate answer
    result = generate_answer(
        query=request.query,
        chunks=chunks,
        openai_api_key=settings.openai_api_key,
        model=settings.llm_model,
        settings=settings,
    )

    latency_ms = int((time.time() - start_time) * 1000)

    # Log query (fire and forget)
    _log_query(
        postgres,
        query_id=query_id,
        query_text=request.query,
        filters=request.filters,
        strategy=request.chunking_strategy,
        chunks_retrieved=len(chunks),
        latency_ms=latency_ms,
        tokens_used=result.tokens_used,
        answer_text=result.text,
        citations=result.citations,
    )

    return QueryResponse(
        answer=result.text,
        citations=result.citations,
        query_id=query_id,
        latency_ms=latency_ms,
        chunks_retrieved=len(chunks),
        tokens_used=result.tokens_used,
    )


def _log_query(
    engine: Engine,
    query_id: str,
    query_text: str,
    filters: QueryFilters | None,
    strategy: ChunkingStrategy,
    chunks_retrieved: int,
    latency_ms: int,
    tokens_used: int,
    answer_text: str,
    citations: list[Citation],
) -> None:
    """Log a query to the database."""
    try:
        from src.database import get_postgres_session
        from src.models import QueryLog

        with get_postgres_session(engine) as session:
            log = QueryLog(
                query_text=query_text,
                filters=filters.model_dump() if filters else None,
                chunking_strategy=strategy.value,
                chunks_retrieved=chunks_retrieved,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                answer_text=answer_text,
                citations=[c.model_dump() for c in citations],
            )
            session.add(log)
    except Exception:
        # Don't fail the request if logging fails
        pass


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantDep,
    postgres: PostgresDep,
    settings: SettingsDep,
) -> IngestResponse:
    """
    Start corpus ingestion.

    Runs asynchronously in the background.
    """
    job_id = f"ingest_{uuid4().hex[:8]}"

    # Start ingestion in background
    background_tasks.add_task(
        _run_ingestion,
        job_id=job_id,
        qdrant=qdrant,
        postgres=postgres,
        settings=settings,
        strategies=request.chunking_strategies,
        limit=request.limit,
        jurisdiction_filter=request.jurisdiction_filter,
    )

    return IngestResponse(
        job_id=job_id,
        status="started",
        estimated_documents=request.limit or 232560,  # Full corpus size
    )


def _run_ingestion(
    job_id: str,
    qdrant: QdrantClient,
    postgres: Engine,
    settings: Settings,
    strategies: list[ChunkingStrategy],
    limit: int | None,
    jurisdiction_filter,
) -> None:
    """Run the ingestion pipeline."""
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        qdrant_client=qdrant,
        postgres_engine=postgres,
        openai_api_key=settings.openai_api_key,
        settings=settings,
    )

    pipeline.ingest(
        strategies=strategies,
        limit=limit,
        jurisdiction_filter=jurisdiction_filter,
    )


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    request: EvaluationRequest,
    qdrant: QdrantDep,
    postgres: PostgresDep,
    settings: SettingsDep,
) -> EvaluationResponse:
    """
    Run evaluation on a chunking strategy.

    Returns retrieval metrics for the specified test set.
    """
    result = run_evaluation(
        qdrant=qdrant,
        postgres=postgres,
        settings=settings,
        strategy=request.chunking_strategy,
        test_set_id=request.test_set_id,
    )

    return EvaluationResponse(
        evaluation_id=result["evaluation_id"],
        strategy=request.chunking_strategy,
        metrics=EvaluationMetrics(**result["metrics"]),
        per_query_results=result["per_query_results"],
    )


def run_evaluation(
    qdrant: QdrantClient,
    postgres: Engine,
    settings: Settings,
    strategy: ChunkingStrategy,
    test_set_id: str,
) -> dict:
    """
    Run evaluation and return results.

    Placeholder implementation - full evaluation in Stage 8.
    """
    evaluation_id = f"eval_{uuid4().hex[:8]}"

    # Placeholder metrics - will be implemented in evaluation module
    return {
        "evaluation_id": evaluation_id,
        "strategy": strategy.value,
        "metrics": {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "avg_latency_ms": 0.0,
        },
        "per_query_results": [],
    }

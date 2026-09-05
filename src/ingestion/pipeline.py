"""Ingestion pipeline for processing documents into the RAG system."""

import logging
import sys
from datetime import date, datetime
from typing import TypedDict
from uuid import uuid4

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
from src.database import (
    create_qdrant_collection,
    get_collection_name,
    get_postgres_session,
)
from src.ingestion.chunkers import Chunk, get_chunker
from src.ingestion.loader import DocumentRecord, load_corpus
from src.models import (
    Chunk as ChunkModel,
)
from src.models import (
    ChunkingStrategy,
    Document,
)


class IngestionStats(TypedDict):
    """Statistics from an ingestion run."""

    documents_processed: int
    chunks_created: int
    errors: int
    started_at: datetime
    completed_at: datetime | None


def generate_embeddings(
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    max_tokens_per_batch: int = 250_000,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.

    Batches by token count to stay under OpenAI's 300k token limit.

    Args:
        texts: List of texts to embed.
        api_key: OpenAI API key.
        model: Embedding model to use.
        dimensions: Embedding dimensions.
        max_tokens_per_batch: Maximum tokens per API call (default 250k, limit is 300k).

    Returns:
        List of embedding vectors.
    """
    import tiktoken

    if not texts:
        return []

    client = OpenAI(api_key=api_key)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    all_embeddings = []

    # Build batches by token count, not item count
    current_batch = []
    current_tokens = 0

    for text in texts:
        text_tokens = len(tokenizer.encode(text))

        # If single text exceeds limit, truncate it
        if text_tokens > max_tokens_per_batch:
            # Truncate to fit (leave some margin)
            tokens = tokenizer.encode(text)[: max_tokens_per_batch - 1000]
            text = tokenizer.decode(tokens)
            text_tokens = len(tokens)

        # If adding this text would exceed limit, send current batch first
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            response = client.embeddings.create(
                input=current_batch,
                model=model,
                dimensions=dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    # Send remaining batch
    if current_batch:
        response = client.embeddings.create(
            input=current_batch,
            model=model,
            dimensions=dimensions,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


class IngestionPipeline:
    """
    Pipeline for ingesting documents into the RAG system.

    Handles loading documents, chunking, embedding generation,
    and storage in both Qdrant and PostgreSQL.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        postgres_engine: Engine,
        openai_api_key: str,
        settings: Settings | None = None,
    ):
        self.qdrant_client = qdrant_client
        self.postgres_engine = postgres_engine
        self.openai_api_key = openai_api_key
        self.settings = settings or get_settings()

    def ingest(
        self,
        strategies: list[ChunkingStrategy],
        limit: int | None = None,
        jurisdiction_filter=None,
        batch_size: int = 50,
    ) -> IngestionStats:
        """
        Run the ingestion pipeline.

        Args:
            strategies: Chunking strategies to use.
            limit: Maximum documents to process.
            jurisdiction_filter: Filter by jurisdiction.
            batch_size: Documents to process before committing.

        Returns:
            Statistics about the ingestion run.
        """
        stats: IngestionStats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "started_at": datetime.utcnow(),
            "completed_at": None,
        }

        # Create collections for each strategy
        logger.info("Creating Qdrant collections...")
        for strategy in strategies:
            create_qdrant_collection(
                self.qdrant_client,
                strategy,
                vector_size=self.settings.embedding_dimensions,
            )
            logger.info(f"  Collection for {strategy.value}: ready")

        # Initialize chunkers
        logger.info("Initializing chunkers...")
        chunkers = {
            strategy: get_chunker(
                strategy,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for strategy in strategies
        }

        # Process documents
        logger.info("Loading corpus from HuggingFace (this may take a moment)...")
        doc_batch = []
        doc_count = 0
        for record in load_corpus(limit=limit, jurisdiction_filter=jurisdiction_filter):
            try:
                doc_batch.append(record)
                doc_count += 1

                if doc_count == 1:
                    logger.info("First document received from corpus stream")

                if len(doc_batch) >= batch_size:
                    logger.info(
                        f"Processing batch of {len(doc_batch)} docs "
                        f"(total seen: {doc_count}, processed: {stats['documents_processed']}, "
                        f"skipped: {stats.get('skipped', 0)}, errors: {stats['errors']})"
                    )
                    self._process_batch(doc_batch, chunkers, stats)
                    doc_batch = []

            except Exception as e:
                logger.warning(f"Error loading document: {e}")
                stats["errors"] += 1
                continue

        # Process remaining documents
        if doc_batch:
            logger.info(f"Processing final batch of {len(doc_batch)} docs")
            self._process_batch(doc_batch, chunkers, stats)

        stats["completed_at"] = datetime.utcnow()
        return stats

    def _process_batch(
        self,
        records: list[DocumentRecord],
        chunkers: dict[ChunkingStrategy, any],
        stats: IngestionStats,
    ) -> None:
        """Process a batch of documents."""
        with get_postgres_session(self.postgres_engine) as session:
            for idx, record in enumerate(records):
                try:
                    # Check if document already exists (idempotent)
                    existing = (
                        session.query(Document)
                        .filter(Document.version_id == record.version_id)
                        .first()
                    )
                    if existing:
                        stats["skipped"] = stats.get("skipped", 0) + 1
                        continue

                    # Parse date if present
                    doc_date = None
                    if record.date:
                        try:
                            doc_date = date.fromisoformat(record.date)
                        except ValueError:
                            pass

                    # Store document metadata
                    doc = Document(
                        version_id=record.version_id,
                        citation=record.citation,
                        document_type=record.document_type.value,
                        jurisdiction=record.jurisdiction.value,
                        source=record.source,
                        date=doc_date,
                        url=record.url,
                        text_length=len(record.text),
                    )
                    session.add(doc)
                    session.flush()  # Get the ID

                    # Process with each chunking strategy
                    for strategy, chunker in chunkers.items():
                        logger.debug(f"  Chunking doc {idx+1}/{len(records)}: {record.citation[:50]}...")
                        chunks = chunker.chunk(record.text)

                        if not chunks:
                            continue

                        logger.debug(f"  Embedding {len(chunks)} chunks...")
                        # Generate embeddings for all chunks
                        chunk_texts = [c.text for c in chunks]
                        embeddings = generate_embeddings(
                            chunk_texts,
                            api_key=self.openai_api_key,
                            model=self.settings.embedding_model,
                            dimensions=self.settings.embedding_dimensions,
                        )

                        logger.debug(f"  Storing in Qdrant + Postgres...")
                        # Store in Qdrant and PostgreSQL
                        self._store_chunks(
                            session,
                            doc.id,
                            record,
                            strategy,
                            chunks,
                            embeddings,
                        )
                        stats["chunks_created"] += len(chunks)

                    stats["documents_processed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 10:
                        logger.error(f"Error processing document {record.citation[:50]}: {e}")
                    session.rollback()
                    continue

    def _store_chunks(
        self,
        session,
        document_id,
        record: DocumentRecord,
        strategy: ChunkingStrategy,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks in Qdrant and PostgreSQL with bidirectional links."""
        collection_name = get_collection_name(strategy)
        points = []
        chunk_records = []

        # First pass: create all chunk records
        for chunk, embedding in zip(chunks, embeddings):
            point_id = uuid4()

            # Create Qdrant point
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding,
                    payload={
                        "document_id": str(document_id),
                        "chunk_index": chunk.index,
                        "jurisdiction": record.jurisdiction.value,
                        "document_type": record.document_type.value,
                        "date": record.date,
                        "citation": record.citation,
                    },
                )
            )

            # Create PostgreSQL chunk record
            chunk_record = ChunkModel(
                document_id=document_id,
                chunking_strategy=strategy.value,
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                token_count=chunk.token_count,
                qdrant_point_id=point_id,
            )
            session.add(chunk_record)
            chunk_records.append(chunk_record)

        # Flush to get IDs assigned
        session.flush()

        # Second pass: link chunks bidirectionally
        for i, chunk_record in enumerate(chunk_records):
            if i > 0:
                chunk_record.prev_chunk_id = chunk_records[i - 1].id
            if i < len(chunk_records) - 1:
                chunk_record.next_chunk_id = chunk_records[i + 1].id

        # Batch upsert to Qdrant (limit batch size to avoid 32MB payload limit)
        if points:
            qdrant_batch_size = 500
            for i in range(0, len(points), qdrant_batch_size):
                batch = points[i : i + qdrant_batch_size]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )

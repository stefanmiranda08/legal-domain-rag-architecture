"""Ingestion pipeline for processing documents into the RAG system."""

from datetime import datetime, date
from typing import TypedDict
from uuid import uuid4

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings
from src.database import (
    create_qdrant_collection,
    get_collection_name,
    get_postgres_session,
)
from src.ingestion.chunkers import get_chunker, Chunk
from src.ingestion.loader import load_corpus, DocumentRecord
from src.models import (
    ChunkingStrategy,
    Document,
    Chunk as ChunkModel,
    IngestionJob,
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
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.

    Args:
        texts: List of texts to embed.
        api_key: OpenAI API key.
        model: Embedding model to use.
        dimensions: Embedding dimensions.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    client = OpenAI(api_key=api_key)
    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            input=batch,
            model=model,
            dimensions=dimensions,
        )

        # Extract embeddings in order
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
        for strategy in strategies:
            create_qdrant_collection(
                self.qdrant_client,
                strategy,
                vector_size=self.settings.embedding_dimensions,
            )

        # Initialize chunkers
        chunkers = {
            strategy: get_chunker(
                strategy,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for strategy in strategies
        }

        # Process documents
        doc_batch = []
        for record in load_corpus(limit=limit, jurisdiction_filter=jurisdiction_filter):
            try:
                doc_batch.append(record)

                if len(doc_batch) >= batch_size:
                    self._process_batch(doc_batch, chunkers, stats)
                    doc_batch = []

            except Exception as e:
                stats["errors"] += 1
                continue

        # Process remaining documents
        if doc_batch:
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
            for record in records:
                try:
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
                        chunks = chunker.chunk(record.text)

                        if not chunks:
                            continue

                        # Generate embeddings for all chunks
                        chunk_texts = [c.text for c in chunks]
                        embeddings = generate_embeddings(
                            chunk_texts,
                            api_key=self.openai_api_key,
                            model=self.settings.embedding_model,
                            dimensions=self.settings.embedding_dimensions,
                        )

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
        """Store chunks in Qdrant and PostgreSQL."""
        collection_name = get_collection_name(strategy)
        points = []

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

        # Batch upsert to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )

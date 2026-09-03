#!/usr/bin/env python3
"""
Validate ingestion data integrity between PostgreSQL and Qdrant.

Checks for:
1. Duplicate chunks in PostgreSQL (same document_id + chunk_index + strategy)
2. Duplicate vectors in Qdrant (same point_id)
3. Orphaned Qdrant vectors (no matching PostgreSQL chunk)
4. Orphaned PostgreSQL chunks (no matching Qdrant vector)
5. Duplicate chunk text within documents

Usage:
    python scripts/validate_ingestion.py
    python scripts/validate_ingestion.py --fix  # Remove duplicates/orphans
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import os

from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from src.database import get_collection_name
from src.models import ChunkingStrategy


@dataclass
class ValidationReport:
    """Results from validation."""

    postgres_chunks: int = 0
    qdrant_vectors: int = 0
    duplicate_postgres_chunks: list = None
    duplicate_qdrant_vectors: list = None
    orphaned_qdrant_vectors: list = None
    orphaned_postgres_chunks: list = None
    duplicate_text_within_docs: list = None

    def __post_init__(self):
        self.duplicate_postgres_chunks = self.duplicate_postgres_chunks or []
        self.duplicate_qdrant_vectors = self.duplicate_qdrant_vectors or []
        self.orphaned_qdrant_vectors = self.orphaned_qdrant_vectors or []
        self.orphaned_postgres_chunks = self.orphaned_postgres_chunks or []
        self.duplicate_text_within_docs = self.duplicate_text_within_docs or []


def get_postgres_chunk_ids(engine) -> dict[str, int]:
    """Get all chunk qdrant_point_ids from PostgreSQL. Returns {point_id: chunk_id}."""
    print("Fetching PostgreSQL chunk point IDs...")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id, qdrant_point_id FROM chunks WHERE qdrant_point_id IS NOT NULL")
        )
        mapping = {str(row[1]): row[0] for row in result}
    print(f"  Found {len(mapping)} chunks with point IDs")
    return mapping


def get_qdrant_point_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Get all point IDs from Qdrant collection."""
    print(f"Fetching Qdrant point IDs from {collection_name}...")

    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    print(f"  Collection has {total_points} points")

    point_ids = set()
    offset = None
    batch_size = 10000

    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        for point in results:
            point_ids.add(str(point.id))

        if next_offset is None:
            break
        offset = next_offset

        if len(point_ids) % 100000 == 0:
            print(f"  Fetched {len(point_ids)} point IDs...")

    print(f"  Total: {len(point_ids)} unique point IDs")
    return point_ids


def find_duplicate_postgres_chunks(engine) -> list[dict]:
    """Find chunks with duplicate (document_id, chunk_index, chunking_strategy)."""
    print("Checking for duplicate PostgreSQL chunks...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT document_id, chunk_index, chunking_strategy, COUNT(*) as cnt,
                   array_agg(id) as chunk_ids
            FROM chunks
            GROUP BY document_id, chunk_index, chunking_strategy
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
        """))

        duplicates = []
        for row in result:
            duplicates.append({
                "document_id": row[0],
                "chunk_index": row[1],
                "chunking_strategy": row[2],
                "count": row[3],
                "chunk_ids": list(row[4]),
            })

    print(f"  Found {len(duplicates)} duplicate chunk groups")
    return duplicates


def find_duplicate_text_within_docs(engine, sample_size: int = 1000) -> list[dict]:
    """Find documents with duplicate chunk text (exact matches)."""
    print("Checking for duplicate text within documents (sampling)...")
    with engine.connect() as conn:
        # Sample documents to check
        result = conn.execute(text("""
            WITH sampled_docs AS (
                SELECT DISTINCT document_id
                FROM chunks
                ORDER BY document_id
                LIMIT :sample_size
            )
            SELECT c.document_id, c.chunk_text, COUNT(*) as cnt, array_agg(c.id) as chunk_ids
            FROM chunks c
            JOIN sampled_docs s ON c.document_id = s.document_id
            GROUP BY c.document_id, c.chunk_text
            HAVING COUNT(*) > 1
        """), {"sample_size": sample_size})

        duplicates = []
        for row in result:
            duplicates.append({
                "document_id": row[0],
                "text_preview": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                "count": row[2],
                "chunk_ids": list(row[3]),
            })

    print(f"  Found {len(duplicates)} documents with duplicate text chunks")
    return duplicates


def find_orphaned_vectors(
    postgres_point_ids: dict[str, int],
    qdrant_point_ids: set[str],
) -> list[str]:
    """Find Qdrant vectors without matching PostgreSQL chunks."""
    print("Finding orphaned Qdrant vectors...")
    postgres_set = set(postgres_point_ids.keys())
    orphaned = list(qdrant_point_ids - postgres_set)
    print(f"  Found {len(orphaned)} orphaned vectors")
    return orphaned


def find_orphaned_chunks(
    postgres_point_ids: dict[str, int],
    qdrant_point_ids: set[str],
) -> list[int]:
    """Find PostgreSQL chunks without matching Qdrant vectors."""
    print("Finding orphaned PostgreSQL chunks...")
    orphaned = []
    for point_id, chunk_id in postgres_point_ids.items():
        if point_id not in qdrant_point_ids:
            orphaned.append(chunk_id)
    print(f"  Found {len(orphaned)} orphaned chunks")
    return orphaned


def delete_orphaned_vectors(client: QdrantClient, collection_name: str, point_ids: list[str]):
    """Delete orphaned vectors from Qdrant."""
    if not point_ids:
        return

    print(f"Deleting {len(point_ids)} orphaned vectors from Qdrant...")
    batch_size = 1000
    for i in range(0, len(point_ids), batch_size):
        batch = point_ids[i:i + batch_size]
        client.delete(
            collection_name=collection_name,
            points_selector=batch,
        )
        print(f"  Deleted {min(i + batch_size, len(point_ids))}/{len(point_ids)}")


def delete_duplicate_chunks(engine, duplicates: list[dict]):
    """Delete duplicate PostgreSQL chunks, keeping the first one."""
    if not duplicates:
        return

    print(f"Deleting duplicate chunks from PostgreSQL...")
    ids_to_delete = []
    for dup in duplicates:
        # Keep the first, delete the rest
        ids_to_delete.extend(dup["chunk_ids"][1:])

    if not ids_to_delete:
        return

    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM chunks WHERE id = ANY(:ids)"),
            {"ids": ids_to_delete}
        )
        conn.commit()
    print(f"  Deleted {len(ids_to_delete)} duplicate chunks")


def validate(fix: bool = False) -> ValidationReport:
    """Run full validation."""
    # Get connection settings from environment or defaults
    postgres_user = os.getenv("POSTGRES_USER", "legal_rag")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "legal_rag_password")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "legal_rag")
    postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    engine = create_engine(postgres_url)
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

    collection_name = get_collection_name(ChunkingStrategy.RECURSIVE)

    report = ValidationReport()

    # Get counts
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
        report.postgres_chunks = result.scalar()

    try:
        info = qdrant.get_collection(collection_name)
        report.qdrant_vectors = info.points_count
    except Exception as e:
        print(f"Warning: Could not get Qdrant collection info: {e}")
        report.qdrant_vectors = 0

    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"PostgreSQL chunks: {report.postgres_chunks:,}")
    print(f"Qdrant vectors:    {report.qdrant_vectors:,}")
    print(f"Difference:        {report.qdrant_vectors - report.postgres_chunks:,}")
    print(f"{'='*60}\n")

    # Find duplicates
    report.duplicate_postgres_chunks = find_duplicate_postgres_chunks(engine)
    report.duplicate_text_within_docs = find_duplicate_text_within_docs(engine)

    # Find orphans
    postgres_point_ids = get_postgres_chunk_ids(engine)
    qdrant_point_ids = get_qdrant_point_ids(qdrant, collection_name)

    report.orphaned_qdrant_vectors = find_orphaned_vectors(postgres_point_ids, qdrant_point_ids)
    report.orphaned_postgres_chunks = find_orphaned_chunks(postgres_point_ids, qdrant_point_ids)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Duplicate chunk groups (same doc+index+strategy): {len(report.duplicate_postgres_chunks)}")
    print(f"Documents with duplicate text chunks:            {len(report.duplicate_text_within_docs)}")
    print(f"Orphaned Qdrant vectors (no PostgreSQL match):   {len(report.orphaned_qdrant_vectors)}")
    print(f"Orphaned PostgreSQL chunks (no Qdrant match):    {len(report.orphaned_postgres_chunks)}")
    print(f"{'='*60}\n")

    # Fix if requested
    if fix:
        print("FIXING ISSUES...")
        delete_orphaned_vectors(qdrant, collection_name, report.orphaned_qdrant_vectors)
        delete_duplicate_chunks(engine, report.duplicate_postgres_chunks)
        print("Done!\n")

    # Output IDs for manual review
    if report.orphaned_qdrant_vectors:
        print(f"\nOrphaned Qdrant vector IDs (first 20):")
        for pid in report.orphaned_qdrant_vectors[:20]:
            print(f"  {pid}")
        if len(report.orphaned_qdrant_vectors) > 20:
            print(f"  ... and {len(report.orphaned_qdrant_vectors) - 20} more")

    if report.duplicate_postgres_chunks:
        print(f"\nDuplicate PostgreSQL chunk IDs to remove (first 10 groups):")
        for dup in report.duplicate_postgres_chunks[:10]:
            print(f"  doc={dup['document_id']}, index={dup['chunk_index']}: keep {dup['chunk_ids'][0]}, remove {dup['chunk_ids'][1:]}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate ingestion data integrity")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete orphaned vectors and duplicate chunks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    args = parser.parse_args()

    report = validate(fix=args.fix)

    if args.json:
        print(json.dumps({
            "postgres_chunks": report.postgres_chunks,
            "qdrant_vectors": report.qdrant_vectors,
            "duplicate_postgres_chunks": len(report.duplicate_postgres_chunks),
            "duplicate_text_within_docs": len(report.duplicate_text_within_docs),
            "orphaned_qdrant_vectors": len(report.orphaned_qdrant_vectors),
            "orphaned_postgres_chunks": len(report.orphaned_postgres_chunks),
            "orphaned_vector_ids": report.orphaned_qdrant_vectors[:100],
            "duplicate_chunk_ids": [d["chunk_ids"][1:] for d in report.duplicate_postgres_chunks[:100]],
        }, indent=2))


if __name__ == "__main__":
    main()

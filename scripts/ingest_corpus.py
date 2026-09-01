#!/usr/bin/env python3
"""
Script to ingest the full Australian Legal Corpus into Qdrant.

This script should be run once locally to generate the vector embeddings.
After ingestion, export a Qdrant snapshot and upload to S3 for deployment.

Usage:
    # Full ingestion (232K documents) - takes several hours
    python scripts/ingest_corpus.py --all

    # Limited ingestion for testing
    python scripts/ingest_corpus.py --limit 1000

    # Specific jurisdiction
    python scripts/ingest_corpus.py --jurisdiction commonwealth --limit 5000

    # Resume from a specific offset
    python scripts/ingest_corpus.py --offset 50000 --limit 10000
"""

import argparse
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.database import get_qdrant_client, get_postgres_engine
from src.ingestion.pipeline import IngestionPipeline
from src.models import ChunkingStrategy, Jurisdiction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Australian Legal Corpus into Qdrant"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest entire corpus (232K documents)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to ingest",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start from this document offset (for resuming)",
    )
    parser.add_argument(
        "--jurisdiction",
        type=str,
        choices=[j.value for j in Jurisdiction],
        default=None,
        help="Filter to specific jurisdiction",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        choices=[s.value for s in ChunkingStrategy],
        default=["recursive"],
        help="Chunking strategies to use (default: recursive)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=None,
        help="Qdrant host (overrides QDRANT_HOST env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.all and args.limit is None:
        print("Error: Must specify --all or --limit")
        print("Use --all for full corpus or --limit N for testing")
        sys.exit(1)

    # Get settings
    settings = get_settings()

    if args.qdrant_host:
        settings.qdrant_host = args.qdrant_host

    print("=" * 60)
    print("Australian Legal Corpus Ingestion")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Strategies: {args.strategies}")
    print(f"Jurisdiction: {args.jurisdiction or 'All'}")
    print(f"Limit: {args.limit or 'None (full corpus)'}")
    print(f"Offset: {args.offset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print("=" * 60)

    # Initialize clients
    print("\nInitializing connections...")
    qdrant_client = get_qdrant_client(settings=settings)
    postgres_engine = get_postgres_engine(settings.postgres_url)

    # Initialize pipeline
    pipeline = IngestionPipeline(
        qdrant_client=qdrant_client,
        postgres_engine=postgres_engine,
        openai_api_key=settings.openai_api_key,
        settings=settings,
    )

    # Parse jurisdiction
    jurisdiction = None
    if args.jurisdiction:
        jurisdiction = Jurisdiction(args.jurisdiction)

    # Parse strategies
    strategies = [ChunkingStrategy(s) for s in args.strategies]

    # Run ingestion
    print("\nStarting ingestion...")
    start_time = time.time()

    try:
        stats = pipeline.ingest(
            strategies=strategies,
            limit=args.limit,
            jurisdiction_filter=jurisdiction,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Ingestion Complete")
        print("=" * 60)
        print(f"Documents processed: {stats.documents_processed}")
        print(f"Chunks created: {stats.chunks_created}")
        print(f"Errors: {stats.errors}")
        print(f"Time elapsed: {elapsed / 60:.1f} minutes")
        print(f"Rate: {stats.documents_processed / elapsed * 60:.1f} docs/minute")
        print("=" * 60)

        if stats.errors > 0:
            print("\nWarning: Some documents failed to process.")
            print("Check logs for details.")

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        print("You can resume using --offset to skip processed documents.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()

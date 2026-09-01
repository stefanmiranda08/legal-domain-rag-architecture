#!/usr/bin/env python3
"""
Script to export/import Qdrant snapshots to/from S3.

This enables persisting the vector database across deployments without
re-running expensive embedding generation.

Usage:
    # Export snapshot to S3
    python scripts/snapshot_qdrant.py export --bucket my-bucket

    # Import snapshot from S3
    python scripts/snapshot_qdrant.py import --bucket my-bucket --snapshot-name snapshot_2024...

    # List available snapshots
    python scripts/snapshot_qdrant.py list --bucket my-bucket
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import boto3
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient

from src.models import ChunkingStrategy

COLLECTIONS = [f"legal_chunks_{s.value}" for s in ChunkingStrategy]


def get_qdrant_client(host: str, port: int) -> QdrantClient:
    """Create Qdrant client."""
    return QdrantClient(host=host, port=port)


def export_snapshots(
    qdrant_client: QdrantClient,
    s3_bucket: str,
    s3_prefix: str = "qdrant-snapshots",
) -> list[str]:
    """
    Export Qdrant collection snapshots to S3.

    Returns list of uploaded S3 keys.
    """
    s3 = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploaded = []

    for collection_name in COLLECTIONS:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            print(f"Skipping {collection_name} (does not exist)")
            continue

        print(f"Creating snapshot for {collection_name}...")

        # Create snapshot
        snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)
        snapshot_name = snapshot_info.name

        print(f"  Snapshot created: {snapshot_name}")

        # Download snapshot from Qdrant
        with tempfile.NamedTemporaryFile(suffix=".snapshot", delete=False) as tmp:
            tmp_path = tmp.name

        # Get snapshot URL and download
        snapshot_url = (
            f"http://{qdrant_client._client._host}:{qdrant_client._client._port}"
            f"/collections/{collection_name}/snapshots/{snapshot_name}"
        )

        import httpx

        response = httpx.get(snapshot_url, follow_redirects=True)
        response.raise_for_status()

        with open(tmp_path, "wb") as f:
            f.write(response.content)

        # Upload to S3
        s3_key = f"{s3_prefix}/{timestamp}/{collection_name}/{snapshot_name}"
        print(f"  Uploading to s3://{s3_bucket}/{s3_key}...")

        s3.upload_file(tmp_path, s3_bucket, s3_key)
        uploaded.append(s3_key)

        # Cleanup
        os.unlink(tmp_path)

        print("  Done!")

    # Write manifest
    manifest_key = f"{s3_prefix}/{timestamp}/manifest.txt"
    manifest_content = "\n".join(uploaded)
    s3.put_object(Bucket=s3_bucket, Key=manifest_key, Body=manifest_content)

    print(f"\nManifest written to s3://{s3_bucket}/{manifest_key}")
    return uploaded


def import_snapshots(
    qdrant_client: QdrantClient,
    s3_bucket: str,
    snapshot_timestamp: str,
    s3_prefix: str = "qdrant-snapshots",
) -> None:
    """
    Import Qdrant collection snapshots from S3.
    """
    s3 = boto3.client("s3")

    # Read manifest
    manifest_key = f"{s3_prefix}/{snapshot_timestamp}/manifest.txt"
    try:
        response = s3.get_object(Bucket=s3_bucket, Key=manifest_key)
        manifest = response["Body"].read().decode("utf-8")
        snapshot_keys = manifest.strip().split("\n")
    except ClientError as e:
        print(f"Error reading manifest: {e}")
        sys.exit(1)

    for s3_key in snapshot_keys:
        # Parse collection name from key
        parts = s3_key.split("/")
        collection_name = parts[-2]
        snapshot_name = parts[-1]

        print(f"Restoring {collection_name} from {snapshot_name}...")

        # Download from S3
        with tempfile.NamedTemporaryFile(suffix=".snapshot", delete=False) as tmp:
            tmp_path = tmp.name

        s3.download_file(s3_bucket, s3_key, tmp_path)

        # Recover snapshot via Qdrant API
        # Note: This requires the snapshot file to be accessible to Qdrant
        # In practice, you'd upload to Qdrant's snapshot directory
        print(f"  Downloaded to {tmp_path}")
        print("  Note: Manual restore may be required for production.")

        # For local testing, you can use:
        # qdrant_client.recover_snapshot(collection_name, location=tmp_path)

        os.unlink(tmp_path)
        print("  Done!")


def list_snapshots(s3_bucket: str, s3_prefix: str = "qdrant-snapshots") -> None:
    """List available snapshots in S3."""
    s3 = boto3.client("s3")

    try:
        response = s3.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_prefix,
            Delimiter="/",
        )

        prefixes = response.get("CommonPrefixes", [])
        if not prefixes:
            print("No snapshots found.")
            return

        print("Available snapshots:")
        for prefix in prefixes:
            timestamp = prefix["Prefix"].rstrip("/").split("/")[-1]
            if timestamp != s3_prefix:
                print(f"  - {timestamp}")

    except ClientError as e:
        print(f"Error listing snapshots: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Export/import Qdrant snapshots to/from S3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export snapshots to S3")
    export_parser.add_argument("--bucket", required=True, help="S3 bucket name")
    export_parser.add_argument("--prefix", default="qdrant-snapshots", help="S3 prefix")
    export_parser.add_argument("--qdrant-host", default="localhost")
    export_parser.add_argument("--qdrant-port", type=int, default=6333)

    # Import command
    import_parser = subparsers.add_parser("import", help="Import snapshots from S3")
    import_parser.add_argument("--bucket", required=True, help="S3 bucket name")
    import_parser.add_argument("--snapshot", required=True, help="Snapshot timestamp to import")
    import_parser.add_argument("--prefix", default="qdrant-snapshots", help="S3 prefix")
    import_parser.add_argument("--qdrant-host", default="localhost")
    import_parser.add_argument("--qdrant-port", type=int, default=6333)

    # List command
    list_parser = subparsers.add_parser("list", help="List available snapshots")
    list_parser.add_argument("--bucket", required=True, help="S3 bucket name")
    list_parser.add_argument("--prefix", default="qdrant-snapshots", help="S3 prefix")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "export":
        client = get_qdrant_client(args.qdrant_host, args.qdrant_port)
        uploaded = export_snapshots(client, args.bucket, args.prefix)
        print(f"\nExported {len(uploaded)} snapshots to S3.")

    elif args.command == "import":
        client = get_qdrant_client(args.qdrant_host, args.qdrant_port)
        import_snapshots(client, args.bucket, args.snapshot, args.prefix)

    elif args.command == "list":
        list_snapshots(args.bucket, args.prefix)


if __name__ == "__main__":
    main()

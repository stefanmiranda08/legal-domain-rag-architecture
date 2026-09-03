"""Load documents from the Open Australian Legal Corpus."""

import logging
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

from src.models import DocumentType, Jurisdiction

logger = logging.getLogger(__name__)

CORPUS_DATASET = "isaacus/open-australian-legal-corpus"


@dataclass
class DocumentRecord:
    """A document record from the corpus."""

    version_id: str
    citation: str
    document_type: DocumentType
    jurisdiction: Jurisdiction
    source: str
    date: str | None
    url: str | None
    text: str


def parse_document_type(type_str: str) -> DocumentType:
    """Parse document type string to enum."""
    mapping = {
        "primary_legislation": DocumentType.PRIMARY_LEGISLATION,
        "secondary_legislation": DocumentType.SECONDARY_LEGISLATION,
        "bill": DocumentType.BILL,
        "decision": DocumentType.DECISION,
    }
    return mapping[type_str]


def parse_jurisdiction(jurisdiction_str: str) -> Jurisdiction:
    """Parse jurisdiction string to enum."""
    mapping = {
        "commonwealth": Jurisdiction.COMMONWEALTH,
        "new_south_wales": Jurisdiction.NEW_SOUTH_WALES,
        "queensland": Jurisdiction.QUEENSLAND,
        "western_australia": Jurisdiction.WESTERN_AUSTRALIA,
        "south_australia": Jurisdiction.SOUTH_AUSTRALIA,
        "tasmania": Jurisdiction.TASMANIA,
        "norfolk_island": Jurisdiction.NORFOLK_ISLAND,
    }
    return mapping[jurisdiction_str]


def parse_document_metadata(raw: dict) -> DocumentRecord:
    """
    Parse a raw document from the HuggingFace dataset.

    Args:
        raw: Raw document dictionary from the dataset.

    Returns:
        Parsed DocumentRecord.
    """
    return DocumentRecord(
        version_id=raw["version_id"],
        citation=raw["citation"],
        document_type=parse_document_type(raw["type"]),
        jurisdiction=parse_jurisdiction(raw["jurisdiction"]),
        source=raw["source"],
        date=raw.get("date"),
        url=raw.get("url"),
        text=raw["text"],
    )


def load_corpus(
    limit: int | None = None,
    jurisdiction_filter: Jurisdiction | None = None,
) -> Iterator[DocumentRecord]:
    """
    Load documents from the Open Australian Legal Corpus.

    Uses streaming mode to handle the large dataset without
    downloading everything to disk.

    Args:
        limit: Maximum number of documents to load (None for all).
        jurisdiction_filter: Only load documents from this jurisdiction.

    Yields:
        DocumentRecord for each document in the corpus.
    """
    logger.info(f"Connecting to HuggingFace dataset: {CORPUS_DATASET}")
    logger.info(f"  Jurisdiction filter: {jurisdiction_filter.value if jurisdiction_filter else 'None'}")
    logger.info(f"  Limit: {limit if limit else 'None'}")

    dataset = load_dataset(
        CORPUS_DATASET,
        split="corpus",
        streaming=True,
    )
    logger.info("Dataset stream initialized, beginning iteration...")

    count = 0
    skipped_jurisdiction = 0
    for item in dataset:
        # Apply jurisdiction filter if specified
        if jurisdiction_filter is not None:
            if item["jurisdiction"] != jurisdiction_filter.value:
                skipped_jurisdiction += 1
                if skipped_jurisdiction % 10000 == 0:
                    logger.info(f"  Skipped {skipped_jurisdiction} non-{jurisdiction_filter.value} docs...")
                continue

        try:
            record = parse_document_metadata(item)
            yield record
            count += 1

            if limit is not None and count >= limit:
                break
        except (KeyError, ValueError) as e:
            # Skip malformed documents
            logger.debug(f"Skipping malformed document: {e}")
            continue

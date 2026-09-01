"""Integration tests for the ingestion pipeline."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from src.database import get_postgres_engine, get_qdrant_client, init_postgres_schema
from src.ingestion.loader import (
    DocumentRecord,
    load_corpus,
    parse_document_metadata,
)
from src.ingestion.pipeline import (
    IngestionPipeline,
    generate_embeddings,
)
from src.models import ChunkingStrategy, DocumentType, Jurisdiction


class TestDocumentRecord:
    """Tests for DocumentRecord dataclass."""

    def test_document_record_creation(self):
        """DocumentRecord should store all fields."""
        record = DocumentRecord(
            version_id="doc_123",
            citation="Smith v Jones [2020]",
            document_type=DocumentType.DECISION,
            jurisdiction=Jurisdiction.NEW_SOUTH_WALES,
            source="nsw_caselaw",
            date="2020-05-15",
            url="https://example.com",
            text="The court held...",
        )

        assert record.version_id == "doc_123"
        assert record.document_type == DocumentType.DECISION


class TestParseDocumentMetadata:
    """Tests for parsing document metadata from HuggingFace."""

    def test_parses_valid_document(self):
        """Should parse a valid document from the corpus."""
        raw = {
            "version_id": "doc_123",
            "citation": "Smith v Jones [2020]",
            "type": "decision",
            "jurisdiction": "new_south_wales",
            "source": "nsw_caselaw",
            "date": "2020-05-15",
            "url": "https://example.com",
            "text": "The court held...",
        }

        record = parse_document_metadata(raw)

        assert record.version_id == "doc_123"
        assert record.document_type == DocumentType.DECISION
        assert record.jurisdiction == Jurisdiction.NEW_SOUTH_WALES

    def test_handles_missing_date(self):
        """Should handle documents with null date."""
        raw = {
            "version_id": "doc_123",
            "citation": "Some Act",
            "type": "primary_legislation",
            "jurisdiction": "commonwealth",
            "source": "federal_register",
            "date": None,
            "url": None,
            "text": "Section 1...",
        }

        record = parse_document_metadata(raw)

        assert record.date is None
        assert record.url is None

    def test_handles_all_document_types(self):
        """Should parse all document types correctly."""
        type_mapping = {
            "primary_legislation": DocumentType.PRIMARY_LEGISLATION,
            "secondary_legislation": DocumentType.SECONDARY_LEGISLATION,
            "bill": DocumentType.BILL,
            "decision": DocumentType.DECISION,
        }

        for type_str, expected_type in type_mapping.items():
            raw = {
                "version_id": "doc",
                "citation": "Test",
                "type": type_str,
                "jurisdiction": "commonwealth",
                "source": "test",
                "date": None,
                "url": None,
                "text": "Test",
            }
            record = parse_document_metadata(raw)
            assert record.document_type == expected_type


class TestLoadCorpus:
    """Tests for corpus loading."""

    @patch("src.ingestion.loader.load_dataset")
    def test_load_corpus_with_limit(self, mock_load_dataset):
        """Should load corpus with limit."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "version_id": f"doc_{i}",
                        "citation": f"Case {i}",
                        "type": "decision",
                        "jurisdiction": "commonwealth",
                        "source": "test",
                        "date": None,
                        "url": None,
                        "text": f"Content {i}",
                    }
                    for i in range(5)
                ]
            )
        )
        mock_load_dataset.return_value = mock_dataset

        records = list(load_corpus(limit=3))

        assert len(records) == 3
        mock_load_dataset.assert_called_once()

    @patch("src.ingestion.loader.load_dataset")
    def test_load_corpus_streaming(self, mock_load_dataset):
        """Should use streaming mode for large dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset

        list(load_corpus(limit=1))

        # Verify streaming was enabled
        call_kwargs = mock_load_dataset.call_args[1]
        assert call_kwargs.get("streaming") is True


class TestGenerateEmbeddings:
    """Tests for embedding generation."""

    @patch("src.ingestion.pipeline.OpenAI")
    def test_generates_embeddings_for_texts(self, mock_openai_class):
        """Should generate embeddings for a list of texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock embedding response
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding, mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["First text", "Second text"]
        embeddings = generate_embeddings(texts, api_key="test-key")

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        mock_client.embeddings.create.assert_called_once()

    @patch("src.ingestion.pipeline.OpenAI")
    def test_handles_empty_list(self, mock_openai_class):
        """Should return empty list for empty input."""
        embeddings = generate_embeddings([], api_key="test-key")

        assert embeddings == []
        mock_openai_class.return_value.embeddings.create.assert_not_called()


class TestIngestionPipeline:
    """Tests for the full ingestion pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with in-memory databases."""
        qdrant = get_qdrant_client(in_memory=True)
        engine = get_postgres_engine(url="sqlite:///:memory:")
        init_postgres_schema(engine)

        return IngestionPipeline(
            qdrant_client=qdrant,
            postgres_engine=engine,
            openai_api_key="test-key",
        )

    def test_pipeline_initialization(self, pipeline):
        """Pipeline should initialize correctly."""
        assert pipeline.qdrant_client is not None
        assert pipeline.postgres_engine is not None

    @patch("src.ingestion.pipeline.generate_embeddings")
    def test_ingest_documents(self, mock_embed, pipeline):
        """Should ingest documents via _process_batch directly."""
        from src.ingestion.chunkers import get_chunker

        # Mock embeddings - return enough for any number of chunks
        def mock_embed_fn(texts, **kwargs):
            return [[0.1] * 1536 for _ in texts]

        mock_embed.side_effect = mock_embed_fn

        # Create test document
        test_doc = DocumentRecord(
            version_id="doc_1",
            citation="Test Case [2020]",
            document_type=DocumentType.DECISION,
            jurisdiction=Jurisdiction.COMMONWEALTH,
            source="test",
            date="2020-01-01",
            url="https://example.com",
            text="This is a test legal document with enough content. " * 50,
        )

        # Create collections
        from src.database import create_qdrant_collection

        create_qdrant_collection(
            pipeline.qdrant_client,
            ChunkingStrategy.RECURSIVE,
            vector_size=1536,
        )

        # Create chunkers
        chunkers = {ChunkingStrategy.RECURSIVE: get_chunker(ChunkingStrategy.RECURSIVE)}

        # Process directly
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": 0,
        }

        pipeline._process_batch([test_doc], chunkers, stats)

        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] > 0
        assert stats["errors"] == 0

    @patch("src.ingestion.pipeline.generate_embeddings")
    @patch("src.ingestion.pipeline.load_corpus")
    def test_creates_collections_for_strategies(self, mock_load, mock_embed, pipeline):
        """Should create Qdrant collections for each strategy."""
        mock_load.return_value = iter([])

        pipeline.ingest(
            strategies=[ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE],
            limit=0,
        )

        # Collections should be created
        assert pipeline.qdrant_client.collection_exists("legal_chunks_fixed")
        assert pipeline.qdrant_client.collection_exists("legal_chunks_recursive")

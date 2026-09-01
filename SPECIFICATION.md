# Legal RAG Architecture - Functional Specification

## 1. Overview

A case law research assistant that enables natural language queries over Australian legal documents. The system retrieves relevant passages from court decisions, legislation, and bills, then generates answers with citations.

This project demonstrates mid-level AI engineering capabilities: RAG pipeline implementation, retrieval evaluation, infrastructure provisioning, and observability.

### 1.1 Core Capabilities

- Natural language queries over 232K Australian legal documents
- Metadata filtering by jurisdiction, document type, and date range
- Comparison of chunking strategies with measured retrieval performance
- Evaluation dashboard showing retrieval metrics and query logs

### 1.2 Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| Orchestration | LangChain |
| Vector Database | Qdrant (self-hosted on ECS) |
| Relational Database | PostgreSQL (RDS) |
| Embeddings | OpenAI text-embedding-3-small |
| Generation | OpenAI GPT-4o-mini |
| Dashboard | Streamlit |
| Infrastructure | AWS ECS Fargate, Terraform |
| CI/CD | GitHub Actions |

---

## 2. Data Source

### 2.1 Open Australian Legal Corpus

- **Source**: https://huggingface.co/datasets/isaacus/open-australian-legal-corpus
- **Size**: 232,560 documents, ~1.4B tokens
- **Format**: JSONL with structured metadata

### 2.2 Document Schema

| Field | Type | Description |
|-------|------|-------------|
| `version_id` | string | Unique document identifier |
| `type` | string | `primary_legislation`, `secondary_legislation`, `bill`, `decision` |
| `jurisdiction` | string | `commonwealth`, `new_south_wales`, `queensland`, `western_australia`, `south_australia`, `tasmania`, `norfolk_island` |
| `source` | string | Origin database (e.g., `federal_court_of_australia`, `nsw_caselaw`) |
| `citation` | string | Document title with jurisdiction |
| `date` | string | ISO 8601 date or null |
| `url` | string | Link to original document |
| `text` | string | Full document text |

### 2.3 Document Distribution

| Type | Count | Percentage |
|------|-------|------------|
| Decisions | 189,216 | 81.4% |
| Secondary Legislation | 31,696 | 13.6% |
| Primary Legislation | 9,059 | 3.9% |
| Bills | 2,589 | 1.1% |

---

## 3. Features

### 3.1 Query Interface

Users submit natural language questions about Australian law. The system retrieves relevant document chunks and generates an answer with citations.

**Example Query**:
> "What are the requirements for a valid will in New South Wales?"

**Response Structure**:
- Generated answer (2-4 paragraphs)
- List of source citations with document title, date, and relevant excerpt
- Metadata: query latency, chunks retrieved, tokens used

**Filtering Options**:
- `jurisdiction`: Limit to specific state/territory or commonwealth
- `document_type`: Filter by legislation, decisions, or bills
- `date_from` / `date_to`: Date range filter

### 3.2 Chunking Strategy Comparison

The system implements multiple chunking strategies and stores chunks from each in separate Qdrant collections. This enables side-by-side retrieval comparison.

**Strategies to Implement**:

| Strategy | Description |
|----------|-------------|
| Fixed-size | Split text every N tokens with M token overlap |
| Paragraph-based | Split on paragraph boundaries, merge small paragraphs |
| Recursive | LangChain RecursiveCharacterTextSplitter with legal-specific separators |

**Comparison Metrics**:
- Recall@5, Recall@10 on evaluation query set
- Mean Reciprocal Rank (MRR)
- Average chunk size and count per document

### 3.3 Evaluation Framework

A set of test queries with known relevant documents. The system runs retrieval against each chunking strategy and records metrics.

**Evaluation Data**:
- 50-100 manually curated query/relevant-document pairs
- Covers different jurisdictions, document types, and query complexities

**Stored Metrics**:
- Per-query: retrieved document IDs, ranks, relevance scores
- Aggregate: recall@k, MRR, latency percentiles

### 3.4 Observability Dashboard

Streamlit application displaying:

| Page | Content |
|------|---------|
| Chunking Comparison | Bar charts comparing recall@k across strategies, table of per-query results |
| Retrieval Metrics | Latency distribution, daily query volume, filter usage breakdown |
| Query Logs | Searchable log of queries with retrieved chunks and generated answers |

---

## 4. Architecture

### 4.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          AWS Cloud                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                        VPC                                 │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │                   ECS Cluster                        │  │ │
│  │  │                                                      │  │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │ │
│  │  │  │  FastAPI │ │  Worker  │ │  Qdrant  │ │Streamlit│ │  │ │
│  │  │  │ Service  │ │ Service  │ │ Service  │ │Dashboard│ │  │ │
│  │  │  │ (Fargate)│ │ (Fargate)│ │ (Fargate)│ │(Fargate)│ │  │ │
│  │  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ │  │ │
│  │  │       │            │            │            │       │  │ │
│  │  └───────┼────────────┼────────────┼────────────┼───────┘  │ │
│  │          │            │            │            │          │ │
│  │  ┌───────▼────────────▼────────────┼────────────▼───────┐  │ │
│  │  │              Private Subnets                         │  │ │
│  │  └─────────────────────────────────┼────────────────────┘  │ │
│  │                                    │                       │ │
│  │          ┌─────────────┐    ┌──────▼──────┐               │ │
│  │          │     RDS     │    │     EFS     │               │ │
│  │          │ (PostgreSQL)│    │  (Qdrant    │               │ │
│  │          │             │    │   storage)  │               │ │
│  │          └─────────────┘    └─────────────┘               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                       ┌──────▼──────┐                          │
│                       │     ALB     │                          │
│                       └──────┬──────┘                          │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                            Internet
```

### 4.2 Services

| Service | Purpose | Resources |
|---------|---------|-----------|
| FastAPI | REST API for queries and ingestion triggers | 0.5 vCPU, 1GB RAM |
| Worker | Background ingestion jobs | 1 vCPU, 4GB RAM |
| Qdrant | Vector storage and search | 2 vCPU, 8GB RAM |
| Dashboard | Streamlit observability UI | 0.25 vCPU, 0.5GB RAM |
| RDS | PostgreSQL for metadata and logs | db.t3.small |
| EFS | Persistent storage for Qdrant data | 20-30GB |

### 4.3 Data Flow

**Ingestion**:
```
HuggingFace Dataset
       │
       ▼
   Worker Service
       │
       ├──► Parse metadata ──► PostgreSQL (documents table)
       │
       ▼
   Chunk document (per strategy)
       │
       ▼
   Generate embeddings (OpenAI API)
       │
       ▼
   Store vectors ──► Qdrant (one collection per strategy)
```

**Query**:
```
User Query + Filters
       │
       ▼
   FastAPI Service
       │
       ├──► Log query ──► PostgreSQL (query_logs table)
       │
       ▼
   Build metadata filter
       │
       ▼
   Vector search ──► Qdrant
       │
       ▼
   Retrieve top-K chunks
       │
       ▼
   Fetch full document context ──► PostgreSQL
       │
       ▼
   Generate answer ──► OpenAI GPT
       │
       ▼
   Format citations
       │
       ▼
   Return response + log metrics ──► PostgreSQL
```

---

## 5. API Specification

### 5.1 Endpoints

#### POST /query

Submit a natural language query.

**Request**:
```json
{
  "query": "What constitutes negligence in medical malpractice cases?",
  "filters": {
    "jurisdiction": "new_south_wales",
    "document_type": "decision",
    "date_from": "2015-01-01",
    "date_to": "2023-12-31"
  },
  "chunking_strategy": "recursive",
  "top_k": 10
}
```

**Response**:
```json
{
  "answer": "In New South Wales, medical negligence is established when...",
  "citations": [
    {
      "document_id": "abc123",
      "citation": "Smith v Sydney Hospital [2019] NSWSC 456",
      "excerpt": "The standard of care required of a medical practitioner...",
      "date": "2019-03-15",
      "relevance_score": 0.89
    }
  ],
  "metadata": {
    "query_id": "q_789",
    "latency_ms": 1250,
    "chunks_retrieved": 10,
    "tokens_used": 2340
  }
}
```

#### POST /ingest

Trigger corpus ingestion. Runs asynchronously.

**Request**:
```json
{
  "chunking_strategies": ["fixed", "paragraph", "recursive"],
  "limit": null,
  "jurisdiction_filter": null
}
```

**Response**:
```json
{
  "job_id": "ingest_001",
  "status": "started",
  "estimated_documents": 232560
}
```

#### GET /ingest/{job_id}

Check ingestion job status.

**Response**:
```json
{
  "job_id": "ingest_001",
  "status": "running",
  "documents_processed": 45000,
  "documents_total": 232560,
  "errors": 12,
  "started_at": "2024-01-15T10:30:00Z"
}
```

#### POST /evaluate

Run evaluation suite against a chunking strategy.

**Request**:
```json
{
  "chunking_strategy": "recursive",
  "test_set_id": "default"
}
```

**Response**:
```json
{
  "evaluation_id": "eval_456",
  "strategy": "recursive",
  "metrics": {
    "recall_at_5": 0.72,
    "recall_at_10": 0.85,
    "mrr": 0.68,
    "avg_latency_ms": 145
  },
  "per_query_results": [
    {
      "query_id": "test_q1",
      "query": "requirements for valid contract",
      "expected_doc_ids": ["doc1", "doc2"],
      "retrieved_doc_ids": ["doc1", "doc3", "doc2"],
      "recall_at_5": 1.0
    }
  ]
}
```

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "qdrant": "connected",
  "postgres": "connected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 6. Data Models

### 6.1 PostgreSQL Schema

```sql
-- Source documents metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id VARCHAR(255) UNIQUE NOT NULL,
    citation TEXT NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    jurisdiction VARCHAR(50) NOT NULL,
    source VARCHAR(100) NOT NULL,
    date DATE,
    url TEXT,
    text_length INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_jurisdiction ON documents(jurisdiction);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_date ON documents(date);

-- Chunk metadata (maps Qdrant vectors back to documents)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    chunking_strategy VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER,
    qdrant_point_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_strategy ON chunks(chunking_strategy);

-- Query logs for observability
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    filters JSONB,
    chunking_strategy VARCHAR(50),
    chunks_retrieved INTEGER,
    latency_ms INTEGER,
    tokens_used INTEGER,
    answer_text TEXT,
    citations JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_query_logs_created ON query_logs(created_at);

-- Evaluation results
CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunking_strategy VARCHAR(50) NOT NULL,
    test_set_id VARCHAR(100) NOT NULL,
    recall_at_5 FLOAT,
    recall_at_10 FLOAT,
    mrr FLOAT,
    avg_latency_ms FLOAT,
    per_query_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eval_runs_strategy ON evaluation_runs(chunking_strategy);

-- Ingestion job tracking
CREATE TABLE ingestion_jobs (
    id VARCHAR(100) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    chunking_strategies TEXT[],
    documents_total INTEGER,
    documents_processed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Evaluation test sets
CREATE TABLE test_queries (
    id VARCHAR(100) PRIMARY KEY,
    test_set_id VARCHAR(100) NOT NULL,
    query_text TEXT NOT NULL,
    expected_document_ids TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 6.2 Qdrant Collections

One collection per chunking strategy, enabling isolated comparison.

**Collection naming**: `legal_chunks_{strategy}`
- `legal_chunks_fixed`
- `legal_chunks_paragraph`
- `legal_chunks_recursive`

**Vector configuration**:
- Dimensions: 1536 (text-embedding-3-small)
- Distance metric: Cosine

**Payload schema** (stored with each vector):
```json
{
  "document_id": "uuid",
  "chunk_id": "uuid",
  "chunk_index": 0,
  "jurisdiction": "new_south_wales",
  "document_type": "decision",
  "date": "2019-03-15",
  "citation": "Smith v Sydney Hospital [2019] NSWSC 456"
}
```

Metadata filtering uses Qdrant payload indexes on `jurisdiction`, `document_type`, and `date`.

---

## 7. Component Specifications

### 7.1 Ingestion Pipeline

**Location**: `src/ingestion/`

**Responsibilities**:
1. Load dataset from HuggingFace using streaming to handle size
2. Parse and validate document metadata
3. Store document records in PostgreSQL
4. Chunk each document using all configured strategies
5. Generate embeddings via OpenAI API (batch requests, rate limiting)
6. Store vectors in appropriate Qdrant collection
7. Track progress and errors in PostgreSQL

**Chunking Configurations**:

| Strategy | Parameters |
|----------|------------|
| Fixed | chunk_size=512 tokens, overlap=50 tokens |
| Paragraph | min_chunk_size=100 tokens, max_chunk_size=1000 tokens |
| Recursive | chunk_size=512, overlap=50, separators=["\n\n", "\n", ". ", " "] |

**Embedding Generation**:
- Model: text-embedding-3-small
- Batch size: 100 texts per API call
- Rate limiting: respect OpenAI limits, implement exponential backoff
- Cost: ~$0.02 per 1M tokens

### 7.2 Retrieval Module

**Location**: `src/retrieval.py`

**Responsibilities**:
1. Parse user query and filters
2. Generate query embedding
3. Build Qdrant filter from metadata parameters
4. Execute vector search with filter
5. Return ranked chunks with scores

**Filter Construction**:
```python
# Example filter for jurisdiction + date range
{
    "must": [
        {"key": "jurisdiction", "match": {"value": "new_south_wales"}},
        {"key": "date", "range": {"gte": "2015-01-01", "lte": "2023-12-31"}}
    ]
}
```

### 7.3 Generation Module

**Location**: `src/generation.py`

**Responsibilities**:
1. Construct prompt with retrieved chunks as context
2. Call OpenAI GPT-4o-mini for answer generation
3. Extract and format citations
4. Handle context length limits (truncate if necessary)

**Prompt Template**:
```
You are a legal research assistant specializing in Australian law.
Answer the user's question based on the provided legal documents.
Cite specific cases or legislation in your answer using the format [Citation].

Documents:
{retrieved_chunks}

Question: {user_query}

Provide a clear, accurate answer with citations to the source documents.
```

### 7.4 Evaluation Module

**Location**: `src/evaluation/`

**Responsibilities**:
1. Load test query set from PostgreSQL
2. Run retrieval for each query against specified chunking strategy
3. Calculate recall@k by comparing retrieved docs to expected docs
4. Calculate MRR (Mean Reciprocal Rank)
5. Measure latency for each query
6. Store results in PostgreSQL

**Metrics Definitions**:
- **Recall@K**: Proportion of expected documents found in top K results
- **MRR**: Average of 1/rank for first relevant document across queries

### 7.5 Dashboard

**Location**: `dashboard/`

**Pages**:

1. **Chunking Comparison** (`pages/chunking_comparison.py`)
   - Select evaluation runs to compare
   - Bar chart: recall@5, recall@10, MRR per strategy
   - Table: per-query breakdown with expandable details

2. **Retrieval Metrics** (`pages/retrieval_metrics.py`)
   - Time series: query volume over time
   - Histogram: latency distribution
   - Pie chart: filter usage (by jurisdiction, document type)

3. **Query Logs** (`pages/query_logs.py`)
   - Searchable table of recent queries
   - Expandable rows showing retrieved chunks and generated answer
   - Export functionality

---

## 8. Infrastructure

### 8.1 AWS Resources

Provisioned via Terraform in `infrastructure/`.

| Resource | Purpose | Terraform File |
|----------|---------|----------------|
| VPC | Network isolation | `vpc.tf` |
| ECS Cluster | Container orchestration | `ecs.tf` |
| ECS Services (4) | FastAPI, Worker, Qdrant, Dashboard | `ecs.tf` |
| RDS PostgreSQL | Relational data | `rds.tf` |
| EFS | Qdrant persistent storage | `efs.tf` |
| ALB | Load balancer for API/Dashboard | `alb.tf` |
| ECR | Container image registry | `ecr.tf` |
| CloudWatch | Logs and metrics | `cloudwatch.tf` |
| S3 | Embeddings backup, Terraform state | `s3.tf` |

### 8.2 Deployment Process

**Initial Setup**:
1. Configure AWS credentials
2. Run `terraform init` and `terraform apply`
3. Build and push Docker images to ECR
4. ECS pulls images and starts services

**Deploy Only When Demoing**:
1. `terraform apply` to create infrastructure
2. Demo the application
3. `terraform destroy` to tear down (EFS data persists in S3 backup)

**CI/CD Pipeline** (GitHub Actions):
1. On push to main: run tests, lint
2. On release tag: build images, push to ECR, update ECS services

### 8.3 Cost Management

Infrastructure is designed for on-demand deployment. Expected costs:

| Scenario | Monthly Cost |
|----------|--------------|
| Always running | ~$150 |
| 10 hours/month demos | ~$3-5 |
| Shut down (only S3 state) | ~$0.50 |

S3 stores:
- Terraform state file
- Pre-computed embeddings backup (to avoid re-embedding on redeploy)

---

## 9. Project Structure

```
legal-rag/
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Environment settings
│   ├── models.py                 # SQLAlchemy + Pydantic schemas
│   ├── database.py               # DB connection management
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # HuggingFace dataset loading
│   │   ├── chunkers.py           # Chunking strategy implementations
│   │   └── pipeline.py           # Ingestion orchestration
│   │
│   ├── retrieval.py              # Vector search with filters
│   ├── generation.py             # LLM answer generation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Recall, MRR calculations
│   │   └── runner.py             # Evaluation orchestration
│   │
│   └── api.py                    # FastAPI application
│
├── dashboard/
│   ├── app.py                    # Streamlit entry point
│   ├── pages/
│   │   ├── chunking_comparison.py
│   │   ├── retrieval_metrics.py
│   │   └── query_logs.py
│   └── Dockerfile
│
├── scripts/
│   ├── ingest.py                 # CLI for ingestion
│   └── evaluate.py               # CLI for evaluation
│
├── tests/
│   ├── conftest.py
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── infrastructure/
│   ├── main.tf
│   ├── variables.tf
│   ├── vpc.tf
│   ├── ecs.tf
│   ├── rds.tf
│   ├── efs.tf
│   ├── alb.tf
│   ├── ecr.tf
│   ├── cloudwatch.tf
│   ├── s3.tf
│   └── outputs.tf
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── pyproject.toml
├── .env.example
├── SPECIFICATION.md
└── README.md
```

---

## 10. Development Workflow

### 10.1 Local Development

```bash
# Start local services
docker-compose up -d

# Run API in development mode
uvicorn src.api:app --reload

# Run dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/
```

### 10.2 Ingestion

```bash
# Full corpus ingestion (all strategies)
python scripts/ingest.py --all-strategies

# Subset for testing
python scripts/ingest.py --limit 1000 --strategy recursive
```

### 10.3 Evaluation

```bash
# Run evaluation on all strategies
python scripts/evaluate.py --all-strategies

# Single strategy
python scripts/evaluate.py --strategy recursive
```

---

## 11. Future Enhancements

Not in scope for initial implementation, but noted for documentation:

- **Hybrid search**: Combine vector similarity with BM25 keyword matching
- **Cross-encoder reranking**: Improve precision with a second-stage ranker
- **User feedback loop**: Collect relevance judgments to improve retrieval
- **Citation graph**: Link cases that cite each other
- **Multi-turn conversation**: Maintain context across queries

# Session Handover - Legal RAG Architecture

## Project Overview

A legal domain RAG (Retrieval-Augmented Generation) system for querying Australian federal law. Built as an AI engineering portfolio project.

**Tech Stack**: Python, FastAPI, LangChain, Qdrant, PostgreSQL, OpenAI, Streamlit, AWS ECS Fargate, Terraform

**GitHub**: https://github.com/stefanmiranda08/legal-domain-rag-architecture

---

## Current State (as of handover)

### In Progress: Data Ingestion

The user is currently running the full Commonwealth dataset ingestion:

```bash
caffeinate -i uv run python scripts/ingest_corpus.py --all
```

**Dataset**: Commonwealth subset of Open Australian Legal Corpus
- ~72,000 documents (Federal Court + High Court + federal legislation)
- ~750M tokens
- Estimated cost: ~$15 in OpenAI embedding API calls
- Estimated time: Several hours

**Check progress**:
```bash
# If running in background
tail -f ingestion.log

# Check database counts
docker exec legal-rag-postgres psql -U legal_rag -d legal_rag -c "SELECT COUNT(*) FROM documents;"
```

---

## What's Been Completed

### Implementation (12 Stages)
1. Project setup with config and data models
2. Database connections (Qdrant, PostgreSQL)
3. Chunking strategies (fixed, paragraph, recursive)
4. Ingestion pipeline
5. Retrieval module with vector search and filtering
6. Generation module with LLM answer synthesis
7. FastAPI application with REST endpoints
8. Evaluation framework with metrics
9. Streamlit observability dashboard + chat interface
10. Docker configuration
11. Terraform AWS infrastructure
12. GitHub Actions CI/CD

**Tests**: 103 passing

### Key Commits
- All code is pushed to GitHub
- CI workflow has lint as non-blocking (continues on error)

---

## What's Next (After Ingestion Completes)

1. **Verify ingestion success**:
   ```bash
   docker exec legal-rag-postgres psql -U legal_rag -d legal_rag -c "SELECT COUNT(*) FROM documents;"
   curl http://localhost:6333/collections/legal_chunks_recursive
   ```

2. **Test the RAG system locally**:
   ```bash
   # Start the API
   uv run uvicorn src.api:app --reload

   # Start the dashboard (in another terminal)
   uv run streamlit run dashboard/app.py

   # Test a query
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the test for negligence?"}'
   ```

3. **Export Qdrant snapshot to S3** (for deployment):
   ```bash
   python scripts/snapshot_qdrant.py export --bucket legal-rag-snapshots-ACCOUNT_ID
   ```

4. **Deploy to AWS**:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with secrets
   terraform init
   terraform apply
   ```

---

## Key Technical Details

### Architecture
- **PostgreSQL**: Document metadata, chunk text, query logs, evaluation results
- **Qdrant**: Vector embeddings (1536 dimensions, cosine similarity)
- **Flow**: Query → embed → Qdrant search → get chunks → LLM generates answer with citations

### Environment Variables Required
```bash
export OPENAI_API_KEY="sk-..."
export POSTGRES_USER=legal_rag
export POSTGRES_PASSWORD=legal_rag_password
export POSTGRES_DB=legal_rag
```

### Docker Services (Local Dev)
```bash
# PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=legal_rag \
  -e POSTGRES_PASSWORD=legal_rag_password \
  -e POSTGRES_DB=legal_rag \
  --name legal-rag-postgres \
  postgres:15-alpine

# Qdrant
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### Database Tables
- `documents` - Document metadata (version_id is unique key)
- `chunks` - Chunk text and Qdrant point ID reference
- `query_logs` - Query history
- `evaluation_runs` - Evaluation metrics

---

## Common Issues & Solutions

### 1. "role does not exist" PostgreSQL error
Stop any local Homebrew postgres that conflicts with Docker:
```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/homebrew.mxcl.postgresql@18.plist
pkill -9 -f postgres
```

### 2. OpenAI API key not found
Must be **exported**, not just set:
```bash
export OPENAI_API_KEY="sk-..."  # Not just OPENAI_API_KEY="sk-..."
```

### 3. HuggingFace network errors
Download dataset locally first - it caches to `~/.cache/huggingface/datasets/`

### 4. Duplicate key errors during ingestion
Pipeline is now idempotent - skips existing documents. Or clear DB:
```bash
docker exec legal-rag-postgres psql -U legal_rag -d legal_rag -c "TRUNCATE documents, chunks CASCADE;"
curl -X DELETE "http://localhost:6333/collections/legal_chunks_recursive"
```

### 5. Git push fails for workflow files
```bash
gh auth refresh -s workflow
TOKEN=$(gh auth token) && git push https://stefanmiranda08:${TOKEN}@github.com/stefanmiranda08/legal-domain-rag-architecture.git main
```

---

## File Structure

```
├── src/                    # Core application
│   ├── api.py             # FastAPI endpoints
│   ├── config.py          # Settings (pydantic-settings)
│   ├── models.py          # SQLAlchemy + Pydantic schemas
│   ├── database.py        # DB connections
│   ├── retrieval.py       # Vector search
│   ├── generation.py      # LLM answer generation
│   ├── ingestion/         # Data pipeline
│   └── evaluation/        # Metrics framework
├── dashboard/             # Streamlit UI
│   └── pages/
│       ├── 0_chat.py      # Chat interface (main UI)
│       ├── 1_chunking_comparison.py
│       ├── 2_retrieval_metrics.py
│       └── 3_query_logs.py
├── scripts/
│   ├── ingest_corpus.py   # Ingestion CLI
│   └── snapshot_qdrant.py # S3 export/import
├── terraform/             # AWS infrastructure
├── tests/                 # 103 tests
├── SPECIFICATION.md       # Full functional spec
└── docker-compose.yml     # Local dev setup
```

---

## Evaluation Test Queries

30 questions defined in SPECIFICATION.md covering:
- Constitutional law (5)
- Administrative law (5)
- Corporations law (5)
- Taxation law (5)
- Immigration law (5)
- Other federal (consumer, employment, native title, copyright) (5)

---

## User Preferences

- No "Co-Authored-By: Claude Code" in commit messages
- Lint failures should not block CI/CD pipeline
- Commonwealth subset only (~$15 budget for embeddings)
- Deploy only when demoing (to minimize AWS costs)

# Session Handover - Legal RAG Architecture

## Project Overview

A legal domain RAG (Retrieval-Augmented Generation) system for querying Australian federal law. Built as an AI engineering portfolio project.

**Tech Stack**: Python, FastAPI, LangChain, Qdrant, PostgreSQL, OpenAI, Streamlit, AWS ECS Fargate, Terraform

**GitHub**: https://github.com/stefanmiranda08/legal-domain-rag-architecture

---

## Current State (as of handover)

### Completed: Data Ingestion

Previous ingestion completed:
- **99,220 Commonwealth documents** ingested
- **1,981,579 chunks** in PostgreSQL
- **1,981,579 vectors** in Qdrant (recursive strategy only)

**Note**: Existing data does not have chunk links. Re-ingestion required to use section reconstruction.

### Completed: Evaluation Harness

Evaluation harness working with fixes:
- `c.score` → `c.relevance_score`
- `gpt-4o-mini` → `gpt-5.4` (API access)
- `max_tokens` → `max_completion_tokens`

**Recursive Chunking Evaluation Results** (without reconstruction):
| Metric | Score |
|--------|-------|
| Faithfulness | 0.567 |
| Answer Relevancy | 0.950 |
| Context Precision | 0.936 |

### Completed: Linked Chunk Architecture

Implemented bottom-up section reconstruction:

**Files changed:**
- `src/models.py` - Added `prev_chunk_id`, `next_chunk_id`, `is_section_start`, `is_section_end` to Chunk model
- `src/ingestion/pipeline.py` - Links chunks bidirectionally during ingestion
- `src/reconstruction.py` - New module for LLM-based section boundary detection
- `src/retrieval.py` - Added `search_with_reconstruction()` function
- `scripts/migrations/001_add_chunk_links.sql` - Database migration
- `scripts/run_migration.py` - Migration runner
- `SPECIFICATION.md` - Updated with new chunking approach
- `dashboard/pages/5_engineering_decisions.py` - Updated documentation

---

## Before Re-Ingestion

Run the database migration to add new columns:

```bash
uv run python scripts/run_migration.py 001_add_chunk_links.sql
```

Then clear existing data and re-ingest:

```bash
# Clear existing chunks (optional - or create new collections)
# Re-run ingestion
uv run python scripts/ingest_corpus.py --strategies recursive --jurisdiction commonwealth
```

---

## How Section Reconstruction Works

1. **Ingestion**: Chunks are linked sequentially (`prev_chunk_id`, `next_chunk_id`)
2. **Retrieval**: Vector search returns initial chunks
3. **Extension**: For each chunk, fetch prev/next and combine
4. **Boundary Detection**: LLM determines if combined text is a complete section
5. **Caching**: Boundary decisions cached in `is_section_start`/`is_section_end`

This avoids brittle document parsing while reconstructing complete semantic units.

---

## Running the System

### Prerequisites
```bash
# Ensure Docker containers are running
docker ps  # Should see postgres and qdrant

# Verify Qdrant
curl http://localhost:6333/healthz

# Set environment variables
export OPENAI_API_KEY="sk-..."
```

### Start the Application
```bash
# Terminal 1: API
uv run uvicorn src.api:app --reload

# Terminal 2: Dashboard
uv run streamlit run dashboard/app.py
```

### Access Points
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Engineering Decisions: http://localhost:8501 (page 5)

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models.py` | Chunk model with navigation links |
| `src/ingestion/pipeline.py` | Ingestion with chunk linking |
| `src/reconstruction.py` | Section boundary detection |
| `src/retrieval.py` | Retrieval with reconstruction |
| `src/evaluation/harness.py` | LLM-as-judge evaluation |
| `dashboard/pages/5_engineering_decisions.py` | Design documentation |
| `SPECIFICATION.md` | Full system specification |

---

## User Preferences

- No "Co-Authored-By: Claude Code" in commit messages
- Lint failures should not block CI/CD pipeline
- Commonwealth subset only (~$15 budget for embeddings)
- Deploy only when demoing (to minimize AWS costs)
- Prefer lightweight frameworks over heavyweight ones
- Use `.env` file for configuration

# Session Handover

## Project Overview

Legal domain RAG system for querying Australian federal law. Portfolio project demonstrating AI engineering capabilities.

**Tech Stack**: Python, FastAPI, Qdrant, PostgreSQL, OpenAI, Streamlit

**GitHub**: https://github.com/stefanmiranda08/legal-domain-rag-architecture

---

## Current State

### Data
- **PostgreSQL**: Empty (truncated)
- **Qdrant**: Empty (collections deleted)
- **Migration applied**: Chunk linking columns added (`prev_chunk_id`, `next_chunk_id`, `is_section_start`, `is_section_end`)

### Ready for Re-Ingestion
The linked chunk architecture is implemented. Next step is to ingest documents with chunk linking enabled.

```bash
uv run python scripts/ingest_corpus.py --strategies recursive --jurisdiction commonwealth
```

This will:
- Ingest ~99,220 Commonwealth documents
- Create ~2M chunks with bidirectional links
- Cost ~$15 in embedding API calls

---

## What Was Implemented This Session

### Linked Chunk Architecture

**Problem**: Low faithfulness (0.567) despite high relevancy/precision. Recursive chunking truncates long legal sections, causing LLM to extrapolate from training data.

**Solution**: Bottom-up section reconstruction
- Chunks linked sequentially during ingestion
- During retrieval, extend bidirectionally and use LLM to detect section boundaries
- No brittle document parsing required

**Key Files**:
| File | Purpose |
|------|---------|
| `src/models.py` | Chunk model with `prev_chunk_id`, `next_chunk_id` |
| `src/ingestion/pipeline.py` | Links chunks during ingestion |
| `src/reconstruction.py` | LLM boundary detection and section reconstruction |
| `src/retrieval.py` | `search_with_reconstruction()` function |

### Evaluation Harness Fixes
- `c.score` → `c.relevance_score`
- Judge model: `gpt-4o-mini` → `gpt-5.4`
- API param: `max_tokens` → `max_completion_tokens`

---

## After Ingestion: Run Evaluation

```bash
uv run python scripts/run_evaluation.py --name "linked_chunks" --chunking-strategy recursive --top-k 10 --llm-model gpt-5.4 --system-prompt professional
```

Compare faithfulness score against baseline (0.567).

---

## Running the System

```bash
# Verify Docker containers
docker ps  # postgres and qdrant

# Start API
uv run uvicorn src.api:app --reload

# Start Dashboard
uv run streamlit run dashboard/app.py
```

**Access**:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## Key Documentation

- `SPECIFICATION.md` — Full system specification
- `PROJECT_TIMELINE.md` — Session history and problem/solution log
- `dashboard/pages/5_engineering_decisions.py` — Design rationale (visible in dashboard)

---

## User Preferences

- No "Co-Authored-By: Claude Code" in commits
- Commonwealth subset only (~$15 embedding budget)
- Deploy only when demoing (minimize AWS costs)
- Lint failures should not block CI/CD

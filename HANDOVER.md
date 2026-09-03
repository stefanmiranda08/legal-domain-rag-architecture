# Session Handover - Legal RAG Architecture

## Project Overview

A legal domain RAG (Retrieval-Augmented Generation) system for querying Australian federal law. Built as an AI engineering portfolio project.

**Tech Stack**: Python, FastAPI, LangChain, Qdrant, PostgreSQL, OpenAI, Streamlit, AWS ECS Fargate, Terraform

**GitHub**: https://github.com/stefanmiranda08/legal-domain-rag-architecture

---

## Current State (as of handover)

### Completed: Data Ingestion

Ingestion completed successfully:
- **99,220 Commonwealth documents** ingested
- **1,981,579 chunks** in PostgreSQL
- **1,981,579 vectors** in Qdrant (validated, no orphans)

### In Progress: Evaluation Harness

Building an evaluation harness to compare experimental variables. The harness is 90% complete but has a minor bug to fix.

**Bug to fix**: In `src/evaluation/harness.py`, line ~117, the `QueryResult` dataclass references `c.score` but `Citation` objects use `relevance_score`. Change:
```python
# From:
"score": c.score,
# To:
"score": c.relevance_score,
```

**After fixing, run evaluation**:
```bash
# Run chunking strategy comparison
python scripts/run_evaluation.py --name "chunking_recursive" --chunking-strategy recursive --top-k 10 --llm-model gpt-5.4 --system-prompt professional

python scripts/run_evaluation.py --name "chunking_fixed" --chunking-strategy fixed --top-k 10 --llm-model gpt-5.4 --system-prompt professional

python scripts/run_evaluation.py --name "chunking_paragraph" --chunking-strategy paragraph --top-k 10 --llm-model gpt-5.4 --system-prompt professional
```

---

## What's Been Completed This Session

### 1. Fixed Ingestion Pipeline Batching
- OpenAI embedding: batch by token count (250k max) instead of item count
- Qdrant upserts: batch to 500 points to avoid 32MB payload limit
- Added logging throughout pipeline

### 2. Created Validation Script
- `scripts/validate_ingestion.py` - detects orphaned vectors and duplicate chunks
- Found and cleaned up 6,582 orphaned Qdrant vectors

### 3. Improved Generation Prompt
- Added "professional" system prompt variant with structured response guidelines
- Multiple prompt variants supported via `prompt_variant` parameter

### 4. Built Evaluation Framework
- `evaluation/test_queries.json` - 30 test queries across 6 legal categories
- `evaluation/experiment_config.py` - experimental variable definitions
- `src/evaluation/harness.py` - LLM-as-judge evaluation (faithfulness, relevancy, context precision)
- `scripts/run_evaluation.py` - CLI for running experiments
- `dashboard/pages/4_evaluation.py` - Streamlit results visualization

### 5. Reduced Dashboard Text Size
- `dashboard/styles.py` - shared CSS applied across all pages

### 6. Engineering Problems Log
- `engineering_problems.md` - documents challenges for interview discussion
  - API payload limit batching
  - Orphaned vectors from transactional mismatch

---

## Running the System

### Prerequisites
```bash
# Ensure Docker containers are running
docker ps  # Should see legal-rag-postgres and qdrant

# Set environment variables (or use .env file)
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
- Chat Interface: http://localhost:8501 (page 0)
- Evaluation Results: http://localhost:8501 (page 4)

---

## Key Files Changed This Session

| File | Purpose |
|------|---------|
| `src/ingestion/pipeline.py` | Token-based batching, Qdrant batch upserts, logging |
| `src/ingestion/loader.py` | Corpus loading logging |
| `src/generation.py` | Multiple system prompt variants |
| `src/evaluation/harness.py` | LLM-as-judge evaluation harness |
| `scripts/validate_ingestion.py` | Data integrity validation |
| `scripts/run_evaluation.py` | Evaluation CLI |
| `evaluation/test_queries.json` | 30 test queries |
| `evaluation/experiment_config.py` | Experimental variables |
| `dashboard/pages/4_evaluation.py` | Evaluation results dashboard |
| `dashboard/styles.py` | Shared CSS styles |
| `engineering_problems.md` | Interview prep documentation |

---

## Experimental Variables for Evaluation

| Variable | Options |
|----------|---------|
| `chunking_strategy` | fixed, paragraph, recursive |
| `top_k` | 5, 10, 15, 20 |
| `llm_model` | gpt-4o-mini, gpt-4o, gpt-5.4 |
| `system_prompt` | default, professional |

---

## Next Steps

1. **Fix evaluation harness bug** (see above)
2. **Run chunking strategy comparison** with fixed variables
3. **View results** in Streamlit dashboard (page 4)
4. **Run additional experiments** varying other parameters
5. **Commit and push** evaluation results

---

## User Preferences

- No "Co-Authored-By: Claude Code" in commit messages
- Lint failures should not block CI/CD pipeline
- Commonwealth subset only (~$15 budget for embeddings)
- Deploy only when demoing (to minimize AWS costs)
- Prefer lightweight frameworks over heavyweight ones
- Use `.env` file for configuration (already created)

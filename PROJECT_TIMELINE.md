# Project Timeline

Historical record of development sessions, problems encountered, and solutions implemented.

---

## Session: 2026-09-05

### Context
Evaluation harness was 90% complete. Needed to fix bugs and run chunking strategy comparison.

### Problems & Solutions

**1. Evaluation harness bug**
- *Problem*: `QueryResult` referenced `c.score` but `Citation` uses `relevance_score`
- *Solution*: Changed to `c.relevance_score` in `harness.py:143`

**2. OpenAI API access**
- *Problem*: Project lacks access to `gpt-4o-mini`
- *Solution*: Changed judge model to `gpt-5.4`

**3. GPT-5.4 API parameter**
- *Problem*: `max_tokens` not supported, requires `max_completion_tokens`
- *Solution*: Updated parameter name in evaluation harness

**4. Low faithfulness score (0.567)**
- *Problem*: High relevancy (0.95) and precision (0.94) but low faithfulness indicated model extrapolation
- *Diagnosis*: Recursive chunking truncates long legal sections at token limit. LLM receives incomplete rules and fills gaps from training data.
- *Solution*: Designed linked chunk architecture with on-demand section reconstruction

### Implementation: Linked Chunk Architecture

**Why not top-down parsing?**
- Legal documents vary in format (historical acts, OCR errors, different document types)
- Rule-based parsers fail silently on unexpected formats
- Creates inconsistent behavior with no way to detect failures

**Bottom-up approach:**
- Ingestion: Link chunks sequentially (`prev_chunk_id`, `next_chunk_id`)
- Retrieval: Extend bidirectionally, use LLM to detect section boundaries
- No document parsing required; structure detected at retrieval time

**Files created/modified:**
- `src/models.py` — Added chunk linking columns
- `src/ingestion/pipeline.py` — Links chunks during ingestion
- `src/reconstruction.py` — LLM boundary detection
- `src/retrieval.py` — `search_with_reconstruction()`
- `scripts/migrations/001_add_chunk_links.sql`
- `dashboard/pages/5_engineering_decisions.py`

### Status at End of Session
- Migration applied
- Existing data cleared (PostgreSQL + Qdrant)
- Ready for re-ingestion with linked chunks

### Next Steps
1. Re-run ingestion with linked chunk architecture
2. Re-run evaluation to measure faithfulness improvement
3. Compare before/after results

---

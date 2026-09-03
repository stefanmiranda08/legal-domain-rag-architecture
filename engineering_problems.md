# Engineering Problems Log

This document tracks engineering challenges encountered during development, useful for discussing real-world problem-solving in interviews.

---

## 1. API Payload Limits During Large-Scale Embedding Ingestion

**Date**: 2026-09-03

**Context**: Ingesting ~99,000 Australian legal documents into a RAG system. Documents are chunked, embedded via OpenAI, and stored in Qdrant vector database.

**Symptoms**:
```
Error: Requested 377009 tokens, max 300000 tokens per request
Error: JSON payload (40352045 bytes) is larger than allowed (limit: 33554432 bytes)
```

Ingestion failed partway through when processing unusually large legal documents.

**Root Cause Analysis**:

Two separate batching issues:

1. **OpenAI Embedding API**: The code batched embedding requests by item count (100 chunks per request), not by token count. Most chunks were ~512 tokens, so 100 chunks ≈ 51k tokens, well under the 300k limit. However, some legal documents had chunks that exceeded the target size (the text splitter's `chunk_size` is a target, not a hard limit when no good split points exist). A batch of 100 larger-than-expected chunks exceeded 300k tokens.

2. **Qdrant Vector Database**: The code upserted all vectors for a document in a single request. Large documents producing thousands of chunks created payloads exceeding Qdrant's 32MB limit. Each point contains a 1536-dimension float vector (~6KB) plus metadata, so ~5,000+ points would exceed the limit.

**Business Impact**:

- **Incomplete data coverage**: The ingestion failed on certain documents, meaning users querying the system would not receive answers from those documents. For a legal research tool, missing case law or legislation could lead to incomplete legal advice.
- **Wasted compute costs**: Failed API calls still incur costs. OpenAI charges per token sent, even for failed requests. At scale, this adds up.
- **Unpredictable failures**: The system worked fine for most documents but failed on edge cases (unusually large documents). This makes the system unreliable for production use where all documents must be searchable.

**Solution**:

1. **Token-based batching for OpenAI**: Changed from batching by count to batching by token count. Accumulate chunks until approaching 250k tokens, then send. Also added truncation for any single chunk exceeding the limit.

2. **Chunked upserts for Qdrant**: Batch upserts to 500 points per request, staying well under the 32MB payload limit.

**Key Takeaway**: When designing batch processing pipelines, consider the actual size constraints of external APIs, not just convenient batch counts. Size-based batching is more robust than count-based batching when item sizes vary.

**Files Changed**: `src/ingestion/pipeline.py`

---

## 2. Orphaned Vectors from Transactional Mismatch Between Databases

**Date**: 2026-09-03

**Context**: The ingestion pipeline writes to two databases: Qdrant (vector embeddings) and PostgreSQL (chunk text and metadata). The system requires a 1:1 relationship between Qdrant vectors and PostgreSQL chunks to function correctly—when a user searches, the system retrieves vectors from Qdrant, then fetches the corresponding text from PostgreSQL to display.

**Symptoms**:

After ingestion completed, validation showed a mismatch:
- PostgreSQL chunks: 1,981,579
- Qdrant vectors: 1,988,161
- Difference: 6,582 orphaned vectors

**Root Cause Analysis**:

The fundamental issue is a transactional mismatch between the two databases:

- **PostgreSQL**: Supports ACID transactions. Multiple inserts are staged in a session and committed atomically—either all succeed or all rollback.
- **Qdrant**: No transaction support. Each upsert batch is immediately permanent with no rollback capability.

The pipeline batched Qdrant writes (500 vectors per API call) but wrapped all PostgreSQL writes in a single transaction. For a document producing 1,500 chunks:

| Step | Qdrant | PostgreSQL Session |
|------|--------|-------------------|
| Batch 1 (0-499) | 500 vectors written permanently | 500 rows staged |
| Batch 2 (500-999) | 1000 vectors written permanently | 1000 rows staged |
| Batch 3 (1000-1499) | Error (timeout/rate limit) | Exception raised |
| **Result** | **1000 orphaned vectors remain** | `rollback()` → 0 rows |

Three Qdrant batches corresponded to one PostgreSQL transaction. When batch 3 failed, PostgreSQL correctly rolled back all 1,500 staged rows, but Qdrant had already permanently written 1,000 vectors from batches 1 and 2. Those vectors became orphans with no corresponding metadata.

**Business Impact**:

- **Degraded search quality**: Orphaned vectors are searched during queries but return no usable content. If a user's query matches an orphaned vector, the system either returns an error or silently omits that result, reducing result quality.
- **Wasted infrastructure costs**: Each vector consumes storage and memory in Qdrant. At 6KB per vector, 6,582 orphans waste ~40MB. At scale (millions of vectors), this becomes significant cloud spend on useless data.
- **Slower query performance**: Qdrant must search through all vectors including orphans. More vectors means longer search times. With 0.3% orphaned vectors, the impact is small, but if the problem compounds over repeated ingestion runs, query latency increases measurably.
- **Data integrity concerns**: A mismatch between databases indicates the system cannot guarantee consistency. For a legal research tool where accuracy matters, this undermines trust in the system.

**Solution**:

Immediate fix:
1. Created `scripts/validate_ingestion.py` to detect orphaned vectors and duplicate chunks
2. Script compares Qdrant point IDs against PostgreSQL `qdrant_point_id` foreign keys
3. Runs with `--fix` flag to delete orphaned vectors

**Proper Solution** (saga pattern / manual two-phase commit):

Implement a custom transaction wrapper that enforces atomicity across both databases:

```python
class DualDbTransaction:
    def __init__(self, qdrant_client, pg_session):
        self.qdrant_client = qdrant_client
        self.pg_session = pg_session
        self.qdrant_point_ids = []  # Track successful Qdrant writes

    def upsert_vectors(self, collection, points):
        self.qdrant_client.upsert(collection, points)
        self.qdrant_point_ids.extend([p.id for p in points])

    def commit(self):
        self.pg_session.commit()
        self.qdrant_point_ids = []  # Clear on success

    def rollback(self):
        self.pg_session.rollback()
        # Compensating action: delete vectors that were written
        if self.qdrant_point_ids:
            self.qdrant_client.delete(collection, self.qdrant_point_ids)
        self.qdrant_point_ids = []
```

This ensures that if any part of the transaction fails, both databases return to their pre-transaction state. The key insight is that Qdrant's lack of rollback can be compensated for by tracking what was written and explicitly deleting it on failure.

**Key Takeaway**: When writing to multiple databases with different transactional guarantees, you cannot rely on implicit rollback behavior. Either align the batch boundaries (1 Qdrant batch = 1 PostgreSQL transaction), or implement explicit compensating actions to maintain consistency. Build validation tooling regardless, because distributed systems will eventually have inconsistencies.

**Files Changed**: `scripts/validate_ingestion.py` (new)

---

## Template for Future Entries

```
## N. [Problem Title]

**Date**: YYYY-MM-DD

**Context**: [What were you building/doing?]

**Symptoms**: [What error messages or unexpected behavior did you observe?]

**Root Cause Analysis**: [What was actually wrong? How did you diagnose it?]

**Business Impact**: [Why does this matter to users/stakeholders? What real-world effects would this cause? Consider: user experience, costs, reliability, data quality, performance.]

**Solution**: [What did you change to fix it?]

**Key Takeaway**: [What general lesson applies to future projects?]

**Files Changed**: [Which files were modified?]
```

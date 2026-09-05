"""Engineering decisions and design rationale for the Legal RAG system."""

import streamlit as st
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from styles import apply_global_styles

st.set_page_config(
    page_title="Engineering Decisions",
    page_icon="",
    layout="wide",
)

apply_global_styles()

st.title("Engineering Decisions")
st.markdown(
    "Key design decisions made during development, with reasoning and trade-offs."
)

st.divider()

# -----------------------------------------------------------------------------
# CHUNKING STRATEGY
# -----------------------------------------------------------------------------
st.header("1. Chunking Strategy")

st.markdown("""
RAG systems split documents into chunks for embedding and retrieval. The chunking
strategy determines what the system can retrieve—and what it cannot.

**The core question**: What is the smallest piece of text that carries complete
meaning in this domain?

For legal documents, the answer is the **section**. A section states a complete
legal rule, often with conditions, exceptions, and definitions in its subsections.
Retrieve only part of a section and the LLM receives incomplete information.
""")

st.subheader("The Problem: Truncation Causes Extrapolation")

st.markdown("""
Standard chunking strategies enforce a maximum chunk size. When a legal section
exceeds this limit, it gets truncated. The LLM then receives an incomplete rule—
for example, a duty without its defences.

Faced with incomplete context, the LLM has two options:
1. State that information is incomplete (unhelpful)
2. Fill in the gaps from its training data (unfaithful to the evidence)

Models choose option 2. The answer may be accurate, but it contains claims not
grounded in the retrieved context. This defeats the purpose of retrieval-augmented
generation.

In evaluation, this manifests as low **faithfulness** scores despite high relevancy
and precision. The retrieval found the right documents, but incomplete chunks forced
the model to extrapolate.
""")

st.subheader("Failed Approach: Top-Down Parsing")

st.markdown("""
The initial approach was to parse document structure during ingestion—detect section
headers, subsection markers, and build a hierarchy. Chunks would link to their parent
and child chunks, enabling reconstruction during retrieval.

This approach has a critical flaw: **it assumes consistent document structure**.

Legal documents vary in formatting. Historical acts use different conventions.
OCR introduces errors. Regulations differ from primary legislation. A parser tuned
for one format fails silently on another. The system would produce correct results
for some documents and incorrect results for others, with no way to know which.
""")

st.subheader("Solution: Bottom-Up Reconstruction")

st.markdown("""
The implemented solution avoids parsing entirely. Instead:

**During ingestion:**
- Chunk documents using standard recursive splitting
- Store bidirectional links between adjacent chunks (`prev_chunk_id`, `next_chunk_id`)
- No structure detection required

**During retrieval:**
- Vector search returns initial chunks
- For each chunk, extend bidirectionally by following the links
- After each extension, ask an LLM: "Does this text begin and end at natural section boundaries?"
- Stop when the LLM confirms completeness
- Pass complete sections to the generation LLM

The document structure is embedded in the text itself. The LLM recognizes section
boundaries without us encoding rules about what those boundaries look like. This
works across legislation, case law, contracts, and other formats without modification.

**Trade-off**: Retrieval requires additional LLM calls for boundary detection. This
adds latency and cost per query. However:
- Boundary decisions can be cached after first discovery
- The cost is paid only for retrieved chunks, not all 2M chunks
- Improved faithfulness justifies the latency cost for legal research
""")

st.divider()

# -----------------------------------------------------------------------------
# DOMAIN-SPECIFIC SEMANTIC UNITS
# -----------------------------------------------------------------------------
st.header("2. Domain-Specific Semantic Units")

st.markdown("""
The minimal semantic unit varies by domain:

| Domain | Semantic Unit | Why |
|--------|---------------|-----|
| Legislation | Section | Rules are drafted as self-contained sections |
| Case Law | Paragraph | Reasoning unfolds paragraph by paragraph |
| Contracts | Clause | Each clause states one obligation or right |
| API Docs | Function/Endpoint | Each endpoint is a complete unit |

Chunking strategy is not a hyperparameter to tune blindly. It requires understanding
how meaning is structured in the target domain.
""")

st.divider()

# -----------------------------------------------------------------------------
# EVALUATION APPROACH
# -----------------------------------------------------------------------------
st.header("3. Evaluation Framework")

st.markdown("""
The system uses LLM-as-judge evaluation with three metrics:

**Faithfulness** — Does the answer contain only claims supported by the context?
Low faithfulness signals extrapolation, often caused by truncated chunks.

**Answer Relevancy** — Does the answer address the question?
Low relevancy indicates off-topic generation.

**Context Precision** — Are the retrieved chunks relevant?
Low precision indicates retrieval problems.

| Pattern | Likely Cause |
|---------|--------------|
| Low faithfulness, high relevancy | Truncated chunks; model extrapolating |
| Low precision, low relevancy | Poor retrieval; wrong documents |
| High precision, low faithfulness | Right documents, incomplete context |
""")

st.divider()

# -----------------------------------------------------------------------------
# INGESTION CHALLENGES
# -----------------------------------------------------------------------------
st.header("4. Ingestion Challenges")

st.markdown("""
**Token-Based Batching**

The initial pipeline batched embedding requests by document count. This failed when
documents varied in size—some batches exceeded API limits. The fix: batch by token
count, flushing when approaching the limit regardless of document count.

**Vector Database Limits**

Qdrant upserts failed silently when payloads exceeded 32MB. The fix: limit batches
to 500 points, well under the threshold.

**Consistency Between Stores**

The pipeline writes to PostgreSQL and Qdrant without distributed transactions.
Failures between writes create orphaned records. A validation script detects and
cleans up inconsistencies by comparing IDs across both systems.
""")

st.divider()

# -----------------------------------------------------------------------------
# ARCHITECTURE DECISIONS
# -----------------------------------------------------------------------------
st.header("5. Architecture")

st.markdown("""
**Hybrid Storage (Qdrant + PostgreSQL)**

Vector search happens in Qdrant. Metadata filtering, chunk text, and relationships
live in PostgreSQL. This separation enables complex filtering before vector search,
reducing the search space and improving relevance.

Trade-off: Requires maintaining consistency between systems.

**Embedding Model**

OpenAI `text-embedding-3-small` (1536 dimensions). Strong performance on legal text,
reasonable cost at scale. Local models would eliminate API costs but require GPU
infrastructure—acceptable trade-off for a portfolio project.
""")

st.divider()

st.caption("Documentation reflects the current system design.")

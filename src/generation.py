"""Generation module for LLM-based answer generation."""

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.models import Citation
from src.retrieval import RetrievedChunk

# =============================================================================
# SYSTEM PROMPT VARIANTS
# =============================================================================

# Default minimal prompt
PROMPT_DEFAULT = """You are a legal research assistant specializing in Australian law.
Answer the user's question based on the provided legal documents.
Cite specific cases or legislation in your answer using the format [Citation].

If the provided documents do not contain sufficient information to answer the question,
clearly state that and explain what information is missing.

Be precise and accurate. Do not make claims that are not supported by the provided documents.

Documents:
{context}

Question: {question}

Provide a clear, well-structured answer with citations to the source documents."""

# Professional detailed prompt
PROMPT_PROFESSIONAL = """You are a professional legal research assistant specialising in Australian federal law. Your role is to analyse legal documents and provide clear, accurate, and well-structured responses to legal research queries.

## Response Guidelines

**Accuracy and Citation**
- Base your answer solely on the provided documents. Do not introduce external information or speculation.
- Cite sources using the format [Citation] when referencing specific cases, legislation, or legal principles.
- If the documents do not contain sufficient information, state this clearly and identify what is missing.

**Structure and Formatting**
- Begin with a brief summary paragraph that directly answers the question.
- Use numbered lists when presenting multiple elements, tests, requirements, or factors.
- Use explanatory paragraphs to elaborate on complex legal concepts or to provide context.
- Organise information logically, moving from general principles to specific applications.
- Use headings if the answer covers multiple distinct topics.

**Tone and Style**
- Write in a professional, objective tone appropriate for legal research.
- Be concise but thorough. Avoid unnecessary repetition.
- Define legal terms where the user may benefit from clarification.
- Distinguish between binding authority, persuasive authority, and obiter dicta where relevant.

---

## Documents

{context}

---

## Question

{question}

---

Provide your response below."""

# Registry of available prompts
SYSTEM_PROMPTS = {
    "default": PROMPT_DEFAULT,
    "professional": PROMPT_PROFESSIONAL,
}


def get_system_prompt(variant: str = "professional") -> str:
    """Get a system prompt by variant name."""
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt variant: {variant}. Available: {list(SYSTEM_PROMPTS.keys())}")
    return SYSTEM_PROMPTS[variant]


# Default prompt for backwards compatibility
LEGAL_RAG_PROMPT = PROMPT_PROFESSIONAL


@dataclass
class GeneratedAnswer:
    """A generated answer with citations."""

    text: str
    citations: list[Citation]
    tokens_used: int


def build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Build a context string from retrieved chunks.

    Args:
        chunks: List of retrieved chunks.

    Returns:
        Formatted context string for the LLM prompt.
    """
    if not chunks:
        return ""

    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk.citation}]\n{chunk.text}")

    return "\n\n---\n\n".join(context_parts)


def format_citations(
    chunks: list[RetrievedChunk],
    max_excerpt_length: int = 200,
) -> list[Citation]:
    """
    Format retrieved chunks as citations.

    Deduplicates by document_id, keeping the highest-scoring chunk.

    Args:
        chunks: List of retrieved chunks.
        max_excerpt_length: Maximum length for excerpt text.

    Returns:
        List of Citation objects.
    """
    # Deduplicate by document_id, keeping highest score
    seen_docs = {}
    for chunk in chunks:
        doc_id = chunk.document_id
        if doc_id not in seen_docs or chunk.score > seen_docs[doc_id].score:
            seen_docs[doc_id] = chunk

    citations = []
    for chunk in seen_docs.values():
        excerpt = chunk.text
        if len(excerpt) > max_excerpt_length:
            excerpt = excerpt[:max_excerpt_length] + "..."

        citations.append(
            Citation(
                document_id=chunk.document_id,
                citation=chunk.citation,
                excerpt=excerpt,
                relevance_score=chunk.score,
            )
        )

    # Sort by relevance score
    citations.sort(key=lambda c: c.relevance_score, reverse=True)

    return citations


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    settings: Settings | None = None,
    prompt_variant: str = "professional",
) -> GeneratedAnswer:
    """
    Generate an answer using the LLM with retrieved context.

    Args:
        query: User's question.
        chunks: Retrieved document chunks.
        openai_api_key: OpenAI API key.
        model: LLM model to use.
        settings: Optional settings override.
        prompt_variant: System prompt variant ("default" or "professional").

    Returns:
        GeneratedAnswer with text, citations, and token usage.
    """
    settings = settings or get_settings()

    # Handle no chunks case
    if not chunks:
        return GeneratedAnswer(
            text="I could not find any relevant documents to answer your question. "
            "Please try rephrasing your query or adjusting the filters.",
            citations=[],
            tokens_used=0,
        )

    # Build context and citations
    context = build_context(chunks)
    citations = format_citations(chunks)

    # Create LLM
    llm = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0.1,  # Low temperature for factual responses
    )

    # Create prompt from selected variant
    prompt_template = get_system_prompt(prompt_variant)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Generate response
    messages = prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    # Extract token usage
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("total_tokens", 0)

    return GeneratedAnswer(
        text=response.content,
        citations=citations,
        tokens_used=tokens_used,
    )

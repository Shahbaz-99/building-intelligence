"""Bounded, labelled evidence context built from semantic retrieval results."""

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CONTEXT_CHARS = 12_000


def build_context(results: list[RetrievedChunk]) -> tuple[list[dict], dict[str, RetrievedChunk]]:
    """Return (context entries, label -> chunk) from the first passages that fit the budget."""
    entries: list[dict] = []
    by_label: dict[str, RetrievedChunk] = {}
    total = 0
    for chunk in results[:MAX_PASSAGES]:
        if total + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        label = f"E{len(entries) + 1}"
        total += len(chunk.text)
        by_label[label] = chunk
        entries.append(
            {
                "label": label,
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "act": chunk.act,
                "act_label": chunk.act_label,
                "heading": chunk.heading,
                "chapter": chunk.chapter,
                "section_number": chunk.section_number,
                "status": chunk.status,
                "source_pdf": chunk.source_pdf,
                "needs_review": chunk.needs_review,
                "text": chunk.text,
            }
        )
    return entries, by_label

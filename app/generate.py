from app.config import settings
from app.store import _get_co

_PREAMBLE = (
    "Answer the question using ONLY the provided sources. "
    "If the sources don't contain the answer, say so. "
    "Be concise.\n\n"
)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have enough information to answer that.",
            "citations": [],
        }

    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Source {i}] ({chunk['source']}, p.{chunk['page']})"
        context_lines.append(f"{label}: {chunk['text']}")
    context_block = "\n".join(context_lines)

    assembled = f"{_PREAMBLE}{context_block}\n\nQuestion: {question}"

    response = _get_co().chat(model=settings.generation_model, message=assembled)
    answer = response.text

    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            citations.append({"source": chunk["source"], "page": chunk["page"]})

    return {"answer": answer, "citations": citations}

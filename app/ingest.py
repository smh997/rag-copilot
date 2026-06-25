from pathlib import Path

from pypdf import PdfReader
import tiktoken

from app.config import settings

DATA_DIR = Path("data")

_ENCODING = "cl100k_base"


def ingest_document(filename: str) -> list[dict]:
    path = (DATA_DIR / filename).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"Invalid filename: {filename}")
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    enc = tiktoken.get_encoding(_ENCODING)
    reader = PdfReader(str(path))

    # Flat list of (token_id, 1-indexed page number)
    token_pages: list[tuple[int, int]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for tok in enc.encode(text):
            token_pages.append((tok, i + 1))

    stride = max(1, settings.chunk_size - settings.chunk_overlap)
    chunks: list[dict] = []
    i = 0
    while i < len(token_pages):
        window = token_pages[i : i + settings.chunk_size]
        chunks.append({
            "text": enc.decode([tp[0] for tp in window]),
            "source": filename,
            "page": window[0][1],
        })
        i += stride

    return chunks


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else "attention.pdf"
    result = ingest_document(filename)
    print(f"Total chunks: {len(result)}")
    for chunk in result[:2]:
        print(f"\n--- Chunk (source={chunk['source']}, page={chunk['page']}) ---")
        print(chunk["text"][:300])

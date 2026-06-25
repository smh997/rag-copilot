from unittest.mock import MagicMock, patch
import pytest
import tiktoken

import app.ingest as ingest_mod
from app.ingest import ingest_document
from app.config import settings


def _make_reader(page_texts: list[str]):
    """Return a mock PdfReader whose .pages list returns the given texts."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.pages = pages
    return reader


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a temp dir and plant a dummy file."""
    monkeypatch.setattr(ingest_mod, "DATA_DIR", tmp_path)
    (tmp_path / "test.pdf").write_bytes(b"fake")
    return tmp_path


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_returns_list_of_dicts(data_dir):
    text = "word " * 600
    with patch("app.ingest.PdfReader", return_value=_make_reader([text])):
        chunks = ingest_document("test.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, dict) for c in chunks)
    assert all({"text", "source", "page"} <= c.keys() for c in chunks)


def test_source_equals_filename(data_dir):
    with patch("app.ingest.PdfReader", return_value=_make_reader(["word " * 600])):
        chunks = ingest_document("test.pdf")
    assert all(c["source"] == "test.pdf" for c in chunks)


def test_text_is_nonempty_string(data_dir):
    with patch("app.ingest.PdfReader", return_value=_make_reader(["word " * 600])):
        chunks = ingest_document("test.pdf")
    assert all(isinstance(c["text"], str) and len(c["text"]) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Page attribution tests
# ---------------------------------------------------------------------------

def test_first_chunk_has_page_1(data_dir):
    with patch("app.ingest.PdfReader", return_value=_make_reader(["word " * 600])):
        chunks = ingest_document("test.pdf")
    assert chunks[0]["page"] == 1


def test_chunk_starting_on_page_2(data_dir):
    # page 1 is tiny (< chunk_size tokens), page 2 is long → a chunk will start on page 2
    short = "hi "          # ~2 tokens
    long = "word " * 600
    with patch("app.ingest.PdfReader", return_value=_make_reader([short, long])):
        chunks = ingest_document("test.pdf")
    pages = [c["page"] for c in chunks]
    assert 2 in pages


def test_pages_are_positive_integers(data_dir):
    with patch("app.ingest.PdfReader", return_value=_make_reader(["word " * 600])):
        chunks = ingest_document("test.pdf")
    assert all(isinstance(c["page"], int) and c["page"] >= 1 for c in chunks)


# ---------------------------------------------------------------------------
# Chunking behaviour tests
# ---------------------------------------------------------------------------

def test_chunk_token_length_does_not_exceed_chunk_size(data_dir):
    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 2000
    with patch("app.ingest.PdfReader", return_value=_make_reader([text])):
        chunks = ingest_document("test.pdf")
    for c in chunks:
        assert len(enc.encode(c["text"])) <= settings.chunk_size


def test_overlap_tokens_shared_between_consecutive_chunks(data_dir):
    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 1200
    with patch("app.ingest.PdfReader", return_value=_make_reader([text])):
        chunks = ingest_document("test.pdf")
    assert len(chunks) >= 2
    tok0 = enc.encode(chunks[0]["text"])
    tok1 = enc.encode(chunks[1]["text"])
    # Last chunk_overlap tokens of chunk[0] == first chunk_overlap tokens of chunk[1]
    assert tok0[-settings.chunk_overlap:] == tok1[:settings.chunk_overlap]


def test_empty_pdf_returns_empty_list(data_dir):
    with patch("app.ingest.PdfReader", return_value=_make_reader([""])):
        chunks = ingest_document("test.pdf")
    assert chunks == []


def test_file_not_found_raises(data_dir):
    with pytest.raises(FileNotFoundError):
        ingest_document("nonexistent.pdf")

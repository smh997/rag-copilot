from unittest.mock import MagicMock, patch

from app.generate import generate_answer


def _make_chunk(text: str, source: str, page: int) -> dict:
    return {"text": text, "source": source, "page": page, "distance": 0.1}


def test_empty_chunks_returns_fallback_without_api_call():
    with patch("app.generate._get_co") as mock_get_co:
        result = generate_answer("What is RLHF?", [])

    assert result["answer"] == "I don't have enough information to answer that."
    assert result["citations"] == []
    mock_get_co.assert_not_called()


def test_non_empty_chunks_calls_chat_with_correct_message():
    chunk = _make_chunk("Transformers use self-attention.", "attention.pdf", 4)
    mock_co = MagicMock()
    mock_co.chat.return_value = MagicMock(text="Self-attention is key.")

    with patch("app.generate._get_co", return_value=mock_co):
        result = generate_answer("What is self-attention?", [chunk])

    mock_co.chat.assert_called_once()
    call_kwargs = mock_co.chat.call_args
    message = call_kwargs.kwargs.get("message") or call_kwargs.args[0]

    assert "Transformers use self-attention." in message
    assert "[Source 1]" in message
    assert "attention.pdf" in message
    assert "p.4" in message
    assert "ONLY the provided sources" in message
    assert result["answer"] == "Self-attention is key."


def test_citations_deduped_by_source_and_page():
    chunks = [
        _make_chunk("chunk A", "paper.pdf", 2),
        _make_chunk("chunk B", "paper.pdf", 2),  # duplicate source+page
        _make_chunk("chunk C", "paper.pdf", 5),
    ]
    mock_co = MagicMock()
    mock_co.chat.return_value = MagicMock(text="Answer.")

    with patch("app.generate._get_co", return_value=mock_co):
        result = generate_answer("Question?", chunks)

    assert result["citations"] == [
        {"source": "paper.pdf", "page": 2},
        {"source": "paper.pdf", "page": 5},
    ]

def retrieve_chunks(question: str, top_k: int) -> list[str]:
    """
    Embed the user's question using the configured embedding model, then
    perform a nearest-neighbour search against the vector store to find
    the top_k most semantically similar text chunks.

    Args:
        question: The user's natural-language query.
        top_k: Number of chunks to return, ordered by relevance descending.

    Returns:
        A list of chunk text strings ready to be used as LLM context.
    """
    raise NotImplementedError

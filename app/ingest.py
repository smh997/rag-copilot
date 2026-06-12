def ingest_document(filename: str) -> int:
    """
    Load a PDF from data/<filename>, split it into overlapping text chunks
    (governed by CHUNK_SIZE and CHUNK_OVERLAP from config), compute an
    embedding vector for each chunk, and upsert those vectors into the
    vector store.

    Args:
        filename: Name of the PDF file located under data/.

    Returns:
        The number of chunks successfully indexed.
    """
    raise NotImplementedError

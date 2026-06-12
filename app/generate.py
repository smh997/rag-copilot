def generate_answer(question: str, context: list[str]) -> str:
    """
    Build a prompt from the retrieved context chunks and the user's
    question, call the Claude API, and return a markdown-formatted answer
    with inline citations referencing the source chunks.

    Args:
        question: The user's natural-language query.
        context: Ordered list of retrieved text chunks to include in the prompt.

    Returns:
        A markdown string containing the answer with inline chunk citations.
    """
    raise NotImplementedError

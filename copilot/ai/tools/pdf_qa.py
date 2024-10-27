import typing as t

from copilot.ai.llama_index_ import load_index


def pdf_qa_tool(
    query: t.Annotated[str, "The user's question"],
) -> str:
    """Answer a question about the uploaded PDFs."""
    vector_store = load_index()
    query_engine = vector_store.as_query_engine()
    return str(query_engine.query(query))

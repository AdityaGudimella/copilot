import typing as t

import pandas as pd
from llama_index.experimental import PandasQueryEngine

from copilot.ai.llama_index_ import initialize_llama_index


def csv_qa_tool(
    csv_path: t.Annotated[str, "The path to the CSV file."],
    query: t.Annotated[str, "The question to answer about the CSV file."],
) -> str:
    """Answer questions about the contents of a CSV file."""
    initialize_llama_index()
    df = pd.read_csv(csv_path)
    query_engine = PandasQueryEngine(df)
    return str(query_engine.query(query))

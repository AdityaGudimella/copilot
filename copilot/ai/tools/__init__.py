"""Folder where all the assistant tools are defined."""

import typing as t

from copilot.ai.tools.csv_df_qa import csv_qa_tool
from copilot.ai.tools.pdf_qa import pdf_qa_tool

TOOL_REGISTRY: dict[str, t.Callable[..., t.Any]] = {
    x.__name__: x for x in [csv_qa_tool, pdf_qa_tool]
}

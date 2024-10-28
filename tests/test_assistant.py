"""Tests the entire assistant functionality."""

import os
from unittest.mock import Mock

import openai
import pytest
import pytest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from pytest_mock import MockerFixture

from copilot.ai.assistant_event_handler import EventHandler
from copilot.ai.llama_index_ import load_parser, initialize_llama_index
from copilot.ai.openai_.clients import create_assistant, get_async_openai_client
from copilot.resources import RESOURCES_ROOT


@pytest.fixture
def file_names() -> str | list[str]:
    return "concept_drift.pdf"


@pytest.fixture
def file_paths(file_names: str | list[str]) -> str | list[str]:
    if isinstance(file_names, str):
        result = [RESOURCES_ROOT / file_names]
    else:
        result = [RESOURCES_ROOT / file_name for file_name in file_names]
    return [str(x) for x in result]


@pytest_asyncio.fixture
async def vector_store(file_paths: str | list[str]) -> VectorStoreIndex:
    initialize_llama_index()
    parser = load_parser()
    result = await parser.aload_data(file_paths)  # type: ignore
    node_parser = MarkdownElementNodeParser(
        num_workers=os.cpu_count() or 1,
    )
    nodes = await node_parser.aget_nodes_from_documents(result)
    vector_store = VectorStoreIndex(nodes)
    return vector_store


@pytest.fixture
def openai_client() -> openai.AsyncOpenAI:
    return get_async_openai_client()


@pytest_asyncio.fixture
async def thread_id(openai_client: openai.AsyncOpenAI) -> str:
    response = await openai_client.beta.threads.create()
    return response.id


@pytest_asyncio.fixture
async def assistant_id(openai_client: openai.AsyncOpenAI) -> str:
    response = await create_assistant(openai_client, model="gpt-4o-mini")
    return response.id


@pytest_asyncio.fixture
async def message_id(
    openai_client: openai.AsyncOpenAI, thread_id: str, query: str
) -> str:
    response = await openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query,
    )
    return response.id


@pytest.fixture
def mock_error_message(mocker: MockerFixture) -> Mock:
    error_msg_mock = mocker.patch("chainlit.ErrorMessage")
    error_msg_instance = mocker.MagicMock()
    error_msg_mock.return_value = error_msg_instance
    error_msg_instance.send = mocker.AsyncMock()
    return error_msg_mock


class MockMessage:
    def __init__(self, author: str, content: str):
        self.author = author
        self.content = content
        self.streamed_content = ""

    async def stream_token(self, token: str) -> None:
        self.streamed_content += token

    async def update(self) -> None:
        self.content = self.streamed_content

    async def send(self) -> "MockMessage":
        return self


@pytest.fixture
def mock_message(mocker: MockerFixture) -> type[MockMessage]:
    mocker.patch("chainlit.Message", MockMessage)
    return MockMessage


class MockStep:
    def __init__(
        self,
        name: str,
        type: str = "tool",
        parent_id: str | None = None,
    ):
        self.name = name
        self.type = type
        self.parent_id = parent_id
        self.show_input = None
        self.start = None
        self.end = None
        self.language = None

    async def send(self) -> "MockStep":
        return self

    async def update(self) -> None:
        pass


class MockContext:
    current_run = None


@pytest.fixture
def mock_chainlit(mocker: MockerFixture):
    # Mock both Message and Step
    mocker.patch("chainlit.Message", MockMessage)
    mocker.patch("chainlit.Step", MockStep)
    mocker.patch("copilot.ai.assistant_event_handler.cl.context", MockContext())
    return MockMessage, MockStep


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, exp_tool_name, exp_content",
    [
        (
            "What is the title of the 'Concept Drift' paper?",
            "pdf_qa_tool",
            "ConceptDrift: Uncovering Biases through the Lens of Foundational Models.",
        )
    ],
)
async def test_assistant(
    openai_client: openai.AsyncOpenAI,
    thread_id: str,
    assistant_id: str,
    message_id: str,
    query: str,
    exp_tool_name: str,
    exp_content: str,
    mock_error_message: Mock,
    vector_store: VectorStoreIndex,
    mock_chainlit: tuple[type[MockMessage], type[MockStep]],
    mocker: MockerFixture,
):
    mocker.patch("copilot.ai.tools.pdf_qa.load_index", return_value=vector_store)
    handler = EventHandler(assistant_name="TestCopilot", client=openai_client)
    async with openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=handler,
    ) as stream:
        await stream.until_done()

    mock_message, mock_step = mock_chainlit

    # Only verify that pdf_qa was called
    assert handler.current_step is not None
    assert isinstance(handler.current_step, mock_step)
    assert handler.current_step.name == exp_tool_name

    # Verify the response content
    assert handler.current_message is not None
    assert isinstance(handler.current_message, mock_message)
    assert exp_content in handler.current_message.content

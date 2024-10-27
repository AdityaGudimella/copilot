import openai
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_create_params import ToolResources
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.vector_store import VectorStore

from copilot import constants
from copilot.ai.openai_.function_calling import get_assistant_tool_metadata
from copilot.ai.tools import TOOL_REGISTRY
from copilot.settings import CopilotSettings
from copilot.utils import persist_str, retrieve_str


def get_openai_client() -> openai.OpenAI:
    settings = CopilotSettings()  # type: ignore
    client = openai.OpenAI(api_key=settings.openai_api_key)
    return client


def get_async_openai_client() -> openai.AsyncOpenAI:
    settings = CopilotSettings()  # type: ignore
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return client


async def get_or_create_thread_id(client: openai.AsyncOpenAI) -> str:
    try:
        return retrieve_str(constants.THREAD_ID_KEY)
    except ValueError:
        thread = await client.beta.threads.create()
        persist_str(constants.THREAD_ID_KEY, thread.id)
        return thread.id


async def get_or_create_vector_store(client: openai.AsyncOpenAI) -> VectorStore:
    from copilot.resources import CONCEPT_DRIFT

    try:
        vector_store_id = retrieve_str(constants.VECTOR_STORE_ID_KEY)
        return await client.beta.vector_stores.retrieve(vector_store_id)
    except ValueError:
        vector_store = await client.beta.vector_stores.create(name="Copilot")

        file_paths = [CONCEPT_DRIFT]
        file_streams = [open(file_path, "rb") for file_path in file_paths]

        # Add to the vector store
        await client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams,
        )

        persist_str(constants.VECTOR_STORE_ID_KEY, vector_store.id)
        return vector_store


async def create_assistant(client: openai.AsyncOpenAI, model: str) -> Assistant:
    instructions = """
    You are a helpful assistant that can answer questions about PDF documents.
    """
    tools: list[AssistantToolParam] = [
        {"type": "file_search"},
        *[
            get_assistant_tool_metadata(tool).as_openai_tool_spec()
            for tool in TOOL_REGISTRY.values()
        ],
    ]

    tool_resources: ToolResources = {
        "file_search": {
            "vector_store_ids": [(await get_or_create_vector_store(client)).id],
        },
    }
    return await client.beta.assistants.create(
        name="Copilot",
        model=model,
        instructions=instructions,
        tools=tools,
        tool_resources=tool_resources,
    )

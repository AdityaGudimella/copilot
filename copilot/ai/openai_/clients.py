import openai
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_tool_param import AssistantToolParam

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


async def create_assistant(client: openai.AsyncOpenAI, model: str) -> Assistant:
    instructions = """
    You are a helpful assistant that can answer questions about PDF documents.
    """
    tools: list[AssistantToolParam] = [
        get_assistant_tool_metadata(tool).as_openai_tool_spec()
        for tool in TOOL_REGISTRY.values()
    ]

    return await client.beta.assistants.create(
        name="Copilot",
        model=model,
        instructions=instructions,
        tools=tools,
    )

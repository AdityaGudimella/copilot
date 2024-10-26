import openai
from openai.types.beta.assistant import Assistant

from copilot import constants
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


async def get_or_create_thread(client: openai.AsyncOpenAI) -> str:
    try:
        return retrieve_str(constants.THREAD_ID_KEY)
    except ValueError:
        thread = await client.beta.threads.create()
        persist_str(constants.THREAD_ID_KEY, thread.id)
        return thread.id


def create_assistant(client: openai.OpenAI, model: str) -> Assistant:
    instructions = """
    You are a helpful assistant that can answer questions about PDF documents.
    """
    return client.beta.assistants.create(
        name="Copilot",
        model=model,
        instructions=instructions,
    )

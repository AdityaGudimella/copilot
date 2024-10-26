import openai
from openai.types.beta.assistant import Assistant

from copilot.settings import CopilotSettings


def get_openai_client() -> openai.OpenAI:
    settings = CopilotSettings()  # type: ignore
    client = openai.OpenAI(api_key=settings.openai_api_key)
    return client


def get_async_openai_client() -> openai.AsyncOpenAI:
    settings = CopilotSettings()  # type: ignore
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return client


def create_assistant(client: openai.OpenAI, model: str) -> Assistant:
    instructions = """
    You are a helpful assistant that can answer questions about PDF documents.
    """
    return client.beta.assistants.create(
        name="Copilot",
        model=model,
        instructions=instructions,
    )

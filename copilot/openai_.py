from copilot.settings import CopilotSettings


import openai


def get_openai_client() -> openai.AsyncOpenAI:
    settings = CopilotSettings()  # type: ignore
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return client

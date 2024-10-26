import chainlit as cl
import openai
import pydantic as pdt
import pydantic_settings as pds

from copilot import REPO_ROOT


class CopilotSettings(pds.BaseSettings):
    model_config = pds.SettingsConfigDict(env_file=REPO_ROOT / ".env")
    openai_api_key: str = pdt.Field(alias="COPILOT_OPENAI_API_KEY")
    model: str = pdt.Field(default="gpt-4o-mini")


@cl.on_message
async def main(message: cl.Message):
    settings = CopilotSettings()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message.content},
        ],
    )
    content = response.choices[0].message.content
    # Send a message back to the user
    await cl.Message(content=content).send()

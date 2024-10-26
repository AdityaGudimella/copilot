import chainlit as cl
import openai
import pydantic as pdt
import pydantic_settings as pds

from copilot import REPO_ROOT


class CopilotSettings(pds.BaseSettings):
    model_config = pds.SettingsConfigDict(env_file=REPO_ROOT / ".env")
    openai_api_key: str = pdt.Field(alias="COPILOT_OPENAI_API_KEY")
    model: str = pdt.Field(default="gpt-4o-mini")


MESSAGE_HISTORY_KEY = "message_history"


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        MESSAGE_HISTORY_KEY,
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )


@cl.on_message
async def main(message: cl.Message):
    settings = CopilotSettings()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    message_history = cl.user_session.get(MESSAGE_HISTORY_KEY)
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    response = await client.chat.completions.create(
        stream=True,
        model=settings.model,
        messages=message_history,
    )
    async for chunk in response:
        if token := chunk.choices[0].delta.content or "":
            await msg.stream_token(token)
    message_history.append({"role": "assistant", "content": msg.content})
    # Send a message back to the user
    await msg.update()

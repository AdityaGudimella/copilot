import typing as t

import chainlit as cl

from openai.types.chat import ChatCompletionMessageParam

from copilot import constants
from copilot.openai_ import get_openai_client


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        constants.MESSAGE_HISTORY_KEY,
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )
    chat_profile = cl.user_session.get(constants.CHAT_PROFILES_KEY)
    await cl.Message(
        content=f"You are using the **{chat_profile}** chat profile."
    ).send()


@cl.set_chat_profiles
async def chat_profiles(user: cl.User | None = None):
    return [
        cl.ChatProfile(
            name=constants.ChatProfiles.GPT4oMini,
            markdown_description="The underlying LLM model is **GPT-4o-mini**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name=constants.ChatProfiles.GPT4o,
            markdown_description="The underlying LLM model is **GPT-4o**.",
            icon="https://picsum.photos/250",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    client = get_openai_client()

    message_history: list[ChatCompletionMessageParam] = cl.user_session.get(  # type: ignore
        constants.MESSAGE_HISTORY_KEY
    )
    message_history.append({"role": "user", "content": message.content})

    chat_profile = cl.user_session.get(constants.CHAT_PROFILES_KEY)
    assert chat_profile in constants.ChatProfiles, chat_profile
    msg = cl.Message(content="")
    response = await client.chat.completions.create(  # type: ignore
        stream=True,
        model=constants.get_model_for_chat_profile(
            t.cast(constants.ChatProfiles, chat_profile)
        ),
        messages=message_history,
    )
    async for chunk in response:
        if token := chunk.choices[0].delta.content or "":
            await msg.stream_token(token)
    message_history.append({"role": "assistant", "content": msg.content})
    # Send a message back to the user
    await msg.update()

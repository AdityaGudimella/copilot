import traceback

import openai
from openai.types.beta.assistant_stream_event import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import RunStep

import chainlit as cl

from copilot import constants


class EventHandler(openai.AsyncAssistantEventHandler):
    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message | None = None
        self.assistant_name = assistant_name

    async def on_run_step_start(self, step: RunStep) -> None:
        cl.user_session.set(constants.CURRENT_RUN_STEP_KEY, step)

    async def on_text_created(self, text: str) -> None:
        self.current_message = await cl.Message(
            author=self.assistant_name,
            content="",
        ).send()

    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        if delta.value:
            assert self.current_message is not None
            await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text: Text) -> None:
        assert self.current_message is not None
        await self.current_message.update()

    async def on_event(
        self,
        event: AssistantStreamEvent,
    ) -> None:
        if event.event == "error":
            await cl.ErrorMessage(
                content=event.data.message,
            ).send()

    async def on_exception(self, exception: Exception) -> None:
        await cl.ErrorMessage(
            content="\n".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
        ).send()

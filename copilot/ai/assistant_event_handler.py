import traceback

import chainlit as cl
import openai
from openai.types.beta.assistant_stream_event import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import RunStep

from copilot import constants
from copilot.resources import RESOURCES_ROOT


class EventHandler(openai.AsyncAssistantEventHandler):
    def __init__(self, assistant_name: str, client: openai.AsyncOpenAI) -> None:
        super().__init__()
        self.current_message: cl.Message | None = None
        self.assistant_name = assistant_name
        self.client = client

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
        citations = []
        for index, annotation in enumerate(text.annotations):
            text.value = text.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = await self.client.files.retrieve(file_citation.file_id)
                citations.append(cited_file)
        elements = []
        for citation in citations:
            if (RESOURCES_ROOT / citation.filename).exists():
                elements.append(
                    cl.Pdf(
                        name=citation.filename,
                        display="inline",
                        path=str(RESOURCES_ROOT / citation.filename),
                    )
                )
        if elements:
            await cl.Message(content="", elements=elements).send()

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

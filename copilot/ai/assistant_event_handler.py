import datetime
import traceback

import chainlit as cl
import openai
from openai.types.beta.assistant_stream_event import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.runs import RunStep, ToolCall, ToolCallDelta

from copilot import REPO_ROOT
from copilot import constants
from copilot.ai.openai_.function_calling import execute_tools
from copilot.resources import RESOURCES_ROOT


class EventHandler(openai.AsyncAssistantEventHandler):
    def __init__(self, assistant_name: str, client: openai.AsyncOpenAI) -> None:
        super().__init__()
        self.current_message: cl.Message | None = None
        self.current_step: cl.Step | None = None
        self.current_tool_call_id: str | None = None
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
            elif (REPO_ROOT / ".files" / citation.filename).exists():
                elements.append(
                    cl.File(
                        name=citation.filename,
                        display="inline",
                        path=str(REPO_ROOT / ".files" / citation.filename),
                    )
                )
        if elements:
            await cl.Message(content="", elements=elements).send()

    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        self.current_tool_call_id = tool_call.id
        self.current_step = cl.Step(
            name=tool_call.function.name
            if tool_call.type == "function"
            else tool_call.type,
            type="tool",
            parent_id=cl.context.current_run.id if cl.context.current_run else None,
        )
        self.current_step.show_input = "python"
        self.current_step.start = str(datetime.datetime.now())
        await self.current_step.send()

    async def on_tool_call_delta(
        self,
        delta: ToolCallDelta,
        snapshot: ToolCall,
    ) -> None:
        if snapshot.id != self.current_tool_call_id:
            self.current_tool_call_id = snapshot.id
            self.current_step = cl.Step(
                name=delta.type,
                type="tool",
                parent_id=cl.context.current_run.id if cl.context.current_run else None,
            )
            self.current_step.start = str(datetime.datetime.now())
            if snapshot.type == "function":
                self.current_step.name = snapshot.function.name
                self.current_step.language = "json"
            await self.current_step.send()

    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        assert self.current_step is not None
        self.current_step.end = str(datetime.datetime.now())
        await self.current_step.update()

    async def handle_requires_action(
        self,
        data: Run,
        run_id: str,
    ) -> None:
        assert data.required_action is not None
        tool_calls = data.required_action.submit_tool_outputs.tool_calls
        tool_outputs = list(execute_tools(tool_calls))
        async with self.client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=data.thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
        ) as stream:
            async for delta in stream.text_deltas:
                if self.current_message is None:
                    self.current_message = await cl.Message(
                        author=self.assistant_name,
                        content="",
                    ).send()
                await self.current_message.stream_token(delta)

    async def on_event(
        self,
        event: AssistantStreamEvent,
    ) -> None:
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            await self.handle_requires_action(event.data, run_id)
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

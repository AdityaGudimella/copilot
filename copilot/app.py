import os
import typing as t

import chainlit as cl
from llama_index.core.node_parser import MarkdownElementNodeParser

from copilot import constants
from copilot.ai.assistant_event_handler import EventHandler
from copilot.ai.llama_index_ import load_index, load_parser, parse_files_if_needed
from copilot.ai.openai_ import (
    create_assistant,
    get_async_openai_client,
    get_or_create_thread_id,
)


@cl.on_chat_start
async def on_chat_start():
    client = get_async_openai_client()
    thread_id = await get_or_create_thread_id(client)
    cl.user_session.set(constants.THREAD_ID_KEY, thread_id)
    chat_profile = cl.user_session.get(constants.CHAT_PROFILES_KEY)
    assistant = await create_assistant(
        client=client,
        model=constants.get_model_for_chat_profile(
            t.cast(constants.ChatProfiles, chat_profile)
        ),
    )
    cl.user_session.set(constants.ASSISTANT_ID_KEY, assistant.id)
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


@cl.on_stop
async def on_stop():
    client = get_async_openai_client()
    current_run_step = cl.user_session.get(constants.CURRENT_RUN_STEP_KEY)
    if current_run_step is not None:
        await client.beta.threads.runs.cancel(
            thread_id=current_run_step.thread_id, run_id=current_run_step.id
        )
    assistant_id = cl.user_session.get(constants.ASSISTANT_ID_KEY)
    assert isinstance(assistant_id, str)
    await client.beta.assistants.delete(assistant_id)


@cl.on_message
async def on_message(message: cl.Message):
    client = get_async_openai_client()
    thread_id = cl.user_session.get(constants.THREAD_ID_KEY)
    assert isinstance(thread_id, str)
    assistant_id = cl.user_session.get(constants.ASSISTANT_ID_KEY)
    assert isinstance(assistant_id, str)

    if message.elements:
        vector_store = load_index()
        parser = load_parser()
        pdf_file_paths = {}
        for element in message.elements:
            if isinstance(element, cl.File):
                if element.mime == "application/pdf":
                    if element.path:
                        pdf_file_paths[element.name] = element.path
                elif element.mime == "text/csv":
                    # CSV QA
                    message.content += f"\n\nCSV file path is: {element.path}"
        if pdf_file_paths:
            message.content += "The user uploaded some PDFs."
            async with cl.Step("Parsing PDFs...") as step:
                results = await parse_files_if_needed(pdf_file_paths, parser)
                await step.update()
            async with cl.Step("Creating nodes...") as step:
                node_parser = MarkdownElementNodeParser(
                    num_workers=os.cpu_count() or 1,
                )
                nodes = node_parser.get_nodes_from_documents(results)
                await step.update()
            async with cl.Step("Inserting nodes into vector store...") as step:
                vector_store.insert_nodes(nodes)
                await step.update()
            vector_store.storage_context.persist()

    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
    )

    async with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=EventHandler(assistant_name="Copilot", client=client),
    ) as stream:
        await stream.until_done()

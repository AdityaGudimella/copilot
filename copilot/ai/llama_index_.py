import json
import typing as t

import chainlit as cl
from chainlit.context import ChainlitContextException
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse, ResultType
from pinecone import Pinecone

from copilot import constants
from copilot.settings import CopilotSettings


_llama_index_initialized = False


def initialize_llama_index() -> None:
    global _llama_index_initialized
    if _llama_index_initialized:
        return
    try:
        chat_profile = cl.user_session.get(constants.CHAT_PROFILES_KEY)
    except ChainlitContextException:
        chat_profile = constants.ChatProfiles.GPT4oMini
    copilot_settings = CopilotSettings()  # type: ignore
    Settings.llm = OpenAI(
        model=constants.get_model_for_chat_profile(
            t.cast(constants.ChatProfiles, chat_profile)
        ),
        api_key=copilot_settings.openai_api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=copilot_settings.openai_api_key,
    )
    _llama_index_initialized = True


async def parse_files_if_needed(
    file_paths: dict[str, str], parser: LlamaParse
) -> list[Document]:
    try:
        parsed_files = json.loads(constants.PARSED_FILES_DATA_PATH.read_text())
    except FileNotFoundError:
        parsed_files = []
    to_parse: dict[str, str] = {}
    for file_name, file_path in file_paths.items():
        if file_name not in parsed_files:
            to_parse[file_name] = file_path
    if not to_parse:
        return []
    else:
        result = await parser.aload_data(list(to_parse.values()))
        parsed_files.extend(to_parse)
        constants.PARSED_FILES_DATA_PATH.write_text(json.dumps(parsed_files))
        return result


def load_index() -> VectorStoreIndex:
    initialize_llama_index()
    copilot_settings = CopilotSettings()  # type: ignore
    pc = Pinecone(api_key=copilot_settings.pinecone_api_key)
    pinecone_index = pc.Index("copilot")
    vector_store = PineconeVectorStore(pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=[], storage_context=storage_context)


def load_parser() -> LlamaParse:
    copilot_settings = CopilotSettings()  # type: ignore
    return LlamaParse(
        api_key=copilot_settings.llama_cloud_api_key,
        result_type=ResultType.MD,
    )

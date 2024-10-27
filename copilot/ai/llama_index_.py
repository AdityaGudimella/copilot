import typing as t

import chainlit as cl
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from copilot import constants
from copilot.settings import CopilotSettings


def initialize_llama_index() -> None:
    chat_profile = cl.user_session.get(constants.CHAT_PROFILES_KEY)
    copilot_settings = CopilotSettings()  # type: ignore
    Settings.llm = OpenAI(
        model=constants.get_model_for_chat_profile(
            t.cast(constants.ChatProfiles, chat_profile)
        ),
        api_key=copilot_settings.openai_api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=copilot_settings.openai_api_key,
    )

import enum
from pathlib import Path

ASSISTANT_ID_KEY = "assistant_id"
CHAT_PROFILES_KEY = "chat_profile"
CURRENT_RUN_STEP_KEY = "current_run_step"
THREAD_ID_KEY = "thread"
VECTOR_STORE_ID_KEY = "vector_store_id"

PERSISTENCE_SETTINGS_PATH = Path.home() / ".copilot" / "settings.json"


class ChatProfiles(str, enum.Enum):
    GPT4oMini = "GPT-4o-mini"
    GPT4o = "GPT-4o"


def get_model_for_chat_profile(chat_profile: ChatProfiles) -> str:
    if chat_profile == ChatProfiles.GPT4oMini:
        return "gpt-4o-mini"
    elif chat_profile == ChatProfiles.GPT4o:
        return "gpt-4o"
    raise ValueError(f"Unknown chat profile: {chat_profile}")

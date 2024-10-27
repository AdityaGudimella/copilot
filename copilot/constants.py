import enum
from pathlib import Path

ASSISTANT_ID_KEY = "assistant_id"
CHAT_PROFILES_KEY = "chat_profile"
CURRENT_RUN_STEP_KEY = "current_run_step"
THREAD_ID_KEY = "thread"

PERSISTENCE_DIR = Path.home() / ".copilot"
PERSISTENCE_SETTINGS_PATH = PERSISTENCE_DIR / "settings.json"
PARSED_FILES_DATA_PATH = PERSISTENCE_DIR / "parsed_files.json"

SUPPROTED_OPENAI_FILE_SEARCH_MIME_TYPES = [
    "text/x-c",
    "text/x-c++",
    "text/x-csharp",
    "text/css",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/x-golang",
    "text/html",
    "text/x-java",
    "text/javascript",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/x-php",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/x-python",
    "text/x-script.python",
    "text/x-ruby",
    "application/x-sh",
    "text/x-tex",
    "application/typescript",
    "text/plain",
]


class ChatProfiles(str, enum.Enum):
    GPT4oMini = "GPT-4o-mini"
    GPT4o = "GPT-4o"


def get_model_for_chat_profile(chat_profile: ChatProfiles) -> str:
    if chat_profile == ChatProfiles.GPT4oMini:
        return "gpt-4o-mini"
    elif chat_profile == ChatProfiles.GPT4o:
        return "gpt-4o"
    raise ValueError(f"Unknown chat profile: {chat_profile}")

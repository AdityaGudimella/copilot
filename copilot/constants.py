import enum


CHAT_PROFILES_KEY = "chat_profile"
MESSAGE_HISTORY_KEY = "message_history"


class ChatProfiles(str, enum.Enum):
    GPT4oMini = "GPT-4o-mini"
    GPT4o = "GPT-4o"


def get_model_for_chat_profile(chat_profile: ChatProfiles) -> str:
    if chat_profile == ChatProfiles.GPT4oMini:
        return "gpt-4o-mini"
    elif chat_profile == ChatProfiles.GPT4o:
        return "gpt-4o"
    raise ValueError(f"Unknown chat profile: {chat_profile}")

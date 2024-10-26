from copilot import REPO_ROOT


import pydantic as pdt
import pydantic_settings as pds


class CopilotSettings(pds.BaseSettings):
    model_config = pds.SettingsConfigDict(env_file=REPO_ROOT / ".env")
    openai_api_key: str = pdt.Field(alias="COPILOT_OPENAI_API_KEY")

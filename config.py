from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # existing keys...
    anthropic_api_key: str
    news_api_key: str
    elevenlabs_api_key: str
    perplexity_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Model config — swap these in .env to change models without touching code
    filter_model: str = "claude-haiku-4-5-20251001"
    synthesis_model: str = "claude-sonnet-4-20250514"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

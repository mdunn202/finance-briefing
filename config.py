from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str
    news_api_key: str
    elevenlabs_api_key: str
    perplexity_api_key: str = ""  # optional at POC stage
    telegram_bot_token: str = ""  # optional at POC stage
    telegram_chat_id: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

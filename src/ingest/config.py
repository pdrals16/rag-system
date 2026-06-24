from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str
    openai_api_key: str

    chroma_persist_dir: str = "chroma_db"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "claude-sonnet-4-6"

    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_summary: bool = False
    enable_subject: bool = True
    enable_format: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()

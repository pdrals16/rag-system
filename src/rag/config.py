from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str
    openai_api_key: str

    chroma_persist_dir: str = "chroma_db"
    llm_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-small"

    top_k: int = 4


def get_settings() -> Settings:
    return Settings()

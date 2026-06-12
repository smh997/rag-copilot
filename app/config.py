from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or a .env file."""

    anthropic_api_key: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

"""Configuration centralisée via Pydantic Settings."""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # API Keys
    openai_api_key: str = ""
    cohere_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_environment: str = ""

    # Paths
    data_dir: Path = Path("./data/reports")
    persist_dir: Path = Path("./data/chroma_db")

    # Models
    llm_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-large"
    clip_model: str = "openai/clip-vit-large-patch14"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    fetch_k: int = 20

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    def model_post_init(self, __context: object) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

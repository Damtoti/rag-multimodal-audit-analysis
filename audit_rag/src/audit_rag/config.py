"""Configuration centralisée via Pydantic Settings."""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemin vers .env à la racine du projet audit_rag
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", case_sensitive=False
    )

    # API Keys
    openai_api_key: str = ""
    cohere_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_environment: str = ""

    # OpenAI API endpoint
    openai_api_base: str = "https://api.openai.com/v1"

    # Paths
    data_dir: Path = Path("./data/reports")
    persist_dir: Path = Path("./data/chroma_db")

    # Models
    llm_model: str = "gpt-3.5-turbo"
    embed_model: str = "text-embedding-3-large"
    clip_model: str = "openai/clip-vit-large-patch14"
    enable_image_extraction: bool = False
    enable_clip_embeddings: bool = False
    enable_image_descriptions: bool = False
    max_images_per_pdf: int = 30

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    fetch_k: int = 20

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    max_upload_mb: int = 10

    def model_post_init(self, __context: object) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

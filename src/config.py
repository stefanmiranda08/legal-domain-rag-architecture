"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "legal_rag"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding model
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM model
    llm_model: str = "gpt-4o-mini"

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 50

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

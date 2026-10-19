"""Central configuration, read from environment variables or a local .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    # Paths
    raw_dir: Path = Path("data/raw")
    eval_path: Path = Path("data/eval/questions.jsonl")
    index_dir: Path = Path("index")
    prompt_path: Path = Path("prompts/answer_v1.txt")
    results_dir: Path = Path("results/v0.1")

    # Chunking and retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_tokens: int = 500
    chunk_overlap: int = 75
    top_k: int = 5

    # Generation (Haiku to iterate, Sonnet for final measurements)
    generation_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-sonnet-5-5"
    max_answer_tokens: int = 600


def get_settings() -> Settings:
    return Settings()

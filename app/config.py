from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8")

    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_NUM_CTX: int = 8192
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_FALLBACK_MODEL: str = "openai/gpt-oss-120b"
    GROQ_API_KEY: str = ""

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    DEVICE: str = "cuda"
    EMBEDDING_DEVICE: str = "cpu"

    CHROMA_DIR: str = str(ROOT / "data" / "chroma")
    CHROMA_COLLECTION: str = "is_kanunu"
    CORPUS_PATH: str = str(ROOT / "data" / "corpus.jsonl")

    CANDIDATES: int = 20
    TOP_K: int = 5
    ASSISTANT_MODE: bool = False
    RELEVANCE_THRESHOLD: float = 0.3


settings = Settings()

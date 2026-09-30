import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Windows without Developer Mode can't symlink; the HF cache still works, just noisier.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_rpm: int = 10
    # Optional: 0 disables "thinking" on Gemini 2.5 models (faster, cheaper). None = model default.
    gemini_thinking_budget: int | None = None

    # Set to use a Qdrant server (e.g. docker compose). Empty = embedded local mode.
    qdrant_url: str = ""
    qdrant_collection: str = "papers"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    data_dir: Path = BACKEND_DIR / "data"

    # Chunking
    chunk_tokens: int = 400
    chunk_overlap: float = 0.15

    # Retrieval
    prefetch_k: int = 50  # candidates per branch (dense / sparse) before fusion
    rerank_candidates: int = 20  # fused candidates scored by the cross-encoder (CPU cost ~linear)
    final_k: int = 8

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def qdrant_local_path(self) -> Path:
        return self.data_dir / "qdrant_local"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'researchrag.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.pdf_dir.mkdir(parents=True, exist_ok=True)
    s.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return s

from datetime import datetime, timezone

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Paper(SQLModel, table=True):
    id: str = Field(primary_key=True)  # arXiv id or sha1 of the uploaded file
    title: str
    authors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    year: int | None = None
    categories: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    abstract: str = ""
    source: str = "upload"  # "upload" | "arxiv"
    pdf_path: str = ""
    num_pages: int = 0
    num_chunks: int = 0
    sections: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = "pending"  # pending | processing | ready | failed
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)


class EvalRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    status: str = "running"  # running | done | failed
    progress: str = ""
    dataset_size: int = 0
    configs: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # {config_name: {metric_name: value}}
    results: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # per-question details for inspection
    details: list = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = None

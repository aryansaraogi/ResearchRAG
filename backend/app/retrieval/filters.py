"""Metadata filters exposed by the API, translated into Qdrant payload filters."""

from __future__ import annotations

from pydantic import BaseModel
from qdrant_client import models


class SearchFilters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    authors: list[str] | None = None
    categories: list[str] | None = None
    paper_ids: list[str] | None = None
    section_types: list[str] | None = None

    def is_empty(self) -> bool:
        return not any(v for v in self.model_dump().values())


def to_qdrant_filter(f: SearchFilters | None) -> models.Filter | None:
    if f is None or f.is_empty():
        return None
    must: list[models.Condition] = []
    if f.year_min is not None or f.year_max is not None:
        must.append(models.FieldCondition(key="year", range=models.Range(gte=f.year_min, lte=f.year_max)))
    if f.authors:
        must.append(models.FieldCondition(key="authors", match=models.MatchAny(any=f.authors)))
    if f.categories:
        must.append(models.FieldCondition(key="categories", match=models.MatchAny(any=f.categories)))
    if f.paper_ids:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchAny(any=f.paper_ids)))
    if f.section_types:
        must.append(models.FieldCondition(key="section_type", match=models.MatchAny(any=f.section_types)))
    return models.Filter(must=must)

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.generation import rag
from google.genai import errors

from app.generation.llm import LLMNotConfigured, describe_error
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Message] = []
    filters: SearchFilters | None = None
    mode: RetrievalMode = RetrievalMode.hybrid
    rerank: bool = False  # cross-encoder adds ~4 s on CPU; see the README evaluation
    top_k: int = Field(default=8, ge=1, le=20)


def _args(req: ChatRequest) -> dict:
    return dict(question=req.question, filters=req.filters, history=[m.model_dump() for m in req.history],
                mode=req.mode, rerank=req.rerank, top_k=req.top_k)


@router.post("", response_model=rag.RAGAnswer)
def chat(req: ChatRequest):
    try:
        return rag.answer(**_args(req))
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except errors.APIError as e:
        raise HTTPException(503 if e.code in (429, 503) else 502, describe_error(e)) from e


@router.post("/stream")
def chat_stream(req: ChatRequest):
    def events():
        try:
            for kind, payload in rag.answer_stream(**_args(req)):
                data = payload.model_dump() if hasattr(payload, "model_dump") else payload
                yield {"event": kind, "data": json.dumps(data)}
        except Exception as e:  # noqa: BLE001 - surface errors to the client as an event
            yield {"event": "error", "data": json.dumps(describe_error(e))}

    return EventSourceResponse(events())

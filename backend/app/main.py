import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, eval, papers, search
from app.config import get_settings
from app.db import init_db
from app.retrieval import vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchrag")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    try:
        vector_store.ensure_collection()
    except Exception as e:  # noqa: BLE001 - app still starts; /health reports the problem
        log.warning("Qdrant not reachable at startup (%s). Check QDRANT_URL or leave it empty for embedded mode.", e)
    yield


app = FastAPI(title="ResearchRAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (papers.router, search.router, chat.router, eval.router):
    app.include_router(router)


@app.get("/health")
def health():
    s = get_settings()
    try:
        vector_store.client().get_collections()
        qdrant = "ok"
    except Exception as e:  # noqa: BLE001
        qdrant = f"unreachable: {e}"
    return {"qdrant": qdrant, "qdrant_mode": "server" if s.qdrant_url else "embedded",
            "llm_configured": bool(s.gemini_api_key), "llm_model": s.gemini_model}

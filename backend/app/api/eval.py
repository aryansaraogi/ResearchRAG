import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.db import get_session
from app.evaluation import dataset as ds
from app.evaluation.runner import CONFIGS, run_in_background
from app.generation.llm import LLMNotConfigured, get_llm
from app.models import EvalRun

router = APIRouter(prefix="/eval", tags=["eval"])

# Dataset generation is a single background job; its status lives in memory.
_gen_status: dict = {"state": "idle", "message": ""}
_gen_lock = threading.Lock()


class GenerateDatasetRequest(BaseModel):
    n: int = Field(default=40, ge=5, le=200)
    paper_ids: list[str] | None = None
    append: bool = False


class RunRequest(BaseModel):
    configs: list[str] | None = None  # default: all
    gen_configs: list[str] = ["hybrid_rrf"]  # LLM-judged generation (slow on free tier)
    # Caps single-passage questions; multi-passage and unanswerable ones are always judged
    max_gen_questions: int | None = Field(default=20, ge=1)


@router.get("/configs")
def configs():
    return {name: {"mode": c.mode.value, "rerank": c.rerank, "multi_query": c.multi_query} for name, c in CONFIGS.items()}


@router.get("/dataset", response_model=list[ds.EvalItem])
def get_dataset():
    return ds.load_dataset()


@router.get("/dataset/status")
def dataset_status():
    return _gen_status


@router.post("/dataset/generate", status_code=202)
def generate_dataset(req: GenerateDatasetRequest, background: BackgroundTasks):
    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "Dataset generation already running")

    def job():
        try:
            _gen_status.update(state="running", message="Sampling chunks…")
            items = ds.generate_dataset(llm, n=req.n, paper_ids=req.paper_ids, append=req.append,
                                        progress=lambda m: _gen_status.update(message=m))
            _gen_status.update(state="done", message=f"{len(items)} questions")
        except Exception as e:  # noqa: BLE001
            _gen_status.update(state="failed", message=str(e)[:300])
        finally:
            _gen_lock.release()

    background.add_task(job)
    return _gen_status


@router.post("/runs", response_model=EvalRun, status_code=202)
def start_run(req: RunRequest, background: BackgroundTasks, session: Session = Depends(get_session)):
    unknown = set((req.configs or []) + req.gen_configs) - set(CONFIGS)
    if unknown:
        raise HTTPException(400, f"Unknown configs: {sorted(unknown)}")
    if not ds.load_dataset():
        raise HTTPException(400, "Evaluation dataset is empty. Generate one first.")
    if req.gen_configs:
        try:
            get_llm()
        except LLMNotConfigured as e:
            raise HTTPException(503, str(e)) from e
    run = EvalRun(configs=req.configs or list(CONFIGS), dataset_size=len(ds.load_dataset()),
                  progress="Queued")
    session.add(run)
    session.commit()
    session.refresh(run)
    background.add_task(run_in_background, run.id, req.configs, req.gen_configs, req.max_gen_questions)
    return run


@router.get("/runs")
def list_runs(session: Session = Depends(get_session)):
    runs = session.exec(select(EvalRun).order_by(EvalRun.id.desc())).all()
    # Omit heavy per-question details from the listing
    return [r.model_dump(exclude={"details"}) for r in runs]


@router.get("/runs/{run_id}", response_model=EvalRun)
def get_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(EvalRun, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run

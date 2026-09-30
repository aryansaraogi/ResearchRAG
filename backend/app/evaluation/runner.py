"""Run the evaluation: retrieval metrics for every retrieval config, and
LLM-judged generation + citation metrics for the selected configs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from sqlmodel import Session

from app.config import get_settings
from app.db import engine
from app.evaluation.citation_metrics import citation_scores
from app.evaluation.dataset import EvalItem, load_dataset
from app.evaluation.generation_metrics import judge
from app.evaluation.retrieval_metrics import mean_scores, retrieval_scores
from app.generation import citations as cit
from app.generation.llm import get_llm
from app.generation.prompts import ANSWER_SYSTEM, answer_prompt
from app.models import EvalRun
from app.retrieval.hybrid import RetrievalMode, search

log = logging.getLogger(__name__)

# name -> (retrieval mode, rerank)
CONFIGS: dict[str, tuple[RetrievalMode, bool]] = {
    "dense": (RetrievalMode.dense, False),
    "sparse_bm25": (RetrievalMode.sparse, False),
    "hybrid_rrf": (RetrievalMode.hybrid, False),
    "hybrid_rrf_rerank": (RetrievalMode.hybrid, True),
}
RETRIEVAL_DEPTH = 10


def _run_generation(item: EvalItem, sources, gold: set[str]) -> dict:
    llm = get_llm()
    raw = llm.generate(answer_prompt(item.question, sources), system=ANSWER_SYSTEM)
    answer = cit.strip_invalid(raw.strip(), len(sources))
    scores, support = judge(llm, item.question, item.reference_answer, answer, sources)
    scores.update(citation_scores(answer, support, [s.id for s in sources], gold))
    return {"answer": answer, **scores}


def evaluate(
    configs: list[str] | None = None,
    gen_configs: list[str] | None = None,
    max_gen_questions: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, list]:
    items = load_dataset()
    if not items:
        raise ValueError("Evaluation dataset is empty. Generate one first.")
    configs = configs or list(CONFIGS)
    gen_configs = [c for c in (gen_configs or []) if c in configs]
    final_k = get_settings().final_k
    gen_items = items[:max_gen_questions] if max_gen_questions else items
    gen_ids = {i.id for i in gen_items}

    per_config_ret: dict[str, list[dict]] = {c: [] for c in configs}
    per_config_gen: dict[str, list[dict]] = {c: [] for c in gen_configs}
    latency: dict[str, list[float]] = {c: [] for c in configs}
    details = []

    total = len(items)
    for qi, item in enumerate(items, start=1):
        gold = set(item.gold_chunk_ids)
        row: dict = {"id": item.id, "question": item.question, "paper_id": item.paper_id, "configs": {}}
        for name in configs:
            mode, rerank = CONFIGS[name]
            t0 = time.perf_counter()
            results = search(item.question, mode=mode, rerank=rerank, top_k=RETRIEVAL_DEPTH)
            latency[name].append((time.perf_counter() - t0) * 1000)
            ranked = [r.id for r in results]
            rs = retrieval_scores(ranked, gold)
            per_config_ret[name].append(rs)
            gold_rank = next((i for i, r in enumerate(ranked, 1) if r in gold), None)
            entry: dict = {"gold_rank": gold_rank, "mrr": rs["mrr"]}

            if name in per_config_gen and item.id in gen_ids:
                try:
                    g = _run_generation(item, results[:final_k], gold)
                    per_config_gen[name].append({k: v for k, v in g.items() if k != "answer"})
                    entry["generation"] = g
                except Exception as e:  # noqa: BLE001 - keep evaluating the rest
                    log.warning("Generation eval failed for %s/%s: %s", item.id, name, e)
                    entry["generation_error"] = str(e)[:300]
            row["configs"][name] = entry
        details.append(row)
        if progress:
            progress(f"Evaluated {qi}/{total} questions")

    results = {}
    for name in configs:
        results[name] = {
            **mean_scores(per_config_ret[name]),
            "latency_ms": round(sum(latency[name]) / max(len(latency[name]), 1), 1),
        }
        if name in per_config_gen:
            results[name].update(mean_scores(per_config_gen[name]))
            results[name]["n_generation"] = len(per_config_gen[name])
    return results, details


def run_in_background(run_id: int, configs: list[str] | None, gen_configs: list[str] | None,
                      max_gen_questions: int | None) -> None:
    def progress(msg: str) -> None:
        with Session(engine) as s:
            run = s.get(EvalRun, run_id)
            run.progress = msg
            s.add(run)
            s.commit()

    try:
        results, details = evaluate(configs, gen_configs, max_gen_questions, progress)
        status, error = "done", None
    except Exception as e:  # noqa: BLE001
        log.exception("Eval run %s failed", run_id)
        results, details, status, error = {}, [], "failed", str(e)[:500]
    with Session(engine) as s:
        run = s.get(EvalRun, run_id)
        run.results, run.details, run.status, run.error = results, details, status, error
        run.dataset_size = len(load_dataset())
        s.add(run)
        s.commit()

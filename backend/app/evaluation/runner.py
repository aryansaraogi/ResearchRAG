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
from app.evaluation.retrieval_metrics import hit_at_k, mean_scores, reciprocal_rank, retrieval_scores
from app.generation import citations as cit
from app.generation.llm import get_judge_llm, get_llm
from app.generation.prompts import ANSWER_SYSTEM, NOT_FOUND, answer_prompt
from app.ingestion.chunker import chunk_id
from app.models import EvalRun
from app.retrieval import vector_store
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

# The answer prompt asks for NOT_FOUND verbatim; a short answer that only says the sources lack it counts too
REFUSAL_MARKERS = ("couldn't find", "could not find", "not in the indexed", "do not contain", "does not contain",
                   "don't contain", "doesn't contain", "no information", "not mentioned", "not covered")


def is_refusal(answer: str) -> bool:
    a = answer.strip().lower()
    return NOT_FOUND.lower() in a or (len(a) < 300 and any(m in a for m in REFUSAL_MARKERS))


def adjacent_gold(items: list[EvalItem]) -> dict[str, set[str]]:
    """Each item's gold chunks plus their neighbours in the same section. Chunks overlap by ~15%, so a
    neighbour often holds the same answer, yet a strict single-label metric scores it as a miss."""
    gold = {str(r.id): r.payload for r in vector_store.get_chunks(sorted({g for i in items for g in i.gold_chunk_ids}))}
    neighbour_of: dict[str, str] = {}  # neighbour id -> gold id
    for gid, p in gold.items():
        for j in (p["chunk_index"] - 1, p["chunk_index"] + 1):
            if j >= 0:
                neighbour_of[chunk_id(p["paper_id"], j)] = gid
    same_section = {
        str(r.id) for r in vector_store.get_chunks(list(neighbour_of))
        if r.payload["section"] == gold[neighbour_of[str(r.id)]]["section"]
    }
    return {
        item.id: set(item.gold_chunk_ids) | {n for n in same_section if neighbour_of[n] in item.gold_chunk_ids}
        for item in items
    }


def _run_generation(item: EvalItem, sources, gold: set[str]) -> dict:
    info: dict = {}
    raw = get_llm().generate(answer_prompt(item.question, sources), system=ANSWER_SYSTEM, info=info)
    answer = cit.strip_invalid(raw.strip(), len(sources))
    out = {"answer": answer, "model": info.get("model")}
    if not item.answerable:
        # Nothing to judge against: the only right answer is a refusal
        return {**out, "refusal_accuracy": 1.0 if is_refusal(answer) else 0.0}
    judge_info: dict = {}
    scores, support = judge(get_judge_llm(), item.question, item.reference_answer, answer, sources, info=judge_info)
    scores.update(citation_scores(answer, support, [s.id for s in sources], gold))
    return {**out, **scores, "false_refusal_rate": 1.0 if is_refusal(answer) else 0.0,
            "judge_model": judge_info.get("model")}


TEXT_FIELDS = ("answer", "model", "judge_model")


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
    # Multi-passage and unanswerable questions are few and are the point of judging, so they always run;
    # max_gen_questions caps the single-passage ones
    singles = [i for i in items if i.answerable and len(i.gold_chunk_ids) == 1]
    single_ids = {i.id for i in singles}
    special = [i for i in items if i.id not in single_ids]
    gen_ids = {i.id for i in special + (singles[:max_gen_questions] if max_gen_questions else singles)}
    lenient = adjacent_gold([i for i in items if i.answerable])

    per_config_ret: dict[str, list[dict]] = {c: [] for c in configs}
    per_config_gen: dict[str, list[dict]] = {c: [] for c in gen_configs}
    latency: dict[str, list[float]] = {c: [] for c in configs}
    details = []

    total = len(items)
    for qi, item in enumerate(items, start=1):
        gold = set(item.gold_chunk_ids)
        row: dict = {"id": item.id, "question": item.question, "paper_id": item.paper_id,
                     "answerable": item.answerable, "configs": {}}
        for name in configs:
            judged = name in per_config_gen and item.id in gen_ids
            if not item.answerable and not judged:
                continue  # no gold passage to rank, nothing to answer
            mode, rerank = CONFIGS[name]
            t0 = time.perf_counter()
            results = search(item.question, mode=mode, rerank=rerank, top_k=RETRIEVAL_DEPTH)
            latency[name].append((time.perf_counter() - t0) * 1000)
            entry: dict = {}
            if item.answerable:
                ranked = [r.id for r in results]
                rs = retrieval_scores(ranked, gold)
                adj = lenient[item.id]
                rs.update({"hit@1_adj": hit_at_k(ranked, adj, 1), "hit@5_adj": hit_at_k(ranked, adj, 5),
                           "mrr_adj": reciprocal_rank(ranked, adj)})
                per_config_ret[name].append(rs)
                entry = {"gold_rank": next((i for i, r in enumerate(ranked, 1) if r in gold), None), "mrr": rs["mrr"]}

            if judged:
                try:
                    g = _run_generation(item, results[:final_k], gold)
                    per_config_gen[name].append({k: v for k, v in g.items() if k not in TEXT_FIELDS})
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
            rows = per_config_gen[name]
            results[name].update(mean_scores(rows))
            results[name]["n_generation"] = len(rows)
            results[name]["n_unanswerable"] = sum(1 for r in rows if "refusal_accuracy" in r)
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

"""Command-line evaluation.

Examples:
  uv run python scripts/eval_cli.py generate --n 40
  uv run python scripts/eval_cli.py run                       # retrieval metrics, all configs
  uv run python scripts/eval_cli.py run --gen hybrid_rrf_rerank --max-gen 20
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db  # noqa: E402
from app.evaluation import dataset as ds  # noqa: E402
from app.evaluation.runner import CONFIGS, evaluate  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402

TABLE_METRICS = ["hit@1", "recall@5", "recall@10", "mrr", "ndcg@10", "mrr_adj", "latency_ms",
                 "faithfulness", "answer_relevance", "correctness", "citation_precision",
                 "citation_coverage", "gold_cited", "refusal_accuracy", "false_refusal_rate"]


def print_table(results: dict) -> None:
    metrics = [m for m in TABLE_METRICS if any(m in r for r in results.values())]
    width = max(len(n) for n in results) + 2
    print("config".ljust(width) + "".join(m.rjust(19) for m in metrics))
    for name, r in results.items():
        cells = "".join((f"{r[m]:.3f}" if m in r else "-").rjust(19) for m in metrics)
        print(name.ljust(width) + cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a synthetic QA dataset from indexed chunks")
    g.add_argument("--n", type=int, default=40)
    g.add_argument("--append", action="store_true")
    r = sub.add_parser("run", help="run evaluation")
    r.add_argument("--configs", nargs="*", choices=list(CONFIGS), default=None)
    r.add_argument("--gen", nargs="*", choices=list(CONFIGS), default=[],
                   help="configs to also evaluate generation/citations for (uses Gemini)")
    r.add_argument("--max-gen", type=int, default=20)
    r.add_argument("--out", type=Path, default=None, help="write full results JSON here")
    args = parser.parse_args()
    init_db()

    if args.cmd == "generate":
        items = ds.generate_dataset(get_llm(), n=args.n, append=args.append, progress=print)
        print(f"Saved {len(items)} questions to {ds.dataset_path()}")
        return

    results, details = evaluate(args.configs, args.gen, args.max_gen, progress=lambda m: print(m, end="\r"))
    print()
    print_table(results)
    if args.out:
        args.out.write_text(json.dumps({"results": results, "details": details}, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

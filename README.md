# ResearchRAG

ResearchRAG is a full-stack retrieval-augmented generation (RAG) platform for research papers. It covers semantic search and question answering across a paper library. Answers cite the exact paper section and page they come from. A built-in evaluation harness measures retrieval quality, answer quality and citation accuracy.

- **Ingestion:** arXiv import (by ID or topic search) or PDF upload. PDFs are parsed into sections using typography cues and two-column reading order, then chunked with section-aware boundaries.
- **Hybrid retrieval:**
  - dense BGE embeddings and sparse BM25, fused with reciprocal rank fusion (RRF) in Qdrant;
  - then a cross-encoder reranker;
  - with metadata filters on year, author, arXiv category, paper and section type.
- **Citation-aware generation:** Gemini answers only from numbered sources and cites every claim. Citation markers are checked against the retrieved sources and mapped back to the paper, section and page. The UI opens the PDF at the cited page.
- **Automated evaluation:**
  - synthetic question sets with gold passages;
  - IR metrics (Hit@k, Recall@k, MRR, nDCG) compared across four retrieval configs;
  - LLM-as-judge scoring of faithfulness, answer relevance and correctness;
  - per-claim citation precision and coverage.

## Architecture

```mermaid
flowchart LR
  subgraph Ingestion
    A[arXiv API / PDF upload] --> B[PyMuPDF parser<br/>sections + reading order]
    B --> C[Section-aware chunker<br/>~300 words, 15% overlap]
    C --> D[FastEmbed<br/>BGE dense + BM25 sparse]
  end
  D --> Q[(Qdrant<br/>named dense + sparse vectors<br/>payload: year, authors, section…)]
  subgraph Query
    U[Question + filters] --> P1[Dense prefetch top-50]
    U --> P2[BM25 prefetch top-50]
    P1 --> F[RRF fusion]
    P2 --> F
    F --> R[Cross-encoder rerank<br/>top-20 → top-8]
    R --> G[Gemini: numbered sources,<br/>cite every claim]
    G --> V[Citation validation<br/>+ source mapping]
  end
  Q --- P1
  Q --- P2
  V --> UI[Next.js UI: answer with<br/>citation chips → PDF page]
```

| Layer | Tech |
|---|---|
| API | FastAPI, Pydantic v2, SQLModel/SQLite (paper metadata and eval runs), SSE streaming |
| Retrieval | Qdrant (server or embedded), FastEmbed: `BAAI/bge-small-en-v1.5` (dense), `Qdrant/bm25` (sparse, IDF), `Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder). All ONNX, CPU-only, no torch. |
| Generation | Google Gemini (`google-genai`), client-side rate limiting and retry for the free tier |
| Frontend | Next.js 16 (App Router), React 19, Tailwind CSS 4, Recharts, react-markdown + KaTeX |

## Quick start

Prerequisites: Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/), Node ≥ 20, and a free [Gemini API key](https://aistudio.google.com/apikey). Docker is optional.

```bash
# 1. Backend
cd backend
uv sync
cp .env.example .env          # then set GEMINI_API_KEY
uv run uvicorn app.main:app --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                   # http://localhost:3000
```

In the UI, go to **Library** and paste some arXiv IDs, for example `1706.03762 2005.11401 1810.04805 2004.04906`. Wait for them to show **Indexed**, then try **Search** and **Ask**.

Without a Gemini key, ingestion, search and retrieval evaluation still work. Chat, question generation and answer judging need the key.

### Vector store: embedded or server

By default `QDRANT_URL` is empty, and Qdrant runs **embedded** in the API process with data in `backend/data/qdrant_local`. That setup needs no Docker, but only one process can open the store at a time.

To use a Qdrant server instead:

```bash
docker compose up -d qdrant
# backend/.env
QDRANT_URL=http://localhost:6333
```

## Evaluation

The evaluation answers two questions:
1. Does each retrieval strategy find the right passage?
2. Are the generated answers faithful, relevant and correctly cited?

1. **Question set.** Gemini writes one self-contained question and a reference answer for each sampled chunk. Sampling is round-robin across papers, and front matter and appendices are skipped. The source chunk is the *gold* passage. Hand-written items with `"source": "manual"` can be appended to `backend/data/eval/dataset.jsonl`.
2. **Retrieval metrics.** These are computed locally for four configs:

   | Config | Description |
   |---|---|
   | `dense` | BGE only |
   | `sparse_bm25` | BM25 only |
   | `hybrid_rrf` | Dense + BM25, RRF |
   | `hybrid_rrf_rerank` | Hybrid, then cross-encoder |

   Reported per config: Hit@{1,3,5,10}, Recall@{5,10}, MRR, nDCG@10, P@5 and mean latency.
3. **Generation and citation metrics.** These use one judge call per answer to stay inside free-tier quotas:

   | Metric | Meaning |
   |---|---|
   | faithfulness | Every claim is supported by the retrieved sources |
   | answer_relevance | The answer addresses the question |
   | context_relevance | The retrieved sources are on-topic |
   | correctness | The answer agrees with the reference answer |
   | citation_precision | Share of (sentence, cited source) pairs where that source actually supports the sentence |
   | citation_coverage | Share of answer sentences that carry a citation |
   | gold_cited | Whether the answer cites the gold passage, when that passage was retrieved |

   Judge scores are 1–5, rescaled to 0–1.

Run the evaluation from the **Evaluation** page, or from the CLI. In embedded mode, stop the API first.

```bash
cd backend
uv run python scripts/eval_cli.py generate --n 40
uv run python scripts/eval_cli.py run                                  # retrieval, all configs
uv run python scripts/eval_cli.py run --gen hybrid_rrf_rerank --max-gen 20 --out results.json
```

### Results

These results cover the four example papers (Transformer, BERT, DPR, RAG), split into 153 chunks, with 66 questions:
- **28 hand-written**, 7 per paper. Each was written from one passage read directly from the chunked PDFs, not picked through search, so the gold labels don't favor any retriever.
- **38 generated by Gemini** with the pipeline described above.

Answers were generated and judged by `gemini-3.5-flash-lite`. Latency is the mean per query on a 4-core CPU.

**Retrieval, all 66 questions**

| Config | Hit@1 | Hit@5 | MRR | nDCG@10 | Latency |
|---|---|---|---|---|---|
| Dense | 0.636 | 0.864 | 0.729 | 0.766 | 59 ms |
| BM25 | 0.667 | **0.970** | 0.799 | 0.846 | 13 ms |
| Hybrid (RRF) | 0.667 | 0.955 | 0.781 | 0.831 | 76 ms |
| Hybrid + rerank | **0.758** | 0.939 | **0.834** | **0.864** | 4.3 s |

**MRR by question source**

| Config | Hand-written (28) | Gemini-written (38) |
|---|---|---|
| Dense | 0.830 | 0.654 |
| BM25 | 0.765 | 0.825 |
| Hybrid (RRF) | **0.851** | 0.729 |
| Hybrid + rerank | 0.808 | **0.852** |

**Answer quality**, Hybrid + rerank, 28 hand-written questions:

| Faithfulness | Answer relevance | Context relevance | Correctness | Citation precision | Citation coverage | Gold passage cited |
|---|---|---|---|---|---|---|
| 0.99 | 1.00 | 1.00 | 0.99 | 0.87 | 0.89 | 1.00 |

- **Overall winner.** Hybrid + rerank has the best Hit@1, MRR and nDCG@10. BM25 has the best Hit@5.
- **The question source changes the winner.** Gemini's questions reuse the passage's wording, so BM25 beats dense by 0.17 MRR on them. The cross-encoder adds 0.12 on top of RRF there.
- **Hand-written questions reverse it.** On these paraphrased questions dense beats BM25, RRF is best, and the reranker costs 0.04. The passages it promotes share the question's key terms but don't hold the answer. One example is an appendix that names "Thorough Decoding" without defining it. `ms-marco-MiniLM-L-6-v2` is a small web-search model, and a larger reranker such as `BAAI/bge-reranker-base` may do better.
- **Which column to trust.** Users rarely know a paper's exact wording, so the hand-written column is the better guide to real queries. That's why both sets are kept.
- **Judge scores sit near the ceiling.** Each question has a single-passage answer, and the judge is the same small model. The citation checks separate answers better: 13% of (sentence, citation) pairs cite a source that doesn't support the sentence, and 11% of sentences carry no citation.
- **Noise.** One question moves MRR by up to 0.036 on 28 questions and up to 0.015 on 66, so treat small differences as noise.

The hand-written questions are in `backend/app/evaluation/manual_questions.jsonl`. Copy them to `backend/data/eval/dataset.jsonl` and run the evaluation to reproduce that column. Chunk IDs are deterministic, so the gold labels match as long as the same PDF versions are indexed. A new Gemini-generated set will contain different questions.

**Caveats.**
- Synthetic questions are written from a single chunk, so they tend to reuse its vocabulary, which favors lexical retrieval (measured above). The prompt asks for paraphrased, self-contained questions to reduce this, and a hand-written set is a useful check.
- Only one gold chunk is labeled per question. Neighboring chunks that also answer it count as misses, which makes the retrieval numbers conservative.
- The judge is the same model family as the generator. Treat absolute judge scores as relative comparisons.

## Design notes

- **PDF parsing.**
  - Headings are detected from bold weight and font size relative to body text, plus numbering patterns. Numbers set as separate spans ("3" + "Model Architecture") are rejoined.
  - Blocks are ordered column by column, so two-column papers don't interleave.
  - Running headers and footers, the vertical arXiv sidebar and the reference list are dropped.
  - Text is NFKC-normalized so ligatures like "ﬁ" match keyword search.
- **Chunking.** Chunks are packed at sentence level (~400 tokens), never cross a section boundary, carry page ranges for citations, and overlap by 15%. Each chunk is embedded with its paper title and section heading as a prefix.
- **Hybrid search.** Both prefetches apply the same payload filter, so filtering happens before fusion rather than after. RRF avoids calibrating dense and sparse scores against each other.
- **Reranking cost.** Cross-encoder cost grows linearly with candidates (~0.15 s each on a 4-core CPU), so only the top `RERANK_CANDIDATES` fused results (default 20) are reranked.
- **Citations.** The model sees sources as `[n] Title — §Section, p. X`. After generation, out-of-range markers are stripped and the rest are resolved to chunk, paper, section and page.
- **Free-tier friendly.** A client-side limiter spaces Gemini calls to `GEMINI_RPM`, and 429 and 5xx responses are retried with exponential backoff.

## Project layout

```
backend/
  app/
    ingestion/    pdf_parser, chunker, arxiv_client, pipeline
    retrieval/    embeddings, vector_store, filters, hybrid, reranker
    generation/   llm (Gemini), prompts, citations, rag
    evaluation/   dataset, retrieval_metrics, generation_metrics, citation_metrics, runner
    api/          papers, search, chat (SSE), eval
  scripts/eval_cli.py
  tests/          parser/chunker, filters, metrics, citation parsing
frontend/
  app/            / (library), /search, /chat, /eval
  components/     FilterPanel, AnswerMarkdown (citation chips), Nav, ui
  lib/api.ts      typed API client and SSE reader
```

Run the tests with `cd backend && uv run pytest`.

## Troubleshooting

- **Python crashes on `import onnxruntime` (Windows).** Newer onnxruntime wheels segfault when the system MSVC runtime (`msvcp140.dll`) is older than 14.40. `pyproject.toml` pins `onnxruntime==1.19.2` for this reason. Installing the latest [VC++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) fixes the root cause, and also lets the native Qdrant Windows binary run.
- **`Storage folder ... is already accessed by another instance`.** Embedded Qdrant is single-process. Stop the API before running the CLI, or switch to server mode.
- **Gemini 429s.**
  - Per-minute limits: lower `GEMINI_RPM`, or evaluate fewer questions with `--max-gen`.
  - Daily quota: the error names the model and its limit. Free-tier quotas are per model per day, and full Flash models allow as few as 20 requests. Waiting won't help until the reset at midnight Pacific time, so the client doesn't retry. Switch `GEMINI_MODEL` to another model the key lists, such as `gemini-3.5-flash-lite`.
- **Gemini 503 "high demand".** The model is overloaded. Calls are retried with backoff. If it persists, switch `GEMINI_MODEL` to a less busy model.

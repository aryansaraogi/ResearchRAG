"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import { FileText, Search as SearchIcon, SlidersHorizontal } from "lucide-react";
import { api, pageLabel, type RetrievalMode, type RetrievedChunk, type SearchFilters } from "@/lib/api";
import { FilterPanel, RetrievalSettings, activeFilterCount, useFacets } from "@/components/FilterPanel";
import { usePdfViewer } from "@/components/PdfViewer";
import { Badge, Button, Card, EmptyState, ErrorNote, PageHeader, Skeleton, inputClass } from "@/components/ui";

const STOPWORDS = new Set("a an the of in on for to and or is are was were how what why which with by from does do".split(" "));
const EXAMPLES = [
  "Why does scaled dot-product attention divide by sqrt(d_k)?",
  "How are hard negatives chosen for dense passage retrieval?",
  "What fraction of tokens does BERT mask during pre-training?",
  "How does RAG marginalize over retrieved documents?",
];

function Highlight({ text, query }: { text: string; query: string }) {
  const terms = query
    .toLowerCase()
    .split(/\W+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
  if (!terms.length) return <>{text}</>;
  // One capture group around term+suffix so split() keeps whole matched words at odd indices
  const re = new RegExp(`\\b((?:${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*)`, "gi");
  const parts = text.split(re);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="rounded-sm bg-accent-soft px-0.5 text-accent-ink">
            {p}
          </mark>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </>
  );
}

function ResultCard({ r, rank, query }: { r: RetrievedChunk; rank: number; query: string }) {
  const openPdf = usePdfViewer();
  const [expanded, setExpanded] = useState(false);
  const text = expanded || r.text.length < 600 ? r.text : r.text.slice(0, 600) + "…";
  return (
    <Card className="animate-in p-4">
      <div className="flex items-start gap-3">
        <span
          className="mt-0.5 grid h-6 min-w-6 shrink-0 place-items-center rounded-md bg-surface-2 px-1 font-mono text-xs text-ink-2"
          title={`Rank ${rank} · fused retrieval score ${r.score.toFixed(4)}`}
        >
          {rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-medium leading-snug">{r.title}</span>
            {r.year && <span className="text-xs text-ink-3">{r.year}</span>}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-ink-2">
            <span>§ {r.section}</span>
            <Badge>{pageLabel(r.page_start, r.page_end)}</Badge>
            <Badge tone="accent">{r.section_type.replace("_", " ")}</Badge>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-ink-2">
            <Highlight text={text} query={query} />
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-4 text-xs">
            {r.text.length >= 600 && (
              <button className="text-accent-ink hover:underline" onClick={() => setExpanded(!expanded)}>
                {expanded ? "Show less" : "Show more"}
              </button>
            )}
            <button
              className="flex items-center gap-1 text-accent-ink hover:underline"
              onClick={() => openPdf({ paperId: r.paper_id, title: r.title, page: r.page_start })}
            >
              <FileText size={12} /> View p. {r.page_start}
            </button>
            {r.rerank_score != null && (
              <span className="ml-auto text-ink-3" title="Cross-encoder relevance score (higher is more relevant)">
                relevance <span className="font-mono text-ink-2">{r.rerank_score.toFixed(2)}</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

function ResultsSkeleton() {
  return (
    <div className="space-y-3" aria-label="Searching">
      {[0, 1, 2].map((i) => (
        <Card key={i} className="space-y-2 p-4">
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-3 w-1/4" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
        </Card>
      ))}
    </div>
  );
}

export default function SearchPage() {
  const facets = useFacets();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [rerank, setRerank] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [results, setResults] = useState<RetrievedChunk[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const nFilters = activeFilterCount(filters);

  // "/" focuses the search box unless the user is already typing somewhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key !== "/" || e.metaKey || e.ctrlKey || el.closest("input, textarea, select, [contenteditable]")) return;
      e.preventDefault();
      inputRef.current?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const run = async (q: string = query) => {
    if (!q.trim()) return;
    setQuery(q);
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      setResults(await api.search({ query: q, filters, mode, rerank, top_k: 10 }));
      setSubmitted(q);
      setLatency(performance.now() - t0);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const settings = (
    <div className="space-y-5">
      <RetrievalSettings mode={mode} rerank={rerank} onMode={setMode} onRerank={setRerank} />
      <FilterPanel filters={filters} onChange={setFilters} facets={facets} />
    </div>
  );

  return (
    <div className="mx-auto grid w-full max-w-7xl gap-6 px-4 py-8 lg:grid-cols-[260px_1fr]">
      <aside className="hidden lg:sticky lg:top-20 lg:block lg:self-start">{settings}</aside>
      <div className="min-w-0 space-y-4">
        <PageHeader icon={SearchIcon} title="Semantic search" subtitle="Find the passages across your papers that match a question or concept." />
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
          className="flex gap-2"
        >
          <div className="relative flex-1">
            <input
              ref={inputRef}
              className={inputClass + " pr-9"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. why does scaled dot-product attention divide by sqrt(d_k)?"
              aria-label="Search query"
              autoFocus
            />
            <kbd className="pointer-events-none absolute right-2 top-1/2 hidden -translate-y-1/2 rounded border border-line px-1.5 font-mono text-[10px] text-ink-3 sm:block">
              /
            </kbd>
          </div>
          <Button type="submit" loading={loading} disabled={!query.trim()}>
            {!loading && <SearchIcon size={14} />} Search
          </Button>
        </form>
        <Button
          variant="secondary"
          className="lg:hidden"
          onClick={() => setShowFilters(!showFilters)}
          aria-expanded={showFilters}
        >
          <SlidersHorizontal size={14} /> {showFilters ? "Hide filters" : "Filters"}
          {nFilters ? ` (${nFilters})` : ""}
        </Button>
        {showFilters && <Card className="animate-in p-4 lg:hidden">{settings}</Card>}
        <ErrorNote message={error} />
        {results && !loading && (
          <p className="text-xs text-ink-3">
            {results.length} passages · {mode === "sparse" ? "BM25" : mode}
            {rerank ? " + rerank" : ""}
            {latency != null && ` · ${Math.round(latency)} ms`}
          </p>
        )}
        {loading ? (
          <ResultsSkeleton />
        ) : results === null ? (
          <Card>
            <EmptyState icon={SearchIcon} title="Search across every indexed paper">
              Hybrid blends meaning-based (dense) and keyword (BM25) matches; rerank reorders the top results with a
              cross-encoder. Try one of these:
            </EmptyState>
            <div className="-mt-4 flex flex-wrap justify-center gap-2 px-4 pb-8">
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  onClick={() => run(q)}
                  className="rounded-full border border-line px-3 py-1 text-xs text-ink-2 transition hover:border-accent/50 hover:text-ink"
                >
                  {q}
                </button>
              ))}
            </div>
          </Card>
        ) : results.length === 0 ? (
          <Card>
            <EmptyState icon={SearchIcon} title="No passages match">
              Try loosening the filters or rephrasing the question.
            </EmptyState>
          </Card>
        ) : (
          <div className="space-y-3">
            {results.map((r, i) => (
              <ResultCard key={r.id} r={r} rank={i + 1} query={submitted} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

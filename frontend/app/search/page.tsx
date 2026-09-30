"use client";

import { Fragment, useState } from "react";
import { ExternalLink, Search as SearchIcon } from "lucide-react";
import { api, pageLabel, pdfUrl, type RetrievalMode, type RetrievedChunk, type SearchFilters } from "@/lib/api";
import { FilterPanel, RetrievalSettings, useFacets } from "@/components/FilterPanel";
import { Badge, Button, Card, ErrorNote, PageHeader, inputClass } from "@/components/ui";

const STOPWORDS = new Set("a an the of in on for to and or is are was were how what why which with by from does do".split(" "));

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
  const [expanded, setExpanded] = useState(false);
  const text = expanded || r.text.length < 600 ? r.text : r.text.slice(0, 600) + "…";
  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 w-6 shrink-0 text-right font-mono text-xs text-ink-3">{rank}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium leading-snug">{r.title}</span>
            {r.year && <span className="text-xs text-ink-3">{r.year}</span>}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-ink-2">
            <span>§ {r.section}</span>
            <span className="text-ink-3">{pageLabel(r.page_start, r.page_end)}</span>
            <Badge>{r.section_type.replace("_", " ")}</Badge>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-ink-2">
            <Highlight text={text} query={query} />
          </p>
          <div className="mt-2 flex items-center gap-4 text-xs">
            {r.text.length >= 600 && (
              <button className="text-accent hover:underline" onClick={() => setExpanded(!expanded)}>
                {expanded ? "Show less" : "Show more"}
              </button>
            )}
            <a href={pdfUrl(r.paper_id, r.page_start)} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-accent hover:underline">
              Open at page {r.page_start} <ExternalLink size={12} />
            </a>
            <span className="ml-auto font-mono text-ink-3" title={r.rerank_score != null ? "Cross-encoder score" : "Retrieval score"}>
              {r.rerank_score != null ? `rerank ${r.rerank_score.toFixed(2)}` : `score ${r.score.toFixed(3)}`}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default function SearchPage() {
  const facets = useFacets();
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [rerank, setRerank] = useState(true);
  const [results, setResults] = useState<RetrievedChunk[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  const run = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      setResults(await api.search({ query, filters, mode, rerank, top_k: 10 }));
      setSubmitted(query);
      setLatency(performance.now() - t0);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto grid w-full max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[260px_1fr]">
      <aside className="space-y-5 lg:sticky lg:top-20 lg:self-start">
        <RetrievalSettings mode={mode} rerank={rerank} onMode={setMode} onRerank={setRerank} />
        <FilterPanel filters={filters} onChange={setFilters} facets={facets} />
      </aside>
      <div className="min-w-0 space-y-4">
        <PageHeader title="Semantic search" subtitle="Find the passages across your papers that match a question or concept." />
        <form onSubmit={run} className="flex gap-2">
          <input
            className={inputClass}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. why does scaled dot-product attention divide by sqrt(d_k)?"
            autoFocus
          />
          <Button type="submit" loading={loading} disabled={!query.trim()}>
            <SearchIcon size={14} /> Search
          </Button>
        </form>
        <ErrorNote message={error} />
        {results && (
          <p className="text-xs text-ink-3">
            {results.length} passages · {mode === "sparse" ? "BM25" : mode}
            {rerank ? " + rerank" : ""}
            {latency != null && ` · ${Math.round(latency)} ms`}
          </p>
        )}
        <div className="space-y-3">
          {results?.map((r, i) => <ResultCard key={r.id} r={r} rank={i + 1} query={submitted} />)}
          {results?.length === 0 && (
            <Card className="p-8 text-center text-sm text-ink-3">No passages match. Try loosening the filters.</Card>
          )}
        </div>
      </div>
    </div>
  );
}

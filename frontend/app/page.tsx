"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BookOpen,
  ChevronDown,
  FileDown,
  FileText,
  FileUp,
  Library,
  List,
  Network,
  Plus,
  Quote,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import { api, type ArxivResult, type Paper, type SummaryPoint } from "@/lib/api";
import { downloadText, toBibtex } from "@/lib/bibtex";
import { CitationGraphView } from "@/components/CitationGraph";
import { usePdfViewer } from "@/components/PdfViewer";
import { useToast } from "@/components/Toast";
import {
  Badge,
  Button,
  Card,
  ConfirmButton,
  EmptyState,
  ErrorNote,
  IconButton,
  PageHeader,
  Skeleton,
  StatTile,
  inputClass,
  inputSmClass,
} from "@/components/ui";

const ARXIV_ID = /^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$|arxiv\.org\/(abs|pdf)\//i;
const CLASSICS = ["1706.03762", "1810.04805", "2004.04906", "2005.11401"];
const FILTER_THRESHOLD = 5;

function authorsShort(a: string[]) {
  if (!a.length) return "Unknown authors";
  return a.length > 3 ? `${a.slice(0, 3).join(", ")} et al.` : a.join(", ");
}

function StatusBadge({ status }: { status: Paper["status"] }) {
  const map = {
    ready: ["good", "Indexed"],
    processing: ["accent", "Indexing…"],
    pending: ["neutral", "Queued"],
    failed: ["critical", "Failed"],
  } as const;
  const [tone, label] = map[status];
  return <Badge tone={tone}>{label}</Badge>;
}

function ArxivImport({ onImported }: { onImported: (n: number) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<ArxivResult[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const tokens = q.split(/[\s,]+/).filter(Boolean);
  const isIdList = tokens.length > 0 && tokens.every((t) => ARXIV_ID.test(t));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (isIdList) {
        const papers = await api.importArxiv(tokens);
        setQ("");
        setResults(null);
        onImported(papers.length);
      } else {
        setResults(await api.searchArxiv(q));
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const importOne = async (id: string) => {
    setImporting(id);
    setError(null);
    try {
      onImported((await api.importArxiv([id])).length);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setImporting(null);
    }
  };

  return (
    <Card className="p-4">
      <h2 className="text-sm font-semibold">Import from arXiv</h2>
      <p className="mt-0.5 text-xs text-ink-3">Paste arXiv IDs (e.g. 1706.03762 2005.11401) or search by topic.</p>
      <form onSubmit={submit} className="mt-3 flex gap-2">
        <input
          className={inputClass}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="IDs or search query"
          aria-label="arXiv IDs or search query"
        />
        <Button type="submit" loading={busy} disabled={!q.trim()}>
          {!busy && (isIdList ? <Plus size={14} /> : <Search size={14} />)}
          {isIdList ? `Import ${tokens.length}` : "Search"}
        </Button>
      </form>
      <div className="mt-2">
        <ErrorNote message={error} />
      </div>
      {results && (
        <ul className="mt-2 max-h-72 divide-y divide-line overflow-y-auto">
          {results.length === 0 && <li className="py-2 text-sm text-ink-3">No results.</li>}
          {results.map((r) => (
            <li key={r.arxiv_id} className="flex items-start gap-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium leading-snug">{r.title}</div>
                <div className="truncate text-xs text-ink-3">
                  {authorsShort(r.authors)} · {r.year} · {r.arxiv_id}
                </div>
              </div>
              <Button variant="secondary" loading={importing === r.arxiv_id} onClick={() => importOne(r.arxiv_id)}>
                Import
              </Button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function Upload({ onUploaded }: { onUploaded: (n: number) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async (files: File[]) => {
    const pdfs = files.filter((f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    if (!pdfs.length) return setError("Only PDF files are supported.");
    setError(null);
    setBusy(true);
    try {
      onUploaded((await api.upload(pdfs)).length);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="p-4">
      <h2 className="text-sm font-semibold">Upload PDFs</h2>
      <p className="mt-0.5 text-xs text-ink-3">arXiv PDFs get their metadata filled in automatically.</p>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => (e.preventDefault(), setDrag(true))}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          send(Array.from(e.dataTransfer.files));
        }}
        className={`mt-3 flex w-full flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-4 py-6 text-sm transition ${
          drag ? "border-accent bg-accent-soft" : "border-line text-ink-2 hover:border-ink-3 hover:bg-surface-2/50"
        }`}
      >
        <FileUp size={20} className="text-ink-3" />
        {busy ? "Uploading…" : "Drop PDFs here or click to browse"}
      </button>
      <input
        ref={input}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={(e) => {
          send(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      <div className="mt-2">
        <ErrorNote message={error} />
      </div>
    </Card>
  );
}

function SummaryList({ label, points }: { label: string; points: SummaryPoint[] }) {
  if (!points.length) return null;
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">{label}</div>
      <ul className="list-disc space-y-1 pl-5 text-ink-2">
        {points.map((p, i) => (
          <li key={i}>
            {p.text} {p.section && <span className="text-xs text-ink-3">(§ {p.section})</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function PaperRow({ paper, onChange }: { paper: Paper; onChange: () => void }) {
  const openPdf = usePdfViewer();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const act = async (fn: () => Promise<unknown>, done: string) => {
    setBusy(true);
    try {
      await fn();
      toast(done, "good");
      onChange();
    } catch (err) {
      toast((err as Error).message, "critical");
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="animate-in px-4 py-3.5 transition-colors hover:bg-surface-2/40">
      {/* Phones: actions wrap under the details so titles keep the full width */}
      <div className="grid grid-cols-[auto_1fr] items-start gap-x-3 sm:grid-cols-[auto_1fr_auto]">
        <button
          onClick={() => setOpen(!open)}
          className="mt-0.5 rounded text-ink-3 hover:text-ink"
          aria-label={open ? "Hide details" : "Show details"}
          aria-expanded={open}
        >
          <ChevronDown size={16} className={`transition ${open ? "" : "-rotate-90"}`} />
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium leading-snug">{paper.title}</span>
            <StatusBadge status={paper.status} />
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
            {paper.source === "arxiv" && <Badge>arXiv:{paper.id}</Badge>}
            {paper.year && <span>{paper.year}</span>}
            <span className="truncate">{authorsShort(paper.authors)}</span>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            {paper.categories.slice(0, 4).map((c) => (
              <Badge key={c} tone="accent">
                {c}
              </Badge>
            ))}
            {paper.status === "ready" && (
              <span className="ml-1 text-xs text-ink-3">
                {paper.num_pages} pages · {paper.sections.length} sections · {paper.num_chunks} chunks
              </span>
            )}
          </div>
          {paper.summary?.tldr && <p className="mt-1.5 line-clamp-2 text-sm text-ink-2">{paper.summary.tldr}</p>}
          {paper.error && <p className="mt-1 text-xs text-critical-ink">{paper.error}</p>}
        </div>
        <div className="col-start-2 -ml-2 mt-1.5 flex items-center gap-0.5 sm:col-start-3 sm:row-start-1 sm:ml-0 sm:mt-0">
          <IconButton
            label="Copy BibTeX"
            onClick={() =>
              navigator.clipboard
                .writeText(toBibtex([paper]))
                .then(() => toast("BibTeX copied", "good"))
                .catch(() => toast("Couldn't copy to the clipboard", "critical"))
            }
          >
            <Quote size={15} />
          </IconButton>
          <IconButton label="View PDF" onClick={() => openPdf({ paperId: paper.id, title: paper.title, page: 1 })}>
            <FileText size={15} />
          </IconButton>
          <IconButton label="Re-index" disabled={busy} onClick={() => act(() => api.reingest(paper.id), "Re-indexing started")}>
            <RefreshCw size={15} className={busy ? "animate-spin" : ""} />
          </IconButton>
          <ConfirmButton
            label={`Delete "${paper.title}"`}
            disabled={busy}
            onConfirm={() => act(() => api.deletePaper(paper.id), "Paper deleted")}
          >
            <Trash2 size={15} />
          </ConfirmButton>
        </div>
      </div>
      {open && (
        <div className="animate-in ml-7 mt-3 space-y-3 text-sm">
          {paper.summary ? (
            <div className="space-y-3 rounded-lg border border-line bg-surface-2/40 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-ink-2">
                  <Sparkles size={13} className="text-accent-ink" /> Summary
                </span>
                <Button variant="ghost" className="px-2! py-1! text-xs" disabled={busy} onClick={() => act(() => api.summarize(paper.id), "Summary regenerated")}>
                  <RefreshCw size={12} /> Regenerate
                </Button>
              </div>
              <p className="text-ink">{paper.summary.tldr}</p>
              <SummaryList label="Key contributions" points={paper.summary.contributions} />
              <SummaryList label="Limitations" points={paper.summary.limitations} />
              {paper.summary.model && <p className="text-xs text-ink-3">Generated by {paper.summary.model} from the paper&apos;s own text.</p>}
            </div>
          ) : (
            paper.status === "ready" && (
              <Button variant="secondary" loading={busy} onClick={() => act(() => api.summarize(paper.id), "Summary ready")}>
                {!busy && <Sparkles size={14} />} Summarize this paper
              </Button>
            )
          )}
          {paper.abstract && <p className="leading-relaxed text-ink-2">{paper.abstract}</p>}
          {!!paper.sections.length && (
            <div>
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">Detected sections</div>
              <div className="flex flex-wrap gap-1">
                {paper.sections.map((s, i) => (
                  <Badge key={i}>{s}</Badge>
                ))}
              </div>
            </div>
          )}
          <div className="text-xs text-ink-3">
            ID: {paper.id} · source: {paper.source}
          </div>
        </div>
      )}
    </li>
  );
}

function PaperListSkeleton() {
  return (
    <ul className="divide-y divide-line" aria-label="Loading papers">
      {[0, 1, 2].map((i) => (
        <li key={i} className="flex gap-3 px-4 py-4">
          <Skeleton className="mt-0.5 h-4 w-4" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-3 w-1/3" />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function LibraryPage() {
  const toast = useToast();
  const [papers, setPapers] = useState<Paper[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [seeding, setSeeding] = useState(false);
  const [view, setView] = useState<"list" | "graph">("list");

  const load = useCallback(() => {
    api
      .papers()
      .then((p) => {
        setPapers(p);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(load, [load]);

  // Poll while anything is still being ingested
  const busy = papers?.some((p) => p.status === "pending" || p.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [busy, load]);

  const added = (verb: string) => (n: number) => {
    toast(`${verb} ${n} paper${n === 1 ? "" : "s"}. Indexing runs in the background.`, "good");
    load();
  };

  const importClassics = async () => {
    setSeeding(true);
    try {
      added("Imported")((await api.importArxiv(CLASSICS)).length);
    } catch (e) {
      toast((e as Error).message, "critical");
    } finally {
      setSeeding(false);
    }
  };

  const ready = papers?.filter((p) => p.status === "ready") ?? [];
  const chunks = ready.reduce((n, p) => n + p.num_chunks, 0);
  const sections = ready.reduce((n, p) => n + p.sections.length, 0);
  const years = ready.map((p) => p.year).filter((y): y is number => !!y);
  const span = years.length ? `${Math.min(...years)}–${Math.max(...years)}` : "—";
  const q = filter.trim().toLowerCase();
  const shown =
    papers?.filter((p) => !q || p.title.toLowerCase().includes(q) || p.authors.some((a) => a.toLowerCase().includes(q))) ?? [];

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8">
      <PageHeader
        icon={BookOpen}
        title="Paper library"
        subtitle="Import papers once. They're parsed into sections, chunked and indexed for search and cited answers."
      />
      <ErrorNote message={error} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {papers === null ? (
          [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-21.5 rounded-xl" />)
        ) : (
          <>
            <StatTile label="Papers indexed" value={ready.length} sub={busy ? "More indexing now…" : `${papers.length} in library`} />
            <StatTile label="Searchable chunks" value={chunks.toLocaleString()} sub="~400 tokens each" />
            <StatTile label="Sections detected" value={sections.toLocaleString()} sub="From headings and fonts" />
            <StatTile label="Publication years" value={span} sub={years.length ? `${new Set(years).size} distinct` : "No dated papers"} />
          </>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ArxivImport onImported={added("Queued")} />
        <Upload onUploaded={added("Uploaded")} />
      </div>

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="grid grid-cols-2 rounded-md border border-line p-0.5 text-xs" role="tablist" aria-label="Library view">
              {([
                ["list", List, "Papers"],
                ["graph", Network, "Citation graph"],
              ] as const).map(([v, Icon, label]) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={view === v}
                  onClick={() => setView(v)}
                  className={`flex items-center justify-center gap-1 rounded px-2.5 py-1 ${view === v ? "bg-accent text-white" : "text-ink-2 hover:text-ink"}`}
                >
                  <Icon size={13} /> {label}
                </button>
              ))}
            </div>
            {papers && view === "list" && (
              <span className="text-xs text-ink-3">{q ? `${shown.length} of ${papers.length}` : papers.length}</span>
            )}
          </div>
          {!!ready.length && (
            <Button
              variant="ghost"
              className="px-2! py-1! text-xs"
              onClick={() => downloadText("researchrag-library.bib", toBibtex(ready) + "\n")}
              title="Download BibTeX for every indexed paper"
            >
              <FileDown size={13} /> Export .bib
            </Button>
          )}
          {view === "list" && papers && papers.length > FILTER_THRESHOLD && (
            <input
              className={inputSmClass + " w-full sm:w-64"}
              placeholder="Filter by title or author…"
              aria-label="Filter papers"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          )}
        </div>
        {view === "graph" ? (
          <CitationGraphView />
        ) : papers === null ? (
          <PaperListSkeleton />
        ) : papers.length === 0 ? (
          <EmptyState
            icon={Library}
            title="No papers yet"
            actions={
              <Button onClick={importClassics} loading={seeding}>
                {!seeding && <Plus size={14} />} Import 4 classic papers
              </Button>
            }
          >
            Import from arXiv or upload PDFs above. Or start with Transformer, BERT, DPR and RAG, the papers the README
            evaluation uses.
          </EmptyState>
        ) : shown.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-ink-3">No papers match “{filter}”.</p>
        ) : (
          <ul className="divide-y divide-line">
            {shown.map((p) => (
              <PaperRow key={p.id} paper={p} onChange={load} />
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

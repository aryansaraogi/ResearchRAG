"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ExternalLink, FileUp, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { api, pdfUrl, type ArxivResult, type Paper } from "@/lib/api";
import { Badge, Button, Card, ErrorNote, PageHeader, inputClass } from "@/components/ui";

const ARXIV_ID = /^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$|arxiv\.org\/(abs|pdf)\//i;

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

function ArxivImport({ onImported }: { onImported: () => void }) {
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
        await api.importArxiv(tokens);
        setQ("");
        setResults(null);
        onImported();
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
      await api.importArxiv([id]);
      onImported();
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
        <input className={inputClass} value={q} onChange={(e) => setQ(e.target.value)} placeholder="IDs or search query" />
        <Button type="submit" loading={busy} disabled={!q.trim()}>
          {isIdList ? <Plus size={14} /> : <Search size={14} />}
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

function Upload({ onUploaded }: { onUploaded: () => void }) {
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
      await api.upload(pdfs);
      onUploaded();
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
        className={`mt-3 flex w-full flex-col items-center justify-center gap-1 rounded-md border border-dashed px-4 py-6 text-sm transition ${
          drag ? "border-accent bg-accent-soft" : "border-line text-ink-2 hover:border-ink-3"
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

function PaperRow({ paper, onChange }: { paper: Paper; onChange: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onChange();
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="px-4 py-3">
      <div className="flex items-start gap-3">
        <button onClick={() => setOpen(!open)} className="mt-0.5 text-ink-3 hover:text-ink" aria-label="Toggle details">
          <ChevronDown size={16} className={`transition ${open ? "" : "-rotate-90"}`} />
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium leading-snug">{paper.title}</span>
            <StatusBadge status={paper.status} />
          </div>
          <div className="mt-0.5 text-xs text-ink-3">
            {authorsShort(paper.authors)}
            {paper.year ? ` · ${paper.year}` : ""}
            {paper.status === "ready" ? ` · ${paper.num_pages} pages · ${paper.num_chunks} chunks` : ""}
          </div>
          {!!paper.categories.length && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {paper.categories.slice(0, 5).map((c) => (
                <Badge key={c}>{c}</Badge>
              ))}
            </div>
          )}
          {paper.error && <p className="mt-1 text-xs text-critical">{paper.error}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <a href={pdfUrl(paper.id)} target="_blank" rel="noreferrer" className="rounded-md p-1.5 text-ink-3 hover:bg-surface-2 hover:text-ink" title="Open PDF">
            <ExternalLink size={15} />
          </a>
          <Button variant="ghost" className="!p-1.5" title="Re-index" disabled={busy} onClick={() => act(() => api.reingest(paper.id))}>
            <RefreshCw size={15} />
          </Button>
          <Button
            variant="danger"
            className="!p-1.5"
            title="Delete"
            disabled={busy}
            onClick={() => confirm(`Delete "${paper.title}" and its index?`) && act(() => api.deletePaper(paper.id))}
          >
            <Trash2 size={15} />
          </Button>
        </div>
      </div>
      {open && (
        <div className="ml-7 mt-3 space-y-3 text-sm">
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
          <div className="text-xs text-ink-3">ID: {paper.id} · source: {paper.source}</div>
        </div>
      )}
    </li>
  );
}

export default function LibraryPage() {
  const [papers, setPapers] = useState<Paper[] | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const ready = papers?.filter((p) => p.status === "ready") ?? [];
  const chunks = ready.reduce((n, p) => n + p.num_chunks, 0);

  return (
    <div className="mx-auto w-full max-w-7xl space-y-5 px-4 py-6">
      <PageHeader
        title="Paper library"
        subtitle={papers ? `${ready.length} indexed papers · ${chunks.toLocaleString()} searchable chunks` : "Loading…"}
      />
      <ErrorNote message={error} />
      <div className="grid gap-4 md:grid-cols-2">
        <ArxivImport onImported={load} />
        <Upload onUploaded={load} />
      </div>
      <Card>
        {papers && papers.length === 0 ? (
          <div className="px-4 py-10 text-center text-sm text-ink-3">
            No papers yet. Import a few from arXiv to get started.
          </div>
        ) : (
          <ul className="divide-y divide-line">
            {papers?.map((p) => <PaperRow key={p.id} paper={p} onChange={load} />)}
          </ul>
        )}
      </Card>
    </div>
  );
}

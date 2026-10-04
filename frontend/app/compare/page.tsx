"use client";

import { useEffect, useState } from "react";
import { Check, Columns3, Copy, FileText, Plus, Quote, X } from "lucide-react";
import { api, pageLabel, type CompareColumn, type CompareResult, type Paper } from "@/lib/api";
import { toBibtex } from "@/lib/bibtex";
import { AnswerMarkdown } from "@/components/AnswerMarkdown";
import { usePdfViewer } from "@/components/PdfViewer";
import { useToast } from "@/components/Toast";
import { Button, Card, EmptyState, ErrorNote, PageHeader, Skeleton, inputSmClass } from "@/components/ui";

const DEFAULT_ASPECTS = ["Problem", "Method", "Data", "Results", "Limitations"];
const MIN_PAPERS = 2;
const MAX_PAPERS = 4;
const MAX_ASPECTS = 8;

/** Markdown table; citation markers become page references so the copy stands on its own. */
function toMarkdown(result: CompareResult) {
  const cell = (col: CompareColumn, aspect: string) =>
    (col.cells[aspect]?.text ?? "")
      .replace(/\[(\d+)\]/g, (m, n) => {
        const s = col.sources[Number(n) - 1];
        return s ? ` (${pageLabel(s.page_start, s.page_end)})` : m;
      })
      .replace(/\s+/g, " ")
      .replace(/\|/g, "\\|")
      .trim();
  const head = `| Aspect | ${result.papers.map((p) => `${p.title}${p.year ? ` (${p.year})` : ""}`).join(" | ")} |`;
  const rule = `|---|${result.papers.map(() => "---").join("|")}|`;
  const rows = result.aspects.map((a) => `| **${a}** | ${result.papers.map((p) => cell(p, a)).join(" | ")} |`);
  return [head, rule, ...rows].join("\n");
}

function ResultTable({ result }: { result: CompareResult }) {
  const openPdf = usePdfViewer();
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] table-fixed border-collapse text-sm">
        <colgroup>
          <col className="w-28" />
          {result.papers.map((p) => (
            <col key={p.paper_id} />
          ))}
        </colgroup>
        <thead>
          <tr className="border-b border-line align-top">
            <th className="p-3 text-left text-xs font-medium text-ink-3">Aspect</th>
            {result.papers.map((p) => (
              <th key={p.paper_id} className="p-3 text-left font-normal">
                <div className="font-semibold leading-snug">{p.title}</div>
                <div className="mt-1 flex items-center gap-2 text-xs text-ink-3">
                  {p.year ?? "n.d."}
                  <button
                    className="flex items-center gap-1 text-accent-ink hover:underline"
                    onClick={() => openPdf({ paperId: p.paper_id, title: p.title, page: 1 })}
                  >
                    <FileText size={12} /> PDF
                  </button>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.aspects.map((aspect) => (
            <tr key={aspect} className="border-b border-line align-top last:border-0">
              <th scope="row" className="p-3 text-left text-xs font-semibold uppercase tracking-wide text-ink-3">
                {aspect}
              </th>
              {result.papers.map((p) => {
                const text = p.cells[aspect]?.text ?? "";
                const missing = text.startsWith("Not covered");
                return (
                  <td key={p.paper_id} className={`p-3 ${missing ? "text-ink-3 italic" : "text-ink-2"}`}>
                    <AnswerMarkdown
                      text={text}
                      sources={p.sources}
                      onCite={(n) => {
                        const s = p.sources[n - 1];
                        if (s) openPdf({ paperId: s.paper_id, title: s.title, page: s.page_start });
                      }}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ComparePage() {
  const toast = useToast();
  const [papers, setPapers] = useState<Paper[] | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [aspects, setAspects] = useState<string[]>(DEFAULT_ASPECTS);
  const [custom, setCustom] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .papers()
      .then((ps) => setPapers(ps.filter((p) => p.status === "ready")))
      .catch((e) => setError(e.message));
  }, []);

  const toggle = (id: string) =>
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : c.length < MAX_PAPERS ? [...c, id] : c));

  const addAspect = () => {
    const a = custom.trim();
    if (a && !aspects.includes(a) && aspects.length < MAX_ASPECTS) setAspects([...aspects, a]);
    setCustom("");
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.compare({ paper_ids: chosen, aspects }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const copy = (text: string, done: string) =>
    navigator.clipboard
      .writeText(text)
      .then(() => toast(done, "good"))
      .catch(() => toast("Couldn't copy to the clipboard", "critical"));

  const ready = chosen.length >= MIN_PAPERS && aspects.length > 0;

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8">
      <PageHeader
        icon={Columns3}
        title="Compare papers"
        subtitle="Line papers up side by side. Every cell is drawn from that paper's own passages and cites them."
      />
      <ErrorNote message={error} />

      <Card className="space-y-4 p-4">
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold">Papers</h2>
            <span className="text-xs text-ink-3">
              {chosen.length}/{MAX_PAPERS} selected · pick at least {MIN_PAPERS}
            </span>
          </div>
          {papers === null ? (
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-12" />
              ))}
            </div>
          ) : papers.length < MIN_PAPERS ? (
            <p className="mt-2 text-sm text-ink-3">Index at least two papers in the Library to compare them.</p>
          ) : (
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {papers.map((p) => {
                const on = chosen.includes(p.id);
                const full = !on && chosen.length >= MAX_PAPERS;
                return (
                  <button
                    key={p.id}
                    type="button"
                    aria-pressed={on}
                    disabled={full}
                    onClick={() => toggle(p.id)}
                    className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-left text-sm transition disabled:opacity-50 ${
                      on ? "border-accent bg-accent-soft" : "border-line hover:border-ink-3"
                    }`}
                  >
                    <span
                      className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded border ${
                        on ? "border-accent bg-accent text-white" : "border-line"
                      }`}
                    >
                      {on && <Check size={11} />}
                    </span>
                    <span className="min-w-0">
                      <span className="line-clamp-2 leading-snug">{p.title}</span>
                      <span className="text-xs text-ink-3">{p.year ?? "n.d."}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div>
          <h2 className="text-sm font-semibold">Aspects</h2>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {aspects.map((a) => (
              <span key={a} className="flex items-center gap-1 rounded-full bg-accent-soft px-2.5 py-0.5 text-xs text-accent-ink">
                {a}
                <button aria-label={`Remove ${a}`} onClick={() => setAspects(aspects.filter((x) => x !== a))}>
                  <X size={12} />
                </button>
              </span>
            ))}
            {DEFAULT_ASPECTS.filter((a) => !aspects.includes(a)).map((a) => (
              <button
                key={a}
                onClick={() => setAspects([...aspects, a])}
                className="flex items-center gap-1 rounded-full border border-dashed border-line px-2.5 py-0.5 text-xs text-ink-3 hover:text-ink"
              >
                <Plus size={11} /> {a}
              </button>
            ))}
            {aspects.length < MAX_ASPECTS && (
              <input
                className={inputSmClass + " w-44 text-xs"}
                placeholder="Add your own, e.g. Compute"
                aria-label="Add a custom aspect"
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addAspect())}
              />
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={run} loading={loading} disabled={!ready}>
            {!loading && <Columns3 size={14} />} Compare {chosen.length || ""} papers
          </Button>
          <span className="text-xs text-ink-3">One Gemini call per paper; cells only use that paper&apos;s passages.</span>
        </div>
      </Card>

      {loading ? (
        <Card className="space-y-3 p-4" aria-label="Comparing">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-14" />
          ))}
        </Card>
      ) : result ? (
        <Card className="animate-in overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
            <h2 className="text-sm font-semibold">
              {result.papers.length} papers × {result.aspects.length} aspects
            </h2>
            <div className="flex gap-1">
              <Button variant="ghost" className="px-2! py-1! text-xs" onClick={() => copy(toMarkdown(result), "Table copied as Markdown")}>
                <Copy size={13} /> Copy as Markdown
              </Button>
              <Button
                variant="ghost"
                className="px-2! py-1! text-xs"
                onClick={() =>
                  copy(
                    toBibtex(result.papers.map((c) => papers?.find((p) => p.id === c.paper_id) ?? { id: c.paper_id, title: c.title, authors: [], year: c.year })),
                    "BibTeX copied",
                  )
                }
              >
                <Quote size={13} /> BibTeX
              </Button>
            </div>
          </div>
          <ResultTable result={result} />
        </Card>
      ) : (
        <Card>
          <EmptyState icon={Columns3} title="Pick papers to compare">
            Choose 2–4 papers and the aspects you care about. Click any citation number to open the PDF at that page.
          </EmptyState>
        </Card>
      )}
    </div>
  );
}

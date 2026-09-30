"use client";

import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from "recharts";
import { ChevronDown, FlaskConical, Loader2, Play, Sparkles } from "lucide-react";
import { api, type DatasetStatus, type EvalDetailRow, type EvalItem, type EvalRun, type Health } from "@/lib/api";
import { AnswerMarkdown } from "@/components/AnswerMarkdown";
import { useFacets } from "@/components/FilterPanel";
import { Badge, Button, Card, EmptyState, ErrorNote, PageHeader, Skeleton, StatTile, inputSmClass } from "@/components/ui";

/* Color follows the config, never its rank: each config owns a fixed series slot. */
const CONFIGS: Record<string, { label: string; hint: string; color: string }> = {
  dense: { label: "Dense", hint: "BGE embeddings only", color: "var(--series-1)" },
  sparse_bm25: { label: "BM25", hint: "Sparse keyword retrieval", color: "var(--series-2)" },
  hybrid_rrf: { label: "Hybrid (RRF)", hint: "Dense + BM25, reciprocal rank fusion", color: "var(--series-3)" },
  hybrid_rrf_rerank: { label: "Hybrid + rerank", hint: "RRF, then cross-encoder rerank", color: "var(--series-4)" },
};
const CONFIG_ORDER = Object.keys(CONFIGS);
const cfgLabel = (c: string) => CONFIGS[c]?.label ?? c;
const cfgColor = (c: string) => CONFIGS[c]?.color ?? "var(--text-3)";
const sortConfigs = (cs: string[]) =>
  [...cs].sort((a, b) => (CONFIG_ORDER.indexOf(a) + 1 || 99) - (CONFIG_ORDER.indexOf(b) + 1 || 99));

const RETRIEVAL_METRICS: [key: string, label: string][] = [
  ["mrr", "MRR"],
  ["hit@1", "Hit@1"],
  ["hit@3", "Hit@3"],
  ["hit@5", "Hit@5"],
  ["hit@10", "Hit@10"],
  ["recall@5", "Recall@5"],
  ["recall@10", "Recall@10"],
  ["ndcg@10", "nDCG@10"],
  ["precision@5", "P@5"],
];
const CHART_METRICS = ["mrr", "hit@1", "hit@5", "recall@10", "ndcg@10"];
const GEN_METRICS: [key: string, label: string, hint: string][] = [
  ["faithfulness", "Faithful", "Claims supported by the retrieved passages"],
  ["answer_relevance", "Answer rel.", "Answer addresses the question"],
  ["context_relevance", "Context rel.", "Retrieved passages are on-topic"],
  ["correctness", "Correct", "Agrees with the reference answer"],
  ["citation_precision", "Cite prec.", "Citations that actually support their sentence"],
  ["citation_coverage", "Cite cover.", "Sentences that carry a supporting citation"],
  ["gold_cited", "Gold cited", "Answer cites the passage the question was written from"],
];

const fmt = (v: number | undefined | null, digits = 3) => (v == null ? "—" : v.toFixed(digits));

/** SQLite drops the timezone; the API stores UTC. */
function parseUtc(s: string) {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z");
}

function runTime(s: string) {
  return parseUtc(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function RunStatus({ status }: { status: EvalRun["status"] }) {
  if (status === "running")
    return (
      <Badge tone="accent">
        <Loader2 size={10} className="animate-spin" /> Running
      </Badge>
    );
  return status === "done" ? <Badge tone="good">Done</Badge> : <Badge tone="critical">Failed</Badge>;
}

/* ---------- Dataset ---------- */

function DatasetCard({
  items,
  status,
  llmReady,
  onGenerate,
}: {
  items: EvalItem[] | null;
  status: DatasetStatus | null;
  llmReady: boolean;
  onGenerate: (n: number, append: boolean) => Promise<void>;
}) {
  const [n, setN] = useState(40);
  const [append, setAppend] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const running = status?.state === "running";
  const papers = new Set(items?.map((i) => i.paper_id)).size;
  const manual = items?.filter((i) => i.source === "manual").length ?? 0;

  const generate = async () => {
    setError(null);
    try {
      await onGenerate(n, append);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <Card className="p-4">
      <h2 className="text-sm font-semibold">Question set</h2>
      <p className="mt-0.5 text-xs text-ink-3">
        Gemini writes one question per sampled passage; that passage becomes the gold answer location.
      </p>
      <div className="mt-3 text-2xl font-semibold">{items ? items.length : "—"}</div>
      <div className="text-xs text-ink-3">
        {items?.length ? `questions across ${papers} paper${papers === 1 ? "" : "s"}` : "questions, none generated yet"}
        {manual > 0 && ` · ${manual} hand-written`}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 text-sm text-ink-2">
          Generate
          <input
            type="number"
            min={5}
            max={200}
            value={n}
            onChange={(e) => setN(Math.min(200, Math.max(5, Number(e.target.value) || 5)))}
            className={inputSmClass + " w-20"}
            aria-label="Number of questions"
          />
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-sm text-ink-2">
          <input type="checkbox" checked={append} onChange={(e) => setAppend(e.target.checked)} className="accent-[var(--accent)]" />
          Append
        </label>
        <Button
          variant="secondary"
          className="ml-auto"
          onClick={generate}
          loading={running}
          disabled={!llmReady}
          title={llmReady ? undefined : "Requires GEMINI_API_KEY"}
        >
          {!running && <Sparkles size={14} />}
          {items?.length && !append ? "Regenerate" : "Generate"}
        </Button>
      </div>
      {manual > 0 && !append && llmReady && (
        <p className="mt-2 text-xs text-ink-3">
          Regenerating replaces the {manual} hand-written questions. Tick Append to keep them.
        </p>
      )}
      {status && status.state !== "idle" && (
        <p className={`mt-2 text-xs ${status.state === "failed" ? "text-critical-ink" : "text-ink-3"}`}>
          {status.state === "done" ? `Last generation: ${status.message}` : status.message}
        </p>
      )}
      <div className="mt-2">
        <ErrorNote message={error} />
      </div>

      {!!items?.length && (
        <div className="mt-3 border-t border-line pt-3">
          <button onClick={() => setOpen(!open)} className="flex items-center gap-1 text-xs text-accent-ink hover:underline">
            <ChevronDown size={13} className={`transition ${open ? "" : "-rotate-90"}`} />
            {open ? "Hide questions" : "Browse questions"}
          </button>
          {open && (
            <ol className="mt-2 max-h-80 divide-y divide-line overflow-y-auto text-sm">
              {items.map((it) => (
                <li key={it.id} className="py-2">
                  <div className="leading-snug">{it.question}</div>
                  <div className="mt-0.5 text-xs text-ink-3">
                    {it.reference_answer}
                    <span className="block">
                      {it.paper_id}
                      {it.section ? ` · § ${it.section}` : ""}
                      {it.source === "manual" ? " · hand-written" : ""}
                    </span>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </Card>
  );
}

/* ---------- New run ---------- */

function NewRunCard({
  available,
  datasetSize,
  llmReady,
  onStart,
}: {
  available: string[];
  datasetSize: number;
  llmReady: boolean;
  onStart: (configs: string[], genConfigs: string[], maxGen: number | null) => Promise<void>;
}) {
  const [configs, setConfigs] = useState<string[] | null>(null);
  const [genConfigs, setGenConfigs] = useState<string[]>(["hybrid_rrf_rerank"]);
  const [maxGen, setMaxGen] = useState<string>("20");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chosen = configs ?? available;
  const gen = llmReady ? genConfigs.filter((c) => chosen.includes(c)) : [];
  const nGen = maxGen ? Math.min(Number(maxGen), datasetSize) : datasetSize;
  const toggle = (list: string[], c: string) => (list.includes(c) ? list.filter((x) => x !== c) : [...list, c]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      await onStart(chosen, gen, maxGen ? Number(maxGen) : null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="p-4">
      <h2 className="text-sm font-semibold">New run</h2>
      <p className="mt-0.5 text-xs text-ink-3">
        Retrieval metrics run locally for every config. Answer quality is judged by Gemini, so it&apos;s rate-limited.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead>
          <tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-ink-3">
            <th className="pb-1 font-semibold">Config</th>
            <th className="w-20 pb-1 text-center font-semibold">Retrieval</th>
            <th className="w-20 pb-1 text-center font-semibold">Answers</th>
          </tr>
        </thead>
        <tbody>
          {sortConfigs(available).map((c) => (
            <tr key={c} className="border-t border-line">
              <td className="py-1.5">
                <span className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: cfgColor(c) }} />
                  <span title={CONFIGS[c]?.hint}>{cfgLabel(c)}</span>
                </span>
              </td>
              <td className="text-center">
                <input
                  type="checkbox"
                  aria-label={`Evaluate retrieval for ${cfgLabel(c)}`}
                  checked={chosen.includes(c)}
                  onChange={() => setConfigs(toggle(chosen, c))}
                  className="accent-[var(--accent)]"
                />
              </td>
              <td className="text-center">
                <input
                  type="checkbox"
                  aria-label={`Judge answers for ${cfgLabel(c)}`}
                  checked={gen.includes(c)}
                  disabled={!llmReady || !chosen.includes(c)}
                  onChange={() => setGenConfigs(toggle(genConfigs, c))}
                  className="accent-[var(--accent)]"
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 whitespace-nowrap text-sm text-ink-2">
          Judge first
          <input
            type="number"
            min={1}
            value={maxGen}
            placeholder="all"
            onChange={(e) => setMaxGen(e.target.value.replace(/\D/g, ""))}
            className={inputSmClass + " w-20"}
            aria-label="Questions to judge"
            disabled={!gen.length}
          />
          questions
        </label>
        <Button className="ml-auto" onClick={start} loading={busy} disabled={!datasetSize || !chosen.length}>
          <Play size={13} /> Start run
        </Button>
      </div>
      <p className="mt-2 text-xs text-ink-3">
        {!datasetSize
          ? "Generate a question set first."
          : gen.length
            ? `≈ ${nGen * gen.length * 2} Gemini calls (answer + judge per question and config).`
            : llmReady
              ? "Retrieval only: no Gemini calls."
              : "No Gemini key: retrieval metrics only."}
      </p>
      <div className="mt-2">
        <ErrorNote message={error} />
      </div>
    </Card>
  );
}

/* ---------- Results ---------- */

function Legend({ configs }: { configs: string[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {configs.map((c) => (
        <span key={c} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: cfgColor(c) }} />
          {cfgLabel(c)}
        </span>
      ))}
    </div>
  );
}

function ChartTooltip({ active, payload, label }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="min-w-40 rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-ink">{label}</div>
      {payload.map((p) => (
        <div key={String(p.dataKey)} className="flex items-center gap-2 py-0.5">
          <span className="h-0.5 w-3 rounded-full" style={{ background: cfgColor(String(p.dataKey)) }} />
          <span className="font-semibold tabular-nums text-ink">{fmt(Number(p.value))}</span>
          <span className="text-ink-2">{cfgLabel(String(p.dataKey))}</span>
        </div>
      ))}
    </div>
  );
}

function RetrievalChart({ run, configs }: { run: EvalRun; configs: string[] }) {
  const label = Object.fromEntries(RETRIEVAL_METRICS);
  const data = CHART_METRICS.map((m) => ({
    metric: label[m],
    ...Object.fromEntries(configs.map((c) => [c, run.results[c]?.[m] ?? 0])),
  }));
  return (
    // On narrow screens the chart keeps a minimum width and scrolls, so no metric label is dropped
    <div className="-mx-1 overflow-x-auto px-1">
    <div className="h-64 min-w-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={2} barCategoryGap="18%" margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="metric"
            interval={0}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            tick={{ fill: "var(--text-2)", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => v.toFixed(2)}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "var(--text-3)", fontSize: 11 }}
          />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={ChartTooltip} />
          {configs.map((c) => (
            <Bar key={c} dataKey={c} name={cfgLabel(c)} fill={cfgColor(c)} radius={[4, 4, 0, 0]} maxBarSize={24} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
    </div>
  );
}

function MetricTable({
  configs,
  results,
  metrics,
  extra,
}: {
  configs: string[];
  results: EvalRun["results"];
  metrics: [key: string, label: string, hint?: string][];
  extra?: { label: string; value: (c: string) => string; best?: (c: string) => boolean };
}) {
  const best = Object.fromEntries(
    metrics.map(([k]) => [k, Math.max(...configs.map((c) => results[c]?.[k] ?? -Infinity))]),
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead>
          <tr className="border-b border-line text-right text-xs text-ink-3">
            <th className="py-2 pr-3 text-left font-medium">Config</th>
            {metrics.map(([k, label, hint]) => (
              <th key={k} className="whitespace-nowrap px-2 py-2 font-medium" title={hint}>
                {label}
              </th>
            ))}
            {extra && <th className="whitespace-nowrap py-2 pl-2 font-medium">{extra.label}</th>}
          </tr>
        </thead>
        <tbody>
          {configs.map((c) => (
            <tr key={c} className="border-b border-line last:border-0">
              <td className="whitespace-nowrap py-2 pr-3">
                <span className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: cfgColor(c) }} />
                  {cfgLabel(c)}
                </span>
              </td>
              {metrics.map(([k]) => {
                const v = results[c]?.[k];
                const isBest = v != null && configs.length > 1 && v === best[k];
                return (
                  <td key={k} className={`px-2 py-2 text-right ${isBest ? "font-semibold text-ink" : "text-ink-2"}`}>
                    {fmt(v)}
                  </td>
                );
              })}
              {extra && (
                <td className={`py-2 pl-2 text-right ${extra.best?.(c) ? "font-semibold text-ink" : "text-ink-2"}`}>
                  {extra.value(c)}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Summary({ run, configs, genConfigs }: { run: EvalRun; configs: string[]; genConfigs: string[] }) {
  const r = run.results;
  const bestCfg = configs.reduce((a, b) => ((r[b]?.mrr ?? 0) > (r[a]?.mrr ?? 0) ? b : a), configs[0]);
  const lift =
    r.hybrid_rrf_rerank?.mrr != null && r.hybrid_rrf?.mrr != null ? r.hybrid_rrf_rerank.mrr - r.hybrid_rrf.mrr : null;
  const g = genConfigs[0];
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatTile label="Questions" value={String(run.dataset_size)} sub={`${configs.length} retrieval configs`} />
      <StatTile label="Best MRR" value={fmt(r[bestCfg]?.mrr)} sub={cfgLabel(bestCfg)} />
      {lift != null ? (
        <StatTile
          label="Reranker lift on MRR"
          value={`${lift >= 0 ? "+" : "−"}${Math.abs(lift).toFixed(3)}`}
          sub={
            <span className={lift >= 0 ? "text-good-ink" : "text-critical-ink"}>
              {lift >= 0 ? "▲ better" : "▼ worse"} than Hybrid (RRF)
            </span>
          }
        />
      ) : (
        <StatTile label="Hit@5" value={fmt(r[bestCfg]?.["hit@5"])} sub={cfgLabel(bestCfg)} />
      )}
      {g ? (
        <StatTile
          label="Faithfulness"
          value={fmt(r[g]?.faithfulness, 2)}
          sub={`${cfgLabel(g)} · ${r[g]?.n_generation ?? 0} judged`}
        />
      ) : (
        <StatTile label="Mean latency" value={`${Math.round(r[bestCfg]?.latency_ms ?? 0)} ms`} sub={cfgLabel(bestCfg)} />
      )}
    </div>
  );
}

function RankCell({ rank }: { rank: number | null | undefined }) {
  if (rank === undefined) return <span className="text-ink-3">—</span>;
  if (rank === null) return <Badge tone="critical">miss</Badge>;
  return <Badge tone={rank === 1 ? "good" : rank <= 5 ? "neutral" : "warning"}>#{rank}</Badge>;
}

function QuestionRow({
  row,
  configs,
  paperTitle,
}: {
  row: EvalDetailRow;
  configs: string[];
  paperTitle: string;
}) {
  const [open, setOpen] = useState(false);
  const gens = configs.filter((c) => row.configs[c]?.generation || row.configs[c]?.generation_error);
  return (
    <>
      <tr className="cursor-pointer border-b border-line align-top hover:bg-surface-2/60" onClick={() => setOpen(!open)}>
        <td className="py-2 pr-3">
          <div className="flex items-start gap-1.5">
            <ChevronDown size={14} className={`mt-0.5 shrink-0 text-ink-3 transition ${open ? "" : "-rotate-90"}`} />
            <div className="min-w-0">
              <div className="leading-snug">{row.question}</div>
              <div className="mt-0.5 truncate text-xs text-ink-3">{paperTitle}</div>
              {/* Phones: ranks sit under the question instead of in columns */}
              <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 sm:hidden">
                {configs.map((c) => (
                  <span key={c} className="flex items-center gap-1 text-xs text-ink-3">
                    <span className="h-2 w-2 rounded-sm" style={{ background: cfgColor(c) }} />
                    {cfgLabel(c)} <RankCell rank={row.configs[c]?.gold_rank} />
                  </span>
                ))}
              </div>
            </div>
          </div>
        </td>
        {configs.map((c) => (
          <td key={c} className="hidden px-2 py-2 text-center sm:table-cell">
            <RankCell rank={row.configs[c]?.gold_rank} />
          </td>
        ))}
      </tr>
      {open && (
        <tr className="border-b border-line bg-surface-2/40">
          <td colSpan={configs.length + 1} className="px-6 py-3">
            {gens.length === 0 ? (
              <p className="text-xs text-ink-3">No generated answer for this question in this run (retrieval only).</p>
            ) : (
              <div className="space-y-4">
                {gens.map((c) => {
                  const e = row.configs[c];
                  return (
                    <div key={c}>
                      <div className="mb-1 flex items-center gap-2 text-xs font-medium">
                        <span className="h-2.5 w-2.5 rounded-sm" style={{ background: cfgColor(c) }} />
                        {cfgLabel(c)}
                      </div>
                      {e.generation_error && <ErrorNote message={e.generation_error} />}
                      {e.generation && (
                        <>
                          <div className="text-ink-2">
                            <AnswerMarkdown text={e.generation.answer} sources={[]} onCite={() => {}} />
                          </div>
                          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
                            {GEN_METRICS.map(([k, label, hint]) => (
                              <span key={k} title={hint}>
                                {label}{" "}
                                <span className="font-semibold tabular-nums text-ink">
                                  {fmt(e.generation![k as keyof typeof e.generation] as number | null, 2)}
                                </span>
                              </span>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function QuestionTable({ run, configs }: { run: EvalRun; configs: string[] }) {
  const facets = useFacets();
  const [missesOnly, setMissesOnly] = useState(false);
  const titles = Object.fromEntries(facets?.papers.map((p) => [p.id, p.title]) ?? []);
  const rows = (run.details ?? []).filter(
    (r) => !missesOnly || configs.some((c) => r.configs[c] && r.configs[c].gold_rank == null),
  );
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold">Per question</h3>
          <p className="text-xs text-ink-3">Rank of the gold passage in the top 10. Click a row for the judged answer.</p>
        </div>
        <label className="flex cursor-pointer items-center gap-1.5 text-sm text-ink-2">
          <input
            type="checkbox"
            checked={missesOnly}
            onChange={(e) => setMissesOnly(e.target.checked)}
            className="accent-[var(--accent)]"
          />
          Only questions with a miss
        </label>
      </div>
      <div className="max-h-[70dvh] overflow-auto px-4">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-surface">
            <tr className="border-b border-line text-xs text-ink-3">
              <th className="py-2 pr-3 text-left font-medium">Question</th>
              {configs.map((c) => (
                <th key={c} className="hidden w-24 whitespace-nowrap px-2 py-2 text-center font-medium sm:table-cell">
                  {cfgLabel(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <QuestionRow key={r.id} row={r} configs={configs} paperTitle={titles[r.paper_id] ?? r.paper_id} />
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={configs.length + 1} className="py-6 text-center text-sm text-ink-3">
                  {missesOnly ? "Every config found the gold passage for every question." : "No per-question data."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function RunResults({ run }: { run: EvalRun }) {
  const configs = sortConfigs(Object.keys(run.results));
  const genConfigs = configs.filter((c) => run.results[c]?.n_generation);
  const [view, setView] = useState<"chart" | "table">("chart");

  if (run.status === "running")
    return (
      <Card className="flex items-center gap-3 p-6 text-sm text-ink-2">
        <Loader2 size={16} className="animate-spin text-accent" />
        Run #{run.id}: {run.progress || "starting…"}
      </Card>
    );
  if (run.status === "failed") return <ErrorNote message={`Run #${run.id} failed: ${run.error ?? "unknown error"}`} />;
  if (!configs.length) return <Card className="p-6 text-sm text-ink-3">This run produced no results.</Card>;

  return (
    <div className="space-y-4">
      <Summary run={run} configs={configs} genConfigs={genConfigs} />

      <Card className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold">Retrieval quality</h3>
            <p className="text-xs text-ink-3">Does the passage each question was written from come back, and how high?</p>
          </div>
          <div className="grid grid-cols-2 rounded-md border border-line p-0.5 text-xs" role="tablist">
            {(["chart", "table"] as const).map((v) => (
              <button
                key={v}
                role="tab"
                aria-selected={view === v}
                onClick={() => setView(v)}
                className={`rounded px-2.5 py-1 capitalize ${view === v ? "bg-accent text-white" : "text-ink-2 hover:text-ink"}`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-3">
          {view === "chart" ? (
            <div className="space-y-2">
              <Legend configs={configs} />
              <RetrievalChart run={run} configs={configs} />
            </div>
          ) : (
            <MetricTable
              configs={configs}
              results={run.results}
              metrics={RETRIEVAL_METRICS}
              extra={{
                label: "Latency",
                value: (c) => `${Math.round(run.results[c]?.latency_ms ?? 0)} ms`,
              }}
            />
          )}
        </div>
      </Card>

      {!!genConfigs.length && (
        <Card className="p-4">
          <h3 className="text-sm font-semibold">Answer quality</h3>
          <p className="mb-3 text-xs text-ink-3">
            LLM-as-judge scores (1–5 rescaled to 0–1) and citation checks on the first{" "}
            {Math.max(...genConfigs.map((c) => run.results[c]?.n_generation ?? 0))} questions. Hover a column for its
            definition.
          </p>
          <MetricTable configs={genConfigs} results={run.results} metrics={GEN_METRICS} />
        </Card>
      )}

      {!!run.details?.length && <QuestionTable run={run} configs={configs} />}
    </div>
  );
}

/* ---------- Page ---------- */

export default function EvalPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [available, setAvailable] = useState<string[]>(CONFIG_ORDER);
  const [dataset, setDataset] = useState<EvalItem[] | null>(null);
  const [dsStatus, setDsStatus] = useState<DatasetStatus | null>(null);
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selected, setSelected] = useState<EvalRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const llmReady = !!health?.llm_configured;

  const loadRuns = useCallback(
    () =>
      api
        .runs()
        .then(setRuns)
        .catch((e) => setError(e.message)),
    [],
  );
  const loadDataset = useCallback(() => api.dataset().then(setDataset).catch((e) => setError(e.message)), []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .evalConfigs()
      .then((c) => setAvailable(Object.keys(c)))
      .catch(() => {});
    api.datasetStatus().then(setDsStatus).catch(() => {});
    loadDataset();
    loadRuns();
  }, [loadDataset, loadRuns]);

  // Default to the newest run once the list arrives
  const shownId = selectedId ?? runs?.[0]?.id ?? null;
  const shownStatus = runs?.find((r) => r.id === shownId)?.status;

  useEffect(() => {
    if (shownId == null) return;
    let stale = false;
    api
      .run(shownId)
      .then((r) => !stale && setSelected(r))
      .catch((e) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [shownId, shownStatus]);

  // Poll while a run is in flight
  const anyRunning = runs?.some((r) => r.status === "running");
  useEffect(() => {
    if (!anyRunning) return;
    const t = setInterval(loadRuns, 3000);
    return () => clearInterval(t);
  }, [anyRunning, loadRuns]);

  // Poll dataset generation, then reload the questions when it finishes
  const generating = dsStatus?.state === "running";
  useEffect(() => {
    if (!generating) return;
    const t = setInterval(
      () =>
        api
          .datasetStatus()
          .then((s) => {
            setDsStatus(s);
            if (s.state !== "running") loadDataset();
          })
          .catch(() => {}),
      2000,
    );
    return () => clearInterval(t);
  }, [generating, loadDataset]);

  const generate = async (n: number, append: boolean) => {
    // The 202 response reflects the state before the background job picks up
    await api.generateDataset(n, append);
    setDsStatus({ state: "running", message: "Starting…" });
  };

  const start = async (configs: string[], genConfigs: string[], maxGen: number | null) => {
    const run = await api.startRun({
      configs: configs.length === available.length ? null : configs,
      gen_configs: genConfigs,
      max_gen_questions: maxGen,
    });
    setSelectedId(run.id);
    await loadRuns();
  };

  const shown = selected && selected.id === shownId ? { ...selected, ...runs?.find((r) => r.id === shownId), details: selected.details } : null;

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8">
      <PageHeader
        icon={FlaskConical}
        title="Evaluation"
        subtitle="Compare retrieval strategies and grade cited answers on questions generated from your own papers."
      />
      <ErrorNote message={error} />
      {health && !llmReady && (
        <div className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-ink-2">
          No Gemini key configured: you can run retrieval metrics on an existing question set, but generating questions
          and judging answers need <code className="font-mono text-xs">GEMINI_API_KEY</code>.
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <DatasetCard items={dataset} status={dsStatus} llmReady={llmReady} onGenerate={generate} />
        <NewRunCard available={available} datasetSize={dataset?.length ?? 0} llmReady={llmReady} onStart={start} />
      </div>

      <div className="grid gap-5 lg:grid-cols-[220px_1fr]">
        <aside className="space-y-2 lg:sticky lg:top-20 lg:self-start">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">Runs</div>
          {runs?.length === 0 && <p className="text-xs text-ink-3">No runs yet.</p>}
          <ul className="flex gap-1.5 overflow-x-auto lg:max-h-[70dvh] lg:flex-col lg:overflow-y-auto">
            {runs?.map((r) => (
              <li key={r.id} className="shrink-0">
                <button
                  onClick={() => setSelectedId(r.id)}
                  className={`w-full rounded-lg border px-3 py-2 text-left shadow-card transition ${
                    r.id === shownId ? "border-accent bg-accent-soft" : "border-line bg-surface hover:border-ink-3"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium">Run #{r.id}</span>
                    <RunStatus status={r.status} />
                  </div>
                  <div className="mt-0.5 whitespace-nowrap text-xs text-ink-3">
                    {runTime(r.created_at)} · {r.dataset_size} q
                  </div>
                  {r.status === "running" && <div className="mt-0.5 text-xs text-ink-2">{r.progress}</div>}
                </button>
              </li>
            ))}
          </ul>
        </aside>
        <div className="min-w-0">
          {shown ? (
            <RunResults key={shown.id} run={shown} />
          ) : runs === null || runs.length ? (
            <div className="space-y-4" aria-label="Loading run">
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-21.5 rounded-xl" />
                ))}
              </div>
              <Skeleton className="h-80 rounded-xl" />
            </div>
          ) : (
            <Card>
              <EmptyState icon={FlaskConical} title="No evaluation runs yet">
                Generate a question set, then start a run to compare Dense, BM25, Hybrid and Hybrid + rerank.
              </EmptyState>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

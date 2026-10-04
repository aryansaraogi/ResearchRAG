"use client";

import { useEffect, useState } from "react";
import { FileText, Network, Table2 } from "lucide-react";
import { api, type CitationGraph as Graph } from "@/lib/api";
import { usePdfViewer } from "@/components/PdfViewer";
import { Button, EmptyState, ErrorNote, Skeleton } from "@/components/ui";

const NODE_W = 184;
const NODE_H = 66;
const COL_GAP = 64;
const ROW_GAP = 22;
const LABELS = 30; // room for the year labels
const ARC_BASE = 22; // how far a same-row arc rises above the boxes it passes over...
const ARC_STEP = 14; // ...plus this much per column skipped, so nested arcs don't overlap
const PAD = 16;
const COL = NODE_W + COL_GAP;

type Node = Graph["nodes"][number];

function layout(graph: Graph) {
  // Oldest year on the left, so "cites" arrows point leftwards (back in time)
  const years = [...new Set(graph.nodes.map((n) => n.year ?? Infinity))].sort((a, b) => a - b);
  const columns = years.map((y) => graph.nodes.filter((n) => (n.year ?? Infinity) === y));
  const cell = new Map<string, { col: number; row: number }>();
  columns.forEach((col, ci) => col.forEach((n, ri) => cell.set(n.id, { col: ci, row: ri })));
  // Top-row links that skip columns arc above the boxes, so reserve a band for the tallest arc
  const tallest = Math.max(
    0,
    ...graph.edges
      .map((e) => [cell.get(e.source)!, cell.get(e.target)!])
      .filter(([a, b]) => a.row === 0 && b.row === 0 && Math.abs(a.col - b.col) > 1)
      .map(([a, b]) => ARC_BASE + ARC_STEP * (Math.abs(a.col - b.col) - 1)),
  );
  const top = LABELS + tallest;
  const pos = new Map<string, { x: number; y: number }>();
  cell.forEach(({ col, row }, id) => pos.set(id, { x: PAD + col * COL, y: top + row * (NODE_H + ROW_GAP) }));
  const width = PAD * 2 + years.length * NODE_W + (years.length - 1) * COL_GAP + 40; // + room for same-year loops
  const height = top + Math.max(...columns.map((c) => c.length)) * (NODE_H + ROW_GAP) + PAD;
  return { years, pos, width, height };
}

function edgePath(a: { x: number; y: number }, b: { x: number; y: number }) {
  const ay = a.y + NODE_H / 2;
  const by = b.y + NODE_H / 2;
  if (a.x === b.x) {
    // Same year: loop out to the right of the column
    const x = a.x + NODE_W;
    return `M ${x} ${ay} C ${x + 44} ${ay}, ${x + 44} ${by}, ${x + 6} ${by}`;
  }
  const skipped = Math.round(Math.abs(a.x - b.x) / COL) - 1;
  if (skipped > 0 && a.y === b.y) {
    // Same row with columns in between: arc over those boxes instead of running behind them
    const arc = a.y - (ARC_BASE + ARC_STEP * skipped) * 1.33; // a cubic peaks at ~3/4 of its control height
    const ax = a.x + NODE_W / 2;
    const bx = b.x + NODE_W / 2;
    return `M ${ax} ${a.y} C ${ax} ${arc}, ${bx} ${arc}, ${bx} ${b.y - 6}`;
  }
  // Citing (newer, right) -> cited (older, left)
  const x1 = a.x;
  const x2 = b.x + NODE_W + 6;
  const mid = (x1 + x2) / 2;
  return `M ${x1} ${ay} C ${mid} ${ay}, ${mid} ${by}, ${x2} ${by}`;
}

function PaperList({ items, empty, onSelect }: { items: Node[]; empty: string; onSelect: (id: string) => void }) {
  if (!items.length) return <p className="text-ink-3">{empty}</p>;
  return (
    <ul className="space-y-1">
      {items.map((n) => (
        <li key={n.id}>
          <button className="text-left text-accent-ink hover:underline" onClick={() => onSelect(n.id)}>
            {n.title}
          </button>
          <span className="text-ink-3"> · {n.year ?? "n.d."}</span>
        </li>
      ))}
    </ul>
  );
}

export function CitationGraphView() {
  const openPdf = usePdfViewer();
  const [graph, setGraph] = useState<Graph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [asTable, setAsTable] = useState(false);

  useEffect(() => {
    api.graph().then(setGraph).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="p-4"><ErrorNote message={error} /></div>;
  if (!graph) return <div className="space-y-3 p-4" aria-label="Loading citation graph"><Skeleton className="h-64" /></div>;
  if (graph.nodes.length < 2)
    return (
      <EmptyState icon={Network} title="Not enough papers yet">
        The graph links papers in your library that cite each other. Index at least two papers to see it.
      </EmptyState>
    );

  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const cites = (id: string) => graph.edges.filter((e) => e.source === id).map((e) => byId.get(e.target)!);
  const citedBy = (id: string) => graph.edges.filter((e) => e.target === id).map((e) => byId.get(e.source)!);
  const focus = hover ?? selected;
  const linked = (e: Graph["edges"][number]) => focus !== null && (e.source === focus || e.target === focus);
  const { years, pos, width, height } = layout(graph);
  const sel = selected ? byId.get(selected) : undefined;

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-ink-3">
          {graph.edges.length} citation{graph.edges.length === 1 ? "" : "s"} between {graph.nodes.length} papers, matched
          from each paper&apos;s reference list. Arrows point from the citing paper to the cited one.
        </p>
        <Button variant="secondary" className="px-2! py-1! text-xs" onClick={() => setAsTable(!asTable)}>
          {asTable ? <Network size={13} /> : <Table2 size={13} />} {asTable ? "Graph" : "Table"}
        </Button>
      </div>

      {asTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-3">
                <th className="py-2 pr-3 font-medium">Paper</th>
                <th className="px-2 py-2 font-medium">Cites (in library)</th>
                <th className="px-2 py-2 font-medium">Cited by (in library)</th>
                <th className="py-2 pl-2 text-right font-medium">References</th>
              </tr>
            </thead>
            <tbody>
              {graph.nodes.map((n) => (
                <tr key={n.id} className="border-b border-line align-top last:border-0">
                  <td className="py-2 pr-3">
                    {n.title} <span className="text-xs text-ink-3">{n.year ?? ""}</span>
                  </td>
                  <td className="px-2 py-2 text-xs text-ink-2">{cites(n.id).map((c) => c.title).join("; ") || "—"}</td>
                  <td className="px-2 py-2 text-xs text-ink-2">{citedBy(n.id).map((c) => c.title).join("; ") || "—"}</td>
                  <td className="py-2 pl-2 text-right tabular-nums text-ink-2">{n.n_references || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-line bg-bg">
          <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`Citation graph of ${graph.nodes.length} papers`}
            className="block"
          >
            <defs>
              {(["muted", "focus"] as const).map((k) => (
                <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={k === "focus" ? "var(--accent)" : "var(--text-3)"} />
                </marker>
              ))}
            </defs>
            {years.map((y, i) => (
              <text key={y} x={PAD + i * (NODE_W + COL_GAP) + NODE_W / 2} y={18} textAnchor="middle" fontSize={11} fill="var(--text-3)">
                {Number.isFinite(y) ? y : "n.d."}
              </text>
            ))}
            {graph.edges.map((e) => {
              const on = linked(e);
              return (
                <path
                  key={`${e.source}>${e.target}`}
                  d={edgePath(pos.get(e.source)!, pos.get(e.target)!)}
                  fill="none"
                  stroke={on ? "var(--accent)" : "var(--text-3)"}
                  strokeWidth={on ? 2 : 1.25}
                  strokeOpacity={focus && !on ? 0.2 : on ? 1 : 0.55}
                  markerEnd={`url(#arrow-${on ? "focus" : "muted"})`}
                />
              );
            })}
            {graph.nodes.map((n) => {
              const p = pos.get(n.id)!;
              const isFocus = focus === n.id;
              const dim = focus !== null && !isFocus && !graph.edges.some((e) => linked(e) && (e.source === n.id || e.target === n.id));
              const out = cites(n.id).length;
              const inc = citedBy(n.id).length;
              return (
                <g
                  key={n.id}
                  tabIndex={0}
                  role="button"
                  aria-label={`${n.title}, ${n.year ?? "no year"}: cites ${out}, cited by ${inc} in your library`}
                  aria-pressed={selected === n.id}
                  className="cursor-pointer outline-none"
                  opacity={dim ? 0.35 : 1}
                  onMouseEnter={() => setHover(n.id)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(n.id)}
                  onBlur={() => setHover(null)}
                  onClick={() => setSelected(selected === n.id ? null : n.id)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setSelected(selected === n.id ? null : n.id))}
                >
                  <rect
                    x={p.x}
                    y={p.y}
                    width={NODE_W}
                    height={NODE_H}
                    rx={10}
                    fill="var(--surface)"
                    stroke={isFocus || selected === n.id ? "var(--accent)" : "var(--border)"}
                    strokeWidth={isFocus || selected === n.id ? 2 : 1}
                  />
                  <foreignObject x={p.x + 10} y={p.y + 7} width={NODE_W - 20} height={NODE_H - 12}>
                    <div className="text-[11.5px] leading-snug">
                      <div className="line-clamp-2 font-medium text-ink">{n.title}</div>
                      <div className="mt-0.5 text-[10.5px] text-ink-3">
                        cites {out} · cited by {inc}
                      </div>
                    </div>
                  </foreignObject>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {sel && !asTable && (
        <div className="animate-in grid gap-4 rounded-lg border border-line bg-surface p-4 text-sm sm:grid-cols-[1fr_1fr_auto]">
          <div>
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">{sel.title} cites</div>
            <PaperList items={cites(sel.id)} empty="No other paper in your library." onSelect={setSelected} />
          </div>
          <div>
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">Cited by</div>
            <PaperList items={citedBy(sel.id)} empty="No other paper in your library." onSelect={setSelected} />
          </div>
          <div className="flex items-start gap-2 sm:flex-col sm:items-end">
            <span className="text-xs text-ink-3">{sel.n_references} references in total</span>
            <Button variant="secondary" className="px-2! py-1! text-xs" onClick={() => openPdf({ paperId: sel.id, title: sel.title, page: 1 })}>
              <FileText size={13} /> View PDF
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api, type Facets, type RetrievalMode, type SearchFilters } from "@/lib/api";
import { inputClass } from "@/components/ui";

export function useFacets() {
  const [facets, setFacets] = useState<Facets | null>(null);
  useEffect(() => {
    api.facets().then(setFacets).catch(() => setFacets(null));
  }, []);
  return facets;
}

export function activeFilterCount(f: SearchFilters) {
  return (
    (f.year_min ? 1 : 0) +
    (f.year_max ? 1 : 0) +
    (f.paper_ids?.length ?? 0) +
    (f.section_types?.length ?? 0) +
    (f.categories?.length ?? 0) +
    (f.authors?.length ?? 0)
  );
}

function toggle(list: string[] | null | undefined, v: string) {
  const cur = list ?? [];
  const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
  return next.length ? next : null;
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-full border px-2.5 py-0.5 text-xs transition ${
        active ? "border-accent bg-accent-soft text-accent-ink" : "border-line text-ink-2 hover:border-ink-3"
      }`}
    >
      {children}
    </button>
  );
}

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">{label}</div>
      {children}
    </div>
  );
}

export function RetrievalSettings({
  mode,
  rerank,
  onMode,
  onRerank,
}: {
  mode: RetrievalMode;
  rerank: boolean;
  onMode: (m: RetrievalMode) => void;
  onRerank: (r: boolean) => void;
}) {
  const modes: { value: RetrievalMode; label: string; hint: string }[] = [
    { value: "hybrid", label: "Hybrid", hint: "Dense + BM25 fused with reciprocal rank fusion" },
    { value: "dense", label: "Dense", hint: "BGE embeddings only" },
    { value: "sparse", label: "BM25", hint: "Keyword (sparse) only" },
  ];
  return (
    <Group label="Retrieval">
      <div className="grid grid-cols-3 rounded-md border border-line p-0.5 text-xs">
        {modes.map((m) => (
          <button
            key={m.value}
            type="button"
            title={m.hint}
            onClick={() => onMode(m.value)}
            className={`rounded px-2 py-1 ${mode === m.value ? "bg-accent text-white" : "text-ink-2 hover:text-ink"}`}
          >
            {m.label}
          </button>
        ))}
      </div>
      <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-2">
        <input type="checkbox" checked={rerank} onChange={(e) => onRerank(e.target.checked)} className="accent-[var(--accent)]" />
        Cross-encoder rerank
      </label>
    </Group>
  );
}

export function FilterPanel({
  filters,
  onChange,
  facets,
}: {
  filters: SearchFilters;
  onChange: (f: SearchFilters) => void;
  facets: Facets | null;
}) {
  const [author, setAuthor] = useState("");
  const set = (patch: Partial<SearchFilters>) => onChange({ ...filters, ...patch });
  const count = activeFilterCount(filters);

  const addAuthor = () => {
    const a = author.trim();
    if (a && !(filters.authors ?? []).includes(a)) set({ authors: [...(filters.authors ?? []), a] });
    setAuthor("");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Filters</span>
        {count > 0 && (
          <button className="text-xs text-accent hover:underline" onClick={() => onChange({})}>
            Clear ({count})
          </button>
        )}
      </div>

      <Group label="Papers">
        {facets?.papers.length ? (
          <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
            {facets.papers.map((p) => (
              <label key={p.id} className="flex cursor-pointer items-start gap-2 text-xs leading-snug text-ink-2">
                <input
                  type="checkbox"
                  className="mt-0.5 accent-[var(--accent)]"
                  checked={filters.paper_ids?.includes(p.id) ?? false}
                  onChange={() => set({ paper_ids: toggle(filters.paper_ids, p.id) })}
                />
                <span className="line-clamp-2">{p.title}</span>
              </label>
            ))}
          </div>
        ) : (
          <p className="text-xs text-ink-3">No indexed papers yet.</p>
        )}
      </Group>

      <Group label="Year">
        <div className="flex items-center gap-2">
          <input
            type="number"
            className={inputClass + " py-1"}
            placeholder={facets?.year_min?.toString() ?? "from"}
            value={filters.year_min ?? ""}
            onChange={(e) => set({ year_min: e.target.value ? Number(e.target.value) : null })}
            aria-label="Year from"
          />
          <span className="text-ink-3">–</span>
          <input
            type="number"
            className={inputClass + " py-1"}
            placeholder={facets?.year_max?.toString() ?? "to"}
            value={filters.year_max ?? ""}
            onChange={(e) => set({ year_max: e.target.value ? Number(e.target.value) : null })}
            aria-label="Year to"
          />
        </div>
      </Group>

      <Group label="Section">
        <div className="flex flex-wrap gap-1.5">
          {(facets?.section_types ?? []).map((s) => (
            <Chip
              key={s}
              active={filters.section_types?.includes(s) ?? false}
              onClick={() => set({ section_types: toggle(filters.section_types, s) })}
            >
              {s.replace("_", " ")}
            </Chip>
          ))}
        </div>
      </Group>

      {!!facets?.categories.length && (
        <Group label="arXiv category">
          <div className="flex flex-wrap gap-1.5">
            {facets.categories.map((c) => (
              <Chip
                key={c}
                active={filters.categories?.includes(c) ?? false}
                onClick={() => set({ categories: toggle(filters.categories, c) })}
              >
                {c}
              </Chip>
            ))}
          </div>
        </Group>
      )}

      <Group label="Author">
        <div className="flex gap-1.5">
          <input
            list="author-options"
            className={inputClass + " py-1"}
            placeholder="Add author…"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addAuthor())}
          />
          <datalist id="author-options">
            {facets?.authors.map((a) => <option key={a} value={a} />)}
          </datalist>
        </div>
        {!!filters.authors?.length && (
          <div className="flex flex-wrap gap-1.5">
            {filters.authors.map((a) => (
              <span key={a} className="flex items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-xs text-accent-ink">
                {a}
                <button aria-label={`Remove ${a}`} onClick={() => set({ authors: toggle(filters.authors, a) })}>
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}
      </Group>
    </div>
  );
}

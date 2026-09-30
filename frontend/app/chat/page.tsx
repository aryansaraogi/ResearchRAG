"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUp, ExternalLink, RotateCcw, SlidersHorizontal, Square } from "lucide-react";
import {
  api,
  pageLabel,
  pdfUrl,
  streamChat,
  type ChatMessage,
  type Citation,
  type Health,
  type RetrievalMode,
  type RetrievedChunk,
  type SearchFilters,
} from "@/lib/api";
import { AnswerMarkdown } from "@/components/AnswerMarkdown";
import { FilterPanel, RetrievalSettings, activeFilterCount, useFacets } from "@/components/FilterPanel";
import { Badge, Button, ErrorNote } from "@/components/ui";

interface Turn {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: RetrievedChunk[];
  citations?: Citation[];
  streaming?: boolean;
  error?: string;
}

const SUGGESTIONS = [
  "What problem does each paper address, and how do their approaches differ?",
  "Summarize the main experimental results, with the numbers reported.",
  "What limitations or failure cases do the authors acknowledge?",
];
const HISTORY_TURNS = 6;

let nextId = 1;

function SourceCard({
  s,
  n,
  cited,
  dimmed,
  active,
  onSelect,
}: {
  s: RetrievedChunk;
  n: number;
  cited: boolean;
  dimmed: boolean;
  active: boolean;
  onSelect: () => void;
}) {
  const ref = useRef<HTMLLIElement>(null);
  // null = follow selection: a cited source opens in full until the reader toggles it
  const [expanded, setExpanded] = useState<boolean | null>(null);
  const open = expanded ?? active;

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  const text = open || s.text.length < 280 ? s.text : s.text.slice(0, 280) + "…";
  return (
    <li
      ref={ref}
      id={`source-${n}`}
      className={`rounded-md border p-3 text-sm transition ${
        active ? "border-accent ring-2 ring-accent/20" : "border-line"
      } ${dimmed ? "opacity-70" : ""}`}
    >
      <button type="button" onClick={onSelect} className="flex w-full items-start gap-2 text-left">
        <span
          className={`grid h-4.5 min-w-4.5 shrink-0 place-items-center rounded px-1 font-mono text-[10.5px] font-semibold ${
            active ? "bg-accent text-white" : "bg-accent-soft text-accent-ink"
          }`}
        >
          {n}
        </span>
        <span className="min-w-0 flex-1">
          <span className="line-clamp-2 font-medium leading-snug">{s.title}</span>
          <span className="mt-0.5 block text-xs text-ink-3">
            § {s.section} · {pageLabel(s.page_start, s.page_end)}
          </span>
        </span>
        {cited && <Badge tone="accent">cited</Badge>}
      </button>
      <p className="mt-2 whitespace-pre-line text-xs leading-relaxed text-ink-2">{text}</p>
      <div className="mt-2 flex items-center gap-3 text-xs">
        {s.text.length >= 280 && (
          <button className="text-accent hover:underline" onClick={() => setExpanded(!open)}>
            {open ? "Show less" : "Show more"}
          </button>
        )}
        <a
          href={pdfUrl(s.paper_id, s.page_start)}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 text-accent hover:underline"
        >
          Open PDF <ExternalLink size={11} />
        </a>
      </div>
    </li>
  );
}

function SourcesList({
  turn,
  activeSource,
  onSelect,
}: {
  turn: Turn;
  activeSource: number | null;
  onSelect: (n: number) => void;
}) {
  const sources = turn.sources ?? [];
  const cited = new Set(turn.citations?.map((c) => c.number));
  if (!sources.length) return <p className="text-xs text-ink-3">No passages were retrieved for this question.</p>;
  return (
    <ol className="space-y-2">
      {sources.map((s, i) => (
        <SourceCard
          key={s.id}
          s={s}
          n={i + 1}
          cited={cited.has(i + 1)}
          // Only fade uncited sources when the answer actually cites something
          dimmed={cited.size > 0 && !cited.has(i + 1) && activeSource !== i + 1}
          active={activeSource === i + 1}
          onSelect={() => onSelect(i + 1)}
        />
      ))}
    </ol>
  );
}

function AssistantTurn({
  turn,
  selected,
  activeSource,
  onCite,
  onClearSource,
}: {
  turn: Turn;
  selected: boolean;
  activeSource: number | null;
  onCite: (n: number) => void;
  onClearSource: () => void;
}) {
  const sources = turn.sources ?? [];
  const [showSources, setShowSources] = useState(false);
  const sourcesOpen = showSources || (selected && activeSource != null);
  const waiting = turn.streaming && !turn.content;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <span className="grid h-5 w-5 place-items-center rounded bg-accent text-[10px] font-bold text-white">R</span>
        {turn.sources === undefined && turn.streaming
          ? "Searching your papers…"
          : `${sources.length} passages retrieved${turn.citations ? ` · ${turn.citations.length} cited` : ""}`}
      </div>
      {waiting && turn.sources !== undefined && <p className="caret text-sm text-ink-3">Reading sources</p>}
      {!!turn.content && (
        <div className={turn.streaming ? "streaming" : ""}>
          <AnswerMarkdown
            text={turn.content}
            sources={sources}
            activeSource={selected ? activeSource : null}
            onCite={onCite}
          />
        </div>
      )}
      <ErrorNote message={turn.error ?? null} />
      {/* Below xl the side panel is hidden, so sources open inline under the answer */}
      {!!sources.length && !turn.streaming && (
        <div className="xl:hidden">
          <button
            className="text-xs text-accent hover:underline"
            onClick={() => {
              if (sourcesOpen && selected) onClearSource();
              setShowSources(!sourcesOpen);
            }}
          >
            {sourcesOpen ? "Hide sources" : `Show ${sources.length} sources`}
          </button>
          {sourcesOpen && (
            <div className="mt-2">
              <SourcesList turn={turn} activeSource={selected ? activeSource : null} onSelect={onCite} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const facets = useFacets();
  const [health, setHealth] = useState<Health | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [rerank, setRerank] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const busy = turns.some((t) => t.streaming);
  const selected = turns.find((t) => t.id === selectedTurn) ?? [...turns].reverse().find((t) => t.role === "assistant");
  const nFilters = activeFilterCount(filters);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    return () => abortRef.current?.abort();
  }, []);

  // Follow the stream while the reader is near the bottom
  const lastContent = turns.at(-1)?.content;
  useEffect(() => {
    const el = scrollRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [turns.length, lastContent]);

  const patch = (id: number, p: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...(typeof p === "function" ? p(t) : p) } : t)));

  const ask = async (question: string) => {
    question = question.trim();
    if (!question || busy) return;
    const history: ChatMessage[] = turns
      .filter((t) => !t.error && t.content)
      .slice(-HISTORY_TURNS)
      .map((t) => ({ role: t.role, content: t.content }));
    const user: Turn = { id: nextId++, role: "user", content: question };
    const bot: Turn = { id: nextId++, role: "assistant", content: "", streaming: true };
    setTurns((ts) => [...ts, user, bot]);
    setSelectedTurn(bot.id);
    setActiveSource(null);
    setInput("");
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        { question, history, filters, mode, rerank },
        {
          onSources: (sources) => patch(bot.id, { sources }),
          onToken: (tok) => patch(bot.id, (t) => ({ content: t.content + tok })),
          onDone: (a) => patch(bot.id, { content: a.answer, citations: a.citations, sources: a.sources }),
          onError: (message) => patch(bot.id, { error: message }),
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") patch(bot.id, { error: (e as Error).message });
    } finally {
      patch(bot.id, { streaming: false });
      abortRef.current = null;
      textareaRef.current?.focus();
    }
  };

  const cite = (turnId: number) => (n: number) => {
    setSelectedTurn(turnId);
    setActiveSource((cur) => (cur === n && selected?.id === turnId ? null : n));
  };

  const reset = () => {
    abortRef.current?.abort();
    setTurns([]);
    setSelectedTurn(null);
    setActiveSource(null);
  };

  const settings = (
    <div className="space-y-5">
      <RetrievalSettings mode={mode} rerank={rerank} onMode={setMode} onRerank={setRerank} />
      <FilterPanel filters={filters} onChange={setFilters} facets={facets} />
    </div>
  );

  return (
    <div className="mx-auto flex h-[calc(100dvh-3.5rem)] w-full max-w-7xl gap-6 px-4">
      <aside className="hidden w-60 shrink-0 overflow-y-auto py-6 lg:block">{settings}</aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="flex-1 overflow-y-auto py-6">
          {turns.length === 0 ? (
            <div className="mx-auto max-w-xl pt-[10vh] text-center">
              <h1 className="text-xl font-semibold tracking-tight">Ask your papers</h1>
              <p className="mt-1 text-sm text-ink-2">
                Answers are grounded in retrieved passages. Every claim carries a numbered citation you can open.
              </p>
              {health && !health.llm_configured && (
                <div className="mt-4 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-left text-sm text-ink-2">
                  No Gemini API key is configured, so answers can&apos;t be generated. Set{" "}
                  <code className="font-mono text-xs">GEMINI_API_KEY</code> in <code className="font-mono text-xs">backend/.env</code>, or
                  use{" "}
                  <Link href="/search" className="text-accent hover:underline">
                    Search
                  </Link>{" "}
                  meanwhile.
                </div>
              )}
              {facets && facets.papers.length === 0 && (
                <p className="mt-4 text-sm text-ink-3">
                  No indexed papers yet.{" "}
                  <Link href="/" className="text-accent hover:underline">
                    Add some to your library
                  </Link>{" "}
                  first.
                </p>
              )}
              <div className="mt-6 grid gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => ask(s)}
                    className="rounded-md border border-line bg-surface px-3 py-2 text-left text-sm text-ink-2 transition hover:border-ink-3 hover:text-ink"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6">
              {turns.map((t) =>
                t.role === "user" ? (
                  <div key={t.id} className="flex justify-end">
                    <div className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-accent-soft px-3.5 py-2 text-sm text-ink">
                      {t.content}
                    </div>
                  </div>
                ) : (
                  <AssistantTurn
                    key={t.id}
                    turn={t}
                    selected={selected?.id === t.id}
                    activeSource={activeSource}
                    onCite={cite(t.id)}
                    onClearSource={() => setActiveSource(null)}
                  />
                ),
              )}
            </div>
          )}
        </div>

        <div className="mx-auto w-full max-w-3xl pb-4">
          {showSettings && (
            <div className="mb-2 max-h-[50dvh] overflow-y-auto rounded-lg border border-line bg-surface p-4 lg:hidden">
              {settings}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="rounded-lg border border-line bg-surface p-2 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20"
          >
            <textarea
              ref={textareaRef}
              rows={2}
              value={input}
              autoFocus
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              placeholder={turns.length ? "Ask a follow-up…" : "Ask a question about your papers…"}
              className="block max-h-40 w-full resize-none bg-transparent px-1.5 py-1 text-sm outline-none placeholder:text-ink-3"
            />
            <div className="flex items-center gap-1 pt-1">
              <Button
                type="button"
                variant="ghost"
                className="lg:hidden"
                onClick={() => setShowSettings(!showSettings)}
                aria-expanded={showSettings}
              >
                <SlidersHorizontal size={14} />
                Filters{nFilters ? ` (${nFilters})` : ""}
              </Button>
              <span className="hidden text-xs text-ink-3 lg:inline">
                {mode === "sparse" ? "BM25" : mode}
                {rerank ? " + rerank" : ""}
                {nFilters ? ` · ${nFilters} filter${nFilters > 1 ? "s" : ""}` : ""}
              </span>
              <div className="ml-auto flex items-center gap-1">
                {!!turns.length && (
                  <Button type="button" variant="ghost" onClick={reset} title="New conversation">
                    <RotateCcw size={14} />
                    <span className="hidden sm:inline">New</span>
                  </Button>
                )}
                {busy ? (
                  <Button type="button" variant="secondary" onClick={() => abortRef.current?.abort()}>
                    <Square size={12} className="fill-current" /> Stop
                  </Button>
                ) : (
                  <Button type="submit" disabled={!input.trim()} aria-label="Send">
                    <ArrowUp size={14} /> Ask
                  </Button>
                )}
              </div>
            </div>
          </form>
        </div>
      </div>

      <aside className="hidden w-85 shrink-0 overflow-y-auto py-6 xl:block">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="text-sm font-medium">Sources</span>
          {selected?.sources && <span className="text-xs text-ink-3">for the selected answer</span>}
        </div>
        {selected?.sources ? (
          <SourcesList
            turn={selected}
            activeSource={activeSource}
            onSelect={(n) => setActiveSource((cur) => (cur === n ? null : n))}
          />
        ) : (
          <p className="text-xs leading-relaxed text-ink-3">
            Retrieved passages appear here. Click a citation number in an answer to jump to its source.
          </p>
        )}
      </aside>
    </div>
  );
}

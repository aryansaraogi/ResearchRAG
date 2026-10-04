"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Copy,
  FileText,
  MessageSquareText,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  Square,
} from "lucide-react";
import {
  api,
  pageLabel,
  streamChat,
  type ChatMessage,
  type Citation,
  type Health,
  type RetrievalMode,
  type RetrievedChunk,
  type SearchFilters,
} from "@/lib/api";
import { readJSON, writeJSON } from "@/lib/storage";
import { AnswerMarkdown } from "@/components/AnswerMarkdown";
import { FilterPanel, RetrievalSettings, activeFilterCount, useFacets } from "@/components/FilterPanel";
import { usePdfViewer } from "@/components/PdfViewer";
import { useToast } from "@/components/Toast";
import { Badge, Button, EmptyState, ErrorNote } from "@/components/ui";

interface Turn {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: RetrievedChunk[];
  citations?: Citation[];
  streaming?: boolean;
  error?: string;
  /** What retrieval searched for, when not the question as asked (rewritten follow-up, or one query per part) */
  queries?: string[];
  /** Single rewritten query saved by earlier versions */
  query?: string;
  /** Gemini model that wrote the answer */
  model?: string;
}

const SUGGESTIONS = [
  "What problem does each paper address, and how do their approaches differ?",
  "Summarize the main experimental results, with the numbers reported.",
  "What limitations or failure cases do the authors acknowledge?",
];
const HISTORY_TURNS = 6;
const ARXIV_ID = /^\d{4}\.\d{4,5}(v\d+)?$/;

/* ---------- Saved conversation (this browser only) ---------- */

const STORE_KEY = "researchrag.chat.v1";
const MAX_SAVED = 40;
const EMPTY: Turn[] = [];
let savedCache: Turn[] | undefined;

function getSaved(): Turn[] {
  if (savedCache === undefined) {
    const raw = readJSON<unknown>(STORE_KEY, []);
    savedCache = Array.isArray(raw)
      ? raw.filter((t): t is Turn => (t?.role === "user" || t?.role === "assistant") && typeof t?.content === "string")
      : [];
  }
  return savedCache;
}

function persist(turns: Turn[]) {
  savedCache = turns.filter((t) => !t.streaming).slice(-MAX_SAVED);
  writeJSON(STORE_KEY, savedCache);
}

const noSubscribe = () => () => {};

/** Answer text plus a numbered reference list for the sources it actually cites. */
function withReferences(turn: Turn) {
  const sources = turn.sources ?? [];
  const numbers = [...new Set((turn.citations ?? []).map((c) => c.number))].sort((a, b) => a - b);
  const refs = numbers
    .filter((n) => sources[n - 1])
    .map((n) => {
      const s = sources[n - 1];
      const arxiv = ARXIV_ID.test(s.paper_id) ? `, arXiv:${s.paper_id}` : "";
      return `[${n}] ${s.title}${arxiv} — §${s.section}, ${pageLabel(s.page_start, s.page_end)}`;
    });
  return { text: refs.length ? `${turn.content}\n\nReferences\n${refs.join("\n")}` : turn.content, count: refs.length };
}

/* ---------- Sources ---------- */

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
  const openPdf = usePdfViewer();
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
      className={`rounded-lg border bg-surface p-3 text-sm transition ${
        active ? "border-accent ring-2 ring-accent/20" : "border-line"
      } ${dimmed ? "opacity-70" : ""}`}
    >
      <button type="button" onClick={onSelect} className="flex w-full items-start gap-2 rounded text-left">
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
          <button className="text-accent-ink hover:underline" onClick={() => setExpanded(!open)}>
            {open ? "Show less" : "Show more"}
          </button>
        )}
        <button
          className="flex items-center gap-1 text-accent-ink hover:underline"
          onClick={() => openPdf({ paperId: s.paper_id, title: s.title, page: s.page_start })}
        >
          <FileText size={12} /> View p. {s.page_start}
        </button>
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

/* ---------- Turns ---------- */

function AssistantTurn({
  turn,
  selected,
  isLast,
  busy,
  primaryModel,
  activeSource,
  onCite,
  onClearSource,
  onCopy,
  onRegenerate,
}: {
  turn: Turn;
  selected: boolean;
  isLast: boolean;
  busy: boolean;
  primaryModel?: string;
  activeSource: number | null;
  onCite: (n: number) => void;
  onClearSource: () => void;
  onCopy: () => void;
  onRegenerate: () => void;
}) {
  const sources = turn.sources ?? [];
  const [showSources, setShowSources] = useState(false);
  const sourcesOpen = showSources || (selected && activeSource != null);
  const waiting = turn.streaming && !turn.content;
  const done = !turn.streaming;
  const searched = turn.queries ?? (turn.query ? [turn.query] : undefined);

  return (
    <div className="group animate-in space-y-2">
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <span className="grid h-5 w-5 place-items-center rounded-md bg-accent text-[10px] font-bold text-white">R</span>
        {turn.sources === undefined && turn.streaming
          ? "Searching your papers…"
          : `${sources.length} passages retrieved${turn.citations ? ` · ${turn.citations.length} cited` : ""}`}
        {turn.model && primaryModel && turn.model !== primaryModel && (
          <span title={`${primaryModel} was unavailable, so a fallback model answered`}>· answered by {turn.model}</span>
        )}
      </div>
      {!!searched?.length && (
        <p
          className="text-xs text-ink-3"
          title="Follow-ups are made self-contained, and questions about several things are searched one part at a time"
        >
          {searched.length > 1 ? "Searched separately for: " : "Searched for: "}
          {searched.map((q, i) => (
            <span key={i}>
              {i > 0 && <span className="text-ink-3"> · </span>}
              <span className="italic text-ink-2">{q}</span>
            </span>
          ))}
        </p>
      )}
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

      {done && (turn.content || (turn.error && isLast)) && (
        // Older answers show their actions on hover/focus (desktop); the latest always shows them
        <div
          className={`-ml-2 flex flex-wrap items-center gap-0.5 ${
            isLast ? "" : "lg:opacity-0 lg:transition lg:group-hover:opacity-100 lg:group-focus-within:opacity-100"
          }`}
        >
          {turn.content && !turn.error && (
            <Button variant="ghost" className="px-2! py-1! text-xs" onClick={onCopy}>
              <Copy size={13} /> Copy
            </Button>
          )}
          {isLast && (
            <Button variant="ghost" className="px-2! py-1! text-xs" onClick={onRegenerate} disabled={busy}>
              <RefreshCw size={13} /> {turn.error ? "Retry" : "Regenerate"}
            </Button>
          )}
          {!!sources.length && (
            // Below xl the side panel is hidden, so sources open inline under the answer
            <Button
              variant="ghost"
              className="px-2! py-1! text-xs xl:hidden"
              aria-expanded={sourcesOpen}
              onClick={() => {
                if (sourcesOpen && selected) onClearSource();
                setShowSources(!sourcesOpen);
              }}
            >
              <BookOpen size={13} /> {sourcesOpen ? "Hide sources" : `${sources.length} sources`}
            </Button>
          )}
        </div>
      )}
      {!!sources.length && done && sourcesOpen && (
        <div className="xl:hidden">
          <SourcesList turn={turn} activeSource={selected ? activeSource : null} onSelect={onCite} />
        </div>
      )}
    </div>
  );
}

/* ---------- Page ---------- */

export default function ChatPage() {
  const facets = useFacets();
  const toast = useToast();
  const [health, setHealth] = useState<Health | null>(null);
  // Saved turns are read after hydration (the server renders an empty conversation)
  const saved = useSyncExternalStore(noSubscribe, getSaved, () => EMPTY);
  const [live, setLive] = useState<Turn[] | null>(null);
  const turns = live ?? saved;
  const restored = live === null && saved.length > 0;
  const [input, setInput] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [rerank, setRerank] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const busy = turns.some((t) => t.streaming);
  const lastAssistant = [...turns].reverse().find((t) => t.role === "assistant");
  const selected = turns.find((t) => t.id === selectedTurn) ?? lastAssistant;
  const nFilters = activeFilterCount(filters);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    return () => abortRef.current?.abort();
  }, []);

  // Save finished conversations to this browser
  useEffect(() => {
    if (live !== null && !busy) persist(live);
  }, [live, busy]);

  // Open a restored conversation at its end
  useEffect(() => {
    const el = scrollRef.current;
    if (restored && el) el.scrollTop = el.scrollHeight;
  }, [restored]);

  // Follow the stream while the reader is near the bottom
  const lastContent = turns.at(-1)?.content;
  useEffect(() => {
    const el = scrollRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [turns.length, lastContent]);

  const patch = (id: number, p: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setLive((ts) => (ts ?? saved).map((t) => (t.id === id ? { ...t, ...(typeof p === "function" ? p(t) : p) } : t)));

  const scrollToEnd = (behavior: ScrollBehavior = "auto") =>
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior }));

  /** Ask `question` after the turns in `base` (defaults to the whole conversation). */
  const ask = async (question: string, base: Turn[] = turns) => {
    question = question.trim();
    if (!question || busy) return;
    const history: ChatMessage[] = base
      .filter((t) => !t.error && t.content)
      .slice(-HISTORY_TURNS)
      .map((t) => ({ role: t.role, content: t.content }));
    const id = Math.max(0, ...base.map((t) => t.id)) + 1;
    const user: Turn = { id, role: "user", content: question };
    const bot: Turn = { id: id + 1, role: "assistant", content: "", streaming: true };
    setLive([...base, user, bot]);
    setSelectedTurn(bot.id);
    setActiveSource(null);
    setInput("");
    scrollToEnd();

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        { question, history, filters, mode, rerank },
        {
          onQueries: (queries) => patch(bot.id, { queries }),
          onSources: (sources) => patch(bot.id, { sources }),
          onToken: (tok) => patch(bot.id, (t) => ({ content: t.content + tok })),
          onDone: (a) =>
            patch(bot.id, {
              content: a.answer,
              citations: a.citations,
              sources: a.sources,
              queries: a.queries ?? undefined,
              model: a.model ?? undefined,
            }),
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

  const regenerate = (turnId: number) => {
    const i = turns.findIndex((t) => t.id === turnId);
    const question = turns[i - 1];
    if (i < 1 || question?.role !== "user") return;
    ask(question.content, turns.slice(0, i - 1));
  };

  const copy = async (turn: Turn) => {
    const { text, count } = withReferences(turn);
    try {
      await navigator.clipboard.writeText(text);
      toast(count ? `Copied with ${count} reference${count === 1 ? "" : "s"}` : "Copied answer", "good");
    } catch {
      toast("Couldn't copy to the clipboard", "critical");
    }
  };

  const cite = (turnId: number) => (n: number) => {
    setSelectedTurn(turnId);
    setActiveSource((cur) => (cur === n && selected?.id === turnId ? null : n));
  };

  const reset = () => {
    abortRef.current?.abort();
    setLive([]);
    persist([]);
    setSelectedTurn(null);
    setActiveSource(null);
    textareaRef.current?.focus();
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
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto py-6"
          onScroll={(e) => {
            const el = e.currentTarget;
            setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 160);
          }}
        >
          {turns.length === 0 ? (
            <div className="mx-auto max-w-xl pt-[6vh]">
              <EmptyState icon={MessageSquareText} title="Ask your papers">
                Answers come only from retrieved passages, and every claim carries a numbered citation you can open at
                its page.
              </EmptyState>
              {health && !health.llm_configured && (
                <div className="mb-4 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-ink-2">
                  No Gemini API key is configured, so answers can&apos;t be generated. Set{" "}
                  <code className="font-mono text-xs">GEMINI_API_KEY</code> in <code className="font-mono text-xs">backend/.env</code>, or
                  use{" "}
                  <Link href="/search" className="text-accent-ink hover:underline">
                    Search
                  </Link>{" "}
                  meanwhile.
                </div>
              )}
              {facets && facets.papers.length === 0 && (
                <p className="mb-4 text-center text-sm text-ink-3">
                  No indexed papers yet.{" "}
                  <Link href="/" className="text-accent-ink hover:underline">
                    Add some to your library
                  </Link>{" "}
                  first.
                </p>
              )}
              <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">Try asking</div>
              <div className="mt-2 grid gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => ask(s)}
                    className="rounded-lg border border-line bg-surface px-3 py-2.5 text-left text-sm text-ink-2 shadow-card transition hover:border-accent/50 hover:text-ink"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-7">
              {restored && (
                <p className="flex items-center justify-center gap-2 text-xs text-ink-3">
                  Restored your last conversation.
                  <button className="text-accent-ink hover:underline" onClick={reset}>
                    Start a new one
                  </button>
                </p>
              )}
              {turns.map((t) =>
                t.role === "user" ? (
                  <div key={t.id} className="animate-in flex justify-end">
                    <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-sm text-ink">
                      {t.content}
                    </div>
                  </div>
                ) : (
                  <AssistantTurn
                    key={t.id}
                    turn={t}
                    selected={selected?.id === t.id}
                    isLast={t.id === lastAssistant?.id}
                    busy={busy}
                    primaryModel={health?.llm_model}
                    activeSource={activeSource}
                    onCite={cite(t.id)}
                    onClearSource={() => setActiveSource(null)}
                    onCopy={() => copy(t)}
                    onRegenerate={() => regenerate(t.id)}
                  />
                ),
              )}
            </div>
          )}
        </div>

        <div className="relative mx-auto w-full max-w-3xl pb-4">
          {!atBottom && turns.length > 0 && (
            <button
              onClick={() => scrollToEnd("smooth")}
              className="animate-in absolute -top-11 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-ink-2 shadow-pop hover:text-ink"
            >
              <ArrowDown size={13} /> {busy ? "Jump to live answer" : "Jump to latest"}
            </button>
          )}
          {showSettings && (
            <div className="mb-2 max-h-[50dvh] overflow-y-auto rounded-xl border border-line bg-surface p-4 shadow-pop lg:hidden">
              {settings}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="rounded-xl border border-line bg-surface p-2 shadow-card transition focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20"
          >
            <textarea
              ref={textareaRef}
              rows={2}
              value={input}
              autoFocus
              aria-label="Question"
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
                <span className="hidden xl:inline"> · Enter to send, Shift+Enter for a new line</span>
              </span>
              <div className="ml-auto flex items-center gap-1">
                {!!turns.length && (
                  <Button type="button" variant="ghost" onClick={reset} title="Start a new conversation">
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

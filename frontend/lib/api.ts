export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type PaperStatus = "pending" | "processing" | "ready" | "failed";

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  categories: string[];
  abstract: string;
  source: "upload" | "arxiv";
  num_pages: number;
  num_chunks: number;
  sections: string[];
  status: PaperStatus;
  error: string | null;
  created_at: string;
}

export interface ArxivResult {
  arxiv_id: string;
  title: string;
  authors: string[];
  year: number;
  categories: string[];
  abstract: string;
}

export interface SearchFilters {
  year_min?: number | null;
  year_max?: number | null;
  authors?: string[] | null;
  categories?: string[] | null;
  paper_ids?: string[] | null;
  section_types?: string[] | null;
}

export type RetrievalMode = "hybrid" | "dense" | "sparse";

export interface RetrievedChunk {
  id: string;
  paper_id: string;
  title: string;
  authors: string[];
  year: number | null;
  section: string;
  section_type: string;
  page_start: number;
  page_end: number;
  text: string;
  score: number;
  rerank_score: number | null;
}

export interface Citation {
  number: number;
  chunk_id: string;
  paper_id: string;
  title: string;
  section: string;
  page_start: number;
  page_end: number;
  snippet: string;
}

export interface RAGAnswer {
  answer: string;
  citations: Citation[];
  sources: RetrievedChunk[];
  /** What retrieval searched for, when a follow-up was rewritten into a standalone question */
  query?: string | null;
  /** The Gemini model that wrote the answer (may be a fallback) */
  model?: string | null;
}

export interface Facets {
  authors: string[];
  categories: string[];
  year_min: number | null;
  year_max: number | null;
  papers: { id: string; title: string }[];
  section_types: string[];
}

export interface Health {
  qdrant: string;
  qdrant_mode: "server" | "embedded";
  llm_configured: boolean;
  llm_model: string;
  llm_fallbacks?: string[];
}

export interface EvalItem {
  id: string;
  question: string;
  reference_answer: string;
  gold_chunk_ids: string[];
  paper_id: string;
  section: string;
  source: string;
}

export interface GenerationDetail {
  answer: string;
  faithfulness: number;
  answer_relevance: number;
  context_relevance: number;
  correctness: number;
  citation_precision: number;
  citation_coverage: number;
  gold_cited: number | null;
}

export interface EvalDetailRow {
  id: string;
  question: string;
  paper_id: string;
  configs: Record<
    string,
    { gold_rank: number | null; mrr: number; generation?: GenerationDetail; generation_error?: string }
  >;
}

export interface EvalRun {
  id: number;
  created_at: string;
  status: "running" | "done" | "failed";
  progress: string;
  dataset_size: number;
  configs: string[];
  results: Record<string, Record<string, number>>;
  details?: EvalDetailRow[];
  error: string | null;
}

export interface DatasetStatus {
  state: "idle" | "running" | "done" | "failed";
  message: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, `Can't reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),

  papers: () => request<Paper[]>("/papers"),
  facets: () => request<Facets>("/papers/facets"),
  deletePaper: (id: string) => request<void>(`/papers/${encodeURIComponent(id)}`, { method: "DELETE" }),
  reingest: (id: string) => post<Paper>(`/papers/${encodeURIComponent(id)}/reingest`, {}),
  importArxiv: (ids: string[]) => post<Paper[]>("/papers/arxiv", { ids }),
  searchArxiv: (q: string) => request<ArxivResult[]>(`/papers/arxiv/search?q=${encodeURIComponent(q)}&max_results=10`),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<Paper[]>("/papers/upload", { method: "POST", body: form });
  },

  search: (body: { query: string; filters?: SearchFilters; mode?: RetrievalMode; rerank?: boolean; top_k?: number }) =>
    post<RetrievedChunk[]>("/search", body),

  evalConfigs: () => request<Record<string, { mode: string; rerank: boolean }>>("/eval/configs"),
  dataset: () => request<EvalItem[]>("/eval/dataset"),
  datasetStatus: () => request<DatasetStatus>("/eval/dataset/status"),
  generateDataset: (n: number, append = false) => post<DatasetStatus>("/eval/dataset/generate", { n, append }),
  runs: () => request<EvalRun[]>("/eval/runs"),
  run: (id: number) => request<EvalRun>(`/eval/runs/${id}`),
  startRun: (body: { configs?: string[] | null; gen_configs: string[]; max_gen_questions: number | null }) =>
    post<EvalRun>("/eval/runs", body),
};

export function pdfUrl(paperId: string, page?: number) {
  return `${API_URL}/papers/${encodeURIComponent(paperId)}/pdf${page ? `#page=${page}` : ""}`;
}

export function pageLabel(start: number, end: number) {
  return start === end ? `p. ${start}` : `pp. ${start}–${end}`;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  history: ChatMessage[];
  filters?: SearchFilters;
  mode?: RetrievalMode;
  rerank?: boolean;
}

export interface StreamHandlers {
  onQuery?: (query: string) => void;
  onSources: (sources: RetrievedChunk[]) => void;
  onToken: (token: string) => void;
  onDone: (answer: RAGAnswer) => void;
  onError: (message: string) => void;
}

/** POST /chat/stream and dispatch server-sent events (EventSource can't POST). */
export async function streamChat(req: ChatRequest, h: StreamHandlers, signal?: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    if ((e as Error).name !== "AbortError") h.onError(`Can't reach the API at ${API_URL}.`);
    return;
  }
  if (!res.ok || !res.body) {
    h.onError(`Chat request failed (${res.status})`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() ?? "";
    for (const raw of events) {
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue; // keep-alive comments
      const payload = JSON.parse(data.join("\n"));
      if (event === "query") h.onQuery?.(payload);
      else if (event === "sources") h.onSources(payload);
      else if (event === "token") h.onToken(payload);
      else if (event === "done") h.onDone(payload);
      else if (event === "error") h.onError(payload);
    }
  }
}

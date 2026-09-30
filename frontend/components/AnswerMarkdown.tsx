"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { RetrievedChunk } from "@/lib/api";
import { pageLabel } from "@/lib/api";

/** Turn [1], [1, 3] and [1][2] markers into links the renderer swaps for citation chips. */
function linkCitations(text: string, nSources: number) {
  return text.replace(/\[(\d+(?:\s*[,;]\s*\d+)*)\](?!\()/g, (m, group: string) => {
    const nums = group.split(/\s*[,;]\s*/).map(Number);
    if (!nums.every((n) => n >= 1 && n <= nSources)) return m;
    return nums.map((n) => `[${n}](#cite-${n})`).join("");
  });
}

export function AnswerMarkdown({
  text,
  sources,
  activeSource,
  onCite,
}: {
  text: string;
  sources: RetrievedChunk[];
  activeSource?: number | null;
  onCite: (n: number) => void;
}) {
  return (
    <div className="prose-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children, ...props }) => {
            const m = href?.match(/^#cite-(\d+)$/);
            if (!m) {
              return (
                <a href={href} target="_blank" rel="noreferrer" className="text-accent underline" {...props}>
                  {children}
                </a>
              );
            }
            const n = Number(m[1]);
            const s = sources[n - 1];
            return (
              <button
                type="button"
                onClick={() => onCite(n)}
                title={s ? `${s.title} — § ${s.section}, ${pageLabel(s.page_start, s.page_end)}` : undefined}
                className={`mx-0.5 inline-flex h-[18px] min-w-[18px] items-center justify-center rounded px-1 align-[2px] font-mono text-[10.5px] font-semibold transition ${
                  activeSource === n ? "bg-accent text-white" : "bg-accent-soft text-accent-ink hover:bg-accent hover:text-white"
                }`}
              >
                {n}
              </button>
            );
          },
        }}
      >
        {linkCitations(text, sources.length)}
      </ReactMarkdown>
    </div>
  );
}

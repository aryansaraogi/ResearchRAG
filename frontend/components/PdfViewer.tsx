"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { ExternalLink, FileText, X } from "lucide-react";
import { pdfUrl } from "@/lib/api";
import { IconButton, iconButtonClass } from "@/components/ui";

export interface PdfTarget {
  paperId: string;
  title: string;
  page?: number;
}

const PdfContext = createContext<(target: PdfTarget) => void>(() => {});

/** One viewer for the whole app: pages call usePdfViewer()(target) to open a paper at a page. */
export function PdfViewerProvider({ children }: { children: ReactNode }) {
  const [target, setTarget] = useState<PdfTarget | null>(null);
  const opener = useRef<HTMLElement | null>(null);

  const open = useCallback((t: PdfTarget) => {
    if (document.activeElement instanceof HTMLElement) opener.current = document.activeElement;
    setTarget(t);
  }, []);

  const close = useCallback(() => {
    setTarget(null);
    opener.current?.focus();
  }, []);

  useEffect(() => {
    if (!target) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [target, close]);

  return (
    <PdfContext value={open}>
      {children}
      {target && <Viewer target={target} onClose={close} />}
    </PdfContext>
  );
}

export const usePdfViewer = () => useContext(PdfContext);

function Viewer({ target, onClose }: { target: PdfTarget; onClose: () => void }) {
  const src = pdfUrl(target.paperId, target.page);
  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" aria-label={`PDF viewer: ${target.title}`}>
      <div className="absolute inset-0 hidden bg-black/30 lg:block" onClick={onClose} aria-hidden />
      <div className="animate-drawer relative flex h-full w-full flex-col border-l border-line bg-surface shadow-pop lg:w-[min(780px,58vw)]">
        <header className="flex items-center gap-2 border-b border-line px-3 py-2">
          <FileText size={16} className="ml-1 shrink-0 text-ink-3" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium">{target.title}</div>
            <div className="text-xs text-ink-3">Page {target.page ?? 1}</div>
          </div>
          <a href={src} target="_blank" rel="noreferrer" className={iconButtonClass} aria-label="Open in new tab" title="Open in new tab">
            <ExternalLink size={15} />
          </a>
          <IconButton label="Close (Esc)" onClick={onClose} autoFocus>
            <X size={17} />
          </IconButton>
        </header>
        {/* Keyed by page so jumping to another page of the same PDF reloads the viewer */}
        <iframe key={`${target.paperId}#${target.page ?? 1}`} src={src} title={target.title} className="w-full flex-1 bg-surface-2" />
      </div>
    </div>
  );
}

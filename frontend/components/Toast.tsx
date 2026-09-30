"use client";

import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CircleAlert, CircleCheck, Info } from "lucide-react";

type Tone = "neutral" | "good" | "critical";
interface ToastItem {
  id: number;
  message: string;
  tone: Tone;
}

const ToastContext = createContext<(message: string, tone?: Tone) => void>(() => {});
let seq = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const show = useCallback((message: string, tone: Tone = "neutral") => {
    const id = ++seq;
    setToasts((ts) => [...ts.slice(-2), { id, message, tone }]);
    setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), 3500);
  }, []);

  return (
    <ToastContext value={show}>
      {children}
      <div role="status" aria-live="polite" className="pointer-events-none fixed inset-x-0 top-16 z-50 flex flex-col items-center gap-2 px-4">
        {toasts.map((t) => {
          const Icon = t.tone === "good" ? CircleCheck : t.tone === "critical" ? CircleAlert : Info;
          const color = t.tone === "good" ? "text-good-ink" : t.tone === "critical" ? "text-critical-ink" : "text-accent-ink";
          return (
            <div
              key={t.id}
              className="animate-in pointer-events-auto flex max-w-md items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm shadow-pop"
            >
              <Icon size={15} className={`shrink-0 ${color}`} />
              {t.message}
            </div>
          );
        })}
      </div>
    </ToastContext>
  );
}

export const useToast = () => useContext(ToastContext);

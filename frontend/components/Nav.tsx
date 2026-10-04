"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { BookOpen, Columns3, FlaskConical, MessageSquareText, Search } from "lucide-react";
import { api, type Health } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";

const LINKS = [
  { href: "/", label: "Library", icon: BookOpen },
  { href: "/search", label: "Search", icon: Search },
  { href: "/chat", label: "Ask", icon: MessageSquareText },
  { href: "/compare", label: "Compare", icon: Columns3 },
  { href: "/eval", label: "Evaluation", icon: FlaskConical },
];

export function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const check = () =>
      api
        .health()
        .then((h) => {
          setHealth(h);
          setDown(false);
        })
        .catch(() => setDown(true));
    check();
    const t = setInterval(check, 15000);
    return () => clearInterval(t);
  }, []);

  const ok = !down && health?.qdrant === "ok";
  const status = down
    ? "API offline"
    : !health
      ? "Connecting…"
      : health.qdrant !== "ok"
        ? "Vector store unreachable"
        : health.llm_configured
          ? health.llm_model
          : "No Gemini key: search only";

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4 sm:gap-6">
        <Link href="/" className="flex shrink-0 items-center gap-2 font-semibold tracking-tight" aria-label="ResearchRAG home">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white shadow-card">R</span>
          <span className="hidden sm:inline">ResearchRAG</span>
        </Link>
        <nav className="flex items-center gap-0.5 sm:gap-1" aria-label="Main">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-label={label}
                aria-current={active ? "page" : undefined}
                title={label}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition-colors sm:px-3 ${
                  active ? "bg-accent-soft font-medium text-accent-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"
                }`}
              >
                <Icon size={16} />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex items-center gap-2 px-1 text-xs text-ink-3" title={`${status}${health && health.qdrant !== "ok" ? ` (${health.qdrant})` : ""}${
              health?.llm_configured && health.llm_fallbacks?.length ? `\nFallbacks: ${health.llm_fallbacks.join(", ")}` : ""
            }`}>
            <span className={`h-2 w-2 shrink-0 rounded-full ${ok ? (health?.llm_configured ? "bg-good" : "bg-warning") : "bg-critical"}`} />
            <span className="sr-only lg:not-sr-only">{status}</span>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

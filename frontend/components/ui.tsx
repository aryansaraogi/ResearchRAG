"use client";

import { useState } from "react";
import { Loader2, type LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-white hover:opacity-90",
  secondary: "border border-line bg-surface text-ink hover:bg-surface-2",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink",
  danger: "text-critical-ink hover:bg-critical/10",
};

export function Button({
  variant = "primary",
  loading,
  className = "",
  children,
  disabled,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`}
    >
      {loading && <Loader2 size={14} className="animate-spin" />}
      {children}
    </button>
  );
}

export const iconButtonClass =
  "grid h-8 w-8 shrink-0 place-items-center rounded-md text-ink-3 transition hover:bg-surface-2 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40";

/** Icon-only button: the label is required and doubles as the tooltip. */
export function IconButton({
  label,
  tone = "neutral",
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; tone?: "neutral" | "danger" }) {
  const danger = tone === "danger" ? "hover:bg-critical/10! hover:text-critical-ink!" : "";
  return (
    <button type="button" aria-label={label} title={label} {...props} className={`${iconButtonClass} ${danger} ${className}`}>
      {children}
    </button>
  );
}

/** Two-step destructive action: the icon turns into an inline "Delete? / Cancel" prompt. */
export function ConfirmButton({
  label,
  prompt = "Delete?",
  confirmLabel = "Delete",
  onConfirm,
  disabled,
  children,
}: {
  label: string;
  prompt?: string;
  confirmLabel?: string;
  onConfirm: () => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  const [asking, setAsking] = useState(false);
  if (!asking)
    return (
      <IconButton label={label} tone="danger" disabled={disabled} onClick={() => setAsking(true)}>
        {children}
      </IconButton>
    );
  return (
    <span className="animate-in flex items-center gap-1 text-xs" role="group" aria-label={label}>
      <span className="px-1 text-ink-2">{prompt}</span>
      <Button
        variant="danger"
        className="px-2! py-1! text-xs"
        autoFocus
        onClick={() => {
          setAsking(false);
          onConfirm();
        }}
      >
        {confirmLabel}
      </Button>
      <Button variant="ghost" className="px-2! py-1! text-xs" onClick={() => setAsking(false)}>
        Cancel
      </Button>
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-xl border border-line bg-surface shadow-card ${className}`}>{children}</section>;
}

export type Tone = "neutral" | "accent" | "good" | "warning" | "critical";

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: Tone; title?: string }) {
  const tones: Record<Tone, string> = {
    neutral: "bg-surface-2 text-ink-2",
    accent: "bg-accent-soft text-accent-ink",
    good: "bg-good/12 text-good-ink",
    warning: "bg-warning/15 text-warning-ink",
    critical: "bg-critical/12 text-critical-ink",
  };
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function ErrorNote({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded-md border border-critical/30 bg-critical/8 px-3 py-2 text-sm text-critical-ink">
      {message}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  icon: Icon,
  children,
}: {
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div className="flex items-start gap-3">
        {Icon && (
          <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent-ink">
            <Icon size={18} />
          </span>
        )}
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-0.5 text-sm text-ink-2">{subtitle}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

/** Stat tile: sentence-case label, proportional-figure value, optional context line. */
export function StatTile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3 shadow-card">
      <div className="text-xs text-ink-3">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-2">{sub}</div>}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`animate-pulse rounded-md bg-surface-2 motion-reduce:animate-none ${className}`} />;
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  actions,
  className = "",
}: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`animate-in flex flex-col items-center px-4 py-10 text-center ${className}`}>
      <span className="grid h-11 w-11 place-items-center rounded-xl bg-accent-soft text-accent-ink">
        <Icon size={20} />
      </span>
      <h2 className="mt-3 text-base font-semibold">{title}</h2>
      {children && <div className="mt-1 max-w-md text-sm text-ink-2">{children}</div>}
      {actions && <div className="mt-4 flex flex-wrap justify-center gap-2">{actions}</div>}
    </div>
  );
}

const inputBase =
  "rounded-md border border-line bg-surface px-3 text-sm outline-none placeholder:text-ink-3 focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:opacity-50";
/** Full-width, regular height. */
export const inputClass = `${inputBase} w-full py-2`;
/** Compact height, no width: add w-full / w-20 etc. at the call site. */
export const inputSmClass = `${inputBase} py-1`;

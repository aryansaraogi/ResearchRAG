"use client";

import { useSyncExternalStore } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { writeString } from "@/lib/storage";
import { IconButton } from "@/components/ui";

type Theme = "system" | "light" | "dark";

const ORDER: Theme[] = ["system", "light", "dark"];
const ICONS = { system: Monitor, light: Sun, dark: Moon };
const LABELS = { system: "System theme", light: "Light theme", dark: "Dark theme" };
const listeners = new Set<() => void>();

// The inline script in layout.tsx applies the saved choice before paint; the attribute is the source of truth.
function getTheme(): Theme {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "light" || t === "dark" ? t : "system";
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function setTheme(theme: Theme) {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  writeString("theme", theme === "system" ? null : theme);
  listeners.forEach((l) => l());
}

export function ThemeToggle() {
  // Server render and hydration use "system"; the real choice appears right after
  const theme = useSyncExternalStore(subscribe, getTheme, () => "system" as Theme);
  const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length];
  const Icon = ICONS[theme];
  return (
    <IconButton label={`${LABELS[theme]}: switch to ${LABELS[next].toLowerCase()}`} onClick={() => setTheme(next)}>
      <Icon size={16} />
    </IconButton>
  );
}

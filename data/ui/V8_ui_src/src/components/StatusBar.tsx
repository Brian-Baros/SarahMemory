import { useEffect, useMemo, useRef, useState } from "react";
import {
  ExternalLink,
  Heart,
  Github,
  Zap,
  Database,
  Globe,
  Cpu,
  MessageCircle,
  Clock,
  Folder,
  Search,
  Palette,
  User,
  Network,
  Play,
  Settings,
  LayoutGrid,
  CheckCircle2,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { config } from "@/lib/config";
import { api } from "@/lib/api";
import { useSarahStore } from "@/stores/useSarahStore";
import { useWindowStore } from "@/stores/useWindowStore";

interface SystemStatus {
  local: boolean;
  web: boolean;
  api: boolean;
  network: boolean;
}

// Mode display map
const MODE_LABELS: Record<string, { label: string; icon: typeof Zap }> = {
  any: { label: "Any", icon: Zap },
  local: { label: "Local", icon: Database },
  web: { label: "Web", icon: Globe },
  api: { label: "API", icon: Cpu },
};

const MODE_ORDER = ["any", "local", "web", "api"] as const;

// ----------------------------------------------------------------------------
// Taskbar / icon order persistence
// - Phase 1: localStorage fallback
// - Phase 1.5+: if settings.taskbar.items exists, use + update it
// ----------------------------------------------------------------------------
const TASKBAR_ORDER_KEY = "sm_taskbar_order_v1";

// Default built-in taskbar items
const DEFAULT_TASKBAR_IDS = [
  "chat",
  "history",
  "files",
  "research",
  "studio",
  "avatar",
  "sarahnet",
  "media",
  "dlengine",
  "addons",
  "settings",
] as const;

type TaskbarId = (typeof DEFAULT_TASKBAR_IDS)[number] | string;

function safeParseOrder(raw: string | null): TaskbarId[] | null {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw);
    if (!Array.isArray(v)) return null;
    return v.map((x) => String(x));
  } catch {
    return null;
  }
}

function uniqKeepOrder(ids: TaskbarId[]) {
  const out: TaskbarId[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    const k = String(id);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(id);
  }
  return out;
}

function mergeWithDefaults(userOrder: TaskbarId[] | null) {
  const base = userOrder && userOrder.length ? userOrder : [];
  return uniqKeepOrder([...base, ...DEFAULT_TASKBAR_IDS]);
}

function formatClock(d: Date) {
  // compact: 08:41 PM (no seconds)
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function StatusBar() {
  const barRef = useRef<HTMLDivElement>(null);

  const { settings, updateSettings } = useSarahStore();
  const { windows, openWindow, focusWindow, restoreWindow, focusedWindowId } =
    useWindowStore();

  const [status, setStatus] = useState<SystemStatus>({
    local: true,
    web: true,
    api: false,
    network: true,
  });

  // ✅ Clock
  const [now, setNow] = useState<Date>(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // --------------------------------------------------------------------------
  // Taskbar order state (drag-reorder)
  // Priority:
  //  1) settings.taskbar.items (if present)
  //  2) localStorage
  //  3) defaults
  // --------------------------------------------------------------------------
  const initialOrder = useMemo(() => {
    const s = settings as any;
    const fromSettings = Array.isArray(s?.taskbar?.items)
      ? (s.taskbar.items as TaskbarId[]).map((x: any) => String(x))
      : null;

    if (fromSettings && fromSettings.length) return mergeWithDefaults(fromSettings);

    const saved = safeParseOrder(
      typeof window !== "undefined"
        ? window.localStorage.getItem(TASKBAR_ORDER_KEY)
        : null,
    );
    return mergeWithDefaults(saved);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // only once (we don’t want reorder to reset while running)

  const [taskbarOrder, setTaskbarOrder] = useState<TaskbarId[]>(initialOrder);

  // Persist order to localStorage (fallback)
  useEffect(() => {
    try {
      window.localStorage.setItem(TASKBAR_ORDER_KEY, JSON.stringify(taskbarOrder));
    } catch {
      // ignore (private mode / storage disabled)
    }
  }, [taskbarOrder]);

  // Best-effort persist order into settings.taskbar.items (if schema exists)
  useEffect(() => {
    const s: any = settings as any;
    if (!s?.taskbar) return;

    // Only write if it differs (avoid loops)
    const existing = Array.isArray(s.taskbar.items) ? s.taskbar.items.map(String) : [];
    const desired = taskbarOrder.map(String);

    const same =
      existing.length === desired.length &&
      existing.every((v: string, i: number) => v === desired[i]);

    if (same) return;

    try {
      updateSettings({
        taskbar: {
          ...(s.taskbar || {}),
          items: desired,
        },
      } as any);
    } catch {
      // silent (store might be strict until we patch useSarahStore.ts)
    }
  }, [taskbarOrder, settings, updateSettings]);

  // Drag state
  const dragIdRef = useRef<string | null>(null);

  const moveItem = (fromId: string, toId: string) => {
    if (!fromId || !toId || fromId === toId) return;
    setTaskbarOrder((prev) => {
      const next = [...prev];
      const fromIdx = next.findIndex((x) => String(x) === fromId);
      const toIdx = next.findIndex((x) => String(x) === toId);
      if (fromIdx === -1 || toIdx === -1) return prev;
      const [moved] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, moved);
      return next;
    });
  };

  // --------------------------------------------------------------------------
  // Taskbar sizing CSS vars
  // We keep your legacy --bottom-bar-h and add:
  //   --taskbar-dock (default "bottom" until DesktopShell drives it)
  //   --taskbar-size (computed)
  // --------------------------------------------------------------------------
  useEffect(() => {
    const el = barRef.current;
    if (!el) return;

    const apply = () => {
      const dock = (settings as any)?.taskbar?.dock || "bottom";
      const size =
        dock === "left" || dock === "right" ? el.offsetWidth || 56 : el.offsetHeight || 56;

      document.documentElement.style.setProperty("--taskbar-dock", String(dock));
      document.documentElement.style.setProperty("--taskbar-size", `${size}px`);

      // legacy var used by some window maximize logic
      document.documentElement.style.setProperty("--bottom-bar-h", `${el.offsetHeight || 56}px`);
    };

    apply();

    let ro: ResizeObserver | null = null;
    try {
      ro = new ResizeObserver(() => apply());
      ro.observe(el);
    } catch {
      window.addEventListener("resize", apply);
    }

    return () => {
      try {
        ro?.disconnect();
      } catch {}
      window.removeEventListener("resize", apply);
    };
  }, [settings]);

  // Check API health on mount via edge function
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.proxy.call("/api/health");
        setStatus((prev) => ({
          ...prev,
          api: response && !(response as any).fallback,
        }));
      } catch {
        setStatus((prev) => ({ ...prev, api: false }));
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Current mode
  const currentModeId = (settings.mode || "any") as keyof typeof MODE_LABELS;
  const currentMode = MODE_LABELS[currentModeId] || MODE_LABELS.any;
  const ModeIcon = currentMode.icon;

  const nextModeId = useMemo(() => {
    const idx = MODE_ORDER.indexOf(currentModeId as any);
    const next = MODE_ORDER[(idx + 1) % MODE_ORDER.length];
    return next;
  }, [currentModeId]);

  const cycleMode = async () => {
    const newMode = nextModeId;

    // Update store immediately so UI changes right away
    updateSettings({ mode: newMode });

    // Best-effort save to backend (same conventions SettingsModal uses)
    try {
      await api.settings.setSetting("api_mode", newMode);
      await api.settings.setSetting("mode", newMode);
    } catch {
      // silent: local UI still updated
    }
  };

  // Window open/focus behavior (like dock)
  const clickWindow = (id: any) => {
    const win = windows.find((w) => w.id === id);
    if (!win) {
      openWindow(id);
    } else if (win.isMinimized) {
      restoreWindow(id);
    } else {
      focusWindow(id);
    }
  };

  // Taskbar item registry (maps id -> label/icon/action)
  const TASKBAR_ITEMS: Record<
    string,
    { label: string; Icon: any; onClick: () => void }
  > = {
    chat: {
      label: "Chat",
      Icon: MessageCircle,
      onClick: () => clickWindow("chat"),
    },
    history: {
      label: "History",
      Icon: Clock,
      onClick: () => clickWindow("history"),
    },
    files: {
      label: "Files",
      Icon: Folder,
      onClick: () => clickWindow("files"),
    },
    research: {
      label: "Research",
      Icon: Search,
      onClick: () => clickWindow("research"),
    },
    studio: {
      label: "Studios",
      Icon: Palette,
      onClick: () => clickWindow("studio"),
    },
    avatar: {
      label: "Avatar",
      Icon: User,
      onClick: () => clickWindow("avatar"),
    },
    sarahnet: {
      label: "SarahNet",
      Icon: Network,
      onClick: () => clickWindow("sarahnet"),
    },
    media: {
      label: "Media",
      Icon: Play,
      onClick: () => clickWindow("media"),
    },
    dlengine: {
      label: "DL Engine",
      Icon: Cpu,
      onClick: () => clickWindow("dlengine"),
    },
    addons: {
      label: "Addons",
      Icon: LayoutGrid,
      onClick: () => clickWindow("addons"),
    },
    settings: {
      label: "Settings",
      Icon: Settings,
      onClick: () => clickWindow("settings"),
    },
  };

  const renderTaskbarButton = (id: string) => {
    const entry = TASKBAR_ITEMS[id];
    if (!entry) return null; // unknown IDs are allowed later (custom pins), skipped for now

    const isOpen = windows.some((w) => w.id === id);
    const isFocused = focusedWindowId === id;

    return (
      <button
        key={id}
        draggable
        onDragStart={() => {
          dragIdRef.current = id;
        }}
        onDragOver={(e) => {
          e.preventDefault();
        }}
        onDrop={() => {
          const from = dragIdRef.current;
          if (from) moveItem(from, id);
          dragIdRef.current = null;
        }}
        onDragEnd={() => {
          dragIdRef.current = null;
        }}
        onClick={entry.onClick}
        className={cn(
          "relative p-2 rounded-lg transition-all duration-150 select-none",
          "hover:bg-primary/20 hover:scale-110 active:scale-95",
          "cursor-pointer",
          isFocused && "bg-primary/30",
          "focus:outline-none focus:ring-1 focus:ring-primary/40",
        )}
        aria-label={entry.label}
        title={`${entry.label} (drag to reorder)`}
      >
        <entry.Icon
          className={cn(
            "h-5 w-5",
            isFocused
              ? "text-primary"
              : isOpen
                ? "text-foreground"
                : "text-muted-foreground",
          )}
        />
        {isOpen && (
          <span
            className={cn(
              "absolute bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full",
              isFocused ? "bg-primary" : "bg-muted-foreground",
            )}
          />
        )}
      </button>
    );
  };

  // ✅ Compact status pill (replaces LEDs)
  const overall = useMemo(() => {
    if (status.api) return { label: "Ready", tone: "ok" as const };
    return { label: "Connecting", tone: "warn" as const };
  }, [status.api]);

  const statusTitle = useMemo(() => {
    const yes = "✅";
    const no = "❌";
    return [
      `Local: ${status.local ? yes : no}`,
      `Web: ${status.web ? yes : no}`,
      `API: ${status.api ? yes : no}`,
      `Network: ${status.network ? yes : no}`,
    ].join("  |  ");
  }, [status]);

  const StatusIcon =
    overall.tone === "ok" ? CheckCircle2 : overall.tone === "warn" ? AlertTriangle : Loader2;

  return (
    <div
      ref={barRef}
      className="h-14 bg-card/95 backdrop-blur-sm border-t border-border flex items-center justify-between px-3 shrink-0"
      data-taskbar="true"
    >
      {/* LEFT: MODE + GitHub link */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={cycleMode}
          className={cn(
            "flex items-center gap-1.5 px-2 py-1 rounded-md",
            "bg-secondary/40 hover:bg-secondary/60 transition-colors",
            "text-foreground/80 hover:text-foreground",
          )}
          title={`Click to switch to ${MODE_LABELS[nextModeId].label}`}
          aria-label="Toggle Mode"
        >
          <span className="text-sm text-muted-foreground">MODE:</span>
          <ModeIcon className="h-3.5 w-3.5 text-primary" />
          <span className="text-sm font-medium text-foreground">
            {currentMode.label}
          </span>
        </button>

        <div className="h-4 w-px bg-border hidden sm:block" />

        <a
          href={config.githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors truncate"
          title="Backend Source: SarahMemory on GitHub"
        >
          <Github className="h-3.5 w-3.5 shrink-0" />
          <span className="hidden md:inline truncate">
            Backend Source: SarahMemory on GitHub
          </span>
          <span className="md:hidden">GitHub</span>
          <ExternalLink className="h-3 w-3 shrink-0" />
        </a>
      </div>

      {/* CENTER: TASKBAR ICONS (dynamic order) */}
      <div className="flex items-center justify-center flex-1">
        <div className="flex items-center gap-1 px-3 py-1.5 bg-secondary/50 rounded-xl">
          {taskbarOrder.map((id) => renderTaskbarButton(String(id)))}
        </div>
      </div>

      {/* RIGHT: Clock + Donate + Compact Status */}
      <div className="flex items-center gap-3">
        {/* Clock */}
        <div
          className="hidden sm:flex items-center gap-2 px-2 py-1 rounded-md bg-secondary/30"
          title={now.toLocaleString()}
        >
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground tabular-nums">
            {formatClock(now)}
          </span>
        </div>

        <div className="h-4 w-px bg-border hidden sm:block" />

        {/* Donate Button (ONLY HERE) */}
        <Button
          variant="outline"
          size="sm"
          className="h-8 px-3 text-xs border-primary/30 hover:border-primary hover:bg-primary/10"
          asChild
        >
          <a
            href={config.donateUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5"
          >
            <Heart className="h-3 w-3 text-destructive" />
            <span>Donate</span>
          </a>
        </Button>

        <div className="h-4 w-px bg-border hidden sm:block" />

        {/* Compact Status Pill */}
        <div
          className={cn(
            "flex items-center gap-2 px-2.5 py-1 rounded-md",
            "bg-secondary/30 border border-border",
          )}
          title={statusTitle}
          aria-label="System Status"
        >
          <StatusIcon
            className={cn(
              "h-4 w-4",
              overall.tone === "ok"
                ? "text-status-online"
                : overall.tone === "warn"
                  ? "text-status-warning"
                  : "text-status-error",
              StatusIcon === Loader2 && "animate-spin",
            )}
          />
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            {overall.label}
          </span>
        </div>
      </div>
    </div>
  );
}

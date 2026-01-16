import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  MessageCircle,
  Folder,
  Play,
  Search,
  Palette,
  Cpu,
  Network,
  User,
  Settings,
  Clock,
  Heart,
  ExternalLink,
  Wifi,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface DockItem {
  id: WindowId;
  label: string;
  icon: ReactNode;
}

const DOCK_ITEMS: DockItem[] = [
  { id: "chat", label: "Chat", icon: <MessageCircle className="h-5 w-5" /> },
  { id: "history", label: "History", icon: <Clock className="h-5 w-5" /> },
  { id: "files", label: "Files", icon: <Folder className="h-5 w-5" /> },
  { id: "research", label: "Research", icon: <Search className="h-5 w-5" /> },
  { id: "studio", label: "Studios", icon: <Palette className="h-5 w-5" /> },
  { id: "avatar", label: "Avatar", icon: <User className="h-5 w-5" /> },
  { id: "sarahnet", label: "SarahNet", icon: <Network className="h-5 w-5" /> },
  { id: "media", label: "Media", icon: <Play className="h-5 w-5" /> },
  { id: "dlengine", label: "DL Engine", icon: <Cpu className="h-5 w-5" /> },
  { id: "settings", label: "Settings", icon: <Settings className="h-5 w-5" /> },
];

type RouteMode = "Any" | "Local" | "Web" | "API";
const ROUTE_MODES: RouteMode[] = ["Any", "Local", "Web", "API"];

/**
 * Dock (combined bottom bar)
 *
 * We intentionally keep this as the single bottom bar in DesktopShell:
 * - Left: Mode toggle + backend/source links
 * - Center: Dock icons
 * - Right: status indicator + Donate
 *
 * NOTE: we keep this lightweight; anything backend-specific is best-effort only.
 */
export function Dock() {
  const barRef = useRef<HTMLDivElement>(null);

  const {
    windows,
    openWindow,
    focusWindow,
    restoreWindow,
    focusedWindowId,
  } = useWindowStore();

  // ---------------------------------------------------------------------------
  // Route Mode (Any/Local/Web/API)
  // - Stored in localStorage so it survives reloads.
  // - Broadcast via a CustomEvent so other parts of the app can optionally react.
  // ---------------------------------------------------------------------------
  const [routeMode, setRouteMode] = useState<RouteMode>(() => {
    try {
      const v = (localStorage.getItem("route_mode") || "Any") as RouteMode;
      return ROUTE_MODES.includes(v) ? v : "Any";
    } catch {
      return "Any";
    }
  });

  const nextMode = useMemo(() => {
    const idx = ROUTE_MODES.indexOf(routeMode);
    return ROUTE_MODES[(idx + 1) % ROUTE_MODES.length];
  }, [routeMode]);

  const cycleMode = () => {
    const idx = ROUTE_MODES.indexOf(routeMode);
    const v = ROUTE_MODES[(idx + 1) % ROUTE_MODES.length];

    setRouteMode(v);
    try {
      localStorage.setItem("route_mode", v);
    } catch {}

    // Optional: allow other modules to hook in without hard coupling.
    try {
      window.dispatchEvent(
        new CustomEvent("sarah:route_mode", { detail: { mode: v } })
      );
    } catch {}
  };

  // ---------------------------------------------------------------------------
  // Height -> CSS variable for maximized windows (Window.tsx reads this)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const el = barRef.current;
    if (!el) return;

    const apply = () => {
      const h = el.offsetHeight || 56;
      document.documentElement.style.setProperty("--dock-h", `${h}px`);
    };

    apply();

    // ResizeObserver is best; fallback to window resize.
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
  }, []);

  const handleDockClick = (id: WindowId) => {
    const win = windows.find((w) => w.id === id);
    if (!win) {
      openWindow(id);
    } else if (win.isMinimized) {
      restoreWindow(id);
    } else {
      focusWindow(id);
    }
  };

  // Connection indicator (simple UI-only label; real connectivity lives elsewhere)
  const connectionLabel = "Ready";

  return (
    <div
      ref={barRef}
      className={cn(
        "h-14 bg-card/80 backdrop-blur-md border-t border-border",
        "flex items-center px-3 gap-2"
      )}
    >
      <TooltipProvider delayDuration={200}>
        {/* LEFT: Mode + source links */}
        <div className="min-w-0 flex items-center gap-2 text-xs text-muted-foreground">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={cycleMode}
                className={cn(
                  "inline-flex items-center gap-1.5 px-2 py-1 rounded-md",
                  "bg-secondary/40 hover:bg-secondary/60 transition-colors",
                  "text-foreground/80 hover:text-foreground"
                )}
                aria-label="Toggle route mode"
                title={`Click to switch to ${nextMode}`}
              >
                <span className="font-mono">MODE:</span>
                <span className="font-semibold">{routeMode}</span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              Click to cycle modes: Any → Local → Web → API
            </TooltipContent>
          </Tooltip>

          <div className="hidden md:flex items-center gap-2 min-w-0">
            <span className="text-muted-foreground/70">|</span>
            <a
              href="https://github.com/Brian-Baros/SarahMemory"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:text-foreground transition-colors truncate"
              title="Backend Source: SarahMemory on GitHub"
            >
              <span className="truncate">Backend Source: SarahMemory on GitHub</span>
              <ExternalLink className="h-3 w-3 opacity-70" />
            </a>
            <span className="text-muted-foreground/70">|</span>
            <span className="truncate">The SarahMemory Project by Brian Lee Baros</span>
          </div>
        </div>

        {/* CENTER: Dock icons */}
        <div className="flex-1 flex items-center justify-center">
          <div className="flex items-center gap-1 px-3 py-1.5 bg-secondary/50 rounded-xl">
            {DOCK_ITEMS.map((item) => {
              const isOpen = windows.some((w) => w.id === item.id);
              const isFocused = focusedWindowId === item.id;
              const isMinimized = windows.find((w) => w.id === item.id)?.isMinimized;

              return (
                <Tooltip key={item.id}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => handleDockClick(item.id)}
                      className={cn(
                        "relative p-2.5 rounded-lg transition-all duration-150",
                        "hover:bg-primary/20 hover:scale-110",
                        "active:scale-95",
                        isFocused && "bg-primary/30",
                        isMinimized && "opacity-60"
                      )}
                      aria-label={item.label}
                      title={item.label}
                    >
                      <span
                        className={cn(
                          "text-muted-foreground",
                          isFocused && "text-primary",
                          isOpen && !isFocused && "text-foreground"
                        )}
                      >
                        {item.icon}
                      </span>

                      {/* Open indicator dot */}
                      {isOpen && (
                        <span
                          className={cn(
                            "absolute bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full",
                            isFocused ? "bg-primary" : "bg-muted-foreground"
                          )}
                        />
                      )}
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="text-xs">
                    {item.label}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        </div>

        {/* RIGHT: Status + Donate */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Wifi className="h-4 w-4 opacity-70" />
            <span className="hidden sm:inline">{connectionLabel}</span>
          </div>

          {/* Single Donate button */}
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href="https://patreon.com/sarahmemory"
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  "inline-flex items-center gap-2 px-3 py-2 rounded-lg",
                  "bg-gradient-to-r from-pink-500/20 to-rose-500/20",
                  "hover:from-pink-500/30 hover:to-rose-500/30",
                  "transition-all text-pink-400 hover:text-pink-300"
                )}
              >
                <Heart className="h-4 w-4" />
                <span className="text-xs font-medium">Donate</span>
              </a>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              Support SarahMemory
            </TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>
    </div>
  );
}

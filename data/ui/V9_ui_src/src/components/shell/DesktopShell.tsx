import { useEffect, useMemo, useRef, type CSSProperties } from "react";
import {
  Cpu,
  Folder,
  LayoutGrid,
  MessageCircle,
  MonitorCog,
  Network,
  Palette,
  Search,
  Settings,
  Terminal,
  User,
} from "lucide-react";
import { StatusBar } from "@/components/StatusBar";
import { WindowManager } from "./WindowManager";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { useSarahStore } from "@/stores/useSarahStore";
import { cn } from "@/lib/utils";
import sarahLogoUrl from "@/assets/smaios-logo.jpg";

const DESKTOP_SHORTCUTS: { id: WindowId; label: string; Icon: any }[] = [
  { id: "chat", label: "Chat", Icon: MessageCircle },
  { id: "avatar", label: "Avatar", Icon: User },
  { id: "nailde", label: "NAILDE", Icon: MonitorCog },
  { id: "files", label: "Files", Icon: Folder },
  { id: "research", label: "Research", Icon: Search },
  { id: "dlengine", label: "DL Engine", Icon: Cpu },
  { id: "sarahnet", label: "SarahNet", Icon: Network },
  { id: "studio", label: "Studios", Icon: Palette },
  { id: "terminal", label: "Terminal", Icon: Terminal },
  { id: "settings", label: "Settings", Icon: Settings },
];

const WORKSPACE_PRESETS = [
  { id: "chat", label: "Chat" },
  { id: "research", label: "Research" },
  { id: "operator", label: "Operator" },
  { id: "engineer", label: "Engineer" },
  { id: "media", label: "Media" },
] as const;

function clampNumber(value: any, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.round(parsed)));
}

/**
 * DesktopShell
 *
 * Desktop windowed shell with:
 *  - Custom window manager (draggable, resizable windows)
 *  - Single unified StatusBar acting as the Taskbar
 *
 * Taskbar goals:
 *  - Dockable: bottom (default), top, left, right
 *  - Resizable to add rows (Phase 1: integer rows, 1..4)
 *
 * NOTE:
 *  - WindowManager will be patched next to respect --taskbar-dock and --taskbar-size
 */
export function DesktopShell() {
  const { windows, openWindow, applyWorkspacePreset } = useWindowStore();
  const { settings, updateSettings } = useSarahStore();
  const wallpaperUrl = String((settings as any)?.wallpaperUrl || "").trim();
  const wallpaperMode = String((settings as any)?.wallpaperMode || "cover");
  const panelTransparency = String((settings as any)?.panelTransparency || "glass");
  const shellDensity = String((settings as any)?.shellDensity || "comfortable");
  const backgroundBrightness = clampNumber((settings as any)?.backgroundBrightness, 82, 20, 125);
  const backgroundOverlay = clampNumber((settings as any)?.backgroundOverlay, 42, 0, 85);
  const backgroundBlur = clampNumber((settings as any)?.backgroundBlur, 0, 0, 18);
  const panelOpacity = clampNumber((settings as any)?.panelOpacity, 82, 45, 100);
  const activeWorkspace = String((settings as any)?.activeWorkspace || "chat");
  const safeWallpaperCssUrl = wallpaperUrl.replace(/"/g, '\\"');
  const panelAlpha =
    panelTransparency === "solid"
      ? 1
      : panelTransparency === "translucent"
        ? Math.min(panelOpacity / 100, 0.68)
        : panelOpacity / 100;

  // Open Chat window by default if no windows are open
  useEffect(() => {
    if (windows.length === 0) {
      openWindow("chat");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Taskbar layout state (read from settings.taskbar, with safe defaults)
  const dock = (settings as any)?.taskbar?.dock || "bottom";
  const rows = Number((settings as any)?.taskbar?.rows || 1);

  const clampedRows = useMemo(() => {
    if (!Number.isFinite(rows) || rows < 1) return 1;
    // Phase 1 clamp (adjust later)
    return Math.max(1, Math.min(4, Math.floor(rows)));
  }, [rows]);

  // Constant row size (StatusBar height is ~56px; keep consistent)
  const ROW_PX = 56;

  // For horizontal docks (top/bottom): size is height
  // For vertical docks (left/right): size is width
  const taskbarSizePx = clampedRows * ROW_PX;

  // Publish CSS vars so WindowManager can respect the dock bounds
  useEffect(() => {
    try {
      document.documentElement.style.setProperty("--taskbar-dock", String(dock));
      document.documentElement.style.setProperty("--taskbar-size", `${taskbarSizePx}px`);
    } catch {
      // ignore
    }
  }, [dock, taskbarSizePx]);

  // ---- Resize handle drag logic
  const dragRef = useRef<{
    startX: number;
    startY: number;
    startRows: number;
    active: boolean;
  }>({ startX: 0, startY: 0, startRows: clampedRows, active: false });

  const onStartResize = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startRows: clampedRows,
      active: true,
    };

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current.active) return;

      const dx = ev.clientX - dragRef.current.startX;
      const dy = ev.clientY - dragRef.current.startY;

      // Convert drag distance into row changes
      // bottom: drag up increases rows (negative dy)
      // top: drag down increases rows (positive dy)
      // left: drag right increases rows (positive dx)
      // right: drag left increases rows (negative dx)
      let deltaPx = 0;
      if (dock === "bottom") deltaPx = -dy;
      else if (dock === "top") deltaPx = dy;
      else if (dock === "left") deltaPx = dx;
      else if (dock === "right") deltaPx = -dx;

      const deltaRows = Math.round(deltaPx / ROW_PX);
      const nextRows = Math.max(1, Math.min(4, dragRef.current.startRows + deltaRows));

      // Only write when it actually changes
      if (nextRows !== (settings as any)?.taskbar?.rows) {
        updateSettings({
          taskbar: {
            ...((settings as any)?.taskbar || {}),
            rows: nextRows,
          },
        } as any);
      }
    };

    const onUp = () => {
      dragRef.current.active = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ---- Layout mode
  const isVerticalDock = dock === "left" || dock === "right";
  const shellClass = cn(
    isVerticalDock
      ? "min-h-[100dvh] max-h-[100dvh] flex flex-row overflow-hidden bg-background"
      : "min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background",
    shellDensity === "compact" && "text-[13px]",
    shellDensity === "operator" && "tracking-wide",
  );

  const shellVars = {
    "--panel-bg": `hsl(var(--card) / ${panelAlpha.toFixed(2)})`,
    "--wallpaper-brightness": `${backgroundBrightness}%`,
    "--wallpaper-blur": `${backgroundBlur}px`,
    "--wallpaper-overlay": (backgroundOverlay / 100).toFixed(2),
  } as CSSProperties;

  // Dock container placement order
  const taskbarFirst = dock === "top" || dock === "left";

  const TaskbarContainer = (
    <div
      className={cn(
        "relative shrink-0 bg-card/95 backdrop-blur-sm",
        // borders based on dock side
        dock === "bottom" && "border-t border-border",
        dock === "top" && "border-b border-border",
        dock === "left" && "border-r border-border",
        dock === "right" && "border-l border-border",
      )}
      style={
        isVerticalDock
          ? { width: `${taskbarSizePx}px` }
          : { height: `${taskbarSizePx}px` }
      }
      data-dock={dock}
    >
      {/* Resize handle (Phase 1 rows) */}
      <div
        onMouseDown={onStartResize}
        className={cn(
          "absolute z-50",
          // Make handle sit on the inner edge between workspace + taskbar
          dock === "bottom" && "top-0 left-0 right-0 h-2 cursor-ns-resize",
          dock === "top" && "bottom-0 left-0 right-0 h-2 cursor-ns-resize",
          dock === "left" && "top-0 bottom-0 right-0 w-2 cursor-ew-resize",
          dock === "right" && "top-0 bottom-0 left-0 w-2 cursor-ew-resize",
          // subtle visibility on hover
          "bg-transparent hover:bg-primary/10",
        )}
        title="Drag to resize taskbar (rows)"
        aria-label="Resize taskbar"
        role="separator"
      />

      {/* StatusBar fills the container; internal content remains the same for now */}
      <div className="w-full h-full">
        <StatusBar />
      </div>
    </div>
  );

  return (
    <div className={shellClass} style={shellVars}>
      {/* Docked taskbar first (top/left) */}
      {taskbarFirst && TaskbarContainer}

      {/* Desktop workspace area */}
      <div className="flex-1 relative min-h-0">
        {/* Wallpaper/Background pattern */}
        <div className="absolute inset-0 sarah-desktop-ambient" />
        {wallpaperUrl && (
          <div
            className="absolute inset-0 sarah-desktop-wallpaper"
            style={{
              backgroundImage: `url("${safeWallpaperCssUrl}")`,
              backgroundSize:
                wallpaperMode === "stretch"
                  ? "100% 100%"
                  : wallpaperMode === "tile" || wallpaperMode === "center"
                    ? "auto"
                    : wallpaperMode,
              backgroundRepeat: wallpaperMode === "tile" ? "repeat" : "no-repeat",
              backgroundPosition: "center",
              filter: "brightness(var(--wallpaper-brightness)) blur(var(--wallpaper-blur))",
              transform: backgroundBlur > 0 ? "scale(1.02)" : undefined,
            }}
          />
        )}
        <div className="absolute inset-0 sarah-desktop-grid" />
        <div
          className="absolute inset-0 bg-background"
          style={{ opacity: "var(--wallpaper-overlay)" }}
        />

        <div className="absolute left-4 top-4 z-10 hidden max-h-[calc(100%-2rem)] grid-flow-row auto-rows-min grid-cols-1 gap-2 overflow-hidden md:grid">
          {DESKTOP_SHORTCUTS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => openWindow(id)}
              className="group flex w-[86px] flex-col items-center gap-1 rounded-md border border-transparent px-2 py-2 text-center text-xs text-foreground/85 outline-none transition hover:border-primary/35 hover:bg-card/55 focus:border-primary/60 focus:bg-card/70"
              title={`Open ${label}`}
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-border/65 bg-card/70 shadow-lg shadow-black/20 transition group-hover:border-primary/50 group-hover:text-primary">
                <Icon className="h-5 w-5" />
              </span>
              <span className="line-clamp-2 min-h-[2em] leading-tight drop-shadow">{label}</span>
            </button>
          ))}
        </div>

        <div className="absolute right-4 top-4 z-10 hidden max-w-[min(560px,calc(100%-8rem))] items-center gap-2 rounded-xl border border-border/70 bg-[var(--panel-bg)] px-3 py-2 shadow-xl shadow-black/25 backdrop-blur-xl xl:flex">
          <img src={sarahLogoUrl} alt="" className="h-8 w-8 rounded-lg object-cover" />
          <div className="min-w-0 pr-1">
            <div className="truncate text-xs font-semibold uppercase tracking-[0.18em] text-primary/85">
              Governed Cognitive Workstation
            </div>
            <div className="truncate text-[11px] text-muted-foreground">
              Local-first shell with preserved V9 panels
            </div>
          </div>
          <div className="h-8 w-px bg-border" />
          {WORKSPACE_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              onClick={() => {
                updateSettings({ activeWorkspace: preset.id } as any);
                applyWorkspacePreset(preset.id);
              }}
              className={cn(
                "rounded-lg px-2.5 py-1.5 text-xs font-medium transition",
                activeWorkspace === preset.id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-secondary/80 hover:text-foreground",
              )}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Window Manager */}
        <WindowManager />

        {/* Empty state hint */}
        {windows.filter((w) => !w.isMinimized).length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-muted-foreground/50 text-sm">
              Click an icon in the taskbar to open a window
            </p>
          </div>
        )}
      </div>

      {/* Docked taskbar last (bottom/right) */}
      {!taskbarFirst && TaskbarContainer}
    </div>
  );
}

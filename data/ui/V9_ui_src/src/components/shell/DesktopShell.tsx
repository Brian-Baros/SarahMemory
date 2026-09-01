import { useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import {
  Cpu,
  Clock,
  ExternalLink,
  Folder,
  Grid3X3,
  Link2,
  LayoutGrid,
  MessageCircle,
  MonitorCog,
  Network,
  Palette,
  Play,
  Plus,
  Search,
  Settings,
  Terminal,
  Trash2,
  User,
  X,
  type LucideIcon,
} from "lucide-react";
import { StatusBar } from "@/components/StatusBar";
import { WindowManager } from "./WindowManager";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { useSarahStore } from "@/stores/useSarahStore";
import type { DesktopShortcut } from "@/types/sarah";
import { cn } from "@/lib/utils";
import sarahLogoUrl from "@/assets/smaios-logo.jpg";

const WINDOW_IDS: WindowId[] = [
  "chat",
  "history",
  "files",
  "research",
  "studio",
  "avatar",
  "sarahnet",
  "media",
  "dlengine",
  "nailde",
  "terminal",
  "addons",
  "settings",
];

const DEFAULT_DESKTOP_SHORTCUTS: DesktopShortcut[] = [
  { id: "desktop_chat", label: "Sarah Chat", kind: "app", windowId: "chat", icon: "chat" },
  { id: "desktop_avatar", label: "Avatar Core", kind: "app", windowId: "avatar", icon: "avatar" },
  { id: "desktop_nailde", label: "NAILDE", kind: "app", windowId: "nailde", icon: "nailde" },
  { id: "desktop_files", label: "File Cortex", kind: "app", windowId: "files", icon: "files" },
  { id: "desktop_trash", label: "Recovery Bin", kind: "trash", windowId: "files", icon: "trash" },
  { id: "desktop_research", label: "Evidence Lens", kind: "app", windowId: "research", icon: "research" },
  { id: "desktop_dlengine", label: "Model Forge", kind: "app", windowId: "dlengine", icon: "dlengine" },
  { id: "desktop_sarahnet", label: "SarahNet", kind: "app", windowId: "sarahnet", icon: "sarahnet" },
  { id: "desktop_studio", label: "Creation Bay", kind: "app", windowId: "studio", icon: "studio" },
  { id: "desktop_terminal", label: "Operator Terminal", kind: "app", windowId: "terminal", icon: "terminal" },
  { id: "desktop_settings", label: "System Tuning", kind: "app", windowId: "settings", icon: "settings" },
];

const DESKTOP_ICON_MAP: Record<string, LucideIcon> = {
  chat: MessageCircle,
  avatar: User,
  nailde: MonitorCog,
  files: Folder,
  history: Clock,
  trash: Trash2,
  research: Search,
  dlengine: Cpu,
  sarahnet: Network,
  media: Play,
  studio: Palette,
  addons: Grid3X3,
  terminal: Terminal,
  settings: Settings,
  url: ExternalLink,
  link: Link2,
  custom: LayoutGrid,
};

const WORKSPACE_PRESETS = [
  { id: "chat", label: "Dialogue" },
  { id: "research", label: "Evidence" },
  { id: "operator", label: "Operator" },
  { id: "engineer", label: "Builder" },
  { id: "media", label: "Media" },
] as const;

const FILES_PENDING_KEY = "sarahmemory:files:pending";

function clampNumber(value: any, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.round(parsed)));
}

function isWindowId(value: any): value is WindowId {
  return WINDOW_IDS.includes(String(value) as WindowId);
}

function defaultShortcutPosition(index: number) {
  const row = index % 9;
  const col = Math.floor(index / 9);
  return {
    x: 18 + col * 98,
    y: 18 + row * 88,
  };
}

function normalizeCustomShortcut(input: any): DesktopShortcut | null {
  if (!input || typeof input !== "object") return null;
  const id = String(input.id || "").trim();
  const label = String(input.label || "").trim();
  const kind = String(input.kind || "custom") as DesktopShortcut["kind"];
  if (!id || !label) return null;
  return {
    id,
    label: label.slice(0, 48),
    kind: kind === "app" || kind === "trash" || kind === "url" || kind === "custom" ? kind : "custom",
    windowId: input.windowId ? String(input.windowId) : undefined,
    url: input.url ? String(input.url) : undefined,
    icon: input.icon ? String(input.icon) : undefined,
  };
}

/**
 * DesktopShell owns the SarahMemory windowed workspace and command rail.
 * The persisted settings field is still named taskbar for compatibility with
 * older V9 UI saves, but the visible shell language is SarahMemory-native.
 */
export function DesktopShell() {
  const { windows, openWindow, applyWorkspacePreset } = useWindowStore();
  const { settings, updateSettings } = useSarahStore();
  const workspaceRef = useRef<HTMLDivElement>(null);
  const shortcutDragRef = useRef<{
    id: string;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    moved: boolean;
  } | null>(null);
  const [draggingShortcutId, setDraggingShortcutId] = useState<string | null>(null);
  const [trashDropActive, setTrashDropActive] = useState(false);
  const [shortcutManagerOpen, setShortcutManagerOpen] = useState(false);
  const [newShortcutLabel, setNewShortcutLabel] = useState("");
  const [newShortcutTarget, setNewShortcutTarget] = useState<WindowId | "url">("chat");
  const [newShortcutUrl, setNewShortcutUrl] = useState("");
  const wallpaperUrl = String((settings as any)?.wallpaperUrl || "").trim();
  const wallpaperMode = String((settings as any)?.wallpaperMode || "cover");
  const panelTransparency = String((settings as any)?.panelTransparency || "glass");
  const shellDensity = String((settings as any)?.shellDensity || "comfortable");
  const backgroundBrightness = clampNumber((settings as any)?.backgroundBrightness, 82, 20, 125);
  const backgroundOverlay = clampNumber((settings as any)?.backgroundOverlay, 42, 0, 85);
  const backgroundBlur = clampNumber((settings as any)?.backgroundBlur, 0, 0, 18);
  const panelOpacity = clampNumber((settings as any)?.panelOpacity, 82, 45, 100);
  const activeWorkspace = String((settings as any)?.activeWorkspace || "chat");
  const desktopShortcutPositions = ((settings as any)?.desktopShortcutPositions || {}) as Record<string, { x: number; y: number }>;
  const desktopShortcuts = useMemo(() => {
    const defaultsById = new Set(DEFAULT_DESKTOP_SHORTCUTS.map((shortcut) => shortcut.id));
    const custom = Array.isArray((settings as any)?.desktopShortcuts)
      ? ((settings as any).desktopShortcuts as any[])
          .map(normalizeCustomShortcut)
          .filter((item): item is DesktopShortcut => Boolean(item))
          .filter((item) => !defaultsById.has(item.id))
      : [];
    return [...DEFAULT_DESKTOP_SHORTCUTS, ...custom];
  }, [settings]);
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

  // ---- Command rail layout state (stored under settings.taskbar for V9 compatibility)
  const dock = (settings as any)?.taskbar?.dock || "bottom";
  const rows = Number((settings as any)?.taskbar?.rows || 1);

  const clampedRows = useMemo(() => {
    if (!Number.isFinite(rows) || rows < 1) return 1;
    // Phase 1 clamp (adjust later)
    return Math.max(1, Math.min(4, Math.floor(rows)));
  }, [rows]);

  // Constant rail row size; keep consistent with StatusBar geometry.
  const ROW_PX = 56;

  // For horizontal rails: size is height. For vertical rails: size is width.
  const commandRailSizePx = clampedRows * ROW_PX;

  // Publish CSS vars so WindowManager can respect the rail bounds.
  useEffect(() => {
    try {
      document.documentElement.style.setProperty("--taskbar-dock", String(dock));
      document.documentElement.style.setProperty("--taskbar-size", `${commandRailSizePx}px`);
    } catch {
      // ignore
    }
  }, [dock, commandRailSizePx]);

  // ---- Command rail resize gesture
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

      // Convert drag distance into rail row changes.
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

  const shortcutPosition = (id: string, index: number) => {
    const saved = desktopShortcutPositions[id];
    if (saved) {
      return {
        x: clampNumber(saved.x, 18, 0, 5000),
        y: clampNumber(saved.y, 18, 0, 5000),
      };
    }
    return defaultShortcutPosition(index);
  };

  const setShortcutPosition = (id: string, x: number, y: number) => {
    const bounds = workspaceRef.current?.getBoundingClientRect();
    const maxX = Math.max(0, (bounds?.width || window.innerWidth || 1280) - 100);
    const maxY = Math.max(0, (bounds?.height || window.innerHeight || 720) - 92);
    updateSettings({
      desktopShortcutPositions: {
        ...desktopShortcutPositions,
        [id]: {
          x: clampNumber(x, 18, 0, maxX),
          y: clampNumber(y, 18, 0, maxY),
        },
      },
    } as any);
  };

  const isUserShortcut = (shortcut: DesktopShortcut) => shortcut.id.startsWith("custom_");

  const isPointOverTrashShortcut = (clientX: number, clientY: number) => {
    const bounds = workspaceRef.current?.getBoundingClientRect();
    if (!bounds) return false;
    const trashIndex = desktopShortcuts.findIndex((shortcut) => shortcut.id === "desktop_trash");
    if (trashIndex < 0) return false;
    const trash = shortcutPosition("desktop_trash", trashIndex);
    const localX = clientX - bounds.left;
    const localY = clientY - bounds.top;
    return (
      localX >= trash.x - 14 &&
      localX <= trash.x + 102 &&
      localY >= trash.y - 14 &&
      localY <= trash.y + 106
    );
  };

  const openDesktopShortcut = (shortcut: DesktopShortcut) => {
    if (shortcut.kind === "trash") {
      const action = { type: "files_open_trash", payload: { source: "desktop" } };
      openWindow("files");
      try {
        window.sessionStorage.setItem(FILES_PENDING_KEY, JSON.stringify([action]));
      } catch {
        // optional handoff cache
      }
      window.setTimeout(() => {
        window.dispatchEvent(
          new CustomEvent("sarah:ui", {
            detail: {
              source: "desktop-shortcut",
              actions: [action],
            },
          }),
        );
      }, 80);
      return;
    }

    if (shortcut.kind === "url" && shortcut.url) {
      window.open(shortcut.url, "_blank", "noopener,noreferrer");
      return;
    }

    if (isWindowId(shortcut.windowId)) {
      openWindow(shortcut.windowId);
    }
  };

  const onShortcutPointerDown = (
    e: ReactPointerEvent<HTMLButtonElement>,
    shortcut: DesktopShortcut,
    index: number,
  ) => {
    if (e.button !== 0) return;
    const origin = shortcutPosition(shortcut.id, index);
    shortcutDragRef.current = {
      id: shortcut.id,
      startX: e.clientX,
      startY: e.clientY,
      originX: origin.x,
      originY: origin.y,
      moved: false,
    };
    setDraggingShortcutId(shortcut.id);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onShortcutPointerMove = (e: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = shortcutDragRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
      drag.moved = true;
    }
    if (!drag.moved) return;
    e.preventDefault();
    setShortcutPosition(drag.id, drag.originX + dx, drag.originY + dy);
    const shortcut = desktopShortcuts.find((item) => item.id === drag.id);
    setTrashDropActive(Boolean(shortcut && isUserShortcut(shortcut) && isPointOverTrashShortcut(e.clientX, e.clientY)));
  };

  const onShortcutPointerUp = (e: ReactPointerEvent<HTMLButtonElement>, shortcut: DesktopShortcut) => {
    const drag = shortcutDragRef.current;
    const shouldDeleteShortcut = Boolean(
      drag?.moved &&
        isUserShortcut(shortcut) &&
        isPointOverTrashShortcut(e.clientX, e.clientY),
    );
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // pointer capture may already be released
    }
    shortcutDragRef.current = null;
    setDraggingShortcutId(null);
    setTrashDropActive(false);
    if (shouldDeleteShortcut) {
      removeDesktopShortcut(shortcut.id);
      return;
    }
    if (!drag?.moved) openDesktopShortcut(shortcut);
  };

  const onShortcutPointerCancel = (e: ReactPointerEvent<HTMLButtonElement>) => {
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // pointer capture may already be released
    }
    shortcutDragRef.current = null;
    setDraggingShortcutId(null);
    setTrashDropActive(false);
  };

  const addDesktopShortcut = () => {
    const label = newShortcutLabel.trim();
    const isUrl = newShortcutTarget === "url";
    const url = newShortcutUrl.trim();
    if (!label) return;
    if (isUrl && !url) return;

    const nextShortcut: DesktopShortcut = {
      id: `custom_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      label,
      kind: isUrl ? "url" : "app",
      windowId: isUrl ? undefined : newShortcutTarget,
      url: isUrl ? url : undefined,
      icon: isUrl ? "url" : newShortcutTarget,
    };

    updateSettings({
      desktopShortcuts: [...(((settings as any)?.desktopShortcuts || []) as DesktopShortcut[]), nextShortcut],
      desktopShortcutPositions: {
        ...desktopShortcutPositions,
        [nextShortcut.id]: defaultShortcutPosition(desktopShortcuts.length),
      },
    } as any);
    setNewShortcutLabel("");
    setNewShortcutUrl("");
    setNewShortcutTarget("chat");
  };

  const removeDesktopShortcut = (id: string) => {
    updateSettings({
      desktopShortcuts: (((settings as any)?.desktopShortcuts || []) as DesktopShortcut[]).filter(
        (shortcut) => shortcut.id !== id,
      ),
      desktopShortcutPositions: Object.fromEntries(
        Object.entries(desktopShortcutPositions).filter(([key]) => key !== id),
      ),
    } as any);
  };

  const resetDesktopLayout = () => {
    updateSettings({ desktopShortcutPositions: {} } as any);
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
  const commandRailFirst = dock === "top" || dock === "left";

  const CommandRailContainer = (
    <div
      className={cn(
        "relative shrink-0 sarah-material",
        // borders based on dock side
        dock === "bottom" && "border-t border-border",
        dock === "top" && "border-b border-border",
        dock === "left" && "border-r border-border",
        dock === "right" && "border-l border-border",
      )}
      style={
        isVerticalDock
          ? { width: `${commandRailSizePx}px` }
          : { height: `${commandRailSizePx}px` }
      }
      data-dock={dock}
    >
      {/* Resize handle (Phase 1 rows) */}
      <div
        onMouseDown={onStartResize}
        className={cn(
          "absolute z-50",
          // Make handle sit on the inner edge between workspace and command rail.
          dock === "bottom" && "top-0 left-0 right-0 h-2 cursor-ns-resize",
          dock === "top" && "bottom-0 left-0 right-0 h-2 cursor-ns-resize",
          dock === "left" && "top-0 bottom-0 right-0 w-2 cursor-ew-resize",
          dock === "right" && "top-0 bottom-0 left-0 w-2 cursor-ew-resize",
          // subtle visibility on hover
          "bg-transparent hover:bg-primary/10",
        )}
        title="Drag to resize command rail"
        aria-label="Resize command rail"
        role="separator"
      />

      {/* StatusBar fills the command rail container. */}
      <div className="w-full h-full">
        <StatusBar />
      </div>
    </div>
  );

  return (
    <div className={shellClass} style={shellVars}>
      {/* Docked command rail first (top/left). */}
      {commandRailFirst && CommandRailContainer}

      {/* Desktop workspace area */}
      <div ref={workspaceRef} className="sarah-orbit-field flex-1 relative min-h-0 overflow-hidden">
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

        <div className="pointer-events-none absolute inset-0 z-10 hidden md:block" aria-label="Desktop shortcuts">
          {desktopShortcuts.map((shortcut, index) => {
            const Icon = DESKTOP_ICON_MAP[shortcut.icon || shortcut.kind] || DESKTOP_ICON_MAP.custom;
            const position = shortcutPosition(shortcut.id, index);
            const isDragging = draggingShortcutId === shortcut.id;
            const canRemove = isUserShortcut(shortcut);
            const isTrashTarget = shortcut.id === "desktop_trash" && trashDropActive;

            return (
              <div
                key={shortcut.id}
                className="pointer-events-auto absolute"
                style={{ left: position.x, top: position.y }}
              >
                <button
                  type="button"
                  onPointerDown={(e) => onShortcutPointerDown(e, shortcut, index)}
                  onPointerMove={onShortcutPointerMove}
                  onPointerUp={(e) => onShortcutPointerUp(e, shortcut)}
                  onPointerCancel={onShortcutPointerCancel}
                  className={cn(
                    "sarah-focus-ring group flex w-[88px] touch-none select-none flex-col items-center gap-1 rounded-md border border-transparent px-2 py-2 text-center text-xs text-foreground/85 transition",
                    "hover:border-primary/35 hover:bg-card/55 focus:border-primary/60 focus:bg-card/70",
                    isDragging && "scale-[1.03] border-primary/55 bg-card/70 shadow-xl shadow-primary/10",
                    isTrashTarget && "scale-[1.04] border-destructive/70 bg-destructive/20 text-destructive",
                  )}
                  title={
                    canRemove
                      ? `${shortcut.label} - drag to move, drop on Recovery Bin to delete`
                      : `${shortcut.label} - drag to move`
                  }
                  aria-label={`Open ${shortcut.label}`}
                  aria-grabbed={isDragging}
                >
                  <span className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border/65 bg-card/70 shadow-lg shadow-black/20 transition group-hover:border-primary/50 group-hover:text-primary">
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="line-clamp-2 min-h-[2em] leading-tight drop-shadow">{shortcut.label}</span>
                </button>
                {canRemove && (
                  <button
                    type="button"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      removeDesktopShortcut(shortcut.id);
                    }}
                    className="sarah-focus-ring absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-lg transition hover:border-destructive/70 hover:bg-destructive hover:text-destructive-foreground"
                    title={`Delete ${shortcut.label}`}
                    aria-label={`Delete ${shortcut.label}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            );
          })}
        </div>

        <div className="absolute bottom-4 right-4 z-20 hidden flex-col items-end gap-2 md:flex">
          {shortcutManagerOpen && (
            <div className="sarah-material w-[340px] rounded-xl p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-semibold">Shortcut Forge</div>
                  <div className="text-xs text-muted-foreground">Bind apps or URLs to the workspace field.</div>
                  <span className="sr-only">Desktop Shortcuts Add Shortcut</span>
                </div>
                <button
                  type="button"
                  onClick={() => setShortcutManagerOpen(false)}
                  className="rounded-lg border border-border/70 p-1.5 text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
                  aria-label="Close shortcut forge"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-2">
                <input
                  value={newShortcutLabel}
                  onChange={(e) => setNewShortcutLabel(e.target.value)}
                  placeholder="Shortcut name"
                  className="h-9 w-full rounded-lg border border-border bg-background/70 px-3 text-sm outline-none focus:border-primary/70"
                />
                <select
                  value={newShortcutTarget}
                  onChange={(e) => setNewShortcutTarget(e.target.value as WindowId | "url")}
                  className="h-9 w-full rounded-lg border border-border bg-background/70 px-3 text-sm outline-none focus:border-primary/70"
                >
                  {WINDOW_IDS.map((id) => (
                    <option key={id} value={id}>
                      Open {id === "studio" ? "Creation Bay" : id}
                    </option>
                  ))}
                  <option value="url">Open URL</option>
                </select>
                {newShortcutTarget === "url" && (
                  <input
                    value={newShortcutUrl}
                    onChange={(e) => setNewShortcutUrl(e.target.value)}
                    placeholder="https://example.com or /local/path"
                    className="h-9 w-full rounded-lg border border-border bg-background/70 px-3 text-sm outline-none focus:border-primary/70"
                  />
                )}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={addDesktopShortcut}
                    disabled={!newShortcutLabel.trim() || (newShortcutTarget === "url" && !newShortcutUrl.trim())}
                    className="flex h-9 flex-1 items-center justify-center gap-2 rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Plus className="h-4 w-4" />
                    Bind Shortcut
                  </button>
                  <button
                    type="button"
                    onClick={resetDesktopLayout}
                    className="h-9 rounded-lg border border-border px-3 text-sm text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {Array.isArray((settings as any)?.desktopShortcuts) && (settings as any).desktopShortcuts.length > 0 && (
                <div className="mt-3 border-t border-border/60 pt-3">
                  <div className="mb-2 text-xs font-medium text-muted-foreground">User bindings</div>
                  <div className="max-h-32 space-y-1 overflow-auto pr-1">
                    {((settings as any).desktopShortcuts as DesktopShortcut[]).map((shortcut) => (
                      <div key={shortcut.id} className="flex items-center justify-between gap-2 rounded-lg border border-border/50 bg-background/45 px-2 py-1.5">
                        <span className="truncate text-xs">{shortcut.label}</span>
                        <button
                          type="button"
                          onClick={() => removeDesktopShortcut(shortcut.id)}
                          className="rounded-md p-1 text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
                          aria-label={`Remove ${shortcut.label}`}
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <button
            type="button"
            onClick={() => setShortcutManagerOpen((open) => !open)}
            className="sarah-focus-ring flex h-10 items-center gap-2 rounded-xl border border-border/70 bg-[var(--panel-bg)] px-3 text-sm font-medium shadow-xl shadow-black/25 backdrop-blur-xl transition hover:border-primary/45 hover:text-primary"
          >
            <Plus className="h-4 w-4" />
            Shortcut Forge
          </button>
        </div>

        <div className="sarah-shell-ribbon absolute right-4 top-4 z-10 hidden max-w-[min(640px,calc(100%-8rem))] items-center gap-2 rounded-xl px-3 py-2 xl:flex">
          <img src={sarahLogoUrl} alt="" className="h-8 w-8 rounded-lg object-cover" />
          <div className="min-w-0 pr-1">
            <div className="truncate text-xs font-semibold uppercase tracking-[0.18em] text-primary/85">
              SarahMemory Workspace Compass
            </div>
            <div className="truncate text-[11px] text-muted-foreground">
              Local-first interface fabric with governed V9 panels
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
                "sarah-focus-ring rounded-lg px-2.5 py-1.5 text-xs font-medium transition",
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
              Open a surface from the command rail or desktop field.
            </p>
          </div>
        )}
      </div>

      {/* Docked command rail last (bottom/right). */}
      {!commandRailFirst && CommandRailContainer}
    </div>
  );
}

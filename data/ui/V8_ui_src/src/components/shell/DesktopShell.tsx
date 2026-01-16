import { useEffect, useMemo, useRef } from "react";
import { StatusBar } from "@/components/StatusBar";
import { WindowManager } from "./WindowManager";
import { useWindowStore } from "@/stores/useWindowStore";
import { useSarahStore } from "@/stores/useSarahStore";
import { cn } from "@/lib/utils";

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
  const { windows, openWindow } = useWindowStore();
  const { settings, updateSettings } = useSarahStore();

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
  const shellClass = isVerticalDock
    ? "min-h-[100dvh] max-h-[100dvh] flex flex-row overflow-hidden bg-background"
    : "min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background";

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
    <div className={shellClass}>
      {/* Docked taskbar first (top/left) */}
      {taskbarFirst && TaskbarContainer}

      {/* Desktop workspace area */}
      <div className="flex-1 relative min-h-0">
        {/* Wallpaper/Background pattern */}
        <div className="absolute inset-0 bg-gradient-to-br from-background via-muted/20 to-background">
          <div
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, currentColor 1px, transparent 0)`,
              backgroundSize: "32px 32px",
            }}
          />
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

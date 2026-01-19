import { useRef, useCallback, useEffect, useState, ReactNode } from "react";
import { X, Minus, Square, Maximize2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWindowStore, type WindowState } from "@/stores/useWindowStore";

interface WindowProps {
  window: WindowState;
  children: ReactNode;
}

export function Window({ window: win, children }: WindowProps) {
  const {
    focusWindow,
    closeWindow,
    minimizeWindow,
    maximizeWindow,
    restoreWindow,
    moveWindow,
    resizeWindow,
    focusedWindowId,
  } = useWindowStore();

  const windowRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  });

  const isFocused = focusedWindowId === win.id;

  // Handle drag start
  const handleDragStart = useCallback(
    (e: React.MouseEvent) => {
      if (win.isMaximized) return;
      e.preventDefault();
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - win.x,
        y: e.clientY - win.y,
      });
      focusWindow(win.id);
    },
    [win.x, win.y, win.isMaximized, win.id, focusWindow],
  );

  // Handle resize start
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      if (win.isMaximized) return;
      e.preventDefault();
      e.stopPropagation();
      setIsResizing(true);
      setResizeStart({
        x: e.clientX,
        y: e.clientY,
        width: win.width,
        height: win.height,
      });
      focusWindow(win.id);
    },
    [win.width, win.height, win.isMaximized, win.id, focusWindow],
  );

  // Handle mouse move for drag/resize
  useEffect(() => {
    if (!isDragging && !isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const newX = Math.max(0, e.clientX - dragOffset.x);
        const newY = Math.max(0, e.clientY - dragOffset.y);
        moveWindow(win.id, newX, newY);
      }
      if (isResizing) {
        const deltaX = e.clientX - resizeStart.x;
        const deltaY = e.clientY - resizeStart.y;
        resizeWindow(win.id, resizeStart.width + deltaX, resizeStart.height + deltaY);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, isResizing, dragOffset, resizeStart, win.id, moveWindow, resizeWindow]);

  // Handle double-click to maximize/restore
  const handleTitleDoubleClick = useCallback(() => {
    if (win.isMaximized) {
      restoreWindow(win.id);
    } else {
      maximizeWindow(win.id);
    }
  }, [win.id, win.isMaximized, restoreWindow, maximizeWindow]);

  // Don't render minimized windows
  if (win.isMinimized) return null;

  /**
   * IMPORTANT:
   * WindowManager now defines the "workspace" bounds and excludes the taskbar area.
   * Therefore a maximized window should fill the workspace (100% x 100%) with no extra subtraction.
   */
  const windowStyle = win.isMaximized
    ? {
        left: 0,
        top: 0,
        width: "100%",
        height: "100%",
        zIndex: win.zIndex,
      }
    : {
        left: win.x,
        top: win.y,
        width: win.width,
        height: win.height,
        zIndex: win.zIndex,
      };

  return (
    <div
      ref={windowRef}
      className={cn(
        "absolute flex flex-col rounded-lg overflow-hidden shadow-2xl",
        "bg-card border border-border",
        "transition-shadow duration-150",
        isFocused ? "ring-2 ring-primary/50 shadow-primary/10" : "opacity-95",
        isDragging && "cursor-grabbing",
        isResizing && "cursor-se-resize",
      )}
      style={windowStyle}
      onMouseDown={() => focusWindow(win.id)}
    >
      {/* Title Bar */}
      <div
        className={cn(
          "h-9 flex items-center justify-between px-3 shrink-0",
          "bg-muted/80 border-b border-border",
          "select-none",
          !win.isMaximized && "cursor-grab",
        )}
        onMouseDown={handleDragStart}
        onDoubleClick={handleTitleDoubleClick}
      >
        <span className="text-sm font-medium text-foreground truncate">{win.title}</span>

        <div className="flex items-center gap-1">
          {/* Minimize */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              minimizeWindow(win.id);
            }}
            className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Minimize"
            title="Minimize"
          >
            <Minus className="h-3.5 w-3.5" />
          </button>

          {/* Maximize/Restore */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              win.isMaximized ? restoreWindow(win.id) : maximizeWindow(win.id);
            }}
            className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
            aria-label={win.isMaximized ? "Restore" : "Maximize"}
            title={win.isMaximized ? "Restore" : "Maximize"}
          >
            {win.isMaximized ? <Square className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>

          {/* Close */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              closeWindow(win.id);
            }}
            className="p-1 rounded hover:bg-destructive hover:text-destructive-foreground text-muted-foreground transition-colors"
            aria-label="Close"
            title="Close"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Window Content */}
      <div className="flex-1 min-h-0 overflow-auto">{children}</div>

      {/* Resize Handle (bottom-right corner) */}
      {!win.isMaximized && (
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize z-50"
          onMouseDown={handleResizeStart}
          aria-label="Resize"
          title="Resize"
        >
          <svg className="w-full h-full text-muted-foreground/50" viewBox="0 0 16 16">
            <path d="M14 14L8 14L14 8L14 14Z" fill="currentColor" />
          </svg>
        </div>
      )}
    </div>
  );
}

import { useEffect } from "react";
import { StatusBar } from "@/components/StatusBar";
import { WindowManager } from "./WindowManager";
import { Dock } from "./Dock";
import { useWindowStore } from "@/stores/useWindowStore";

/**
 * DesktopShell
 *
 * Desktop windowed shell with:
 *  - Custom window manager (draggable, resizable windows)
 *  - Dock/taskbar at bottom
 *  - Wallpaper/workspace area
 *
 * This is the "AiOS Desktop Shell" - a lightweight OS-like experience.
 */
export function DesktopShell() {
  const { windows, openWindow } = useWindowStore();

  // Open Chat window by default if no windows are open
  useEffect(() => {
    if (windows.length === 0) {
      openWindow("chat");
    }
  }, []);

  return (
    <div className="min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background">
      {/* Desktop workspace area */}
      <div className="flex-1 relative min-h-0">
        {/* Wallpaper/Background pattern */}
        <div className="absolute inset-0 bg-gradient-to-br from-background via-muted/20 to-background">
          <div 
            className="absolute inset-0 opacity-5"
            style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, currentColor 1px, transparent 0)`,
              backgroundSize: '32px 32px',
            }}
          />
        </div>

        {/* Window Manager */}
        <WindowManager />
        
        {/* Empty state hint */}
        {windows.filter(w => !w.isMinimized).length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-muted-foreground/50 text-sm">
              Click an icon in the dock to open a window
            </p>
          </div>
        )}
      </div>

      {/* Status bar */}
      <StatusBar />

      {/* Dock */}
      <Dock />
    </div>
  );
}

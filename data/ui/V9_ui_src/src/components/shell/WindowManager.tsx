import { useEffect, useRef, type CSSProperties } from "react";
import { useWindowStore } from "@/stores/useWindowStore";
import { Window } from "./Window";
import { getFeatureComponent } from "@/features/featureRegistry";

export function WindowManager() {
  const { windows, moveWindow, resizeWindow } = useWindowStore();
  const workspaceRef = useRef<HTMLDivElement>(null);

  const workspaceStyle: CSSProperties = {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    overflow: "hidden",
  };

  useEffect(() => {
    const syncWindowBounds = () => {
      const bounds = workspaceRef.current?.getBoundingClientRect();
      if (!bounds) return;
      for (const win of windows) {
        if (win.isMaximized) continue;
        const nextWidth = Math.max(320, Math.min(win.width, bounds.width));
        const nextHeight = Math.max(240, Math.min(win.height, bounds.height));
        const nextX = Math.max(0, Math.min(win.x, Math.max(0, bounds.width - nextWidth)));
        const nextY = Math.max(0, Math.min(win.y, Math.max(0, bounds.height - nextHeight)));
        if (nextWidth !== win.width || nextHeight !== win.height) {
          resizeWindow(win.id, nextWidth, nextHeight);
        }
        if (nextX !== win.x || nextY !== win.y) {
          moveWindow(win.id, nextX, nextY);
        }
      }
    };

    syncWindowBounds();
    window.addEventListener("resize", syncWindowBounds);
    window.addEventListener("orientationchange", syncWindowBounds);
    return () => {
      window.removeEventListener("resize", syncWindowBounds);
      window.removeEventListener("orientationchange", syncWindowBounds);
    };
  }, [moveWindow, resizeWindow, windows]);

  return (
    <div ref={workspaceRef} style={workspaceStyle}>
      {windows.map((w) => (
        <Window key={w.id} window={w}>
          {getFeatureComponent(w.id)}
        </Window>
      ))}
    </div>
  );
}

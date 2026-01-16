import { useEffect, useState } from "react";
import { useWindowStore } from "@/stores/useWindowStore";
import { Window } from "./Window";

import { ChatPanel } from "@/components/chat/ChatPanel";
import { HistoryScreen } from "@/components/screens/HistoryScreen";
import { FilesScreen } from "@/components/screens/FilesScreen";
import { ResearchScreen } from "@/components/screens/ResearchScreen";
import { StudiosScreen } from "@/components/screens/StudiosScreen";
import { AvatarScreen } from "@/components/screens/AvatarScreen";
import { SarahNetScreen } from "@/components/screens/SarahNetScreen";
import { MediaScreen } from "@/components/screens/MediaScreen";
import { DLEngineScreen } from "@/components/screens/DLEngineScreen";
import { SettingsScreen } from "@/components/screens/SettingsScreen"; // ✅

// NOTE: AddonsScreen will be implemented next; for now render a minimal placeholder
const AddonsPlaceholder = () => (
  <div className="p-4 text-sm text-muted-foreground">
    Addons launcher coming next… (dynamic from ../data/addons)
  </div>
);

const WINDOW_CONTENT: Record<string, JSX.Element> = {
  chat: <ChatPanel />,
  history: <HistoryScreen />,
  files: <FilesScreen />,
  research: <ResearchScreen />,
  studio: <StudiosScreen />,
  avatar: <AvatarScreen />,
  sarahnet: <SarahNetScreen />,
  media: <MediaScreen />,
  dlengine: <DLEngineScreen />,
  addons: <AddonsPlaceholder />, // ✅ prevent undefined render
  settings: <SettingsScreen />, // ✅
};

type Dock = "bottom" | "top" | "left" | "right";

export function WindowManager() {
  const { windows } = useWindowStore();

  const [dock, setDock] = useState<Dock>("bottom");

  // Read the dock value DesktopShell publishes on <html>
  useEffect(() => {
    const readDock = () => {
      try {
        const v = getComputedStyle(document.documentElement)
          .getPropertyValue("--taskbar-dock")
          .trim() as Dock;

        if (v === "top" || v === "left" || v === "right" || v === "bottom") {
          setDock(v);
        } else {
          setDock("bottom");
        }
      } catch {
        setDock("bottom");
      }
    };

    readDock();

    // In case dock changes via settings while app is open,
    // we can re-check on resize + a small interval (cheap, safe).
    window.addEventListener("resize", readDock);
    const t = window.setInterval(readDock, 750);

    return () => {
      window.removeEventListener("resize", readDock);
      window.clearInterval(t);
    };
  }, []);

  // Workspace bounds: taskbar size is published as --taskbar-size
  const workspaceStyle: React.CSSProperties = {
    position: "absolute",
    top: dock === "top" ? "var(--taskbar-size, 56px)" : 0,
    bottom: dock === "bottom" ? "var(--taskbar-size, 56px)" : 0,
    left: dock === "left" ? "var(--taskbar-size, 56px)" : 0,
    right: dock === "right" ? "var(--taskbar-size, 56px)" : 0,
    overflow: "hidden",
  };

  return (
    <div style={workspaceStyle}>
      {windows.map((w) => (
        <Window key={w.id} window={w}>
          {WINDOW_CONTENT[w.id] ?? null}
        </Window>
      ))}
    </div>
  );
}

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
import { SettingsScreen } from "@/components/screens/SettingsScreen";
import { AddonsScreen } from "@/components/screens/AddonsScreen";
import TerminalScreen from "@/components/screens/TerminalScreen";

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
  terminal: <TerminalScreen />,
  addons: <AddonsScreen />,
  settings: <SettingsScreen />,
};

type Dock = "bottom" | "top" | "left" | "right";

export function WindowManager() {
  const { windows } = useWindowStore();

  const [dock, setDock] = useState<Dock>("bottom");

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

    window.addEventListener("resize", readDock);
    const t = window.setInterval(readDock, 750);

    return () => {
      window.removeEventListener("resize", readDock);
      window.clearInterval(t);
    };
  }, []);

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
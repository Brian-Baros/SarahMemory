import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { Window } from "./Window";

// Screen components (reused from mobile)
import { ChatPanel } from "@/components/chat/ChatPanel";
import { HistoryScreen } from "@/components/screens/HistoryScreen";
import { StudiosScreen } from "@/components/screens/StudiosScreen";
import { AvatarScreen } from "@/components/screens/AvatarScreen";
import { SarahNetScreen } from "@/components/screens/SarahNetScreen";
import { ResearchScreen } from "@/components/screens/ResearchScreen";
import { SettingsScreen } from "@/components/screens/SettingsScreen";
import { FilesScreen } from "@/components/screens/FilesScreen";
import { MediaScreen } from "@/components/screens/MediaScreen";
import { DLEngineScreen } from "@/components/screens/DLEngineScreen";

// Map window IDs to their content components
const WINDOW_CONTENT: Record<WindowId, React.ReactNode> = {
  chat: <ChatPanel />,
  history: <HistoryScreen />,
  studio: <StudiosScreen />,
  avatar: <AvatarScreen />,
  sarahnet: <SarahNetScreen />,
  research: <ResearchScreen />,
  settings: <SettingsScreen />,
  files: <FilesScreen />,
  media: <MediaScreen />,
  dlengine: <DLEngineScreen />,
};

export function WindowManager() {
  const { windows } = useWindowStore();

  return (
    <div className="absolute inset-0 overflow-hidden">
      {windows.map((win) => (
        <Window key={win.id} window={win}>
          {WINDOW_CONTENT[win.id]}
        </Window>
      ))}
    </div>
  );
}

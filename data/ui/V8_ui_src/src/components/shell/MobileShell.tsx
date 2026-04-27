import { useEffect } from "react";
import { useSwipeGesture } from "@/hooks/useSwipeGesture";
import { useNavigationStore } from "@/stores/useNavigationStore";
import { cn } from "@/lib/utils";

// Screen components
import { ChatPanel } from "@/components/chat/ChatPanel";
import { HistoryScreen } from "@/components/screens/HistoryScreen";
import { StudiosScreen } from "@/components/screens/StudiosScreen";
import { AvatarScreen } from "@/components/screens/AvatarScreen";
import { SarahNetScreen } from "@/components/screens/SarahNetScreen";
import { ResearchScreen } from "@/components/screens/ResearchScreen";
import { FilesScreen } from "@/components/screens/FilesScreen";
import { MediaScreen } from "@/components/screens/MediaScreen";
import { DLEngineScreen } from "@/components/screens/DLEngineScreen";
import TerminalScreen from "@/components/screens/TerminalScreen";

interface MobileShellProps {
  className?: string;
}

/**
 * Mobile shell container with swipe navigation
 * Handles screen transitions and gesture navigation
 */
export function MobileShell({ className }: MobileShellProps) {
  const { currentScreen, swipeLeft, swipeRight } = useNavigationStore();

  const swipeHandlers = useSwipeGesture({
    onSwipeLeft: swipeRight,
    onSwipeRight: swipeLeft,
    threshold: 75,
  });

  const renderScreen = () => {
    switch (currentScreen) {
      case "history":
        return <HistoryScreen />;
      case "studios":
        return <StudiosScreen />;
      case "chat":
        return <ChatPanel />;
      case "avatar":
        return <AvatarScreen />;
      case "sarahnet":
        return <SarahNetScreen />;
      case "research":
        return <ResearchScreen />;
      case "files":
        return <FilesScreen />;
      case "media":
        return <MediaScreen />;
      case "dlengine":
        return <DLEngineScreen />;
      case "terminal":
        return <TerminalScreen />;
      case "settings":
        return <ChatPanel />;
      default:
        return <ChatPanel />;
    }
  };

  return (
    <div className={cn("flex-1 flex flex-col min-h-0 overflow-hidden", className)} {...swipeHandlers}>
      <div className="flex-1 min-h-0 animate-fade-in">{renderScreen()}</div>
    </div>
  );
}
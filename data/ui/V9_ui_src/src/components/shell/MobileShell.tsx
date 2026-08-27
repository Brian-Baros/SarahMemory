import { useEffect } from "react";
import { Bot, Camera, Files, MessageCircle, Mic, MonitorCog, Settings, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSwipeGesture } from "@/hooks/useSwipeGesture";
import { useNavigationStore } from "@/stores/useNavigationStore";
import { useSarahStore } from "@/stores/useSarahStore";
import { cn } from "@/lib/utils";
import { getFeatureComponent } from "@/features/featureRegistry";
import { useViewportProfile } from "@/hooks/use-mobile";

interface MobileShellProps {
  className?: string;
}

/**
 * Mobile shell container with swipe navigation
 * Handles screen transitions and gesture navigation
 */
export function MobileShell({ className }: MobileShellProps) {
  const { currentScreen, setCurrentScreen, swipeLeft, swipeRight } = useNavigationStore();
  const { mediaState, toggleMicrophone } = useSarahStore();
  const viewport = useViewportProfile();

  const swipeHandlers = useSwipeGesture({
    onSwipeLeft: swipeRight,
    onSwipeRight: swipeLeft,
    threshold: 75,
  });

  return (
    <div className={cn("flex-1 flex flex-col min-h-0 overflow-hidden", className)} {...swipeHandlers}>
      {viewport.isPortrait && (
        <div className="shrink-0 border-b border-border bg-card/75 px-2 py-2 backdrop-blur-xl">
          <div className="flex gap-2 overflow-x-auto overscroll-x-contain pb-0.5">
            {[
              { label: "Chat", icon: MessageCircle, action: () => setCurrentScreen("chat") },
              { label: "Vision", icon: Camera, action: () => window.open("/vision", "_self") },
              { label: "Voice", icon: Mic, action: toggleMicrophone, active: mediaState.microphoneEnabled },
              { label: "Files", icon: Files, action: () => setCurrentScreen("files") },
              { label: "Avatar", icon: User, action: () => setCurrentScreen("avatar") },
              { label: "NAILDE", icon: MonitorCog, action: () => setCurrentScreen("nailde") },
              { label: "Models", icon: Bot, action: () => setCurrentScreen("dlengine") },
              { label: "Settings", icon: Settings, action: () => setCurrentScreen("settings") },
            ].map((item) => {
              const Icon = item.icon;
              const isActive =
                ("active" in item && item.active) ||
                currentScreen.toLowerCase() === item.label.toLowerCase() ||
                (item.label === "Models" && currentScreen === "dlengine");
              return (
                <Button
                  key={item.label}
                  type="button"
                  variant={isActive ? "default" : "outline"}
                  size="sm"
                  onClick={item.action}
                  className="h-12 min-w-[74px] flex-col gap-1 rounded-xl px-2 text-[11px]"
                  title={item.label === "Vision" ? "Open Camera Vision object-recognition HUD" : item.label}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Button>
              );
            })}
          </div>
        </div>
      )}
      <div className="flex-1 min-h-0 animate-fade-in">{getFeatureComponent(currentScreen)}</div>
    </div>
  );
}

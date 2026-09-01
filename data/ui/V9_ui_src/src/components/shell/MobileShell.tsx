import { useState } from "react";
import {
  Bot,
  Camera,
  Clock,
  Files,
  Grid3X3,
  LayoutGrid,
  MessageCircle,
  Mic,
  MonitorCog,
  Network,
  Palette,
  Play,
  Search,
  Settings,
  Terminal,
  User,
  Volume2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AudioMixerPanel } from "@/components/panels/audio-mixer/AudioMixerPanel";
import { useSwipeGesture } from "@/hooks/useSwipeGesture";
import { useNavigationStore, type MobileScreen } from "@/stores/useNavigationStore";
import { useSarahStore } from "@/stores/useSarahStore";
import { cn } from "@/lib/utils";
import { getFeatureComponent, SHELL_FEATURES, type ShellFeatureDefinition } from "@/features/featureRegistry";
import { useViewportProfile } from "@/hooks/use-mobile";

interface MobileShellProps {
  className?: string;
}

/**
 * Mobile shell container with SarahMemory swipe-field navigation.
 */
export function MobileShell({ className }: MobileShellProps) {
  const { currentScreen, setCurrentScreen, swipeLeft, swipeRight } = useNavigationStore();
  const { mediaState, toggleMicrophone } = useSarahStore();
  const viewport = useViewportProfile();
  const [appLauncherOpen, setAppLauncherOpen] = useState(false);
  const [audioMixerOpen, setAudioMixerOpen] = useState(false);

  const swipeHandlers = useSwipeGesture({
    onSwipeLeft: swipeRight,
    onSwipeRight: swipeLeft,
    threshold: 75,
  });

  const featureToMobileScreen = (feature: ShellFeatureDefinition): MobileScreen => {
    return (feature.id === "studio" ? "studios" : feature.id) as MobileScreen;
  };

  const featureIcon = (feature: ShellFeatureDefinition) => {
    const iconMap: Record<string, any> = {
      chat: MessageCircle,
      history: Clock,
      files: Files,
      research: Search,
      studio: Palette,
      avatar: User,
      sarahnet: Network,
      media: Play,
      dlengine: Bot,
      nailde: MonitorCog,
      terminal: Terminal,
      addons: LayoutGrid,
      settings: Settings,
    };
    return iconMap[feature.id] || Grid3X3;
  };

  return (
    <div className={cn("flex-1 flex flex-col min-h-0 overflow-hidden", className)} {...swipeHandlers}>
      {viewport.isPortrait && (
        <div className="shrink-0 border-b border-border bg-card/75 px-2 py-2 backdrop-blur-xl">
          <div className="flex gap-2 overflow-x-auto overscroll-x-contain pb-0.5">
            {[
              { label: "Chat", icon: MessageCircle, action: () => setCurrentScreen("chat") },
              { label: "Vision", icon: Camera, action: () => window.open("/vision", "_self") },
              { label: "Voice", icon: Mic, action: toggleMicrophone, active: mediaState.microphoneEnabled },
              { label: "Audio", icon: Volume2, action: () => setAudioMixerOpen((open) => !open), active: audioMixerOpen },
              { label: "Files", icon: Files, action: () => setCurrentScreen("files") },
              { label: "Avatar", icon: User, action: () => setCurrentScreen("avatar") },
              { label: "NAILDE", icon: MonitorCog, action: () => setCurrentScreen("nailde") },
              { label: "Models", icon: Bot, action: () => setCurrentScreen("dlengine") },
              { label: "Nexus", icon: Grid3X3, action: () => setAppLauncherOpen((open) => !open), active: appLauncherOpen, title: "Apps" },
              { label: "Tuning", icon: Settings, action: () => setCurrentScreen("settings") },
            ].map((item) => {
              const Icon = item.icon;
              const isActive =
                ("active" in item && item.active) ||
                currentScreen.toLowerCase() === item.label.toLowerCase() ||
                (item.label === "Models" && currentScreen === "dlengine") ||
                (item.label === "Tuning" && currentScreen === "settings");
              return (
                <Button
                  key={item.label}
                  type="button"
                  variant={isActive ? "default" : "outline"}
                  size="sm"
                  onClick={item.action}
                  className="h-12 min-w-[74px] flex-col gap-1 rounded-xl px-2 text-[11px]"
                  title={item.label === "Vision" ? "Open Camera Vision object-recognition HUD" : item.title || item.label}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Button>
              );
            })}
          </div>
          {audioMixerOpen && (
            <div className="mt-2 rounded-xl border border-border/70 bg-background/80 p-3 shadow-xl shadow-black/20">
              <AudioMixerPanel />
            </div>
          )}
          {appLauncherOpen && (
            <div className="mt-2 grid grid-cols-4 gap-2 rounded-xl border border-border/70 bg-background/70 p-2 shadow-xl shadow-black/20 sm:grid-cols-6">
              {SHELL_FEATURES.map((feature) => {
                const Icon = featureIcon(feature);
                const screen = featureToMobileScreen(feature);
                const isActive = currentScreen === screen;
                return (
                  <button
                    key={feature.id}
                    type="button"
                    onClick={() => {
                      setCurrentScreen(screen);
                      setAppLauncherOpen(false);
                    }}
                    className={cn(
                    "sarah-focus-ring flex min-h-16 flex-col items-center justify-center gap-1 rounded-xl border px-2 py-2 text-center text-[11px] transition",
                      isActive
                        ? "border-primary/60 bg-primary/15 text-primary"
                        : "border-border/55 bg-card/45 text-foreground/80 hover:border-primary/40 hover:bg-card/80",
                    )}
                    title={feature.purpose}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="line-clamp-2 leading-tight">{feature.title}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
      <div className="sarah-mobile-screen-content flex-1 min-h-0 animate-fade-in overflow-hidden">
        {getFeatureComponent(currentScreen)}
      </div>
    </div>
  );
}

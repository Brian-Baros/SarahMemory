import React from "react";
import {
  Clock,
  Palette,
  MessageCircle,
  User,
  LayoutGrid,
  Folder,
  Search,
  Cpu,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useIsMobile } from "@/hooks/use-mobile";
import { useNavigationStore, BOTTOM_NAV_ITEMS } from "@/stores/useNavigationStore";

const iconMap: Record<string, React.ElementType> = {
  clock: Clock,
  palette: Palette,
  "message-circle": MessageCircle,
  user: User,
  "grid-3x3": LayoutGrid,
  folder: Folder,
  search: Search,
  cpu: Cpu,
};

// Map MobileScreen -> DesktopApp (desktop shell mode)
function screenToDesktopApp(screen: string) {
  switch (screen) {
    case "chat":
      return "chat";
    case "files":
      return "files";
    case "media":
      return "media";
    case "research":
      return "research";
    case "dlengine":
      return "dlengine";
    case "avatar":
      return "avatar";
    case "sarahnet":
      return "sarahnet";
    case "studios":
      return "studio";
    case "history":
    case "settings":
    default:
      // Desktop shell doesn't have a dedicated "history/settings app" yet;
      // default to chat for now.
      return "chat";
  }
}

/**
 * Bottom navigation bar
 * - Mobile: drives MobileShell currentScreen
 * - Desktop (Shell Mode): acts like a Dock, drives activeDesktopApp
 */
export function BottomNav() {
  const isMobile = useIsMobile();

  const {
    currentScreen,
    setCurrentScreen,
    isDesktopShellMode,
    activeDesktopApp,
    setActiveDesktopApp,
  } = useNavigationStore();

  // Show on mobile always, and on desktop only when shell mode is enabled
  const shouldRender = isMobile || isDesktopShellMode;
  if (!shouldRender) return null;

  const isDesktopDock = !isMobile && isDesktopShellMode;

  return (
    <nav
      className={cn(
        "fixed left-0 right-0 z-50",
        // Mobile: bottom nav
        isMobile && "bottom-0 lg:hidden",
        // Desktop shell: also bottom, but allow on lg+
        isDesktopDock && "bottom-0 hidden lg:block",
      )}
    >
      <div className="px-2 pb-[env(safe-area-inset-bottom)]">
        <div
          className={cn(
            // Mobile: max-w-md; Desktop dock: wider
            "mx-auto mb-2 h-14",
            isDesktopDock ? "max-w-5xl" : "max-w-md",
            "rounded-2xl border border-border/60",
            "bg-card/80 backdrop-blur-xl shadow-lg",
          )}
        >
          <div className="flex items-center justify-around h-full px-1">
            {BOTTOM_NAV_ITEMS.map((item) => {
              const Icon = iconMap[item.icon] || MessageCircle;

              const isActive = isDesktopDock
                ? activeDesktopApp === screenToDesktopApp(item.screen)
                : currentScreen === item.screen;

              return (
                <button
                  key={item.screen}
                  onClick={() => {
                    if (isDesktopDock) {
                      setActiveDesktopApp(screenToDesktopApp(item.screen));
                    } else {
                      setCurrentScreen(item.screen);
                    }
                  }}
                  className={cn(
                    "flex flex-col items-center justify-center flex-1 h-full py-1 transition-all",
                    isActive ? "text-primary" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  <div
                    className={cn(
                      "w-9 h-9 rounded-xl flex items-center justify-center transition-all",
                      isActive ? "bg-primary/10" : "bg-muted/40",
                    )}
                  >
                    <Icon className={cn("h-5 w-5 transition-transform", isActive && "scale-110")} />
                  </div>

                  <span className={cn("text-[10px] mt-0.5 font-medium", isActive && "text-primary")}>
                    {item.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}

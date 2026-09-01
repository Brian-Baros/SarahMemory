import React, { useMemo } from "react";
import {
  Clock,
  Palette,
  MessageCircle,
  User,
  LayoutGrid,
  Folder,
  Search,
  Cpu,
  Terminal,
  MonitorCog,
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
  terminal: Terminal,
  "monitor-cog": MonitorCog,
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
    case "nailde":
      return "nailde";
    case "terminal":
      return "terminal";
    case "avatar":
      return "avatar";
    case "sarahnet":
      return "sarahnet";
    case "studios":
      return "studio";
    case "history":
      return "history";
    case "settings":
      return "settings";
    default:
      return "chat";
  }
}

/**
 * Bottom command rail
 * - Mobile shell: drives MobileShell currentScreen
 * - Desktop shell mode: drives activeDesktopApp
 *
 * Hook order must remain invariant. Do not return before all hooks execute.
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

  // Add terminal item even if store list has not been updated yet.
  const navItems = useMemo(() => {
    const base = Array.isArray(BOTTOM_NAV_ITEMS) ? [...BOTTOM_NAV_ITEMS] : [];

    const hasNailde = base.some((item) => String(item?.screen) === "nailde");
    if (!hasNailde) {
      base.push({
        screen: "nailde",
        label: "NAILDE",
        icon: "monitor-cog",
      } as any);
    }

    const hasTerminal = base.some((item) => String(item?.screen) === "terminal");
    if (!hasTerminal) {
      base.push({
        screen: "terminal",
        label: "Ops",
        icon: "terminal",
      } as any);
    }

    return base;
  }, []);

  const shouldRender = isMobile || isDesktopShellMode;
  const isDesktopDock = !isMobile && isDesktopShellMode;

  if (!shouldRender) return null;

  return (
    <nav
      className={cn(
        "fixed left-0 right-0 z-50",
        isMobile && "bottom-0 lg:hidden",
        isDesktopDock && "bottom-0 hidden lg:block",
      )}
      data-bottom-nav="true"
    >
      <div className="px-2 pb-[env(safe-area-inset-bottom)]">
        <div
          className={cn(
            "mx-auto mb-2 h-[var(--dock-h,56px)] min-h-14",
            isDesktopDock ? "max-w-5xl" : "max-w-md",
            "rounded-2xl border border-border/60",
            "sarah-material shadow-lg",
          )}
        >
          <div className="flex items-center justify-around h-full px-1 overflow-x-auto overscroll-x-contain">
            {navItems.map((item) => {
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
                    "sarah-focus-ring flex flex-col items-center justify-center flex-1 min-w-[48px] h-full py-1 transition-all touch-manipulation",
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

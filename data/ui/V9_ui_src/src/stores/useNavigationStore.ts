import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Mobile navigation screens - swipe map (Concept 2 aligned):
 * LEFT → RIGHT:
 *   History → Chat → Files → Research → DL Engine → Assistant (Avatar) → Studios
 */
export type MobileScreen =
  | "history"
  | "studios"
  | "chat"
  | "avatar"
  | "sarahnet"
  | "research"
  | "files"
  | "media"
  | "dlengine"
  | "nailde"
  | "terminal"
  | "addons"
  | "settings";

export type DesktopApp =
  | "chat"
  | "files"
  | "media"
  | "research"
  | "studio"
  | "dlengine"
  | "nailde"
  | "terminal"
  | "history"
  | "addons"
  | "settings"
  | "sarahnet"
  | "avatar";

// Screen order for swipe navigation (left to right)
export const SCREEN_ORDER: MobileScreen[] = [
  "history",
  "chat",
  "files",
  "research",
  "dlengine",
  "nailde",
  "terminal",
  "avatar",
  "sarahnet",
  "media",
  "studios",
  "addons",
  "settings",
];

// Bottom nav items (5 max)
export const BOTTOM_NAV_ITEMS: { screen: MobileScreen; label: string; icon: string }[] = [
  { screen: "chat", label: "Chat", icon: "message-circle" },
  { screen: "history", label: "History", icon: "clock" },
  { screen: "files", label: "Files", icon: "folder" },
  { screen: "research", label: "Research", icon: "search" },
  { screen: "avatar", label: "Assistant", icon: "user" },
];

export type UiAction = { type: string; payload?: any };

interface NavigationState {
  // Mobile
  currentScreen: MobileScreen;
  setCurrentScreen: (screen: MobileScreen) => void;
  swipeLeft: () => void;
  swipeRight: () => void;
  goHome: () => void;

  // Desktop
  activeDesktopApp: DesktopApp;
  setActiveDesktopApp: (app: DesktopApp) => void;

  // Status
  connectionStatus: "connected" | "degraded" | "offline";
  setConnectionStatus: (status: "connected" | "degraded" | "offline") => void;

  // Desktop shell toggle
  isDesktopShellMode: boolean;
  setDesktopShellMode: (enabled: boolean) => void;

  // UI Control Bus hook
  applyUiAction: (action: UiAction) => boolean;
}


export const useNavigationStore = create<NavigationState>()(
  persist(
    (set, get) => ({
      // Mobile
      currentScreen: "chat",
      setCurrentScreen: (screen) => set({ currentScreen: screen }),

      swipeLeft: () => {
        const { currentScreen } = get();
        const idx = SCREEN_ORDER.indexOf(currentScreen);
        if (idx === -1) return set({ currentScreen: "chat" }); // safety fallback
        if (idx > 0) set({ currentScreen: SCREEN_ORDER[idx - 1] });
      },

      swipeRight: () => {
        const { currentScreen } = get();
        const idx = SCREEN_ORDER.indexOf(currentScreen);
        if (idx === -1) return set({ currentScreen: "chat" }); // safety fallback
        if (idx < SCREEN_ORDER.length - 1) set({ currentScreen: SCREEN_ORDER[idx + 1] });
      },

      goHome: () => set({ currentScreen: "chat" }),

      // Desktop
      activeDesktopApp: "chat",
      setActiveDesktopApp: (app) => set({ activeDesktopApp: app }),

      // Status
      connectionStatus: "connected",
      setConnectionStatus: (status) => set({ connectionStatus: status }),

      // Desktop shell (default ON so landscape dock behavior works)
      isDesktopShellMode: true,
      setDesktopShellMode: (enabled) => set({ isDesktopShellMode: enabled }),

      applyUiAction: (action) => {
        const type = String(action?.type || "");
        const p = action?.payload || {};

        if (type === "navigate") {
          const screen = p?.screen || p?.route || p?.app;
          if (!screen) return false;
          const s = String(screen);
          if ((SCREEN_ORDER as any).includes(s)) {
            set({ currentScreen: s as MobileScreen });
            return true;
          }
          const desktopMap: Record<string, DesktopApp> = {
            chat: "chat",
            files: "files",
            media: "media",
            research: "research",
            studio: "studio",
            studios: "studio",
            dlengine: "dlengine",
            nailde: "nailde",
            terminal: "terminal",
            history: "history",
            addons: "addons",
            settings: "settings",
            sarahnet: "sarahnet",
            avatar: "avatar",
          };
          if (desktopMap[s]) {
            set({ activeDesktopApp: desktopMap[s] });
            return true;
          }
          return false;
        }

        if (type === "nav.set_screen") {
          const s = String(p?.screen || "");
          if ((SCREEN_ORDER as any).includes(s)) {
            set({ currentScreen: s as MobileScreen });
            return true;
          }
          return false;
        }

        if (type === "nav.swipe_left") {
          get().swipeLeft();
          return true;
        }
        if (type === "nav.swipe_right") {
          get().swipeRight();
          return true;
        }
        if (type === "nav.home") {
          get().goHome();
          return true;
        }

        if (type === "desktop.set_app") {
          const s = String(p?.app || "");
          const desktopMap: Record<string, DesktopApp> = {
            chat: "chat",
            files: "files",
            media: "media",
            research: "research",
            studio: "studio",
            dlengine: "dlengine",
            nailde: "nailde",
            terminal: "terminal",
            history: "history",
            addons: "addons",
            settings: "settings",
            sarahnet: "sarahnet",
            avatar: "avatar",
          };
          if (desktopMap[s]) {
            set({ activeDesktopApp: desktopMap[s] });
            return true;
          }
          return false;
        }

        if (type === "connection.set") {
          const s = String(p?.status || "");
          if (s === "connected" || s === "degraded" || s === "offline") {
            set({ connectionStatus: s as any });
            return true;
          }
          return false;
        }

        if (type === "desktop.shell_mode") {
          set({ isDesktopShellMode: !!p?.enabled });
          return true;
        }

        return false;
      },
    }),
    {
      name: "sarah-navigation-storage",
      partialize: (state) => ({
        isDesktopShellMode: state.isDesktopShellMode,
        activeDesktopApp: state.activeDesktopApp,
      }),
    },
  ),
);

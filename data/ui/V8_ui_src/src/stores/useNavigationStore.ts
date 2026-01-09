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
  | "settings";

export type DesktopApp =
  | "chat"
  | "files"
  | "media"
  | "research"
  | "studio"
  | "dlengine"
  | "sarahnet"
  | "avatar";

// Screen order for swipe navigation (left to right)
export const SCREEN_ORDER: MobileScreen[] = [
  "history",
  "chat",
  "files",
  "research",
  "dlengine",
  "avatar",
  "studios",
];

// Bottom nav items (5 max)
export const BOTTOM_NAV_ITEMS: { screen: MobileScreen; label: string; icon: string }[] = [
  { screen: "chat", label: "Chat", icon: "message-circle" },
  { screen: "history", label: "History", icon: "clock" },
  { screen: "files", label: "Files", icon: "folder" },
  { screen: "research", label: "Research", icon: "search" },
  { screen: "avatar", label: "Assistant", icon: "user" },
];

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

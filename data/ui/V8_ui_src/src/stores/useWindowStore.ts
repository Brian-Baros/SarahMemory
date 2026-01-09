import { create } from "zustand";
import { persist } from "zustand/middleware";

export type WindowId = 
  | "chat" 
  | "files" 
  | "media" 
  | "research" 
  | "studio" 
  | "dlengine" 
  | "sarahnet" 
  | "avatar"
  | "settings"
  | "history";

export interface WindowState {
  id: WindowId;
  title: string;
  icon: string;
  x: number;
  y: number;
  width: number;
  height: number;
  isMinimized: boolean;
  isMaximized: boolean;
  zIndex: number;
}

interface WindowStore {
  windows: WindowState[];
  focusedWindowId: WindowId | null;
  nextZIndex: number;
  
  // Actions
  openWindow: (id: WindowId) => void;
  closeWindow: (id: WindowId) => void;
  focusWindow: (id: WindowId) => void;
  minimizeWindow: (id: WindowId) => void;
  maximizeWindow: (id: WindowId) => void;
  restoreWindow: (id: WindowId) => void;
  moveWindow: (id: WindowId, x: number, y: number) => void;
  resizeWindow: (id: WindowId, width: number, height: number) => void;
  isWindowOpen: (id: WindowId) => boolean;
}

// Default window configurations
const WINDOW_DEFAULTS: Record<WindowId, Omit<WindowState, "x" | "y" | "zIndex" | "isMinimized" | "isMaximized">> = {
  chat: { id: "chat", title: "Chat", icon: "message-circle", width: 480, height: 600 },
  files: { id: "files", title: "Files", icon: "folder", width: 500, height: 450 },
  media: { id: "media", title: "Media Player", icon: "play", width: 450, height: 400 },
  research: { id: "research", title: "Research", icon: "search", width: 550, height: 500 },
  studio: { id: "studio", title: "Studios", icon: "palette", width: 600, height: 550 },
  dlengine: { id: "dlengine", title: "DL Engine", icon: "cpu", width: 500, height: 450 },
  sarahnet: { id: "sarahnet", title: "SarahNet", icon: "network", width: 520, height: 480 },
  avatar: { id: "avatar", title: "Avatar", icon: "user", width: 400, height: 500 },
  settings: { id: "settings", title: "Settings", icon: "settings", width: 450, height: 500 },
  history: { id: "history", title: "History", icon: "clock", width: 400, height: 500 },
};

// Calculate initial position with cascade effect
const getInitialPosition = (windowCount: number) => {
  const baseX = 80;
  const baseY = 60;
  const offset = 30;
  return {
    x: baseX + (windowCount % 8) * offset,
    y: baseY + (windowCount % 8) * offset,
  };
};

export const useWindowStore = create<WindowStore>()(
  persist(
    (set, get) => ({
      windows: [],
      focusedWindowId: null,
      nextZIndex: 100,

      openWindow: (id) => {
        const { windows, nextZIndex } = get();
        const existing = windows.find((w) => w.id === id);
        
        if (existing) {
          // If already open, just focus and restore if minimized
          set((state) => ({
            windows: state.windows.map((w) =>
              w.id === id
                ? { ...w, isMinimized: false, zIndex: state.nextZIndex }
                : w
            ),
            focusedWindowId: id,
            nextZIndex: state.nextZIndex + 1,
          }));
          return;
        }

        // Create new window
        const defaults = WINDOW_DEFAULTS[id];
        const pos = getInitialPosition(windows.length);
        
        const newWindow: WindowState = {
          ...defaults,
          x: pos.x,
          y: pos.y,
          isMinimized: false,
          isMaximized: false,
          zIndex: nextZIndex,
        };

        set((state) => ({
          windows: [...state.windows, newWindow],
          focusedWindowId: id,
          nextZIndex: state.nextZIndex + 1,
        }));
      },

      closeWindow: (id) => {
        set((state) => ({
          windows: state.windows.filter((w) => w.id !== id),
          focusedWindowId: state.focusedWindowId === id ? null : state.focusedWindowId,
        }));
      },

      focusWindow: (id) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, zIndex: state.nextZIndex } : w
          ),
          focusedWindowId: id,
          nextZIndex: state.nextZIndex + 1,
        }));
      },

      minimizeWindow: (id) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, isMinimized: true } : w
          ),
          focusedWindowId: state.focusedWindowId === id ? null : state.focusedWindowId,
        }));
      },

      maximizeWindow: (id) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, isMaximized: true, isMinimized: false } : w
          ),
          focusedWindowId: id,
        }));
      },

      restoreWindow: (id) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, isMaximized: false, isMinimized: false, zIndex: state.nextZIndex } : w
          ),
          focusedWindowId: id,
          nextZIndex: state.nextZIndex + 1,
        }));
      },

      moveWindow: (id, x, y) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, x, y } : w
          ),
        }));
      },

      resizeWindow: (id, width, height) => {
        set((state) => ({
          windows: state.windows.map((w) =>
            w.id === id ? { ...w, width: Math.max(300, width), height: Math.max(200, height) } : w
          ),
        }));
      },

      isWindowOpen: (id) => {
        return get().windows.some((w) => w.id === id);
      },
    }),
    {
      name: "sarah-window-storage",
      partialize: (state) => ({
        windows: state.windows.map((w) => ({
          ...w,
          // Reset z-index on reload
          zIndex: 100,
        })),
      }),
    }
  )
);

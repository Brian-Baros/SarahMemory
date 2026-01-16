import { create } from "zustand";
import { persist } from "zustand/middleware";

export type WindowId =
  | "chat"
  | "history"
  | "files"
  | "research"
  | "studio"
  | "avatar"
  | "sarahnet"
  | "media"
  | "dlengine"
  | "addons"   // ✅ ADDED (Apps/Addons launcher)
  | "settings"; // ✅ ADDED

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

  openWindow: (id: WindowId) => void;
  closeWindow: (id: WindowId) => void;
  focusWindow: (id: WindowId) => void;
  minimizeWindow: (id: WindowId) => void;
  maximizeWindow: (id: WindowId) => void;
  restoreWindow: (id: WindowId) => void;
  moveWindow: (id: WindowId, x: number, y: number) => void;
  resizeWindow: (id: WindowId, width: number, height: number) => void;
}

const WINDOW_DEFAULTS: Record<
  WindowId,
  Omit<WindowState, "x" | "y" | "zIndex" | "isMinimized" | "isMaximized">
> = {
  chat: { id: "chat", title: "Chat", icon: "message-circle", width: 520, height: 620 },
  history: { id: "history", title: "History", icon: "clock", width: 460, height: 520 },
  files: { id: "files", title: "Files", icon: "folder", width: 520, height: 480 },
  research: { id: "research", title: "Research", icon: "search", width: 560, height: 540 },
  studio: { id: "studio", title: "Studios", icon: "palette", width: 620, height: 560 },
  avatar: { id: "avatar", title: "Avatar", icon: "user", width: 420, height: 520 },
  sarahnet: { id: "sarahnet", title: "SarahNet", icon: "network", width: 540, height: 500 },
  media: { id: "media", title: "Media", icon: "play", width: 480, height: 420 },
  dlengine: { id: "dlengine", title: "DL Engine", icon: "cpu", width: 520, height: 460 },

  // ✅ ADDONS LAUNCHER WINDOW
  addons: {
    id: "addons",
    title: "Addons",
    icon: "layout-grid",
    width: 720,
    height: 520,
  },

  // ✅ SETTINGS WINDOW
  settings: {
    id: "settings",
    title: "Settings",
    icon: "settings",
    width: 640,
    height: 560,
  },
};

const getInitialPosition = (count: number) => ({
  x: 80 + (count % 6) * 30,
  y: 60 + (count % 6) * 30,
});

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
          set({
            windows: windows.map((w) =>
              w.id === id ? { ...w, isMinimized: false, zIndex: nextZIndex } : w,
            ),
            focusedWindowId: id,
            nextZIndex: nextZIndex + 1,
          });
          return;
        }

        const base = WINDOW_DEFAULTS[id];
        const pos = getInitialPosition(windows.length);

        set({
          windows: [
            ...windows,
            {
              ...base,
              x: pos.x,
              y: pos.y,
              isMinimized: false,
              isMaximized: false,
              zIndex: nextZIndex,
            },
          ],
          focusedWindowId: id,
          nextZIndex: nextZIndex + 1,
        });
      },

      closeWindow: (id) =>
        set((s) => ({
          windows: s.windows.filter((w) => w.id !== id),
          focusedWindowId: s.focusedWindowId === id ? null : s.focusedWindowId,
        })),

      focusWindow: (id) =>
        set((s) => ({
          windows: s.windows.map((w) =>
            w.id === id ? { ...w, zIndex: s.nextZIndex } : w,
          ),
          focusedWindowId: id,
          nextZIndex: s.nextZIndex + 1,
        })),

      minimizeWindow: (id) =>
        set((s) => ({
          windows: s.windows.map((w) =>
            w.id === id ? { ...w, isMinimized: true } : w,
          ),
        })),

      maximizeWindow: (id) =>
        set((s) => ({
          windows: s.windows.map((w) =>
            w.id === id ? { ...w, isMaximized: true } : w,
          ),
          focusedWindowId: id,
        })),

      restoreWindow: (id) =>
        set((s) => ({
          windows: s.windows.map((w) =>
            w.id === id
              ? { ...w, isMaximized: false, isMinimized: false, zIndex: s.nextZIndex }
              : w,
          ),
          focusedWindowId: id,
          nextZIndex: s.nextZIndex + 1,
        })),

      moveWindow: (id, x, y) =>
        set((s) => ({
          windows: s.windows.map((w) => (w.id === id ? { ...w, x, y } : w)),
        })),

      resizeWindow: (id, width, height) =>
        set((s) => ({
          windows: s.windows.map((w) =>
            w.id === id
              ? { ...w, width: Math.max(320, width), height: Math.max(240, height) }
              : w,
          ),
        })),
    }),
    { name: "sarah-windows" },
  ),
);
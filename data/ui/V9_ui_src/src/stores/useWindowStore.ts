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
  | "nailde"
  | "device-manager"
  | "addons"
  | "settings"
  | "terminal";

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

export type UiAction = { type: string; payload?: any };

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
  applyWorkspacePreset: (preset: "chat" | "research" | "operator" | "engineer" | "media") => void;

  // UI Control Bus hook
  applyUiAction: (action: UiAction) => boolean;
}


const WINDOW_DEFAULTS: Record<
  WindowId,
  Omit<WindowState, "x" | "y" | "zIndex" | "isMinimized" | "isMaximized">
> = {
  chat: { id: "chat", title: "Sarah Chat", icon: "message-circle", width: 520, height: 620 },
  history: { id: "history", title: "Memory Trail", icon: "clock", width: 460, height: 520 },
  files: { id: "files", title: "File Cortex", icon: "folder", width: 520, height: 480 },
  research: { id: "research", title: "Evidence Lens", icon: "search", width: 560, height: 540 },
  studio: { id: "studio", title: "Creation Bay", icon: "palette", width: 620, height: 560 },
  avatar: { id: "avatar", title: "Avatar Core", icon: "user", width: 420, height: 520 },
  sarahnet: { id: "sarahnet", title: "SarahNet", icon: "network", width: 540, height: 500 },
  media: { id: "media", title: "Media Deck", icon: "play", width: 480, height: 420 },
  dlengine: { id: "dlengine", title: "Model Forge", icon: "cpu", width: 520, height: 460 },
  nailde: { id: "nailde", title: "NAILDE", icon: "monitor-cog", width: 1180, height: 720 },
  "device-manager": { id: "device-manager", title: "Device Manager", icon: "monitor-cog", width: 940, height: 620 },
  terminal: { id: "terminal", title: "Operator Terminal", icon: "terminal", width: 620, height: 520 },

  addons: {
    id: "addons",
    title: "Addons",
    icon: "layout-grid",
    width: 720,
    height: 520,
  },

  settings: {
    id: "settings",
    title: "System Tuning",
    icon: "settings",
    width: 640,
    height: 560,
  },
};

const getInitialPosition = (count: number) => ({
  x: 80 + (count % 6) * 30,
  y: 60 + (count % 6) * 30,
});

const MIN_WINDOW_WIDTH = 320;
const MIN_WINDOW_HEIGHT = 240;

function workspaceBounds() {
  if (typeof window === "undefined") {
    return { width: 1280, height: 720 };
  }
  try {
    const root = getComputedStyle(document.documentElement);
    const dock = root.getPropertyValue("--taskbar-dock").trim();
    const taskbarSize = parseInt(root.getPropertyValue("--taskbar-size").trim() || "56", 10) || 56;
    const width = Math.max(MIN_WINDOW_WIDTH, window.innerWidth - (dock === "left" || dock === "right" ? taskbarSize : 0));
    const height = Math.max(MIN_WINDOW_HEIGHT, window.innerHeight - (dock === "top" || dock === "bottom" ? taskbarSize : 0));
    return { width, height };
  } catch {
    return { width: Math.max(MIN_WINDOW_WIDTH, window.innerWidth || 1280), height: Math.max(MIN_WINDOW_HEIGHT, window.innerHeight || 720) };
  }
}

function clampWindowRect(x: number, y: number, width: number, height: number) {
  const bounds = workspaceBounds();
  const safeWidth = Math.max(MIN_WINDOW_WIDTH, Math.min(Math.floor(width), bounds.width));
  const safeHeight = Math.max(MIN_WINDOW_HEIGHT, Math.min(Math.floor(height), bounds.height));
  const safeX = Math.max(0, Math.min(Math.floor(x), Math.max(0, bounds.width - safeWidth)));
  const safeY = Math.max(0, Math.min(Math.floor(y), Math.max(0, bounds.height - safeHeight)));
  return { x: safeX, y: safeY, width: safeWidth, height: safeHeight };
}

const WORKSPACE_PRESETS: Record<string, WindowId[]> = {
  chat: ["chat", "history", "avatar"],
  research: ["research", "files", "chat"],
  operator: ["avatar", "sarahnet", "media", "device-manager", "settings"],
  engineer: ["nailde", "terminal", "dlengine", "device-manager", "addons", "settings"],
  media: ["studio", "media", "files"],
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

        const rect = clampWindowRect(pos.x, pos.y, base.width, base.height);

        set({
          windows: [
            ...windows,
            {
              ...base,
              ...rect,
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
          windows: s.windows.map((w) => {
            if (w.id !== id) return w;
            const rect = clampWindowRect(x, y, w.width, w.height);
            return { ...w, x: rect.x, y: rect.y };
          }),
        })),


      applyUiAction: (action) => {
        const type = String(action?.type || "");
        const p = action?.payload || {};
        const id = (p?.id || p?.window || p?.name) as WindowId;

        if (type === "window.open" && id) {
          get().openWindow(id);
          return true;
        }
        if (type === "window.close" && id) {
          get().closeWindow(id);
          return true;
        }
        if (type === "window.focus" && id) {
          get().focusWindow(id);
          return true;
        }
        if (type === "window.minimize" && id) {
          get().minimizeWindow(id);
          return true;
        }
        if (type === "window.maximize" && id) {
          get().maximizeWindow(id);
          return true;
        }
        if (type === "window.restore" && id) {
          get().restoreWindow(id);
          return true;
        }
        if (type === "window.move" && id) {
          const x = Number(p?.x);
          const y = Number(p?.y);
          if (Number.isFinite(x) && Number.isFinite(y)) get().moveWindow(id, x, y);
          return true;
        }
        if (type === "window.resize" && id) {
          const w = Number(p?.width);
          const h = Number(p?.height);
          if (Number.isFinite(w) && Number.isFinite(h)) get().resizeWindow(id, w, h);
          return true;
        }
        if ((type === "workspace.apply" || type === "workspace.preset") && p?.preset) {
          get().applyWorkspacePreset(String(p.preset) as any);
          return true;
        }
        return false;
      },

      resizeWindow: (id, width, height) =>
        set((s) => ({
          windows: s.windows.map((w) => {
            if (w.id !== id) return w;
            const rect = clampWindowRect(w.x, w.y, width, height);
            return { ...w, ...rect };
          }),
        })),

      applyWorkspacePreset: (preset) => {
        const ids = WORKSPACE_PRESETS[preset] || WORKSPACE_PRESETS.chat;
        for (const id of ids) get().openWindow(id);
      },
    }),
    { name: "sarah-windows" },
  ),
);

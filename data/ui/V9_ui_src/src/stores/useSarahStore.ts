import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  Message,
  ChatThread,
  Contact,
  Reminder,
  MediaState,
  Settings,
  VoiceOption,
  ThemeOption,
} from "@/types/sarah";
import type { BootstrapResponse, AvatarSpeechCue } from "@/lib/api";

// Avatar pose types
export type AvatarPose = "stand" | "sit" | "wave" | "walk" | "turn_left" | "turn_right" | "idle";

// Right panel page types
export type RightPanelPage = "contacts" | "keypad" | "tools" | "settings";


// ------------------------------------------------------------
// UI Control Bus + Automation (Chat-driven orchestration)
// ------------------------------------------------------------
export type UiAction = { type: string; payload?: any };

export type AutomationAuditEntry = {
  id: string;
  ts: number;
  source: string;
  actionType: string;
  payload?: any;
  status: "applied" | "skipped" | "error";
  error?: string;
};

function nowTs() {
  return Date.now();
}

function safeString(x: any) {
  try {
    return String(x);
  } catch {
    return "";
  }
}

function coerceBool(v: any, fallback = false): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return ["1", "true", "yes", "on"].includes(v.toLowerCase());
  if (typeof v === "number") return v !== 0;
  return fallback;
}

function normalizeUiActions(input: any): UiAction[] {
  if (!input) return [];
  if (Array.isArray(input)) return input.filter(Boolean) as UiAction[];
  if (typeof input === "object") return [input as UiAction];
  return [];
}

const PANEL_PENDING_PREFIX = "sarahmemory:panel:pending:";

function canonicalActionType(type: any): string {
  return safeString(type || "").trim().toLowerCase().replace(/\./g, "_");
}

function queuePanelAction(panel: "research" | "history", action: UiAction): void {
  if (typeof window === "undefined") return;
  try {
    const key = `${PANEL_PENDING_PREFIX}${panel}`;
    const raw = window.sessionStorage.getItem(key);
    const existing = raw ? JSON.parse(raw) : [];
    const batch = Array.isArray(existing) ? existing : [];
    batch.push({ ...action, queuedAt: Date.now() });
    window.sessionStorage.setItem(key, JSON.stringify(batch.slice(-50)));
    window.dispatchEvent(new CustomEvent(`sarah:${panel}`, { detail: { actions: [action], source: "store" } }));
  } catch {
    // non-fatal UI bridge
  }
}

function normalizeChatThreadMessage(message: any, id: string): any {
  return {
    ...message,
    id,
    timestamp: message?.timestamp instanceof Date ? message.timestamp : new Date(),
  };
}

function deriveThreadTitle(firstText: string): string {
  const clean = safeString(firstText || "").replace(/\s+/g, " ").trim();
  if (!clean) return "Conversation";
  return clean.length > 54 ? `${clean.slice(0, 54).trim()}…` : clean;
}
// ------------------------------------------------------------
// Taskbar settings (kept inside Settings for now)
// We do NOT modify @/types/sarah in this patch; we safely extend at runtime.
// ------------------------------------------------------------
type TaskbarDock = "bottom" | "top" | "left" | "right";

const DEFAULT_TASKBAR_ITEMS = [
  "chat",
  "history",
  "files",
  "research",
  "studio",
  "avatar",
  "sarahnet",
  "media",
  "dlengine",
  "addons",
  "settings",
];

const DEFAULT_TASKBAR = {
  dock: "bottom" as TaskbarDock,
  rows: 1,
  items: DEFAULT_TASKBAR_ITEMS,
};

function ensureTaskbarSettings(s: any): any {
  // s is Settings-like
  const next = { ...(s || {}) };

  if (!next.taskbar || typeof next.taskbar !== "object") {
    next.taskbar = { ...DEFAULT_TASKBAR };
    return next;
  }

  // Merge missing pieces only
  next.taskbar = {
    ...DEFAULT_TASKBAR,
    ...next.taskbar,
  };

  // Sanitize known fields (defensive)
  const dock = String(next.taskbar.dock || "bottom") as TaskbarDock;
  if (!["bottom", "top", "left", "right"].includes(dock)) {
    next.taskbar.dock = "bottom";
  }

  const rowsNum = Number(next.taskbar.rows);
  next.taskbar.rows = Number.isFinite(rowsNum) && rowsNum >= 1 ? Math.floor(rowsNum) : 1;

  if (!Array.isArray(next.taskbar.items)) {
    next.taskbar.items = [...DEFAULT_TASKBAR_ITEMS];
  } else {
    next.taskbar.items = next.taskbar.items.map((x: any) => String(x));
  }

  return next;
}

interface SarahState {
  // Messages
  messages: Message[];
  isTyping: boolean;
  addMessage: (message: Omit<Message, "id" | "timestamp">) => string;
  clearMessages: () => void;
  setTyping: (typing: boolean) => void;

  // Chat threads
  threads: ChatThread[];
  activeThreadId: string | null;
  setActiveThread: (id: string | null) => void;
  setThreads: (threads: ChatThread[]) => void;

  // Contacts
  contacts: Contact[];
  addContact: (contact: Omit<Contact, "id">) => string;
  updateContact: (id: string, updates: Partial<Contact>) => void;
  deleteContact: (id: string) => void;
  setContacts: (contacts: Contact[]) => void;

  // Reminders
  reminders: Reminder[];
  addReminder: (reminder: Omit<Reminder, "id">) => string;
  updateReminder: (id: string, updates: Partial<Reminder>) => void;
  deleteReminder: (id: string) => void;
  toggleReminderComplete: (id: string) => void;
  setReminders: (reminders: Reminder[]) => void;

  // Media state
  mediaState: MediaState;
  toggleWebcam: () => void;
  toggleMicrophone: () => void;
  toggleVoice: () => void;
  setScreenMode: (mode: MediaState["screenMode"]) => void;

  // Settings
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;

  // Available options
  voices: VoiceOption[];
  themes: ThemeOption[];
  setVoices: (voices: VoiceOption[]) => void;
  setThemes: (themes: ThemeOption[]) => void;

  // Bootstrap data from backend
  bootstrapData: BootstrapResponse | null;
  setBootstrapData: (data: BootstrapResponse) => void;

  // UI State
  leftSidebarCollapsed: boolean;
  rightSidebarCollapsed: boolean;
  leftDrawerOpen: boolean;
  rightDrawerOpen: boolean;
  settingsOpen: boolean;
  rightPanelPage: RightPanelPage;

  hasPlayedWelcome: boolean;
  backendReady: boolean;

  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setLeftDrawerOpen: (open: boolean) => void;
  setRightDrawerOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setRightPanelPage: (page: RightPanelPage) => void;
  setHasPlayedWelcome: (played: boolean) => void;
  setBackendReady: (ready: boolean) => void;

  // Welcome / Intro
  playWelcomeIfNeeded: () => Promise<void>;

  
  // Automation + UI bus
  uiAutomationEnabled: boolean;
  setUiAutomationEnabled: (enabled: boolean) => void;

  // Action queue (supports multi-step autonomous flows)
  actionQueue: UiAction[];
  processingActions: boolean;
  enqueueUiActions: (actions: UiAction[] | UiAction, source?: string) => void;
  dispatchUiActions: (actions: UiAction[] | UiAction, source?: string) => Promise<void>;

  // Audit log (bounded)
  automationAudit: AutomationAuditEntry[];
  clearAutomationAudit: () => void;
  lastActionQueueDrainTs: number | null;
// Avatar animation state
  avatarSpeaking: boolean;
  speechCues: AvatarSpeechCue[];
  speechStartTime: number | null;
  avatarPose: AvatarPose;
  setAvatarSpeaking: (speaking: boolean) => void;
  setSpeechCues: (cues: AvatarSpeechCue[]) => void;
  setSpeechStartTime: (time: number | null) => void;
  setAvatarPose: (pose: AvatarPose) => void;
  triggerWave: () => void;
}

const generateId = () => Math.random().toString(36).slice(2, 11);

// A solid greeting pool (randomized, cycles through all before repeating)
const GREETINGS: string[] = [
  "I'm Sarah, your AI companion. I'm online and ready.",
  "I'm Sarah, here and fully operational.",
  "I'm Sarah. Everything is running and I'm ready when you are.",
  "I'm Sarah, your assistant, standing by.",
  "I'm Sarah, online and listening.",
  "I'm Sarah. You can start whenever you're ready.",
  "I'm Sarah, and I'm here to help.",
  "I'm Sarah, active and prepared.",
  "I'm Sarah. All systems are ready.",
  "I'm Sarah, your AI interface, now available.",
  "I'm Sarah, connected and responsive.",
  "I'm Sarah, ready for whatever you need.",
  "I'm Sarah, and I'm fully awake.",
  "I'm Sarah, your digital assistant, online now.",
  "I'm Sarah, operational and standing by.",
  "I'm Sarah. Feel free to begin.",
  "I'm Sarah, active and ready to respond.",
  "I'm Sarah, connected and waiting for input.",
  "I'm Sarah, initialized and ready.",
  "I'm Sarah. You have my attention.",
  "I'm Sarah, your AI system, now online.",
  "I'm Sarah, prepared and listening.",
  "I'm Sarah. Let me know how you'd like to proceed.",
  "I'm Sarah, here and responsive.",
  "I'm Sarah, your assistant, ready at any time.",
  "I'm Sarah. Everything is set.",
  "I'm Sarah, active and available.",
  "I'm Sarah, standing by for input.",
  "I'm Sarah, and I'm ready to engage.",
  "I'm Sarah, your AI companion, now active.",
  "I'm Sarah. Go ahead whenever you're ready.",
  "I'm Sarah, listening and ready.",
  "I'm Sarah, fully operational and here.",
  "I'm Sarah, your assistant, ready to begin.",
  "I'm Sarah. I'm here when you need me.",
  "I'm Sarah, online and attentive.",
  "I'm Sarah, available and responsive.",
  "I'm Sarah. Feel free to start.",
  "I'm Sarah, your AI system, ready.",
  "I'm Sarah, active and waiting.",
  "I'm Sarah, here and prepared.",
  "I'm Sarah. I'm ready to assist.",
  "I'm Sarah, connected and alert.",
  "I'm Sarah, your assistant, online.",
  "I'm Sarah. Let me know what you'd like to do.",
  "I'm Sarah, awake and ready.",
  "I'm Sarah, ready whenever you are.",
  "I'm Sarah, fully online.",
  "I'm Sarah. I'm here and listening.",
];

const GREETING_HISTORY_KEY = "sarah_greeting_history";

// Per-page-load guard (refresh resets this)
let welcomeFiredThisLoad = false;

function pickGreeting(): string {
  try {
    const historyRaw = localStorage.getItem(GREETING_HISTORY_KEY);
    let shownIndices: number[] = historyRaw ? JSON.parse(historyRaw) : [];

    if (!Array.isArray(shownIndices)) shownIndices = [];
    if (shownIndices.length >= GREETINGS.length) shownIndices = [];

    const availableIndices = GREETINGS
      .map((_, idx) => idx)
      .filter((idx) => !shownIndices.includes(idx));
    const chosenIdx = availableIndices[Math.floor(Math.random() * availableIndices.length)];

    shownIndices.push(chosenIdx);
    localStorage.setItem(GREETING_HISTORY_KEY, JSON.stringify(shownIndices));

    return GREETINGS[chosenIdx];
  } catch {
    return GREETINGS[Math.floor(Math.random() * GREETINGS.length)];
  }
}

function buildDataAudioUrl(base64: string): string {
  // Most backends return mp3; if yours returns wav/ogg, browser may still play it anyway.
  return `data:audio/mpeg;base64,${base64}`;
}

async function playVoiceResponseAudio(res: any): Promise<void> {
  const audioUrl: string | undefined = res?.audio_url || res?.audioUrl;
  const audioBase64: string | undefined = res?.audio_base64 || res?.audioBase64;

  const src = audioUrl ? audioUrl : audioBase64 ? buildDataAudioUrl(audioBase64) : null;
  if (!src) return;

  const audio = new Audio();
  audio.src = src;
  audio.preload = "auto";
  audio.crossOrigin = "anonymous";

  try {
    await audio.play();
  } catch {
    // Autoplay restrictions (mobile) are common.
    // Greeting text will still show; audio will work after first user interaction.
  }
}

export const useSarahStore = create<SarahState>()(
  persist(
    (set, get) => ({
      // Messages (START EMPTY)
      messages: [],
      isTyping: false,

      addMessage: (message) => {
        const id = generateId();
        const normalizedMessage = normalizeChatThreadMessage(message, id);

        set((state) => {
          const activeId = state.activeThreadId || `thread_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
          const existingThread = (state.threads || []).find((t: any) => String(t?.id) === activeId) as any;
          const existingMessages = Array.isArray(existingThread?.messages) ? existingThread.messages : [];
          const nextThreadMessages = [...existingMessages, normalizedMessage];
          const firstUser = nextThreadMessages.find((m: any) => String(m?.role || "") === "user");
          const title = existingThread?.title || deriveThreadTitle(firstUser?.content || normalizedMessage.content || "Conversation");

          const updatedThread: any = {
            ...(existingThread || {}),
            id: activeId,
            title,
            preview: safeString(normalizedMessage.content || ""),
            timestamp: normalizedMessage.timestamp,
            messageCount: nextThreadMessages.length,
            messages: nextThreadMessages,
          };

          const otherThreads = (state.threads || []).filter((t: any) => String(t?.id) !== activeId);

          return {
            activeThreadId: activeId,
            messages: [...state.messages, normalizedMessage],
            threads: [updatedThread, ...otherThreads],
          };
        });

        return id;
      },

      clearMessages: () => set({ messages: [] }),
      setTyping: (typing) => set({ isTyping: typing }),

      // Chat threads
      threads: [],
      activeThreadId: null,
      setActiveThread: (id) => set({ activeThreadId: id }),
      setThreads: (threads) => set({ threads }),

      // Contacts
      contacts: [],
      addContact: (contact) => {
        const id = generateId();
        set((state) => ({ contacts: [...state.contacts, { ...contact, id }] }));
        return id;
      },
      updateContact: (id, updates) =>
        set((state) => ({
          contacts: state.contacts.map((c) =>
            c.id === id ? { ...c, ...updates } : c,
          ),
        })),
      deleteContact: (id) =>
        set((state) => ({
          contacts: state.contacts.filter((c) => c.id !== id),
        })),
      setContacts: (contacts) => set({ contacts }),

      // Reminders
      reminders: [],
      addReminder: (reminder) => {
        const id = generateId();
        set((state) => ({ reminders: [...state.reminders, { ...reminder, id }] }));
        return id;
      },
      updateReminder: (id, updates) =>
        set((state) => ({
          reminders: state.reminders.map((r) =>
            r.id === id ? { ...r, ...updates } : r,
          ),
        })),
      deleteReminder: (id) =>
        set((state) => ({
          reminders: state.reminders.filter((r) => r.id !== id),
        })),
      toggleReminderComplete: (id) =>
        set((state) => ({
          reminders: state.reminders.map((r) =>
            r.id === id ? { ...r, completed: !r.completed } : r,
          ),
        })),
      setReminders: (reminders) => set({ reminders }),

      // Media state
      mediaState: {
        webcamEnabled: true,
        microphoneEnabled: true,
        voiceEnabled: true,
        screenMode: "avatar_2d",
      },
      toggleWebcam: () =>
        set((state) => ({
          mediaState: {
            ...state.mediaState,
            webcamEnabled: !state.mediaState.webcamEnabled,
          },
        })),
      toggleMicrophone: () =>
        set((state) => ({
          mediaState: {
            ...state.mediaState,
            microphoneEnabled: !state.mediaState.microphoneEnabled,
          },
        })),
      toggleVoice: () =>
        set((state) => ({
          mediaState: {
            ...state.mediaState,
            voiceEnabled: !state.mediaState.voiceEnabled,
          },
        })),
      setScreenMode: (mode) =>
        set((state) => ({
          mediaState: { ...state.mediaState, screenMode: mode },
        })),

      // Settings
      settings: ensureTaskbarSettings({
        selectedVoice: "sarah",
        selectedTheme: "default",
        autoSpeak: true,
        soundEffects: true,
        notifications: true,
        mode: "any",
        advancedStudioMode: false,
        uiMode: "simple",
        localOnlyMode: false,
        wallpaperUrl: "",
        wallpaperMode: "cover",
        panelTransparency: "glass",
        shellDensity: "comfortable",
        activeWorkspace: "chat",
      }),
      updateSettings: (updates) =>
        set((state) => ({
          settings: ensureTaskbarSettings({ ...state.settings, ...updates }),
        })),

      // Fallback options
      voices: [
        { id: "sarah", name: "Sarah (Default)", language: "en-US", gender: "female" },
        { id: "emma", name: "Emma", language: "en-GB", gender: "female" },
        { id: "alex", name: "Alex", language: "en-US", gender: "male" },
      ],
      themes: [
        { id: "default", name: "Default Dark", filename: "Dark_Theme.css" },
        { id: "light", name: "Light", filename: "Light_Theme.css" },
        { id: "matrix", name: "Matrix", filename: "Matrix_Theme.css" },
        { id: "tron", name: "Tron", filename: "Tron.css" },
        { id: "hal2000", name: "HAL 2000", filename: "HAL2000_Theme.css" },
        { id: "skynet", name: "Skynet", filename: "Skynet_Theme.css" },
        { id: "vibrant", name: "Vibrant", filename: "Vibrant_Theme.css" },
      ],
      setVoices: (voices) => set({ voices }),
      setThemes: (themes) => set({ themes }),

      // Bootstrap data
      bootstrapData: null,

      // ✅ IMPORTANT: when bootstrap arrives, mark backend ready AND trigger welcome
      setBootstrapData: (data) => {
        const ready = !!data?.ok;
        set({ bootstrapData: data, backendReady: ready });

        if (ready) {
          // fire welcome after state is committed
          queueMicrotask(() => {
            get()
              .playWelcomeIfNeeded()
              .catch(() => {});
          });
        }
      },

      // UI state
      leftSidebarCollapsed: true,
      rightSidebarCollapsed: true,
      leftDrawerOpen: false,
      rightDrawerOpen: false,
      settingsOpen: false,
      rightPanelPage: "contacts" as RightPanelPage,

      hasPlayedWelcome: false,
      backendReady: false,

      toggleLeftSidebar: () =>
        set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),
      toggleRightSidebar: () =>
        set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),
      setLeftDrawerOpen: (open) => set({ leftDrawerOpen: open }),
      setRightDrawerOpen: (open) => set({ rightDrawerOpen: open }),
      setSettingsOpen: (open) => set({ settingsOpen: open }),
      setRightPanelPage: (page) => set({ rightPanelPage: page }),
      setHasPlayedWelcome: (played) => set({ hasPlayedWelcome: played }),

      // ✅ IMPORTANT: if you use setBackendReady() elsewhere, also trigger welcome
      setBackendReady: (ready) => {
        set({ backendReady: ready });
        if (ready) {
          queueMicrotask(() => {
            get()
              .playWelcomeIfNeeded()
              .catch(() => {});
          });
        }
      },

      // ✅ Random welcome each frontend load + web audio playback
      playWelcomeIfNeeded: async () => {
        // Must be after backend is marked ready
        if (!get().backendReady) return;

        // Prevent multiple triggers during the same page load
        if (welcomeFiredThisLoad) return;
        welcomeFiredThisLoad = true;

        // Also guard state
        if (get().hasPlayedWelcome) return;

        const greeting = pickGreeting();

        // Add greeting message to chat (THIS is what fixes “blank”)
        get().addMessage({ role: "assistant", content: greeting });

        // Mark played for this runtime
        set({ hasPlayedWelcome: true });

        // Speak if enabled
        const { settings, mediaState } = get();
        if (!settings.autoSpeak || !mediaState.voiceEnabled) return;

        try {
          const { api } = await import("@/lib/api");

          // Flip avatar speaking on while we do TTS (best-effort)
          get().setAvatarSpeaking(true);

          const res = await api.voice.speak(greeting, settings.selectedVoice);

          // Actually PLAY the audio in browser (may be blocked until tap on mobile)
          await playVoiceResponseAudio(res);

          get().setAvatarSpeaking(false);
        } catch {
          get().setAvatarSpeaking(false);
        }
      },


      // Automation + UI bus
      uiAutomationEnabled: true,
      setUiAutomationEnabled: (enabled) => set({ uiAutomationEnabled: enabled }),

      actionQueue: [],
      processingActions: false,

      enqueueUiActions: (actions, source = "ui") => {
        const batch = normalizeUiActions(actions);
        if (batch.length === 0) return;
        set((s) => ({ actionQueue: [...s.actionQueue, ...batch] }));
        queueMicrotask(() => {
          get().dispatchUiActions(batch, source).catch(() => {});
        });
      },

      clearAutomationAudit: () => set({ automationAudit: [] }),

      dispatchUiActions: async (actions, source = "ui") => {
        const batch = normalizeUiActions(actions);
        if (batch.length === 0) return;
        if (!get().uiAutomationEnabled) return;
        if (get().processingActions) return;

        set({ processingActions: true });

        try {
          const [{ useNavigationStore }, { useWindowStore }, { usePreviewStore }, { useCreativeCacheStore }] =
            await Promise.all([
              import("@/stores/useNavigationStore"),
              import("@/stores/useWindowStore"),
              import("@/stores/usePreviewStore"),
              import("@/stores/useCreativeCacheStore"),
            ]);

          const nav = useNavigationStore.getState();
          const win = useWindowStore.getState();
          const preview = usePreviewStore.getState();
          const cache = useCreativeCacheStore.getState();

          const audit: AutomationAuditEntry[] = [];

          for (const a of batch) {
            const actionType = safeString(a?.type || "");
            const payload = a?.payload;
            if (!actionType) continue;

            const entry: AutomationAuditEntry = {
              id: generateId(),
              ts: nowTs(),
              source,
              actionType,
              payload,
              status: "applied",
            };

            try {
              if (nav.applyUiAction && nav.applyUiAction(a as any)) {
                // applied
              } else if (win.applyUiAction && win.applyUiAction(a as any)) {
                // applied
              } else if (preview.applyUiAction && preview.applyUiAction(a as any)) {
                // applied
              } else if (cache.applyUiAction && cache.applyUiAction(a as any)) {
                // applied
              } else {
                switch (actionType) {
                  case "settings.open":
                    get().setSettingsOpen(true);
                    break;
                  case "settings.close":
                    get().setSettingsOpen(false);
                    break;
                  case "settings.update":
                    if (payload && typeof payload === "object") get().updateSettings(payload);
                    break;

                  case "right_panel.set":
                    if (payload?.page) get().setRightPanelPage(payload.page);
                    if (payload?.open != null) get().setRightDrawerOpen(coerceBool(payload.open));
                    break;

                  case "right_drawer.open":
                    get().setRightDrawerOpen(true);
                    break;
                  case "right_drawer.close":
                    get().setRightDrawerOpen(false);
                    break;

                  case "left_drawer.open":
                    get().setLeftDrawerOpen(true);
                    break;
                  case "left_drawer.close":
                    get().setLeftDrawerOpen(false);
                    break;

                  case "contacts.set":
                    if (Array.isArray(payload)) get().setContacts(payload);
                    break;
                  case "contacts.add":
                    if (payload && typeof payload === "object") get().addContact(payload);
                    break;
                  case "contacts.update":
                    if (payload?.id) get().updateContact(payload.id, payload.updates || {});
                    break;
                  case "contacts.delete":
                    if (payload?.id) get().deleteContact(payload.id);
                    break;

                  case "reminders.set":
                    if (Array.isArray(payload)) get().setReminders(payload);
                    break;
                  case "reminders.add":
                    if (payload && typeof payload === "object") get().addReminder(payload);
                    break;
                  case "reminders.update":
                    if (payload?.id) get().updateReminder(payload.id, payload.updates || {});
                    break;
                  case "reminders.delete":
                    if (payload?.id) get().deleteReminder(payload.id);
                    break;
                  case "reminders.toggle_complete":
                    if (payload?.id) get().toggleReminderComplete(payload.id);
                    break;

                  case "avatar.pose":
                    if (payload?.pose) get().setAvatarPose(payload.pose);
                    break;
                  case "avatar.wave":
                    get().triggerWave();
                    break;

                  case "research_open":
                  case "research.open":
                  case "browser.open":
                  case "research_search":
                  case "research.search":
                  case "browser.search":
                  case "research_back":
                  case "research.back":
                  case "research_forward":
                  case "research.forward":
                  case "research_reload":
                  case "research.reload":
                  case "research_read_current":
                  case "research.read_current":
                  case "browser.read_current":
                    queuePanelAction("research", a);
                    break;

                  case "history_refresh":
                  case "history.refresh":
                  case "history_open":
                  case "history.open":
                  case "history_search_date":
                  case "history.search_date":
                    queuePanelAction("history", a);
                    break;

                  case "toast":
                    try {
                      window.dispatchEvent(new CustomEvent("sarah:toast", { detail: payload || {} }));
                    } catch {}
                    break;

                  default:
                    if (canonicalActionType(actionType).startsWith("research_")) {
                      queuePanelAction("research", a);
                    } else if (canonicalActionType(actionType).startsWith("history_")) {
                      queuePanelAction("history", a);
                    } else {
                      entry.status = "skipped";
                    }
                    break;
                }
              }
            } catch (err: any) {
              entry.status = "error";
              entry.error = safeString(err?.message || err);
            }

            audit.push(entry);
          }

          set((s) => {
            const next = [...(s.automationAudit || []), ...audit];
            const MAX = 300;
            return { automationAudit: next.length > MAX ? next.slice(next.length - MAX) : next };
          });
        } finally {
          // BigBang governance: actions are drained exactly once after a bounded dispatch pass.
          // This prevents Research/History pending actions from looping or duplicating after panel open.
          set({ processingActions: false, actionQueue: [], lastActionQueueDrainTs: nowTs() });
        }
      },

      automationAudit: [],
      lastActionQueueDrainTs: null,
      // Avatar animation state
      avatarSpeaking: false,
      speechCues: [],
      speechStartTime: null,
      avatarPose: "stand" as AvatarPose,
      setAvatarSpeaking: (speaking) => set({ avatarSpeaking: speaking }),
      setSpeechCues: (cues) => set({ speechCues: cues }),
      setSpeechStartTime: (time) => set({ speechStartTime: time }),
      setAvatarPose: (pose) => set({ avatarPose: pose }),
      triggerWave: () => {
        set({ avatarPose: "wave" });
        setTimeout(() => set({ avatarPose: "stand" }), 2000);
      },
    }),
    {
      name: "sarah-memory-storage",
      partialize: (state) => ({
        settings: state.settings,
        contacts: state.contacts,
        reminders: state.reminders,
      }),

      // ✅ If store rehydrates and backend is already ready, try to welcome.
      onRehydrateStorage: () => (state, error) => {
        if (error) return;

        // Ensure settings has taskbar defaults after rehydrate
        try {
          const s: any = (state as any)?.settings;
          if (state && (state as any).updateSettings) {
            const ensured = ensureTaskbarSettings(s);
            if (JSON.stringify(ensured?.taskbar) !== JSON.stringify(s?.taskbar)) {
              (state as any).updateSettings({ taskbar: ensured.taskbar } as any);
            }
          }
        } catch {
          // ignore
        }

        queueMicrotask(() => {
          state?.playWelcomeIfNeeded?.().catch(() => {});
        });
      },
    },
  ),
);


// -----------------------------------------------------------------------------
// Global UI Control Bus installer
// -----------------------------------------------------------------------------
let __uiBusInstalled = false;

export function installSarahUiBus() {
  if (__uiBusInstalled) return;
  if (typeof window === "undefined") return;

  __uiBusInstalled = true;

  window.addEventListener("sarah:ui", (ev: any) => {
    const actions = ev?.detail?.actions || [];
    useSarahStore.getState().dispatchUiActions(actions, ev?.detail?.source || "event").catch(() => {});
  });
}

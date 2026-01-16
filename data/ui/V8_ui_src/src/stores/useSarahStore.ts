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
export type AvatarPose = "stand" | "sit" | "wave";

// Right panel page types
export type RightPanelPage = "contacts" | "keypad" | "tools" | "settings";

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
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id,
              timestamp: new Date(),
            },
          ],
        }));
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

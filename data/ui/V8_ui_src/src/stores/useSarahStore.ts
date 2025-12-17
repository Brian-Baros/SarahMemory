import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, ChatThread, Contact, Reminder, MediaState, Settings, VoiceOption, ThemeOption } from '@/types/sarah';
import type { BootstrapResponse } from '@/lib/api';

// Right panel page types
export type RightPanelPage = 'contacts' | 'keypad' | 'tools' | 'settings';

interface SarahState {
  // Messages
  messages: Message[];
  isTyping: boolean;
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  clearMessages: () => void;
  setTyping: (typing: boolean) => void;
  
  // Chat threads
  threads: ChatThread[];
  activeThreadId: string | null;
  setActiveThread: (id: string | null) => void;
  
  // Contacts
  contacts: Contact[];
  addContact: (contact: Omit<Contact, 'id'>) => void;
  updateContact: (id: string, updates: Partial<Contact>) => void;
  deleteContact: (id: string) => void;
  
  // Reminders
  reminders: Reminder[];
  addReminder: (reminder: Omit<Reminder, 'id'>) => void;
  updateReminder: (id: string, updates: Partial<Reminder>) => void;
  deleteReminder: (id: string) => void;
  toggleReminderComplete: (id: string) => void;
  
  // Media state
  mediaState: MediaState;
  toggleWebcam: () => void;
  toggleMicrophone: () => void;
  toggleVoice: () => void;
  setScreenMode: (mode: MediaState['screenMode']) => void;
  
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
}

const generateId = () => Math.random().toString(36).substr(2, 9);

export const useSarahStore = create<SarahState>()(
  persist(
    (set, get) => ({
      // Messages
      messages: [
        {
          id: '1',
          role: 'assistant',
          content: "Hi! I'm Sarah — ready when you are. Try asking me anything.",
          timestamp: new Date(),
        }
      ],
      isTyping: false,
      addMessage: (message) => set((state) => ({
        messages: [...state.messages, {
          ...message,
          id: generateId(),
          timestamp: new Date(),
        }]
      })),
      clearMessages: () => set({ messages: [] }),
      setTyping: (typing) => set({ isTyping: typing }),
      
      // Chat threads
      threads: [
        { id: '1', title: 'Getting Started', preview: "Hi! I'm Sarah — ready when you are.", timestamp: new Date(), messageCount: 1 },
        { id: '2', title: 'Previous Chat', preview: 'Thanks for the help!', timestamp: new Date(Date.now() - 86400000), messageCount: 15 },
        { id: '3', title: 'Research Session', preview: 'Let me look that up for you...', timestamp: new Date(Date.now() - 172800000), messageCount: 8 },
      ],
      activeThreadId: '1',
      setActiveThread: (id) => set({ activeThreadId: id }),
      
      // Contacts
      contacts: [
        { id: '1', name: 'John Doe', email: 'john@example.com', status: 'online' },
        { id: '2', name: 'Jane Smith', phone: '+1 555-0123', status: 'away' },
        { id: '3', name: 'Bob Wilson', address: 'bob@company.com', status: 'offline' },
      ],
      addContact: (contact) => set((state) => ({
        contacts: [...state.contacts, { ...contact, id: generateId() }]
      })),
      updateContact: (id, updates) => set((state) => ({
        contacts: state.contacts.map(c => c.id === id ? { ...c, ...updates } : c)
      })),
      deleteContact: (id) => set((state) => ({
        contacts: state.contacts.filter(c => c.id !== id)
      })),
      
      // Reminders
      reminders: [
        { id: '1', title: 'Team meeting', dueDate: new Date(Date.now() + 3600000), completed: false, priority: 'high' },
        { id: '2', title: 'Review documents', dueDate: new Date(Date.now() + 86400000), completed: false, priority: 'medium' },
        { id: '3', title: 'Send weekly report', dueDate: new Date(Date.now() + 172800000), completed: true, priority: 'low' },
      ],
      addReminder: (reminder) => set((state) => ({
        reminders: [...state.reminders, { ...reminder, id: generateId() }]
      })),
      updateReminder: (id, updates) => set((state) => ({
        reminders: state.reminders.map(r => r.id === id ? { ...r, ...updates } : r)
      })),
      deleteReminder: (id) => set((state) => ({
        reminders: state.reminders.filter(r => r.id !== id)
      })),
      toggleReminderComplete: (id) => set((state) => ({
        reminders: state.reminders.map(r => 
          r.id === id ? { ...r, completed: !r.completed } : r
        )
      })),
      
      // Media state
      mediaState: {
        webcamEnabled: true,
        microphoneEnabled: true,
        voiceEnabled: true,
        screenMode: 'avatar_2d',
      },
      toggleWebcam: () => set((state) => ({
        mediaState: { ...state.mediaState, webcamEnabled: !state.mediaState.webcamEnabled }
      })),
      toggleMicrophone: () => set((state) => ({
        mediaState: { ...state.mediaState, microphoneEnabled: !state.mediaState.microphoneEnabled }
      })),
      toggleVoice: () => set((state) => ({
        mediaState: { ...state.mediaState, voiceEnabled: !state.mediaState.voiceEnabled }
      })),
      setScreenMode: (mode) => set((state) => ({
        mediaState: { ...state.mediaState, screenMode: mode }
      })),
      
      // Settings
      settings: {
        selectedVoice: 'default',
        selectedTheme: 'dark',
        autoSpeak: true,
        soundEffects: true,
        notifications: true,
        mode: 'any',
      },
      updateSettings: (updates) => set((state) => ({
        settings: { ...state.settings, ...updates }
      })),
      
      // Available options
      voices: [
        { id: 'default', name: 'Sarah (Default)', language: 'en-US', gender: 'female' },
        { id: 'emma', name: 'Emma', language: 'en-GB', gender: 'female' },
        { id: 'alex', name: 'Alex', language: 'en-US', gender: 'male' },
      ],
      themes: [
        { id: 'dark', name: 'Default Dark', filename: 'Dark_Theme.css' },
        { id: 'light', name: 'Light', filename: 'Light_Theme.css' },
        { id: 'matrix', name: 'Matrix', filename: 'Matrix_Theme.css' },
        { id: 'tron', name: 'Tron', filename: 'Tron.css' },
        { id: 'hal2000', name: 'HAL 2000', filename: 'HAL2000_Theme.css' },
        { id: 'skynet', name: 'Skynet', filename: 'Skynet_Theme.css' },
        { id: 'vibrant', name: 'Vibrant', filename: 'Vibrant_Theme.css' },
      ],
      setVoices: (voices) => set({ voices }),
      setThemes: (themes) => set({ themes }),
      
      // Bootstrap data
      bootstrapData: null,
      setBootstrapData: (data) => set({ bootstrapData: data }),
      
      // UI State
      leftSidebarCollapsed: true,
      rightSidebarCollapsed: true,
      leftDrawerOpen: false,
      rightDrawerOpen: false,
      settingsOpen: false,
      rightPanelPage: 'contacts' as RightPanelPage,
      hasPlayedWelcome: false,
      backendReady: false,
      toggleLeftSidebar: () => set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),
      toggleRightSidebar: () => set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),
      setLeftDrawerOpen: (open) => set({ leftDrawerOpen: open }),
      setRightDrawerOpen: (open) => set({ rightDrawerOpen: open }),
      setSettingsOpen: (open) => set({ settingsOpen: open }),
      setRightPanelPage: (page) => set({ rightPanelPage: page }),
      setHasPlayedWelcome: (played) => set({ hasPlayedWelcome: played }),
      setBackendReady: (ready) => set({ backendReady: ready }),
    }),
    {
      name: 'sarah-memory-storage',
      partialize: (state) => ({
        settings: state.settings,
        contacts: state.contacts,
        reminders: state.reminders,
      }),
    }
  )
);

// SarahMemory WebUI Type Definitions

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
}

export interface Attachment {
  id: string;
  name: string;
  type: string;
  url?: string;
  size?: number;
}

export interface ChatThread {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messageCount: number;
}

export interface Contact {
  id: string;
  name: string;
  address?: string;
  phone?: string;
  email?: string;
  notes?: string;
  avatar?: string;
  status?: 'online' | 'offline' | 'busy' | 'away';
}

export interface Reminder {
  id: string;
  title: string;
  description?: string;
  dueDate: Date;
  completed: boolean;
  repeat?: 'none' | 'daily' | 'weekly' | 'monthly';
  priority?: 'low' | 'medium' | 'high';
}

export interface VoiceOption {
  id: string;
  name: string;
  language?: string;
  gender?: 'male' | 'female' | 'neutral';
}

export interface ThemeOption {
  id: string;
  name: string;
  filename: string;
}

export interface MediaState {
  webcamEnabled: boolean;
  microphoneEnabled: boolean;
  voiceEnabled: boolean;
  screenMode: 'avatar_2d' | 'avatar_3d' | 'desktop_mirror' | 'media' | 'idle';
}

export interface SystemStatus {
  local: boolean;
  web: boolean;
  api: boolean;
  network: boolean;
  mode: string;
}

export type DesktopShortcutKind = 'app' | 'trash' | 'url' | 'custom';

export interface DesktopShortcut {
  id: string;
  label: string;
  kind: DesktopShortcutKind;
  windowId?: string;
  url?: string;
  icon?: string;
}

export interface DesktopShortcutPosition {
  x: number;
  y: number;
}

export interface Settings {
  selectedVoice: string;
  selectedTheme: string;
  autoSpeak: boolean;
  soundEffects: boolean;
  notifications: boolean;
  mode?: string;
  advancedStudioMode?: boolean;
  uiMode?: 'simple' | 'operator' | 'engineer';
  localOnlyMode?: boolean;
  wallpaperUrl?: string;
  wallpaperMode?: 'cover' | 'contain' | 'stretch' | 'tile' | 'center';
  panelTransparency?: 'solid' | 'glass' | 'translucent';
  backgroundBrightness?: number;
  backgroundOverlay?: number;
  backgroundBlur?: number;
  panelOpacity?: number;
  shellDensity?: 'compact' | 'comfortable' | 'operator';
  activeWorkspace?: 'chat' | 'research' | 'operator' | 'engineer' | 'media';
  desktopShortcuts?: DesktopShortcut[];
  desktopShortcutPositions?: Record<string, DesktopShortcutPosition>;
  masterVolume?: number;
  outputVolume?: number;
  inputVolume?: number;
  bassLevel?: number;
  trebleLevel?: number;
  balance?: number;
  spatialAudio?: boolean;
  noiseSuppression?: boolean;
  manualClockDate?: string;
  manualClockTime?: string;
  manualClockTimezone?: string;
}

import { create } from 'zustand';

/**
 * Preview Display Router Store
 * Manages what the Preview Surface shows at any given time.
 * Modules send intents to change the preview without fighting.
 */

export type PreviewType = 
  | 'avatar'      // 2D/3D avatar
  | 'image'       // Image preview
  | 'audio'       // Audio player
  | 'video'       // Video player
  | 'call'        // Dual video call view
  | 'mirror';     // Desktop mirror

export interface PreviewState {
  type: PreviewType;
  mode?: string;        // e.g., '2d' | '3d' for avatar
  mediaId?: string;     // ID of the media to display
  mediaUrl?: string;    // URL of the media
  mediaBase64?: string; // Base64 data if URL unavailable
  callId?: string;      // ID for active call
  metadata?: Record<string, any>;
}

interface PreviewStoreState {
  current: PreviewState;
  previous: PreviewState | null;
  
  // Set preview with intent pattern
  setPreview: (type: PreviewType, options?: Omit<PreviewState, 'type'>) => void;
  
  // Restore previous preview (e.g., after call ends)
  restorePrevious: () => void;
  
  // Reset to default avatar view
  resetToAvatar: (mode?: '2d' | '3d') => void;
  
  // Quick setters for common cases
  showImage: (mediaId: string, url?: string, base64?: string) => void;
  showAudio: (mediaId: string, url?: string, base64?: string) => void;
  showVideo: (mediaId: string, url?: string) => void;
  showCall: (callId: string) => void;
  endCall: () => void;
}

const defaultPreview: PreviewState = {
  type: 'avatar',
  mode: '2d',
};

export const usePreviewStore = create<PreviewStoreState>((set, get) => ({
  current: { ...defaultPreview },
  previous: null,
  
  setPreview: (type, options = {}) => {
    const current = get().current;
    set({
      previous: current,
      current: { type, ...options },
    });
  },
  
  restorePrevious: () => {
    const prev = get().previous;
    if (prev) {
      set({
        current: prev,
        previous: null,
      });
    } else {
      // Fallback to avatar if no previous
      set({
        current: { ...defaultPreview },
        previous: null,
      });
    }
  },
  
  resetToAvatar: (mode = '2d') => {
    set({
      current: { type: 'avatar', mode },
      previous: null,
    });
  },
  
  showImage: (mediaId, url, base64) => {
    const current = get().current;
    set({
      previous: current,
      current: { type: 'image', mediaId, mediaUrl: url, mediaBase64: base64 },
    });
  },
  
  showAudio: (mediaId, url, base64) => {
    const current = get().current;
    set({
      previous: current,
      current: { type: 'audio', mediaId, mediaUrl: url, mediaBase64: base64 },
    });
  },
  
  showVideo: (mediaId, url) => {
    const current = get().current;
    set({
      previous: current,
      current: { type: 'video', mediaId, mediaUrl: url },
    });
  },
  
  showCall: (callId) => {
    const current = get().current;
    set({
      previous: current,
      current: { type: 'call', callId },
    });
  },
  
  endCall: () => {
    // Restore previous preview when call ends
    get().restorePrevious();
  },
}));

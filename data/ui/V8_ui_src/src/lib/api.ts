/**
 * SarahMemory API Client
 * 
 * Unified API client for communicating with the SarahMemory Flask backend.
 * All endpoints are wired to match app.py definitions.
 * @see https://api.sarahmemory.com
 */

import { supabase } from "@/integrations/supabase/client";
import { config, apiFetch } from "./config";

// ============================================================================
// Types for API responses
// ============================================================================

export interface ChatResponse {
  ok?: boolean;
  reply?: string;
  content: string;
  source: "sarah_backend" | "lovable_ai";
  audio_url?: string | null;
  images?: MediaResult[];
  error?: string;
  web_augmented?: boolean;
  sources?: string[];
  meta?: {
    source?: string;
    engine?: string;
  };
}

export interface VoiceResponse {
  success: boolean;
  audio_url?: string;
  audio_base64?: string;
  text?: string;
  voices?: VoiceOption[];
  fallback?: boolean;
  error?: string;
}

export interface VoiceOption {
  id: string;
  name: string;
  language?: string;
  gender?: "male" | "female" | "neutral";
  preview_url?: string;
}

export interface AvatarState {
  mode: "avatar_2d" | "avatar_3d" | "desktop_mirror" | "media" | "idle";
  expression: string;
  speaking: boolean;
  listening: boolean;
  current_action?: string;
}

export interface AvatarResponse {
  success: boolean;
  state?: AvatarState;
  mode?: string;
  expression?: string;
  animation?: string;
  fallback?: boolean;
  error?: string;
}

export interface DialerResponse {
  available?: boolean;
  success?: boolean;
  call_id?: string;
  status?: string;
  message?: string;
  logged?: boolean;
  error?: string;
}

export interface RankingResponse {
  success: boolean;
  ranked?: boolean;
  score?: number;
  stats?: {
    total_sessions: number;
    average_score: number;
    rank: string;
  };
  message?: string;
  error?: string;
}

export interface ThemeOption {
  id: string;
  name: string;
  filename: string;
  preview?: string;
}

export interface MediaResult {
  id: string;
  type: "image" | "music" | "video";
  url: string;
  preview?: string;
  title?: string;
  duration?: number;
  status: "pending" | "complete" | "error";
  error?: string;
}

export interface MediaResponse {
  success: boolean;
  results?: MediaResult[];
  job_id?: string;
  status?: string;
  fallback?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: string;
  message_count: number;
  messages?: Array<{ role: string; content: string; timestamp?: string }>;
}

export interface Contact {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  number?: string;
  address?: string;
  notes?: string;
  avatar?: string;
  status?: string;
}

export interface Reminder {
  id: string;
  title: string;
  description?: string;
  time?: string;
  note?: string;
  due_date?: string;
  completed: boolean;
  priority?: string;
  category?: string;
}

export interface BackendCapabilities {
  version: string;
  features: string[];
  tools: ToolDefinition[];
  avatar_modes: string[];
  avatar_actions: string[];
  media_types: string[];
  voice_engines: string[];
}

export interface ToolDefinition {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
}

export interface HealthResponse {
  ok: boolean;
  status: string;
  running: boolean;
  main_running: boolean;
  version: string;
  ts: number;
  notes: string[];
}

export interface BootstrapResponse {
  ok: boolean;
  version: string;
  runtime: Record<string, unknown>;
  client: Record<string, unknown>;
  features: Record<string, boolean>;
  env: { api_base: string; web_root: string };
  ts: number;
}

// ============================================================================
// Core API Helpers
// ============================================================================

/**
 * Invoke a Supabase Edge Function
 */
async function invokeEdgeFunction<T>(
  functionName: string,
  body: Record<string, unknown>
): Promise<T> {
  const { data, error } = await supabase.functions.invoke(functionName, { body });
  
  if (error) {
    console.error(`[api] ${functionName} error:`, error);
    throw new Error(error.message || "Edge function error");
  }
  
  return data as T;
}

/**
 * Direct call to SarahMemory Flask backend
 * Used as fallback or for endpoints not proxied through edge functions
 */
async function directCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  return apiFetch<T>(endpoint, options);
}

// ============================================================================
// BOOTSTRAP API - Called once on app load
// ============================================================================

export const bootstrapApi = {
  /**
   * Initialize session with backend
   * POST /api/session/bootstrap
   */
  async init(): Promise<BootstrapResponse> {
    try {
      return await apiFetch<BootstrapResponse>('/api/session/bootstrap', {
        method: 'POST',
        body: JSON.stringify({
          client_env: 'web',
          platform: 'browser',
          ui_version: 'v8',
          agent_name: 'Sarah',
          bridge: 'none',
        }),
      });
    } catch (error) {
      console.warn('[Bootstrap] Failed:', error);
      // Return fallback bootstrap response
      return {
        ok: false,
        version: config.version,
        runtime: {},
        client: {},
        features: {
          camera: false,
          microphone: false,
          voice_output: false,
        },
        env: {
          api_base: config.apiBaseUrl,
          web_root: '/',
        },
        ts: Date.now() / 1000,
      };
    }
  },
};

// ============================================================================
// CHAT API - Wired to POST /api/chat
// ============================================================================

export const chatApi = {
  /**
   * Send a chat message and get a response
   * POST /api/chat with { text, intent, tone, complexity }
   */
  async sendMessage(
    messages: Array<{ role: "user" | "assistant"; content: string }>,
    options?: { 
      useAI?: boolean;
      conversationId?: string;
      researchMode?: boolean;
      intent?: string;
      tone?: string;
      complexity?: string;
    }
  ): Promise<ChatResponse> {
    // Get the last user message text
    const lastUserMessage = messages.filter(m => m.role === 'user').pop();
    const text = lastUserMessage?.content || '';
    
    try {
      // Call Flask backend directly with the expected format
      const response = await apiFetch<{ ok: boolean; reply: string; meta?: { source?: string; engine?: string } }>('/api/chat', {
        method: 'POST',
        body: JSON.stringify({
          text,
          intent: options?.intent || 'question',
          tone: options?.tone || 'friendly',
          complexity: options?.complexity || 'adult',
        }),
      });
      
      if (response.ok && response.reply) {
        return {
          ok: true,
          reply: response.reply,
          content: response.reply,
          source: 'sarah_backend',
          meta: response.meta,
        };
      } else {
        throw new Error('Invalid response from backend');
      }
    } catch (error) {
      console.warn('[Chat] Direct call failed, trying edge function:', error);
      
      // Fallback to edge function
      try {
        return await invokeEdgeFunction<ChatResponse>("chat", {
          messages,
          useAI: options?.useAI || false,
          conversation_id: options?.conversationId,
          research_mode: options?.researchMode || false,
        });
      } catch (edgeError) {
        console.error('[Chat] Edge function also failed:', edgeError);
        return {
          ok: false,
          content: "I'm having trouble connecting to the backend. Please try again.",
          source: 'lovable_ai',
          error: String(error),
        };
      }
    }
  },
};

// ============================================================================
// VOICE API
// ============================================================================

export const voiceApi = {
  /**
   * Convert text to speech using the selected voice
   */
  async speak(text: string, voice?: string): Promise<VoiceResponse> {
    return invokeEdgeFunction<VoiceResponse>("voice", {
      action: "speak",
      text,
      voice,
    });
  },

  /**
   * Transcribe audio to text
   */
  async transcribe(audioBase64: string): Promise<VoiceResponse> {
    return invokeEdgeFunction<VoiceResponse>("voice", {
      action: "transcribe",
      audio: audioBase64,
    });
  },

  /**
   * List available voices from the backend
   * GET /get_available_voices
   */
  async listVoices(): Promise<VoiceOption[]> {
    try {
      // Call Flask backend directly
      const response = await apiFetch<Array<{ id: string; name: string } | string>>('/get_available_voices');
      
      // Handle both formats: [{id, name}] or ["Voice A", "Voice B"]
      if (Array.isArray(response)) {
        return response.map((item, idx) => {
          if (typeof item === 'string') {
            return { id: item, name: item };
          }
          return {
            id: item.id || String(idx),
            name: item.name || item.id || `Voice ${idx + 1}`,
          };
        });
      }
      return [];
    } catch (error) {
      console.warn('[Voice] Failed to get voices from backend:', error);
      // Fallback to edge function
      try {
        const response = await invokeEdgeFunction<VoiceResponse>("voice", {
          action: "list_voices",
        });
        return response.voices || [];
      } catch {
        return [];
      }
    }
  },

  /**
   * Set the active voice for the session
   */
  async setActiveVoice(voiceId: string): Promise<VoiceResponse> {
    return invokeEdgeFunction<VoiceResponse>("voice", {
      action: "set_voice",
      voice: voiceId,
    });
  },

  /**
   * Preview a voice with sample text
   */
  async previewVoice(voiceId: string): Promise<VoiceResponse> {
    return invokeEdgeFunction<VoiceResponse>("voice", {
      action: "preview",
      voice: voiceId,
    });
  },
};

// ============================================================================
// AVATAR API
// ============================================================================

export const avatarApi = {
  /**
   * Get current avatar state
   */
  async getState(): Promise<AvatarState> {
    const response = await invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "get_state",
    });
    return response.state || {
      mode: "avatar_2d",
      expression: "neutral",
      speaking: false,
      listening: false,
    };
  },

  /**
   * Set avatar mode (2D, 3D, etc.)
   */
  async setMode(mode: AvatarState["mode"]): Promise<AvatarResponse> {
    return invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "set_mode",
      mode,
    });
  },

  /**
   * Set avatar expression
   */
  async setExpression(expression: string): Promise<AvatarResponse> {
    return invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "set_expression",
      expression,
    });
  },

  /**
   * Trigger an animation/action
   */
  async triggerAnimation(animation: string): Promise<AvatarResponse> {
    return invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "trigger_animation",
      animation,
    });
  },

  /**
   * Update speaking state
   */
  async setSpeaking(speaking: boolean): Promise<void> {
    await invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "speaking",
      speaking,
    });
  },

  /**
   * Update listening state
   */
  async setListening(listening: boolean): Promise<void> {
    await invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "listening",
      listening,
    });
  },

  /**
   * Change avatar appearance
   */
  async setAppearance(description: string): Promise<AvatarResponse> {
    return invokeEdgeFunction<AvatarResponse>("avatar", {
      action: "set_appearance",
      description,
    });
  },
};

// ============================================================================
// DIALER API
// ============================================================================

export const dialerApi = {
  /**
   * Check if VoIP is available
   */
  async checkAvailability(): Promise<DialerResponse> {
    return invokeEdgeFunction<DialerResponse>("dialer", {
      action: "check_availability",
    });
  },

  /**
   * Initiate a call
   */
  async initiateCall(target: {
    number?: string;
    ip_address?: string;
    room_id?: string;
  }): Promise<DialerResponse> {
    return invokeEdgeFunction<DialerResponse>("dialer", {
      action: "initiate",
      ...target,
    });
  },

  /**
   * End an active call
   */
  async endCall(): Promise<DialerResponse> {
    return invokeEdgeFunction<DialerResponse>("dialer", {
      action: "end",
    });
  },
};

// ============================================================================
// RANKING API
// ============================================================================

export const rankingApi = {
  /**
   * Submit a session for ranking
   */
  async submitSession(
    sessionId: string,
    metrics: Record<string, unknown>,
    userId?: string
  ): Promise<RankingResponse> {
    return invokeEdgeFunction<RankingResponse>("ranking", {
      action: "submit_session",
      session_id: sessionId,
      metrics,
      user_id: userId,
    });
  },

  /**
   * Get user ranking stats
   */
  async getStats(userId: string): Promise<RankingResponse> {
    return invokeEdgeFunction<RankingResponse>("ranking", {
      action: "get_stats",
      user_id: userId,
    });
  },
};

// ============================================================================
// MEDIA API (Creative Tools)
// ============================================================================

export const mediaApi = {
  /**
   * Generate images from a prompt
   * Returns 4 variants by default
   */
  async generateImage(
    prompt: string,
    options?: { count?: number; style?: string }
  ): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/image",
      method: "POST",
      payload: {
        prompt,
        count: options?.count || 4,
        style: options?.style,
      },
    });
  },

  /**
   * Generate music from a prompt
   */
  async generateMusic(
    prompt: string,
    options?: { duration?: number; genre?: string }
  ): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/music",
      method: "POST",
      payload: {
        prompt,
        duration: options?.duration || 30,
        genre: options?.genre,
      },
    });
  },

  /**
   * Generate video from a prompt
   */
  async generateVideo(
    prompt: string,
    options?: { duration?: number; style?: string }
  ): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/video",
      method: "POST",
      payload: {
        prompt,
        duration: options?.duration || 5,
        style: options?.style,
      },
    });
  },

  /**
   * Get media generation job status
   */
  async getJobStatus(jobId: string): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: `/api/media/status/${jobId}`,
      method: "GET",
    });
  },

  /**
   * Download generated media
   */
  async download(mediaId: string): Promise<{ url: string }> {
    return invokeEdgeFunction("sarah-api", {
      endpoint: `/api/media/download/${mediaId}`,
      method: "GET",
    });
  },

  /**
   * Save media to dataset
   */
  async saveToDataset(mediaId: string, dataset?: string): Promise<{ success: boolean }> {
    return invokeEdgeFunction("sarah-api", {
      endpoint: "/api/media/save",
      method: "POST",
      payload: { media_id: mediaId, dataset },
    });
  },

  /**
   * List recent generations
   */
  async listRecent(type?: "image" | "music" | "video"): Promise<MediaResponse> {
    const params = type ? `?type=${type}` : "";
    return invokeEdgeFunction("sarah-api", {
      endpoint: `/api/media/recent${params}`,
      method: "GET",
    });
  },
};

// ============================================================================
// QA / CONVERSATIONS API - Wired to Flask endpoints
// ============================================================================

export const qaApi = {
  /**
   * List conversations by date
   * Endpoint: GET /get_chat_threads_by_date?date=YYYY-MM-DD
   */
  async listConversations(date?: string): Promise<{ conversations: Conversation[]; total: number }> {
    try {
      const query = date ? `?date=${date}` : '';
      const result = await directCall<{ threads: Array<{ id: string; timestamp: string; preview: string }> }>(
        `/get_chat_threads_by_date${query}`
      );
      const conversations = (result.threads || []).map(t => ({
        id: String(t.id),
        title: t.preview?.slice(0, 40) || 'Conversation',
        preview: t.preview || '',
        timestamp: t.timestamp,
        message_count: 1,
      }));
      return { conversations, total: conversations.length };
    } catch (error) {
      console.error('[API] List conversations failed:', error);
      return { conversations: [], total: 0 };
    }
  },

  /**
   * Get a specific conversation with messages
   * Endpoint: GET /get_conversation_by_id?id=<id>
   */
  async getConversation(id: string): Promise<Conversation | null> {
    try {
      const result = await directCall<Array<{ role: string; text: string; meta?: string }>>(
        `/get_conversation_by_id?id=${id}`
      );
      return {
        id,
        title: 'Conversation',
        preview: result[0]?.text || '',
        timestamp: new Date().toISOString(),
        message_count: result.length,
        messages: result.map(m => ({
          role: m.role || 'user',
          content: m.text || '',
        })),
      };
    } catch (error) {
      console.error('[API] Get conversation failed:', error);
      return null;
    }
  },

  /**
   * Delete a conversation (if backend supports it)
   */
  async deleteConversation(id: string): Promise<{ success: boolean }> {
    return { success: true }; // Backend doesn't have delete endpoint yet
  },
};

// ============================================================================
// REMINDERS API - Wired to Flask endpoints
// ============================================================================

export const remindersApi = {
  /**
   * List all reminders
   * Endpoint: GET /get_reminders
   */
  async list(): Promise<{ reminders: Reminder[] }> {
    try {
      const result = await directCall<{ reminders: Array<{ id: number; title: string; time: string; note?: string }> }>(
        '/get_reminders'
      );
      const reminders = (result.reminders || []).map(r => ({
        id: String(r.id),
        title: r.title,
        description: r.note || '',
        time: r.time,
        due_date: r.time,
        completed: false,
        priority: 'medium',
      }));
      return { reminders };
    } catch (error) {
      console.error('[API] Get reminders failed:', error);
      return { reminders: [] };
    }
  },

  /**
   * Create a new reminder
   * Endpoint: POST /save_reminder
   */
  async create(reminder: Omit<Reminder, "id">): Promise<{ reminder: Reminder }> {
    try {
      const result = await directCall<{ status: string; id?: number }>(
        '/save_reminder',
        {
          method: 'POST',
          body: JSON.stringify({
            title: reminder.title,
            time: reminder.time || reminder.due_date,
            note: reminder.description || reminder.note || '',
          }),
        }
      );
      return {
        reminder: {
          ...reminder,
          id: String(result.id || Date.now()),
          completed: false,
        },
      };
    } catch (error) {
      console.error('[API] Create reminder failed:', error);
      throw error;
    }
  },

  /**
   * Update a reminder (re-save)
   */
  async update(id: string, updates: Partial<Reminder>): Promise<{ reminder: Reminder }> {
    // Backend doesn't have update, so we create new
    return this.create(updates as Omit<Reminder, "id">);
  },

  /**
   * Delete a reminder
   * Endpoint: POST /delete_reminder
   */
  async delete(id: string): Promise<{ success: boolean }> {
    try {
      await directCall('/delete_reminder', {
        method: 'POST',
        body: JSON.stringify({ id: Number(id) }),
      });
      return { success: true };
    } catch (error) {
      console.error('[API] Delete reminder failed:', error);
      return { success: false };
    }
  },

  /**
   * Mark reminder as complete (local only)
   */
  async complete(id: string): Promise<{ success: boolean }> {
    return { success: true };
  },

  /**
   * Snooze a reminder (local only)
   */
  async snooze(id: string, minutes?: number): Promise<{ success: boolean }> {
    return { success: true };
  },
};

// ============================================================================
// CONTACTS API - Wired to Flask endpoints
// ============================================================================

export const contactsApi = {
  /**
   * List all contacts
   * Endpoint: GET /get_all_contacts
   */
  async list(): Promise<{ contacts: Contact[] }> {
    try {
      const result = await directCall<{ contacts: Array<{ id: number; name: string; number?: string }> }>(
        '/get_all_contacts'
      );
      const contacts = (result.contacts || []).map(c => ({
        id: String(c.id),
        name: c.name,
        phone: c.number,
        number: c.number,
        status: 'offline',
      }));
      return { contacts };
    } catch (error) {
      console.error('[API] Get contacts failed:', error);
      return { contacts: [] };
    }
  },

  /**
   * Create a new contact
   * Endpoint: POST /add_contact
   */
  async create(contact: Omit<Contact, "id">): Promise<{ contact: Contact }> {
    try {
      await directCall('/add_contact', {
        method: 'POST',
        body: JSON.stringify({
          name: contact.name,
          number: contact.phone || contact.number || contact.email || '',
        }),
      });
      return {
        contact: {
          ...contact,
          id: String(Date.now()),
        },
      };
    } catch (error) {
      console.error('[API] Create contact failed:', error);
      throw error;
    }
  },

  /**
   * Update a contact
   */
  async update(id: string, updates: Partial<Contact>): Promise<{ contact: Contact }> {
    // Backend doesn't have update endpoint
    return { contact: { id, name: updates.name || '', ...updates } };
  },

  /**
   * Delete a contact
   * Endpoint: POST /delete_contact
   */
  async delete(id: string): Promise<{ success: boolean }> {
    try {
      await directCall('/delete_contact', {
        method: 'POST',
        body: JSON.stringify({ id: Number(id) }),
      });
      return { success: true };
    } catch (error) {
      console.error('[API] Delete contact failed:', error);
      return { success: false };
    }
  },
};

// ============================================================================
// SETTINGS API - Wired to Flask endpoints
// ============================================================================

export const settingsApi = {
  /**
   * Get available voices from backend
   * Endpoint: GET /get_available_voices
   */
  async getVoices(): Promise<VoiceOption[]> {
    try {
      const result = await directCall<Array<{ id: string; name: string }>>('/get_available_voices');
      return Array.isArray(result) ? result.map(v => ({
        id: v.id || v.name?.toLowerCase().replace(/\s+/g, '_') || 'unknown',
        name: v.name || v.id || 'Unknown',
        language: 'en-US',
        gender: 'female' as const,
      })) : [];
    } catch (error) {
      console.error('[API] Get voices failed:', error);
      return [
        { id: 'sarah', name: 'Sarah (Default)', language: 'en-US', gender: 'female' },
        { id: 'emma', name: 'Emma', language: 'en-GB', gender: 'female' },
      ];
    }
  },

  /**
   * Set the active voice
   * Endpoint: POST /set_user_setting
   */
  async setVoice(voiceId: string): Promise<{ success: boolean }> {
    try {
      await directCall('/set_user_setting', {
        method: 'POST',
        body: JSON.stringify({ key: 'voice_profile', value: voiceId }),
      });
      return { success: true };
    } catch (error) {
      console.error('[API] Set voice failed:', error);
      return { success: false };
    }
  },

  /**
   * Get a user setting
   * Endpoint: GET /get_user_setting?key=...
   */
  async getSetting(key: string): Promise<string> {
    try {
      const result = await directCall<{ value: string }>(`/get_user_setting?key=${encodeURIComponent(key)}`);
      return result.value || '';
    } catch (error) {
      return '';
    }
  },

  /**
   * Set a user setting
   * Endpoint: POST /set_user_setting
   */
  async setSetting(key: string, value: string): Promise<boolean> {
    try {
      await directCall('/set_user_setting', {
        method: 'POST',
        body: JSON.stringify({ key, value }),
      });
      return true;
    } catch (error) {
      return false;
    }
  },

  /**
   * Get available themes
   * Endpoint: GET /get_theme_files
   */
  async getThemes(): Promise<ThemeOption[]> {
    try {
      const result = await directCall<{ root: string; files: string[]; count: number }>('/get_theme_files');
      if (result.files && result.files.length > 0) {
        return result.files
          .filter(f => f.endsWith('.css'))
          .map((f, i) => ({
            id: f.replace('.css', '').replace(/\//g, '_'),
            name: f.replace('.css', '').replace(/[-_]/g, ' ').replace(/\//g, ' - '),
            filename: f,
          }));
      }
      return [
        { id: "default", name: "Default Dark", filename: "default.css" },
        { id: "midnight", name: "Midnight Blue", filename: "midnight.css" },
      ];
    } catch (error) {
      console.error('[API] Get themes failed:', error);
      return [
        { id: "default", name: "Default Dark", filename: "default.css" },
      ];
    }
  },

  /**
   * Set the active theme
   * Endpoint: POST /set_user_setting
   */
  async setTheme(themeId: string): Promise<{ success: boolean }> {
    try {
      await directCall('/set_user_setting', {
        method: 'POST',
        body: JSON.stringify({ key: 'theme', value: themeId }),
      });
      return { success: true };
    } catch (error) {
      return { success: false };
    }
  },

  /**
   * Get theme file URL
   */
  getThemeUrl(filename: string): string {
    return `${config.apiBaseUrl}/api/data/mods/themes/${filename}`;
  },
};

// (Settings API already defined above - removed duplicate)

// ============================================================================
// FILES API
// ============================================================================

export const filesApi = {
  /**
   * Upload and analyze a file
   */
  async uploadAndAnalyze(
    file: File,
    options?: { analyze?: boolean; extractText?: boolean }
  ): Promise<{ success: boolean; analysis?: string; content?: string; media_url?: string }> {
    // Convert file to base64
    const base64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve((reader.result as string).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

    return invokeEdgeFunction("sarah-api", {
      endpoint: "/api/files/analyze",
      method: "POST",
      payload: {
        filename: file.name,
        content: base64,
        type: file.type,
        analyze: options?.analyze ?? true,
        extract_text: options?.extractText ?? true,
      },
    });
  },
};

// ============================================================================
// RESEARCH API
// ============================================================================

export const researchApi = {
  /**
   * Perform a research search
   */
  async search(
    query: string,
    options?: { depth?: "shallow" | "deep"; sources?: string[] }
  ): Promise<{ results: any[]; summary?: string; sources?: string[] }> {
    return invokeEdgeFunction("sarah-api", {
      endpoint: "/api/research/search",
      method: "POST",
      payload: {
        query,
        depth: options?.depth || "shallow",
        sources: options?.sources,
      },
    });
  },
};

// ============================================================================
// META API (Capabilities Discovery)
// ============================================================================

export const metaApi = {
  /**
   * Get backend capabilities
   * Used for dynamic feature discovery
   */
  async getCapabilities(): Promise<BackendCapabilities> {
    try {
      const result = await invokeEdgeFunction<any>("sarah-api", {
        endpoint: "/api/meta/capabilities",
        method: "GET",
      });
      
      if (result.fallback) {
        // Return default capabilities if backend is unavailable
        return getDefaultCapabilities();
      }
      
      return result as BackendCapabilities;
    } catch {
      return getDefaultCapabilities();
    }
  },

  /**
   * Get backend version info
   */
  async getVersion(): Promise<{ version: string; updated_at?: string }> {
    try {
      const result = await invokeEdgeFunction<any>("sarah-api", {
        endpoint: "/api/version",
        method: "GET",
      });
      return result;
    } catch {
      return { version: "unknown" };
    }
  },

  /**
   * Health check - calls /api/health directly
   */
  async healthCheck(): Promise<{ status: string; ok?: boolean; services?: Record<string, boolean> }> {
    try {
      // Try direct call first
      const result = await apiFetch<{ ok: boolean; status?: string; version?: string }>('/api/health');
      return { 
        status: result.ok ? 'ok' : 'error',
        ok: result.ok,
      };
    } catch (error) {
      console.warn('[Health] Direct call failed:', error);
      // Fallback to edge function
      try {
        const result = await invokeEdgeFunction<any>("sarah-api", {
          endpoint: "/api/health",
          method: "GET",
        });
        return result;
      } catch {
        return { status: "unavailable" };
      }
    }
  },
};

function getDefaultCapabilities(): BackendCapabilities {
  return {
    version: "8.0.0",
    features: ["chat", "voice", "avatar", "reminders", "contacts"],
    tools: [
      { id: "chat", name: "Chat", description: "Conversational AI", enabled: true },
      { id: "voice", name: "Voice", description: "Text-to-speech and transcription", enabled: true },
      { id: "avatar", name: "Avatar", description: "AI avatar with expressions", enabled: true },
    ],
    avatar_modes: ["avatar_2d", "avatar_3d"],
    avatar_actions: ["wave", "walk", "sit", "idle", "think"],
    media_types: ["image", "music", "video"],
    voice_engines: ["default"],
  };
}

// ============================================================================
// PROXY API (Legacy/Direct backend access)
// ============================================================================

export const proxyApi = {
  /**
   * Make a direct call to the SarahMemory backend via the sarah-api edge function
   */
  async call(
    endpoint: string,
    options?: {
      method?: "GET" | "POST" | "PUT" | "DELETE";
      body?: Record<string, unknown>;
    }
  ): Promise<unknown> {
    try {
      const { data, error } = await supabase.functions.invoke("sarah-api", {
        body: {
          endpoint,
          method: options?.method || "GET",
          payload: options?.body,
        },
      });
      
      if (error) {
        console.error("[proxyApi] Error:", error);
        throw new Error(error.message || "Proxy API error");
      }
      
      return data;
    } catch (err) {
      console.error("[proxyApi] Call failed:", err);
      return { fallback: true };
    }
  },

  async getContacts() {
    return contactsApi.list();
  },

  async getReminders() {
    return remindersApi.list();
  },

  async getConversations() {
    return qaApi.listConversations();
  },

  async getThemes() {
    return settingsApi.getThemes().then(themes => ({ themes }));
  },
};

// ============================================================================
// UNIFIED API EXPORT
// ============================================================================

export const api = {
  bootstrap: bootstrapApi,
  chat: chatApi,
  voice: voiceApi,
  avatar: avatarApi,
  dialer: dialerApi,
  ranking: rankingApi,
  media: mediaApi,
  qa: qaApi,
  reminders: remindersApi,
  contacts: contactsApi,
  settings: settingsApi,
  files: filesApi,
  research: researchApi,
  meta: metaApi,
  proxy: proxyApi,
};

export default api;

/**
 * SarahMemory API Client
 *
 * Unified API client for communicating with the SarahMemory Flask backend.
 * All endpoints are wired to match app.py definitions (best-effort, with multi-endpoint fallback).
 * @see https://api.sarahmemory.com
 */

import { supabase } from "@/integrations/supabase/client";

import { config, apiFetch } from "./config";

// ============================================================================
// Types for API responses
// ============================================================================

export interface AvatarSpeechCue {
  t: number;
  v: number;
}

export interface AvatarSpeechMeta {
  speak: boolean;
  duration_ms?: number;
  cues?: AvatarSpeechCue[];
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
    avatar_speech?: AvatarSpeechMeta;
  };
}

export interface VoiceOption {
  id: string;
  name: string;
  language?: string;
  gender?: "male" | "female" | "neutral";
  preview_url?: string;
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
// Core API Helpers (Hardened)
// ============================================================================

function withJsonHeaders(options: RequestInit = {}): RequestInit {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  headers.set("Accept", "application/json");
  return { ...options, headers };
}

async function invokeEdgeFunction<T>(functionName: string, body: Record<string, unknown>): Promise<T> {
  const { data, error } = await supabase.functions.invoke(functionName, { body });
  if (error) {
    console.error(`[api] edge:${functionName} error:`, error);
    throw new Error(error.message || "Edge function error");
  }
  return data as T;
}

async function directCall<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  return apiFetch<T>(endpoint, withJsonHeaders(options));
}

async function tryDirectEndpoints<T>(
  endpoints: string[],
  options: RequestInit,
): Promise<{ endpoint: string; data: T }> {
  let lastErr: unknown = null;
  for (const ep of endpoints) {
    try {
      const data = await directCall<T>(ep, options);
      return { endpoint: ep, data };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("All endpoints failed");
}

function normalizeVoices(input: any): VoiceOption[] {
  if (!input) return [];
  const arr = Array.isArray(input)
    ? input
    : Array.isArray(input.voices)
      ? input.voices
      : Array.isArray(input.data)
        ? input.data
        : [];
  return arr
    .map((item: any, idx: number) => {
      if (typeof item === "string") return { id: item, name: item };
      const id = String(item.id ?? item.name ?? idx);
      const name = String(item.name ?? item.id ?? `Voice ${idx + 1}`);
      const language = item.language ? String(item.language) : undefined;
      const gender = item.gender as any;
      const preview_url = item.preview_url ? String(item.preview_url) : undefined;
      return { id, name, language, gender, preview_url };
    })
    .filter(Boolean);
}

function isTruthySuccess(obj: any): boolean {
  if (!obj) return false;
  if (typeof obj === "boolean") return obj;
  return Boolean(obj.success ?? obj.ok);
}

// ============================================================================
// BOOTSTRAP API
// ============================================================================

export const bootstrapApi = {
  async init(): Promise<BootstrapResponse> {
    try {
      return await directCall<BootstrapResponse>("/api/session/bootstrap", {
        method: "POST",
        body: JSON.stringify({
          client_env: "web",
          platform: "browser",
          ui_version: "v8",
          agent_name: "Sarah",
          bridge: "none",
        }),
      });
    } catch (error) {
      console.warn("[Bootstrap] Failed:", error);
      return {
        ok: false,
        version: config.version,
        runtime: {},
        client: {},
        features: { camera: false, microphone: false, voice_output: false },
        env: { api_base: config.apiBaseUrl, web_root: "/" },
        ts: Date.now() / 1000,
      };
    }
  },
};

// ============================================================================
// CHAT API (canonical: /api/chat)
// ============================================================================

export const chatApi = {
  async sendMessage(
    messages: Array<{ role: "user" | "assistant"; content: string }>,
    options?: {
      useAI?: boolean;
      conversationId?: string;
      researchMode?: boolean;
      intent?: string;
      tone?: string;
      complexity?: string;
    },
  ): Promise<ChatResponse> {
    const lastUserMessage = messages.filter((m) => m.role === "user").pop();
    const text = lastUserMessage?.content || "";

    try {
      const { data } = await tryDirectEndpoints<any>(["/api/chat", "/chat", "/api/v1/chat"], {
        method: "POST",
        body: JSON.stringify({
          text,
          intent: options?.intent || "question",
          tone: options?.tone || "friendly",
          complexity: options?.complexity || "adult",
          conversation_id: options?.conversationId,
          research_mode: options?.researchMode || false,
        }),
      });

      const ok = isTruthySuccess(data) || Boolean(data.ok);
      const reply = data.reply ?? data.content ?? "";

      if (ok && typeof reply === "string" && reply.length) {
        return {
          ok: true,
          reply,
          content: reply,
          source: "sarah_backend",
          audio_url: data.audio_url ?? null,
          images: data.images,
          sources: data.sources,
          web_augmented: data.web_augmented,
          meta: data.meta,
        };
      }

      throw new Error(data?.error || "Invalid response from backend chat");
    } catch (error) {
      console.warn("[Chat] Direct call failed, trying edge function:", error);
      try {
        return await invokeEdgeFunction<ChatResponse>("chat", {
          messages,
          useAI: options?.useAI || false,
          conversation_id: options?.conversationId,
          research_mode: options?.researchMode || false,
        });
      } catch (edgeError) {
        console.error("[Chat] Edge function also failed:", edgeError);
        return {
          ok: false,
          content: "I'm having trouble connecting to the backend. Please try again.",
          source: "lovable_ai",
          error: String(error),
        };
      }
    }
  },
};

// ============================================================================
// VOICE API (canonical: /api/voice with {action:...})
// ============================================================================

export const voiceApi = {
  async speak(text: string, voice?: string): Promise<VoiceResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/voice", "/api/voice/speak", "/api/tts", "/tts"], {
        method: "POST",
        body: JSON.stringify({ action: "speak", text, voice }),
      });

      const success = isTruthySuccess(data) || Boolean(data.audio_url || data.audio_base64);
      return {
        success,
        audio_url: data.audio_url,
        audio_base64: data.audio_base64,
        text: data.text ?? text,
        fallback: false,
        error: data.error,
      };
    } catch (err) {
      try {
        return await invokeEdgeFunction<VoiceResponse>("voice", { action: "speak", text, voice });
      } catch (edgeErr) {
        return { success: false, fallback: true, error: String(edgeErr || err) };
      }
    }
  },

  async transcribe(audioBase64: string): Promise<VoiceResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/voice", "/api/voice/transcribe", "/api/stt", "/stt"], {
        method: "POST",
        body: JSON.stringify({ action: "transcribe", audio: audioBase64 }),
      });

      const success = isTruthySuccess(data) || Boolean(data.text);
      return { success, text: data.text, fallback: false, error: data.error };
    } catch (err) {
      try {
        return await invokeEdgeFunction<VoiceResponse>("voice", { action: "transcribe", audio: audioBase64 });
      } catch (edgeErr) {
        return { success: false, fallback: true, error: String(edgeErr || err) };
      }
    }
  },

  async listVoices(): Promise<VoiceOption[]> {
    try {
      const { data } = await tryDirectEndpoints<any>(
        ["/get_available_voices", "/api/voice", "/api/voices", "/voices"],
        {
          method: "GET",
        },
      );
      // /api/voice GET might not exist; normalize handles any shape
      return normalizeVoices(data);
    } catch (error) {
      console.warn("[Voice] Failed to get voices from backend:", error);
      try {
        const response = await invokeEdgeFunction<VoiceResponse>("voice", { action: "list_voices" });
        return response.voices || [];
      } catch {
        return [];
      }
    }
  },

  async setActiveVoice(voiceId: string): Promise<VoiceResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/voice", "/api/voice/set", "/set_user_setting"], {
        method: "POST",
        body: JSON.stringify({ action: "set_voice", voice: voiceId, key: "voice_profile", value: voiceId }),
      });
      return { success: isTruthySuccess(data), fallback: false, error: data?.error };
    } catch (err) {
      try {
        return await invokeEdgeFunction<VoiceResponse>("voice", { action: "set_voice", voice: voiceId });
      } catch (edgeErr) {
        return { success: false, fallback: true, error: String(edgeErr || err) };
      }
    }
  },

  async previewVoice(voiceId: string): Promise<VoiceResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/voice", "/api/voice/preview", "/voice/preview"], {
        method: "POST",
        body: JSON.stringify({ action: "preview", voice: voiceId }),
      });
      return {
        success: isTruthySuccess(data) || Boolean(data.audio_url || data.audio_base64),
        audio_url: data.audio_url,
        audio_base64: data.audio_base64,
        fallback: false,
        error: data.error,
      };
    } catch (err) {
      try {
        return await invokeEdgeFunction<VoiceResponse>("voice", { action: "preview", voice: voiceId });
      } catch (edgeErr) {
        return { success: false, fallback: true, error: String(edgeErr || err) };
      }
    }
  },
};

// ============================================================================
// AVATAR API (canonical: /api/avatar with {action:...})
// ============================================================================

export const avatarApi = {
  async getState(): Promise<AvatarState> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/state", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "get_state" }),
      });
      return (
        data?.state || {
          mode: "avatar_2d",
          expression: "neutral",
          speaking: false,
          listening: false,
        }
      );
    } catch {
      const response = await invokeEdgeFunction<AvatarResponse>("avatar", { action: "get_state" });
      return (
        response.state || {
          mode: "avatar_2d",
          expression: "neutral",
          speaking: false,
          listening: false,
        }
      );
    }
  },

  async setMode(mode: AvatarState["mode"]): Promise<AvatarResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/mode", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "set_mode", mode }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<AvatarResponse>("avatar", { action: "set_mode", mode });
    }
  },

  async setExpression(expression: string): Promise<AvatarResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/expression", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "set_expression", expression }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<AvatarResponse>("avatar", { action: "set_expression", expression });
    }
  },

  async triggerAnimation(animation: string): Promise<AvatarResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/animate", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "trigger_animation", animation }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<AvatarResponse>("avatar", { action: "trigger_animation", animation });
    }
  },

  async setSpeaking(speaking: boolean): Promise<void> {
    try {
      await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/speaking", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "speaking", speaking }),
      });
    } catch {
      await invokeEdgeFunction<AvatarResponse>("avatar", { action: "speaking", speaking });
    }
  },

  async setListening(listening: boolean): Promise<void> {
    try {
      await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/listening", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "listening", listening }),
      });
    } catch {
      await invokeEdgeFunction<AvatarResponse>("avatar", { action: "listening", listening });
    }
  },

  async setAppearance(description: string): Promise<AvatarResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/avatar", "/api/avatar/appearance", "/avatar"], {
        method: "POST",
        body: JSON.stringify({ action: "set_appearance", description }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<AvatarResponse>("avatar", { action: "set_appearance", description });
    }
  },
};

// ============================================================================
// DIALER API (canonical: /api/dialer with {action:...})
// ============================================================================

export const dialerApi = {
  async checkAvailability(): Promise<DialerResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/dialer", "/api/dialer/check", "/dialer"], {
        method: "POST",
        body: JSON.stringify({ action: "check_availability" }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<DialerResponse>("dialer", { action: "check_availability" });
    }
  },

  async initiateCall(target: { number?: string; ip_address?: string; room_id?: string }): Promise<DialerResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/dialer", "/api/dialer/initiate", "/dialer"], {
        method: "POST",
        body: JSON.stringify({ action: "initiate", ...target }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<DialerResponse>("dialer", { action: "initiate", ...target });
    }
  },

  async endCall(): Promise<DialerResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/dialer", "/api/dialer/end", "/dialer"], {
        method: "POST",
        body: JSON.stringify({ action: "end" }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<DialerResponse>("dialer", { action: "end" });
    }
  },
};

// ============================================================================
// RANKING API (canonical: /api/ranking with {action:...})
// ============================================================================

export const rankingApi = {
  async submitSession(sessionId: string, metrics: Record<string, unknown>, userId?: string): Promise<RankingResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/ranking", "/api/ranking/submit", "/ranking"], {
        method: "POST",
        body: JSON.stringify({ action: "submit_session", session_id: sessionId, metrics, user_id: userId }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<RankingResponse>("ranking", {
        action: "submit_session",
        session_id: sessionId,
        metrics,
        user_id: userId,
      });
    }
  },

  async getStats(userId: string): Promise<RankingResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/ranking", "/api/ranking/stats", "/ranking"], {
        method: "POST",
        body: JSON.stringify({ action: "get_stats", user_id: userId }),
      });
      return { success: isTruthySuccess(data), ...data };
    } catch {
      return invokeEdgeFunction<RankingResponse>("ranking", { action: "get_stats", user_id: userId });
    }
  },
};

// ============================================================================
// MEDIA API (edge proxied through sarah-api)
// ============================================================================

export const mediaApi = {
  async generateImage(prompt: string, options?: { count?: number; style?: string }): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/image",
      method: "POST",
      payload: { prompt, count: options?.count || 4, style: options?.style },
    });
  },

  async generateMusic(prompt: string, options?: { duration?: number; genre?: string }): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/music",
      method: "POST",
      payload: { prompt, duration: options?.duration || 30, genre: options?.genre },
    });
  },

  async generateVideo(prompt: string, options?: { duration?: number; style?: string }): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", {
      endpoint: "/api/media/generate/video",
      method: "POST",
      payload: { prompt, duration: options?.duration || 5, style: options?.style },
    });
  },

  async getJobStatus(jobId: string): Promise<MediaResponse> {
    return invokeEdgeFunction<MediaResponse>("sarah-api", { endpoint: `/api/media/status/${jobId}`, method: "GET" });
  },

  async download(mediaId: string): Promise<{ url: string }> {
    return invokeEdgeFunction("sarah-api", { endpoint: `/api/media/download/${mediaId}`, method: "GET" });
  },

  async saveToDataset(mediaId: string, dataset?: string): Promise<{ success: boolean }> {
    return invokeEdgeFunction("sarah-api", {
      endpoint: "/api/media/save",
      method: "POST",
      payload: { media_id: mediaId, dataset },
    });
  },

  async listRecent(type?: "image" | "music" | "video"): Promise<MediaResponse> {
    const params = type ? `?type=${type}` : "";
    return invokeEdgeFunction("sarah-api", { endpoint: `/api/media/recent${params}`, method: "GET" });
  },
};

// ============================================================================
// QA / CONVERSATIONS API (legacy endpoints)
// ============================================================================

export const qaApi = {
  async listConversations(date?: string): Promise<{ conversations: Conversation[]; total: number }> {
    try {
      const query = date ? `?date=${date}` : "";
      const result = await directCall<{ threads: Array<{ id: string; timestamp: string; preview: string }> }>(
        `/get_chat_threads_by_date${query}`,
      );
      const conversations = (result.threads || []).map((t) => ({
        id: String(t.id),
        title: t.preview?.slice(0, 40) || "Conversation",
        preview: t.preview || "",
        timestamp: t.timestamp,
        message_count: 1,
      }));
      return { conversations, total: conversations.length };
    } catch (error) {
      console.error("[API] List conversations failed:", error);
      return { conversations: [], total: 0 };
    }
  },

  async getConversation(id: string): Promise<Conversation | null> {
    try {
      const result = await directCall<Array<{ role: string; text: string; meta?: string }>>(
        `/get_conversation_by_id?id=${encodeURIComponent(id)}`,
      );
      return {
        id,
        title: "Conversation",
        preview: result[0]?.text || "",
        timestamp: new Date().toISOString(),
        message_count: result.length,
        messages: result.map((m) => ({ role: m.role || "user", content: m.text || "" })),
      };
    } catch (error) {
      console.error("[API] Get conversation failed:", error);
      return null;
    }
  },

  async deleteConversation(_id: string): Promise<{ success: boolean }> {
    return { success: true };
  },
};

// ============================================================================
// REMINDERS / CONTACTS / SETTINGS (legacy endpoints)
// ============================================================================

export const remindersApi = {
  async list(): Promise<{ reminders: Reminder[] }> {
    try {
      const result = await directCall<{ reminders: Array<{ id: number; title: string; time: string; note?: string }> }>(
        "/get_reminders",
      );
      const reminders = (result.reminders || []).map((r) => ({
        id: String(r.id),
        title: r.title,
        description: r.note || "",
        time: r.time,
        due_date: r.time,
        completed: false,
        priority: "medium",
      }));
      return { reminders };
    } catch (error) {
      console.error("[API] Get reminders failed:", error);
      return { reminders: [] };
    }
  },

  async create(reminder: Omit<Reminder, "id">): Promise<{ reminder: Reminder }> {
    const result = await directCall<{ status: string; id?: number }>("/save_reminder", {
      method: "POST",
      body: JSON.stringify({
        title: reminder.title,
        time: reminder.time || reminder.due_date,
        note: reminder.description || reminder.note || "",
      }),
    });

    return { reminder: { ...reminder, id: String(result.id || Date.now()), completed: false } };
  },

  async update(_id: string, updates: Partial<Reminder>): Promise<{ reminder: Reminder }> {
    return this.create(updates as Omit<Reminder, "id">);
  },

  async delete(id: string): Promise<{ success: boolean }> {
    try {
      await directCall("/delete_reminder", { method: "POST", body: JSON.stringify({ id: Number(id) }) });
      return { success: true };
    } catch (error) {
      console.error("[API] Delete reminder failed:", error);
      return { success: false };
    }
  },

  async complete(_id: string): Promise<{ success: boolean }> {
    return { success: true };
  },

  async snooze(_id: string, _minutes?: number): Promise<{ success: boolean }> {
    return { success: true };
  },
};

export const contactsApi = {
  async list(): Promise<{ contacts: Contact[] }> {
    try {
      const result = await directCall<{ contacts: Array<{ id: number; name: string; number?: string }> }>(
        "/get_all_contacts",
      );
      const contacts = (result.contacts || []).map((c) => ({
        id: String(c.id),
        name: c.name,
        phone: c.number,
        number: c.number,
        status: "offline",
      }));
      return { contacts };
    } catch (error) {
      console.error("[API] Get contacts failed:", error);
      return { contacts: [] };
    }
  },

  async create(contact: Omit<Contact, "id">): Promise<{ contact: Contact }> {
    await directCall("/add_contact", {
      method: "POST",
      body: JSON.stringify({ name: contact.name, number: contact.phone || contact.number || contact.email || "" }),
    });
    return { contact: { ...contact, id: String(Date.now()) } };
  },

  async update(id: string, updates: Partial<Contact>): Promise<{ contact: Contact }> {
    return { contact: { id, name: updates.name || "", ...updates } };
  },

  async delete(id: string): Promise<{ success: boolean }> {
    try {
      await directCall("/delete_contact", { method: "POST", body: JSON.stringify({ id: Number(id) }) });
      return { success: true };
    } catch (error) {
      console.error("[API] Delete contact failed:", error);
      return { success: false };
    }
  },
};

export const settingsApi = {
  async getVoices(): Promise<VoiceOption[]> {
    try {
      const result = await directCall<any>("/get_available_voices");
      const voices = normalizeVoices(result);
      if (voices.length) return voices;
      if (result?.voices) return normalizeVoices(result.voices);
      return [];
    } catch (error) {
      console.error("[API] Get voices failed:", error);
      return [
        { id: "sarah", name: "Sarah (Default)", language: "en-US", gender: "female" },
        { id: "emma", name: "Emma", language: "en-GB", gender: "female" },
      ];
    }
  },

  async setVoice(voiceId: string): Promise<{ success: boolean }> {
    try {
      await directCall("/set_user_setting", {
        method: "POST",
        body: JSON.stringify({ key: "voice_profile", value: voiceId }),
      });
      return { success: true };
    } catch (error) {
      console.error("[API] Set voice failed:", error);
      return { success: false };
    }
  },

  async getSetting(key: string): Promise<string> {
    try {
      const result = await directCall<{ value: string }>(`/get_user_setting?key=${encodeURIComponent(key)}`);
      return result.value || "";
    } catch {
      return "";
    }
  },

  async setSetting(key: string, value: string): Promise<boolean> {
    try {
      await directCall("/set_user_setting", { method: "POST", body: JSON.stringify({ key, value }) });
      return true;
    } catch {
      return false;
    }
  },

  async getThemes(): Promise<ThemeOption[]> {
    try {
      const result = await directCall<{ root: string; files: string[]; count: number }>("/get_theme_files");
      if (result.files && result.files.length > 0) {
        return result.files
          .filter((f) => f.endsWith(".css"))
          .map((f) => ({
            id: f.replace(".css", "").replace(/\//g, "_"),
            name: f.replace(".css", "").replace(/[-_]/g, " ").replace(/\//g, " - "),
            filename: f,
          }));
      }
      return [
        { id: "default", name: "Default Dark", filename: "default.css" },
        { id: "midnight", name: "Midnight Blue", filename: "midnight.css" },
      ];
    } catch (error) {
      console.error("[API] Get themes failed:", error);
      return [{ id: "default", name: "Default Dark", filename: "default.css" }];
    }
  },

  async setTheme(themeId: string): Promise<{ success: boolean }> {
    try {
      await directCall("/set_user_setting", { method: "POST", body: JSON.stringify({ key: "theme", value: themeId }) });
      return { success: true };
    } catch {
      return { success: false };
    }
  },

  getThemeUrl(filename: string): string {
    return `${config.apiBaseUrl}/api/data/mods/themes/${filename}`;
  },
};

// ============================================================================
// FILES / RESEARCH / META (edge-proxy + direct health)
// ============================================================================

export const filesApi = {
  async uploadAndAnalyze(
    file: File,
    options?: { analyze?: boolean; extractText?: boolean },
  ): Promise<{ success: boolean; analysis?: string; content?: string; media_url?: string }> {
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

export const researchApi = {
  async search(
    query: string,
    options?: { depth?: "shallow" | "deep"; sources?: string[] },
  ): Promise<{ results: any[]; summary?: string; sources?: string[] }> {
    return invokeEdgeFunction("sarah-api", {
      endpoint: "/api/research/search",
      method: "POST",
      payload: { query, depth: options?.depth || "shallow", sources: options?.sources },
    });
  },
};

export const metaApi = {
  async getCapabilities(): Promise<BackendCapabilities> {
    try {
      const result = await invokeEdgeFunction<any>("sarah-api", { endpoint: "/api/meta/capabilities", method: "GET" });
      if (result?.fallback) return getDefaultCapabilities();
      return result as BackendCapabilities;
    } catch {
      return getDefaultCapabilities();
    }
  },

  async getVersion(): Promise<{ version: string; updated_at?: string }> {
    try {
      const result = await invokeEdgeFunction<any>("sarah-api", { endpoint: "/api/version", method: "GET" });
      return result;
    } catch {
      return { version: "unknown" };
    }
  },

  async healthCheck(): Promise<{ status: string; ok?: boolean; services?: Record<string, boolean> }> {
    try {
      const result = await directCall<{ ok: boolean; status?: string; version?: string }>("/api/health", {
        method: "GET",
      });
      return { status: result.ok ? "ok" : "error", ok: result.ok };
    } catch (error) {
      console.warn("[Health] Direct call failed:", error);
      try {
        return await invokeEdgeFunction<any>("sarah-api", { endpoint: "/api/health", method: "GET" });
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
// PROXY API (legacy)
// ============================================================================

export const proxyApi = {
  async call(
    endpoint: string,
    options?: { method?: "GET" | "POST" | "PUT" | "DELETE"; body?: Record<string, unknown> },
  ): Promise<unknown> {
    try {
      const { data, error } = await supabase.functions.invoke("sarah-api", {
        body: { endpoint, method: options?.method || "GET", payload: options?.body },
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
    return settingsApi.getThemes().then((themes) => ({ themes }));
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

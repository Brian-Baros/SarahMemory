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
  spec?: {
    renderMode?: "procedural_holo" | "gltf_model" | "video_sprite";
    modelUrl?: string;
    videoUrl?: string;
    backgroundUrl?: string;
    backgroundType?: "none" | "image" | "video";
    pose?: "stand" | "sit" | "wave";
    gesture?: "none" | "wave" | "point" | "nod" | "shake";
    lookAt?: { x: number; y: number; z: number };
    expression?: string;
    speaking?: boolean;
    listening?: boolean;
  };
  avatar_3d?: {
    ok?: boolean;
    model_file?: string;
    model_url?: string;
    files?: string[];
  };
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

export interface ModelManagerCategory {
  id: string;
  label: string;
}

export interface ModelManagerModel {
  id: string;
  display_name?: string;
  simple_label?: string;
  repo?: string;
  path?: string;
  source?: string;
  category?: string;
  category_label?: string;
  detected_category?: string;
  domain?: string;
  domain_label?: string;
  adapter_type?: string;
  status?: string;
  status_label?: string;
  installed?: boolean;
  verified?: boolean;
  missing?: boolean;
  is_active?: boolean;
  can_activate?: boolean;
  size_gb?: number;
  known_repo?: boolean;
  user_classified?: boolean;
}

export interface ModelManagerStatus {
  ok: boolean;
  version?: string;
  registry_path?: string;
  models_dir?: string;
  external_roots?: string[];
  categories?: ModelManagerCategory[];
  domains?: ModelManagerCategory[];
  adapter_types?: string[];
  models?: ModelManagerModel[];
  groups?: Record<string, ModelManagerModel[]>;
  active_models?: Record<string, string>;
  active_records?: Record<string, ModelManagerModel>;
  hardware?: Record<string, unknown>;
  live_scan_interval_sec?: number;
  recommended_poll_interval_sec?: number;
  model_count?: number;
  ready_count?: number;
  missing_count?: number;
  unclassified_count?: number;
  active_count?: number;
  scan?: {
    started_at?: string;
    completed_at?: string;
    live_interval_sec?: number;
    model_count?: number;
    ready_count?: number;
    missing_count?: number;
    unclassified_count?: number;
    active_count?: number;
    new_model_ids?: string[];
    removed_model_ids?: string[];
    source?: string;
  };
  updated_at?: string;
  error?: string;
}

export interface DlEngineModelWeights {
  reasoning: number;
  coding: number;
  memory: number;
  research: number;
  creativity: number;
  safety: number;
  autonomy: number;
  precision: number;
  speed: number;
}

export interface DlEngineWeightProfileResponse {
  ok: boolean;
  category?: string;
  model_id?: string;
  context?: Record<string, unknown>;
  profile?: Record<string, unknown>;
  weights?: DlEngineModelWeights;
  raw_tensor_edit?: boolean;
  note?: string;
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


export interface VisionHudTarget {
  id?: string;
  class?: string;
  label?: string;
  bbox?: number[];
  bbox_px?: number[];
  center?: number[];
  confidence?: number;
  vectors?: { dx?: number; dy?: number; dz_est?: number | string | null };
  motion?: { angular_velocity?: number; velocity_px_s?: number[] };
  color?: Record<string, unknown> | null;
  model?: string;
}

export interface VisionHudPacket {
  schema?: string;
  packet_id?: string;
  timestamp?: string;
  ttl_ms?: number;
  mode?: string;
  display_profile?: string;
  frame?: Record<string, any>;
  active_targets?: VisionHudTarget[];
  vision?: Record<string, any>;
  compute_integrity?: Record<string, any>;
  kinetic_integrity?: Record<string, any>;
  smget_state?: Record<string, any>;
  authority?: Record<string, any>;
  source?: string;
}

export interface VisionFrameStatus {
  ok: boolean;
  session_id?: string;
  has_frame?: boolean;
  frame_id?: string;
  ts?: string;
  source?: string;
  width?: number;
  height?: number;
  hud_schema?: string;
  hud_packet_id?: string;
  target_count?: number;
  [key: string]: any;
}

export interface VisionFrameSubmitResponse {
  ok: boolean;
  frame?: Record<string, any>;
  frame_status?: VisionFrameStatus;
  hud_packet?: VisionHudPacket;
  source?: string;
  error?: string;
  [key: string]: any;
}



export interface VrRuntimeResponse {
  ok: boolean;
  running?: boolean;
  runtime?: Record<string, any>;
  probe?: Record<string, any>;
  source?: string;
  error?: string;
  [key: string]: any;
}

export interface VisionFrameLatestResponse {
  ok: boolean;
  has_frame?: boolean;
  frame_id?: string;
  ts?: string;
  source?: string;
  width?: number;
  height?: number;
  mime?: string;
  image_b64?: string;
  data_url?: string;
  image_cached_ts?: string;
  error?: string;
  [key: string]: any;
}


export interface DevBridgeStatusResponse {
  ok: boolean;
  enabled?: boolean;
  version?: string;
  apply_gate?: Record<string, unknown>;
  cmd_tickets?: Record<string, number>;
  cmd_ticket_inventory?: Record<string, number>;
  repair_counts?: Record<string, unknown>;
}

export interface DevBridgeRepairSummaryResponse {
  ok: boolean;
  generated_at?: string;
  cmd_tickets?: Record<string, number>;
  tickets?: { total?: number; by_status?: Record<string, number>; by_severity?: Record<string, number>; by_target_file?: Record<string, number> };
  batches?: { total?: number; by_status?: Record<string, number>; by_severity?: Record<string, number>; by_target_file?: Record<string, number> };
  note?: string;
}

export interface GovernanceResponse {
  ok: boolean;
  api_domain?: string;
  route_base?: string;
  governance?: Record<string, unknown>;
  [key: string]: unknown;
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

function isLocalOnlyRuntime(): boolean {
  try {
    const raw = window.localStorage.getItem("sarah-memory-storage");
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    return Boolean(parsed?.state?.settings?.localOnlyMode);
  } catch {
    return false;
  }
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
// VOICE API (canonical local route: /api/tts/speak)
// ============================================================================

export const voiceApi = {
  async speak(text: string, voice?: string): Promise<VoiceResponse> {
    try {
      const { data } = await tryDirectEndpoints<any>(["/api/tts/speak", "/api/tts", "/tts"], {
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
// DIALER API (canonical local routes: /api/comm/voip/*)
// ============================================================================

export const dialerApi = {
  async checkAvailability(): Promise<DialerResponse> {
    try {
      const data: any = await directCall("/api/comm/voip/capabilities", { method: "GET" });
      return {
        success: data?.ok !== false,
        available: Boolean(data?.available || data?.enabled || data?.capabilities?.available),
        message: data?.message || data?.status || data?.reason || "VoIP control surface checked.",
        ...data,
      };
    } catch (error) {
      console.error("[API] VoIP capability check failed:", error);
      return { available: false, success: false, message: "Local VoIP backend unavailable" };
    }
  },

  async initiateCall(target: { number?: string; ip_address?: string; room_id?: string }): Promise<DialerResponse> {
    try {
      const data: any = await directCall("/api/comm/voip/call/start", {
        method: "POST",
        body: JSON.stringify({
          ...target,
          target: target.number || target.ip_address || target.room_id || "",
          source: "webui_dialer",
        }),
      });
      return { success: data?.ok !== false, ...data };
    } catch (error) {
      console.error("[API] Start call failed:", error);
      return { success: false, available: false, message: "Local VoIP start route unavailable" };
    }
  },

  async endCall(callId?: string): Promise<DialerResponse> {
    try {
      const data: any = await directCall("/api/comm/voip/call/update", {
        method: "POST",
        body: JSON.stringify({ call_id: callId || "active", status: "ended", source: "webui_dialer" }),
      });
      return { success: data?.ok !== false, ...data };
    } catch (error) {
      console.error("[API] End call failed:", error);
      return { success: false, message: "Local VoIP update route unavailable" };
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
  async capabilities(): Promise<any> {
    return directCall<any>("/api/media/capabilities", { method: "GET" });
  },

  async render(kind: "image" | "music" | "video" | "audio", payload: Record<string, unknown>): Promise<MediaResponse> {
    const data: any = await directCall("/api/media/job/render", {
      method: "POST",
      body: JSON.stringify({ kind, ...payload }),
    });
    return data as MediaResponse;
  },

  async generateImage(prompt: string, options?: { count?: number; style?: string }): Promise<MediaResponse> {
    return this.render("image", { prompt, style: options?.style });
  },

  async generateMusic(prompt: string, options?: { duration?: number; genre?: string }): Promise<MediaResponse> {
    return this.render("music", { prompt, duration: options?.duration || 30, style: options?.genre });
  },

  async generateVideo(prompt: string, options?: { duration?: number; style?: string }): Promise<MediaResponse> {
    return this.render("video", { prompt, duration: options?.duration || 5, style: options?.style });
  },

  async getJobStatus(jobId: string): Promise<MediaResponse> {
    return directCall<MediaResponse>(`/api/media/job/status?job_id=${encodeURIComponent(jobId)}`, { method: "GET" });
  },

  async download(jobId: string, filename = ""): Promise<{ url: string }> {
    const suffix = filename ? `&filename=${encodeURIComponent(filename)}` : "";
    return { url: `/api/media/job/download?job_id=${encodeURIComponent(jobId)}${suffix}` };
  },

  async saveToDataset(_mediaId: string, _dataset?: string): Promise<{ success: boolean }> {
    return { success: false };
  },

  async listRecent(_type?: "image" | "music" | "video"): Promise<MediaResponse> {
    return { success: false, status: "unavailable" } as any;
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
      const result: any = await directCall("/api/comm/reminders/list", { method: "GET" });
      const reminders = (result.reminders || result.data?.reminders || []).map((r: any) => ({
        id: String(r.id || r.reminder_id || r.uuid || Date.now()),
        title: r.title || r.text || "",
        description: r.description || r.body || "",
        time: r.time || r.due_date || (r.due_ts ? new Date(Number(r.due_ts) * 1000).toISOString() : undefined),
        due_date: r.due_date || (r.due_ts ? new Date(Number(r.due_ts) * 1000).toISOString() : undefined),
        dueDate: new Date(r.due_date || (r.due_ts ? Number(r.due_ts) * 1000 : Date.now())),
        completed: Boolean(r.completed || r.is_completed || r.status === "completed" || r.status === "done"),
        priority: r.priority || "medium",
      }));
      return { reminders };
    } catch (error) {
      console.error("[API] Get reminders failed:", error);
      return { reminders: [] };
    }
  },

  async create(reminder: Omit<Reminder, "id">): Promise<{ reminder: Reminder }> {
    const payload: any = {
      title: reminder.title,
      body: reminder.description || (reminder as any).note || "",
      due_date: (reminder as any).time || (reminder as any).due_date || reminder.dueDate,
      priority: reminder.priority || "medium",
      source: "webui_reminders",
    };
    const result: any = await directCall("/api/comm/reminders/save", { method: "POST", body: JSON.stringify(payload) });
    const r = result.reminder || result.data?.reminder || payload;
    return { reminder: { ...reminder, id: String(r.id || r.reminder_id || Date.now()), completed: false } as Reminder };
  },

  async update(id: string, updates: Partial<Reminder>): Promise<{ reminder: Reminder }> {
    const payload: any = { reminder_id: id, ...updates, source: "webui_reminders" };
    const result: any = await directCall("/api/comm/reminders/save", { method: "POST", body: JSON.stringify(payload) });
    const r = result.reminder || result.data?.reminder || payload;
    return { reminder: { id, title: r.title || updates.title || "", description: r.body || updates.description || "", dueDate: new Date(r.due_date || r.due_ts || Date.now()), completed: Boolean(r.completed || updates.completed), priority: r.priority || updates.priority || "medium" } };
  },

  async delete(id: string): Promise<{ success: boolean }> {
    try {
      const result: any = await directCall("/api/comm/reminders/delete", { method: "POST", body: JSON.stringify({ reminder_id: id }) });
      return { success: result?.ok !== false };
    } catch (error) {
      console.error("[API] Delete reminder failed:", error);
      return { success: false };
    }
  },

  async complete(id: string): Promise<{ success: boolean }> {
    try {
      await this.update(id, { completed: true } as any);
      return { success: true };
    } catch {
      return { success: false };
    }
  },

  async snooze(id: string, minutes = 60): Promise<{ success: boolean }> {
    try {
      const dueDate = new Date(Date.now() + minutes * 60000);
      await this.update(id, { dueDate } as any);
      return { success: true };
    } catch {
      return { success: false };
    }
  },
};

export const contactsApi = {
  async list(): Promise<{ contacts: Contact[] }> {
    try {
      const result: any = await directCall("/api/comm/contacts/list", { method: "GET" });
      const contacts = (result.contacts || result.data?.contacts || []).map((c: any) => ({
        id: String(c.id || c.contact_id || c.uuid || Date.now()),
        name: c.name || "",
        phone: c.phone || c.number || "",
        number: c.phone || c.number || "",
        email: c.email || "",
        address: c.address || "",
        notes: c.notes || "",
        status: c.status || "offline",
      }));
      return { contacts };
    } catch (error) {
      console.error("[API] Get contacts failed:", error);
      return { contacts: [] };
    }
  },

  async create(contact: Omit<Contact, "id">): Promise<{ contact: Contact }> {
    const result: any = await directCall("/api/comm/contacts/save", {
      method: "POST",
      body: JSON.stringify({ ...contact, source: "webui_contacts" }),
    });
    const c = result.contact || result.data?.contact || contact;
    return { contact: { ...contact, id: String(c.id || c.contact_id || Date.now()) } as Contact };
  },

  async update(id: string, updates: Partial<Contact>): Promise<{ contact: Contact }> {
    const result: any = await directCall("/api/comm/contacts/save", {
      method: "POST",
      body: JSON.stringify({ contact_id: id, ...updates, source: "webui_contacts" }),
    });
    const c = result.contact || result.data?.contact || updates;
    return { contact: { id, name: c.name || updates.name || "", ...c } as Contact };
  },

  async delete(id: string): Promise<{ success: boolean }> {
    try {
      const result: any = await directCall("/api/comm/contacts/delete", {
        method: "POST",
        body: JSON.stringify({ contact_id: id }),
      });
      return { success: result?.ok !== false };
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
    version: "9.0.0",
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


export const visionApi = {
  async policy(): Promise<any> {
    return directCall<any>("/api/vision/policy", { method: "GET" });
  },

  async frameStatus(): Promise<VisionFrameStatus> {
    return directCall<VisionFrameStatus>("/api/vision/frame/status", { method: "GET" });
  },

  async frameLatest(): Promise<VisionFrameLatestResponse> {
    return directCall<VisionFrameLatestResponse>("/api/vision/frame/latest", { method: "GET" });
  },

  async hudStatus(): Promise<any> {
    return directCall<any>("/api/vision/hud/status", { method: "GET" });
  },

  async hudPacket(): Promise<{ ok: boolean; hud_packet?: VisionHudPacket; source?: string; [key: string]: any }> {
    return directCall<{ ok: boolean; hud_packet?: VisionHudPacket; source?: string; [key: string]: any }>("/api/vision/hud/packet", { method: "GET" });
  },

  async submitFrame(
    dataUrl: string,
    options?: { source?: string; width?: number; height?: number; analyze?: boolean; question?: string; ts?: number },
  ): Promise<VisionFrameSubmitResponse> {
    return directCall<VisionFrameSubmitResponse>("/api/vision/frame/submit", {
      method: "POST",
      body: JSON.stringify({
        imageBase64: dataUrl,
        data_url: dataUrl,
        source: options?.source || "frontend_frame_submit",
        width: options?.width,
        height: options?.height,
        mime: "image/jpeg",
        analyze: Boolean(options?.analyze),
        question: options?.question || "VR HUD observation pass",
        ts: options?.ts || Date.now(),
      }),
    });
  },

  async analyzeFrame(dataUrl: string, question = "VR HUD observation pass", learningAllowed = false): Promise<VisionFrameSubmitResponse> {
    return directCall<VisionFrameSubmitResponse>("/api/vision/analyze", {
      method: "POST",
      body: JSON.stringify({
        imageBase64: dataUrl,
        data_url: dataUrl,
        question,
        learning_allowed: Boolean(learningAllowed),
        source: "vision_screen_analyze",
        ts: Date.now(),
      }),
    });
  },
};


export const vrApi = {
  async status(refresh = false): Promise<VrRuntimeResponse> {
    return directCall<VrRuntimeResponse>(`/api/vr/status?refresh=${refresh ? "1" : "0"}`, { method: "GET" });
  },

  async probe(): Promise<VrRuntimeResponse> {
    return directCall<VrRuntimeResponse>("/api/vr/probe", { method: "POST", body: JSON.stringify({}) });
  },

  async start(options: Record<string, unknown> = {}): Promise<VrRuntimeResponse> {
    return directCall<VrRuntimeResponse>("/api/vr/start", {
      method: "POST",
      body: JSON.stringify({
        mirror_preview: true,
        headset_surface: true,
        auto_start_on_headset_connected: true,
        auto_stop_on_headset_disconnected: true,
        ...options,
      }),
    });
  },

  async stop(reason = "frontend_stop"): Promise<VrRuntimeResponse> {
    return directCall<VrRuntimeResponse>("/api/vr/stop", {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  },
};

export const devBridgeApi = {
  async status(): Promise<DevBridgeStatusResponse> {
    return directCall<DevBridgeStatusResponse>("/api/devbridge/status", { method: "GET" });
  },

  async repairSummary(): Promise<DevBridgeRepairSummaryResponse> {
    return directCall<DevBridgeRepairSummaryResponse>("/api/devbridge/repair-summary", { method: "GET" });
  },

  async repairTickets(): Promise<any> {
    return directCall<any>("/api/devbridge/repair-tickets", { method: "GET" });
  },

  async repairBatches(): Promise<any> {
    return directCall<any>("/api/devbridge/repair-batches", { method: "GET" });
  },

  async cmdTickets(): Promise<any> {
    return directCall<any>("/api/devbridge/cmd-tickets", { method: "GET" });
  },

  async processCmdTickets(limit = 50, dryRun = false): Promise<any> {
    return directCall<any>("/api/devbridge/cmd-tickets/process", {
      method: "POST",
      body: JSON.stringify({ limit, dry_run: dryRun }),
    });
  },
};

export const governanceApi = {
  async get(domain: "net" | "net2" | "comm" | "media" | "drivers" | "store" | "browser"): Promise<GovernanceResponse> {
    const routeMap: Record<string, string> = {
      net: "/api/net/governance",
      net2: "/api/net2/governance",
      comm: "/api/comm/governance",
      media: "/api/media/governance",
      drivers: "/api/drivers/governance",
      store: "/api/store/governance",
      browser: "/api/browser/governance",
    };
    return directCall<GovernanceResponse>(routeMap[domain], { method: "GET" });
  },

  async all(): Promise<Record<string, GovernanceResponse | { ok: false; error: string }>> {
    const domains = ["net", "net2", "comm", "media", "drivers", "store", "browser"] as const;
    const entries = await Promise.all(
      domains.map(async (domain) => {
        try {
          return [domain, await this.get(domain)] as const;
        } catch (error: any) {
          return [domain, { ok: false, error: String(error?.message || error) }] as const;
        }
      })
    );
    return Object.fromEntries(entries);
  },

  async addonCandidates(): Promise<any> {
    return directCall<any>("/api/store/addons/candidates", { method: "GET" });
  },

  async driverManifestAudit(): Promise<any> {
    return directCall<any>("/api/drivers/manifest/audit", { method: "GET" });
  },
};

// ============================================================================
// MODEL MANAGER API
// ============================================================================

export const modelsApi = {
  async status(refresh = true): Promise<ModelManagerStatus> {
    return directCall<ModelManagerStatus>(`/api/models/status?refresh=${refresh ? "1" : "0"}`, { method: "GET" });
  },

  async scan(): Promise<ModelManagerStatus> {
    return directCall<ModelManagerStatus>("/api/models/scan", { method: "POST", body: JSON.stringify({}) });
  },

  async select(category: string, modelId: string): Promise<any> {
    return directCall<any>("/api/models/select", {
      method: "POST",
      body: JSON.stringify({ category, model_id: modelId }),
    });
  },

  async classify(
    modelId: string,
    category: string,
    domain = "general",
    adapterType = "",
    displayName = "",
  ): Promise<any> {
    return directCall<any>("/api/models/classify", {
      method: "POST",
      body: JSON.stringify({
        model_id: modelId,
        category,
        domain,
        adapter_type: adapterType,
        display_name: displayName,
      }),
    });
  },

  async verify(modelId: string): Promise<any> {
    return directCall<any>("/api/models/verify", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
  },

  async addExternalPath(path: string): Promise<any> {
    return directCall<any>("/api/models/external-path", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
  },

  async reset(category: string): Promise<any> {
    return directCall<any>("/api/models/reset", {
      method: "POST",
      body: JSON.stringify({ category }),
    });
  },

  async download(category: string, repo: string, modelId = ""): Promise<any> {
    return directCall<any>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ category, repo, model_id: modelId }),
    });
  },
};


// ============================================================================
// DL ENGINE MODEL WEIGHT API
// ============================================================================

export const dlengineApi = {
  async getWeightProfile(category = "reasoning", modelId = ""): Promise<DlEngineWeightProfileResponse> {
    const qs = new URLSearchParams();
    qs.set("category", category);
    if (modelId) qs.set("model_id", modelId);
    return directCall<DlEngineWeightProfileResponse>(`/api/dlengine/weights?${qs.toString()}`, { method: "GET" });
  },

  async saveWeightProfile(
    category: string,
    modelId: string,
    weights: Partial<DlEngineModelWeights>,
    context: Record<string, unknown> = {},
  ): Promise<DlEngineWeightProfileResponse> {
    return directCall<DlEngineWeightProfileResponse>("/api/dlengine/weights", {
      method: "POST",
      body: JSON.stringify({
        category,
        model_id: modelId,
        context,
        weights,
        source: "frontend:dlengine_model_weight_controller",
      }),
    });
  },

  async resetWeightProfile(category: string, modelId = ""): Promise<DlEngineWeightProfileResponse> {
    return directCall<DlEngineWeightProfileResponse>("/api/dlengine/weights/reset", {
      method: "POST",
      body: JSON.stringify({ category, model_id: modelId }),
    });
  },
};

// ============================================================================
// PROXY API (legacy)
// ============================================================================

export const proxyApi = {
  async call(
    endpoint: string,
    options?: { method?: "GET" | "POST" | "PUT" | "DELETE"; body?: Record<string, unknown> },
  ): Promise<unknown> {
    const looksLikeRequestOptions = Boolean(options && ("method" in options || "body" in options));
    const method = looksLikeRequestOptions ? (options?.method || "GET") : (options ? "POST" : "GET");
    const payload = looksLikeRequestOptions ? (options?.body || {}) : (options || {});
    const requestOptions: RequestInit = {
      method,
      body: method === "GET" ? undefined : JSON.stringify(payload),
    };

    // Local-first path.  The UI must never require Supabase/cloud just to talk to
    // the locally running SarahMemory backend.
    try {
      return await directCall(endpoint, requestOptions);
    } catch (localErr) {
      if (isLocalOnlyRuntime() || config.isLocalMode) {
        console.warn("[proxyApi] Local backend call failed and cloud proxy is disabled:", localErr);
        return { ok: false, fallback: true, error: "local_backend_unavailable", endpoint };
      }
    }

    try {
      const { data, error } = await supabase.functions.invoke("sarah-api", {
        body: { endpoint, method, payload },
      });
      if (error) {
        console.error("[proxyApi] Error:", error);
        throw new Error(error.message || "Proxy API error");
      }
      return data;
    } catch (err) {
      console.error("[proxyApi] Call failed:", err);
      return { ok: false, fallback: true, error: "proxy_unavailable", endpoint };
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
  devbridge: devBridgeApi,
  governance: governanceApi,
  vision: visionApi,
  vr: vrApi,
  models: modelsApi,
  dlengine: dlengineApi,
};

export default api;

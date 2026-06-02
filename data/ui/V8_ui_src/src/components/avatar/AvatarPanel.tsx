// AvatarPanel.tsx
import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Camera, ExternalLink, Maximize2, Monitor, RefreshCw, UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

import { useSarahStore } from "@/stores/useSarahStore";
import { api } from "@/lib/api";

import { AvatarBackground } from "./AvatarBackground";
import { WebcamOverlay } from "./WebcamOverlay";
import { Avatar3D, AvatarSpec } from "./Avatar3D";

import sarahAvatarPng from "@/assets/sarah-avatar.png?w=640";

type LocalAvatarState = {
  mode: "avatar_2d" | "avatar_3d" | "desktop_mirror" | "media" | "idle";
  expression: string;
  speaking: boolean;
  listening: boolean;
  current_action?: string;
  spec?: AvatarSpec;
};

type MirrorKind = "mjpeg" | "video" | "image" | "unknown";

const DEFAULT_3D_SPEC: AvatarSpec = {
  renderMode: "gltf_model",
  modelUrl: "/api/avatar/3d/sarahmemory_3d_avatar.glb",
  backgroundType: "none",
  pose: "stand",
  gesture: "none",
};


function joinUrl(base: string, path: string): string {
  if (!base) return path;
  const b = base.endsWith("/") ? base.slice(0, -1) : base;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

async function probeUrl(url: string): Promise<{ ok: boolean; contentType?: string }> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 2000);
  try {
    const res = await fetch(url, { method: "GET", signal: controller.signal, cache: "no-store" });
    const ct = res.headers.get("content-type") || undefined;
    return { ok: res.ok, contentType: ct };
  } catch {
    return { ok: false };
  } finally {
    clearTimeout(t);
  }
}

function classifyContentType(ct?: string): MirrorKind {
  const c = (ct || "").toLowerCase();
  if (!c) return "unknown";
  if (c.includes("multipart/x-mixed-replace")) return "mjpeg";
  if (c.includes("video/")) return "video";
  if (c.includes("image/")) return "image";
  return "unknown";
}

function DesktopMirrorView({ apiBase, enabled }: { apiBase: string; enabled: boolean }) {
  const [mirrorUrl, setMirrorUrl] = useState<string | null>(null);
  const [mirrorKind, setMirrorKind] = useState<MirrorKind>("unknown");
  const [error, setError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);
  const lastChosenRef = useRef<string | null>(null);

  const candidates = useMemo(() => {
    const paths = [
      "/api/desktop/mjpeg",
      "/api/desktop/stream",
      "/api/desktop_mirror",
      "/api/desktop_mirror/stream",
      "/api/screen/stream",
      "/api/screen/mjpeg",
      "/api/desktop",
      "/desktop/mjpeg",
      "/desktop/stream",
      "/screen/stream",
    ];

    const base = apiBase || window.location.origin;
    return paths.map((p) => joinUrl(base, p));
  }, [apiBase]);

  const detect = async () => {
    if (!enabled) return;

    setProbing(true);
    setError(null);

    const ordered = lastChosenRef.current
      ? [lastChosenRef.current, ...candidates.filter((c) => c !== lastChosenRef.current)]
      : candidates;

    for (const url of ordered) {
      const testUrl = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
      const res = await probeUrl(testUrl);
      if (!res.ok) continue;

      const kind = classifyContentType(res.contentType);
      if (kind === "mjpeg" || kind === "video" || kind === "image" || kind === "unknown") {
        lastChosenRef.current = url;
        setMirrorUrl(testUrl);
        setMirrorKind(kind);
        setProbing(false);
        return;
      }
    }

    setMirrorUrl(null);
    setMirrorKind("unknown");
    setProbing(false);
    setError("No desktop mirror stream endpoint responded yet.");
  };

  useEffect(() => {
    detect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, apiBase]);

  if (!enabled) return null;

  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-background">
      {mirrorUrl ? (
        <>
          {mirrorKind === "video" ? (
            <video className="h-full w-full object-cover" src={mirrorUrl} autoPlay playsInline muted controls={false} />
          ) : (
            <img className="h-full w-full object-cover" src={mirrorUrl} alt="Desktop Mirror" draggable={false} />
          )}

          <div className="absolute left-3 top-3 flex items-center gap-2">
            <div className="rounded-md border border-cyan-500/30 bg-slate-950/95 px-2 py-1 text-xs text-cyan-100 shadow-lg backdrop-blur">
              Desktop Mirror {mirrorKind !== "unknown" ? `(${mirrorKind})` : ""}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 border border-cyan-500/20 bg-slate-950/95 text-cyan-100 backdrop-blur hover:bg-slate-900"
              onClick={detect}
              disabled={probing}
              title="Retry mirror detection"
            >
              <RefreshCw className={cn("h-4 w-4", probing && "animate-spin")} />
            </Button>
          </div>
        </>
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-3 bg-slate-950/70 text-muted-foreground">
          <Monitor className="h-8 w-8 text-cyan-400/70" />
          <div className="text-sm">Desktop Mirror not available</div>
          {error && <div className="max-w-[80%] text-center text-xs opacity-80">{error}</div>}
          <Button variant="outline" size="sm" onClick={detect} disabled={probing}>
            {probing ? "Detecting…" : "Retry"}
          </Button>
        </div>
      )}
    </div>
  );
}


function useCompactAvatarViewport(): boolean {
  const [compact, setCompact] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(max-width: 768px), (orientation: portrait)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const query = window.matchMedia("(max-width: 768px), (orientation: portrait)");
    const update = () => setCompact(query.matches);
    update();
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", update);
      return () => query.removeEventListener("change", update);
    }
    query.addListener(update);
    return () => query.removeListener(update);
  }, []);

  return compact;
}

type AvatarSurfaceBoundaryProps = {
  children: React.ReactNode;
  fallback: React.ReactNode;
};

type AvatarSurfaceBoundaryState = {
  hasError: boolean;
};

class AvatarSurfaceBoundary extends React.Component<AvatarSurfaceBoundaryProps, AvatarSurfaceBoundaryState> {
  constructor(props: AvatarSurfaceBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): AvatarSurfaceBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    // Prevent a 3D/mobile render failure from taking down the whole mobile shell.
    // The avatar falls back to the 2D surface and the backend remains authoritative.
    // eslint-disable-next-line no-console
    console.warn("[AvatarPanel] Render surface fallback activated:", error);
  }


  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

function Avatar2DImageSurface({ src, onLoad, onError }: { src: string; onLoad: () => void; onError: () => void }) {
  return (
    <div className="flex h-full w-full items-center justify-center overflow-hidden bg-slate-950">
      <img
        src={src}
        alt="Sarah AI Avatar"
        fetchPriority="high"
        width={1254}
        height={1254}
        className="h-full w-full max-h-full max-w-full object-contain opacity-100 transition-opacity duration-100"
        draggable={false}
        onLoad={onLoad}
        onError={onError}
      />
    </div>
  );
}

function ScreenHeader({ icon, title, right }: { icon: React.ReactNode; title: string; right?: React.ReactNode }) {
  return (
    <div className="pointer-events-none absolute left-3 right-3 top-3 z-30 flex items-center justify-between">
      <div className="flex items-center gap-2 rounded-md border border-cyan-500/20 bg-slate-950/95 px-2 py-1 text-xs text-cyan-100 shadow-lg backdrop-blur">
        {icon}
        <span>{title}</span>
      </div>
      {right}
    </div>
  );
}

export function AvatarPanel() {
  const mediaState = useSarahStore((s) => s.mediaState);
  const setScreenMode = useSarahStore((s) => s.setScreenMode);

  const avatarSpeaking = useSarahStore((s) => s.avatarSpeaking);
  const avatarListening = useSarahStore((s) => (s as any).avatarListening ?? false);
  const bootstrapData = useSarahStore((s) => s.bootstrapData);

  type LiveAvatarState = LocalAvatarState & {
    current_image?: string;
    avatar_image?: string;
    avatar_image_url?: string;
    emotion?: string;
    sequence?: number;
    life_state?: string;
    heartbeat_count?: number;
    idle_seconds?: number;
    night_mode?: boolean;
    busy?: boolean;
    diagnostics?: boolean;
    manifest?: { files?: string[]; base_url?: string; role_map?: Record<string, string>; supports_dynamic_29?: boolean };
  };

  const [avatarState, setAvatarState] = useState<LiveAvatarState>({
    mode: "avatar_2d",
    expression: "neutral",
    emotion: "neutral",
    speaking: false,
    listening: false,
    current_action: "idle",
    current_image: "sarah-avatar.png",
    spec: DEFAULT_3D_SPEC,
  });

  const [webcamVisible, setWebcamVisible] = useState(true);
  const [frameTick, setFrameTick] = useState(0);
  const [imageLoadFailed, setImageLoadFailed] = useState(false);

  const screenModes = useMemo(
    () => [
      { value: "avatar_2d", label: "Avatar 2D" },
      { value: "avatar_3d", label: "Avatar 3D" },
      { value: "desktop_mirror", label: "Desktop Mirror" },
      { value: "media", label: "Media Display" },
      { value: "idle", label: "Idle" },
    ],
    [],
  );

  const currentMode = (mediaState.screenMode || avatarState.mode || "avatar_2d") as LocalAvatarState["mode"];
  const is3DMode = currentMode === "avatar_3d";
  const isDesktopMirror = currentMode === "desktop_mirror";
  const isCallMode = currentMode === "media" && String(avatarState.current_action || "").toLowerCase().includes("call");
  const apiBase = bootstrapData?.env?.api_base || "";
  const isCompactViewport = useCompactAvatarViewport();
  const render3DInline = is3DMode;

  const normalizeAvatarUrl = (url: string): string => {
    if (!url) return sarahAvatarPng;
    if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
    return joinUrl(apiBase || window.location.origin, url);
  };

  const normalizeAssetUrl = (url?: string): string | undefined => {
    if (!url) return undefined;
    if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
    return joinUrl(apiBase || window.location.origin, url);
  };

  const avatar3DSpec = useMemo<AvatarSpec>(() => {
    const incoming = avatarState.spec && typeof avatarState.spec === "object" ? avatarState.spec : DEFAULT_3D_SPEC;
    const merged: AvatarSpec = {
      ...DEFAULT_3D_SPEC,
      ...incoming,
      expression: avatarState.expression || incoming.expression || "neutral",
      speaking: avatarState.speaking ?? incoming.speaking ?? false,
      listening: avatarState.listening ?? incoming.listening ?? false,
    };

    if (merged.renderMode === "gltf_model" && !merged.modelUrl) {
      merged.modelUrl = DEFAULT_3D_SPEC.modelUrl;
    }

    return {
      ...merged,
      modelUrl: normalizeAssetUrl(merged.modelUrl),
      videoUrl: normalizeAssetUrl(merged.videoUrl),
      backgroundUrl: normalizeAssetUrl(merged.backgroundUrl),
    };
  }, [apiBase, avatarState.spec, avatarState.expression, avatarState.speaking, avatarState.listening]);

  const resolveRoleFile = (role: string, fallback: string): string => {
    const key = String(role || "").toLowerCase();
    const roleMap = avatarState.manifest?.role_map || {};
    const mapped = roleMap[key];
    return typeof mapped === "string" && mapped.trim() ? mapped : fallback;
  };

  const localRoleFile = useMemo(() => {
    const expression = String(avatarState.expression || avatarState.emotion || "neutral").toLowerCase();
    if (avatarState.speaking) {
      return frameTick % 2 === 0
        ? resolveRoleFile("speaking_soft", "07_speaking_soft.png")
        : resolveRoleFile("speaking_open", "08_speaking_open.png");
    }
    if (avatarState.listening) return resolveRoleFile("listening", "09_listening_thinking.png");
    if (expression === "joy" || expression === "happy") return resolveRoleFile("happy", "11_happy_open_smile.png");
    if (expression === "surprise") return resolveRoleFile("surprise", "13_surprised_open_mouth.png");
    if (expression === "anger" || expression === "angry") return resolveRoleFile("angry", "16_angry_yelling.png");
    if (expression === "sad" || expression === "sadness") return resolveRoleFile("sad", "05_sad_worried.png");
    if (expression === "thinking") return resolveRoleFile("thinking", "09_listening_thinking.png");
    if (expression === "sleepy" || expression === "rem_dreaming") return resolveRoleFile("sleepy", "02_sleepy_half_lidded.png");
    if (expression === "asleep" || expression === "rem_sleep") return resolveRoleFile("asleep", "01_relaxed_closed_eyes.png");
    if (expression === "very_sleepy" || expression === "exhausted") return resolveRoleFile("very_sleepy", "25_exhausted_sleepy.png");
    if (expression === "success" || expression === "victory") return resolveRoleFile("success", "26_victory_celebration.png");
    if (expression === "thumbs_up" || expression === "approval") return resolveRoleFile("thumbs_up", "21_thumbs_up_smile.png");
    if (expression === "pondering" || expression === "rem_sandbox" || expression === "rem_review") return resolveRoleFile("pondering", "24_pondering_stare.png");
    if (expression === "wondering" || expression === "planning") return resolveRoleFile("wondering", "28_wondering_planning_stare.png");
    if (expression === "state_29" || expression === "extra_29" || expression === "random_29") {
      return resolveRoleFile("state_29", avatarState.current_image || "19_neutral_forward.png");
    }
    return resolveRoleFile("neutral", "19_neutral_forward.png");
  }, [
    avatarState.speaking,
    avatarState.listening,
    avatarState.expression,
    avatarState.emotion,
    avatarState.manifest?.role_map,
    avatarState.current_image,
    frameTick,
  ]);

  const avatarImageSrc = useMemo(() => {
    if (imageLoadFailed) return sarahAvatarPng;

    // Speaking/listening animation must be driven by the local frame clock.
    // Do not let the backend's last avatar_image_url freeze the mouth frame
    // between /api/avatar/state polls while Sarah is still speaking.
    const liveAnimationActive = Boolean(avatarState.speaking || avatarState.listening);
    const backendUrl = liveAnimationActive ? "" : avatarState.avatar_image_url;
    const raw = backendUrl || `/api/avatar/2d/${localRoleFile}`;
    const absolute = normalizeAvatarUrl(raw);
    const seq = liveAnimationActive ? frameTick : avatarState.sequence ?? frameTick;
    return `${absolute}${absolute.includes("?") ? "&" : "?"}seq=${seq}`;
  }, [avatarState.avatar_image_url, avatarState.sequence, avatarState.speaking, avatarState.listening, frameTick, imageLoadFailed, localRoleFile]);

  const fetchAvatarState = async (): Promise<Partial<LiveAvatarState>> => {
    const url = joinUrl(apiBase || window.location.origin, "/api/avatar/state");
    const res = await fetch(`${url}?t=${Date.now()}`, {
      method: "GET",
      credentials: "include",
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`Avatar state failed: ${res.status}`);
    return (await res.json()) as Partial<LiveAvatarState>;
  };

  const postAvatarState = async (payload: Record<string, unknown>): Promise<Partial<LiveAvatarState>> => {
    const url = joinUrl(apiBase || window.location.origin, "/api/avatar/state/live");
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Avatar update failed: ${res.status}`);
    return (await res.json()) as Partial<LiveAvatarState>;
  };

  useEffect(() => {
    let alive = true;

    const syncState = async () => {
      try {
        const state = await fetchAvatarState();
        if (!alive) return;
        setImageLoadFailed(false);
        setAvatarState((prev) => ({
          ...prev,
          ...state,
          mode: (state.mode as LocalAvatarState["mode"]) || prev.mode || "avatar_2d",
          expression: String(state.expression || state.emotion || prev.expression || "neutral"),
          speaking: Boolean(state.speaking ?? prev.speaking),
          listening: Boolean(state.listening ?? prev.listening),
          spec: (state.spec as AvatarSpec) || prev.spec || DEFAULT_3D_SPEC,
        }));
      } catch {
        // Keep UI alive with the hardcoded sarah-avatar.png / local role-file fallback.
      }
    };

    void syncState();
    const interval = window.setInterval(syncState, avatarState.speaking ? 300 : 900);
    return () => {
      alive = false;
      window.clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase, avatarState.speaking]);

  useEffect(() => {
    let alive = true;
    const heartbeatUrl = joinUrl(apiBase || window.location.origin, "/api/avatar/heartbeat");

    const sendHeartbeat = async (payload: Record<string, unknown> = {}) => {
      try {
        const res = await fetch(heartbeatUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          cache: "no-store",
          body: JSON.stringify(payload),
        });
        if (!res.ok) return;
        const state = (await res.json()) as Partial<LiveAvatarState>;
        if (!alive) return;
        setImageLoadFailed(false);
        setAvatarState((prev) => ({
          ...prev,
          ...state,
          mode: (state.mode as LocalAvatarState["mode"]) || prev.mode || "avatar_2d",
          expression: String(state.expression || state.emotion || prev.expression || "neutral"),
          speaking: Boolean(state.speaking ?? prev.speaking),
          listening: Boolean(state.listening ?? prev.listening),
          spec: (state.spec as AvatarSpec) || prev.spec || DEFAULT_3D_SPEC,
        }));
      } catch {
        // Heartbeat is advisory. UI state polling above remains authoritative.
      }
    };

    void sendHeartbeat({ event: "boot" });
    const interval = window.setInterval(() => {
      if (!avatarState.speaking && !avatarState.listening) {
        void sendHeartbeat({ touch: false });
      }
    }, 2500);

    return () => {
      alive = false;
      window.clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase, avatarState.speaking, avatarState.listening]);

  useEffect(() => {
    if (!is3DMode) return;
    let alive = true;
    const specUrl = joinUrl(apiBase || window.location.origin, "/api/avatar/3d/spec");

    const load3DSpec = async () => {
      try {
        const res = await fetch(`${specUrl}?t=${Date.now()}`, {
          method: "GET",
          credentials: "include",
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`Avatar 3D spec failed: ${res.status}`);
        const data = await res.json();
        const spec = (data?.spec || data) as AvatarSpec;
        if (!alive || !spec) return;
        setAvatarState((prev) => ({
          ...prev,
          spec: {
            ...DEFAULT_3D_SPEC,
            ...spec,
          },
        }));
      } catch {
        if (!alive) return;
        setAvatarState((prev) => ({
          ...prev,
          spec: prev.spec || DEFAULT_3D_SPEC,
        }));
      }
    };

    void load3DSpec();
    return () => {
      alive = false;
    };
  }, [apiBase, is3DMode]);

  useEffect(() => {
    if (!avatarState.speaking && !avatarState.listening) return;
    const interval = window.setInterval(() => setFrameTick((v) => v + 1), avatarState.speaking ? 180 : 420);
    return () => window.clearInterval(interval);
  }, [avatarState.speaking, avatarState.listening]);

  useEffect(() => {
    setAvatarState((prev) => ({ ...prev, speaking: !!avatarSpeaking, listening: avatarSpeaking ? false : prev.listening }));
    postAvatarState({ speaking: !!avatarSpeaking }).catch(() => {
      api.avatar.setSpeaking(!!avatarSpeaking).catch(() => {});
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [avatarSpeaking]);

  useEffect(() => {
    setAvatarState((prev) => ({ ...prev, listening: !!avatarListening, speaking: avatarListening ? false : prev.speaking }));
    postAvatarState({ listening: !!avatarListening }).catch(() => {
      api.avatar.setListening(!!avatarListening).catch(() => {});
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [avatarListening]);

  const handleModeChange = async (mode: LocalAvatarState["mode"]) => {
    setScreenMode(mode as any);
    setAvatarState((p) => ({ ...p, mode }));

    try {
      const state = await postAvatarState({ mode, touch: true });
      setAvatarState((p) => ({ ...p, ...state, mode }));
    } catch {
      try {
        await api.avatar.setMode(mode);
      } catch {
        // backend mode sync is optional; local UI still switches immediately
      }
    }
  };

  const openFullscreen = () => toast.info("Wire fullscreen to your layout manager when ready.");
  const openExternal = () => toast.info("Wire external view routing when ready.");
  const statusLabel = avatarState.speaking
    ? "Speaking"
    : avatarState.listening
      ? "Listening"
      : avatarState.busy
        ? "Busy"
        : avatarState.diagnostics
          ? "Diagnostics"
          : avatarState.life_state
            ? String(avatarState.life_state).replaceAll("_", " ")
            : "Ready";

  return (
    <div className="flex h-full min-h-0 max-h-[100svh] flex-col overflow-hidden rounded-none border border-cyan-500/20 bg-slate-950 text-foreground shadow-[0_0_30px_rgba(0,148,255,0.10)] sm:rounded-xl md:min-h-[520px]">
      {/* TOP SCREEN: Avatar / Desktop Mirror / Remote Conference Surface */}
      <section className="relative min-h-[210px] flex-[0_0_52%] overflow-hidden border-b border-cyan-500/15 bg-slate-950 sm:min-h-[260px] md:flex-[0_0_54%]">
        <AvatarBackground />

        <ScreenHeader
          icon={isDesktopMirror ? <Monitor className="h-3.5 w-3.5" /> : <UserRound className="h-3.5 w-3.5" />}
          title={isCallMode ? "Remote Conference" : isDesktopMirror ? "Desktop Vision" : "Avatar Surface"}
          right={
            <div className="rounded-md border border-cyan-500/20 bg-slate-950/95 px-2 py-1 text-xs text-muted-foreground shadow-lg backdrop-blur">
              <span
                className={cn(
                  "mr-1.5 inline-block h-1.5 w-1.5 rounded-full",
                  avatarState.speaking
                    ? "bg-status-warning animate-pulse"
                    : avatarState.listening
                      ? "bg-status-info animate-pulse"
                      : "bg-status-online animate-pulse",
                )}
              />
              {statusLabel}
            </div>
          }
        />

        <DesktopMirrorView apiBase={apiBase} enabled={isDesktopMirror} />

        {!isDesktopMirror && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            {render3DInline ? (
              <AvatarSurfaceBoundary
                fallback={
                  <Avatar2DImageSurface
                    src={avatarImageSrc}
                    onLoad={() => setImageLoadFailed(false)}
                    onError={() => setImageLoadFailed(true)}
                  />
                }
              >
                <Suspense
                  fallback={
                    <div className="flex h-full items-center justify-center text-muted-foreground">Loading 3D Avatar…</div>
                  }
                >
                  <Avatar3D
                    speaking={avatarState.speaking}
                    listening={avatarState.listening}
                    expression={avatarState.expression}
                    spec={avatar3DSpec}
                  />
                </Suspense>
              </AvatarSurfaceBoundary>
            ) : (
              <Avatar2DImageSurface
                src={avatarImageSrc}
                onLoad={() => setImageLoadFailed(false)}
                onError={() => setImageLoadFailed(true)}
              />
            )}
          </div>
        )}

        {/* Opaque mode controls. Kept inside top screen but locked above all render layers. */}
        <div className="absolute bottom-2 left-2 right-2 z-40 flex flex-wrap items-center gap-2 pb-[env(safe-area-inset-bottom)] sm:bottom-3 sm:left-3 sm:right-3 sm:flex-nowrap sm:pb-0">
          <Select value={currentMode} onValueChange={(v) => handleModeChange(v as LocalAvatarState["mode"])}>
            <SelectTrigger className="h-10 min-w-0 flex-1 border-cyan-500/40 bg-slate-950 text-sm text-cyan-50 shadow-xl backdrop-blur-none hover:bg-slate-900 focus:ring-2 focus:ring-cyan-500">
              <SelectValue />
            </SelectTrigger>
            <SelectContent
              position="popper"
              sideOffset={6}
              className="z-[9999] border-cyan-500/40 bg-slate-950 text-cyan-50 shadow-2xl backdrop-blur-none"
            >
              {screenModes.map((mode) => (
                <SelectItem
                  key={mode.value}
                  value={mode.value}
                  className="text-sm text-cyan-50 focus:bg-cyan-500/20 focus:text-white"
                >
                  {mode.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {is3DMode && (
            <div className="basis-full rounded-md border border-cyan-500/20 bg-slate-950/95 px-2 py-1 text-[11px] text-cyan-100 sm:hidden">
              3D Avatar runtime loaded from resources/avatars/3D.
            </div>
          )}

          <Button
            variant="ghost"
            size="icon"
            className="h-10 w-10 border border-cyan-500/20 bg-slate-950 text-cyan-100 shadow-xl hover:bg-slate-900"
            onClick={openFullscreen}
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-10 w-10 border border-cyan-500/20 bg-slate-950 text-cyan-100 shadow-xl hover:bg-slate-900"
            onClick={openExternal}
          >
            <ExternalLink className="h-4 w-4" />
          </Button>
        </div>
      </section>

      {/* BOTTOM SCREEN: Local Webcam / Local Vision / User Conference Surface */}
      <section className="relative min-h-[180px] flex-1 overflow-hidden bg-slate-950/95 sm:min-h-[220px]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,212,255,0.10),transparent_55%)]" />
        <ScreenHeader
          icon={<Camera className="h-3.5 w-3.5" />}
          title={isCallMode ? "Local User" : "Local Webcam Vision"}
          right={
            <div className="pointer-events-auto flex items-center gap-2">
              {mediaState.webcamEnabled && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 border border-cyan-500/20 bg-slate-950/95 px-2 text-xs text-cyan-100 hover:bg-slate-900"
                  onClick={() => setWebcamVisible((v) => !v)}
                >
                  {webcamVisible ? "Hide" : "Show"}
                </Button>
              )}
            </div>
          }
        />

        <div className="absolute inset-0 z-10 pt-12">
          {mediaState.webcamEnabled ? (
            <WebcamOverlay
              enabled={mediaState.webcamEnabled}
              visible={webcamVisible}
              onToggleVisible={() => setWebcamVisible((v) => !v)}
              streamToBackend={true}
              maxFps={4}
              layout="inline"
              className="h-full w-full"
            />
          ) : (
            <div className="flex h-full w-full flex-col items-center justify-center gap-3 text-muted-foreground">
              <Camera className="h-8 w-8 text-cyan-400/50" />
              <div className="text-sm">Camera is off</div>
              <div className="max-w-[80%] text-center text-xs opacity-80">
                Enable Camera in Media Controls to activate local vision under the avatar screen.
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

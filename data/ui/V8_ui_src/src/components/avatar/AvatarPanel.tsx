// AvatarPanel.tsx
import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { ExternalLink, Maximize2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

import { useSarahStore } from "@/stores/useSarahStore";
import { api } from "@/lib/api";

import { AvatarBackground } from "./AvatarBackground";
import { WebcamOverlay } from "./WebcamOverlay";
import { Avatar3D, AvatarSpec } from "./Avatar3D";

import sarahAvatarWebp from "@/assets/sarah-avatar.png?format=webp&w=640";
import sarahAvatarPng from "@/assets/sarah-avatar.png?w=640";

type LocalAvatarState = {
  mode: "avatar_2d" | "avatar_3d" | "desktop_mirror" | "media" | "idle";
  expression: string;
  speaking: boolean;
  listening: boolean;
  current_action?: string;

  // LOCAL ONLY for now (because backend AvatarState doesn't include spec yet)
  spec?: AvatarSpec;
};

type MirrorKind = "mjpeg" | "video" | "image" | "unknown";

function joinUrl(base: string, path: string): string {
  if (!base) return path;
  const b = base.endsWith("/") ? base.slice(0, -1) : base;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

async function probeUrl(url: string): Promise<{ ok: boolean; contentType?: string }> {
  // Some servers block HEAD; try GET with small timeout and abort early.
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
    // Common endpoint guesses (you can add your real one here)
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

    // Build absolute URLs
    const base = apiBase || window.location.origin;
    return paths.map((p) => joinUrl(base, p));
  }, [apiBase]);

  const detect = async () => {
    if (!enabled) return;

    setProbing(true);
    setError(null);

    // If we already found one this session, prefer it first
    const ordered = lastChosenRef.current
      ? [lastChosenRef.current, ...candidates.filter((c) => c !== lastChosenRef.current)]
      : candidates;

    for (const url of ordered) {
      const testUrl = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`; // cache-bust
      const res = await probeUrl(testUrl);
      if (!res.ok) continue;

      const kind = classifyContentType(res.contentType);

      // Accept if it looks like a stream or image
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
    setError(
      "No desktop mirror endpoint responded. Add/confirm a backend stream endpoint (MJPEG/video/image) and include it in the candidate list.",
    );
  };

  useEffect(() => {
    // When entering desktop mirror mode, auto-detect
    detect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, apiBase]);

  if (!enabled) return null;

  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center">
      {mirrorUrl ? (
        <>
          {mirrorKind === "video" ? (
            <video className="w-full h-full object-cover" src={mirrorUrl} autoPlay playsInline muted controls={false} />
          ) : (
            // MJPEG streams are usually best rendered as an <img>
            <img className="w-full h-full object-cover" src={mirrorUrl} alt="Desktop Mirror" draggable={false} />
          )}

          <div className="absolute top-2 left-2 flex items-center gap-2">
            <div className="px-2 py-1 rounded bg-background/80 backdrop-blur text-xs text-muted-foreground">
              Desktop Mirror {mirrorKind !== "unknown" ? `(${mirrorKind})` : ""}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 bg-background/80 backdrop-blur"
              onClick={detect}
              disabled={probing}
              title="Retry mirror detection"
            >
              <RefreshCw className={cn("h-4 w-4", probing && "animate-spin")} />
            </Button>
          </div>
        </>
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-muted-foreground">
          <div className="text-sm">Desktop Mirror not available</div>
          {error && <div className="text-xs max-w-[80%] text-center opacity-80">{error}</div>}
          <Button variant="outline" size="sm" onClick={detect} disabled={probing}>
            {probing ? "Detecting…" : "Retry"}
          </Button>
        </div>
      )}
    </div>
  );
}

export function AvatarPanel() {
  const mediaState = useSarahStore((s) => s.mediaState);
  const setScreenMode = useSarahStore((s) => s.setScreenMode);

  const avatarSpeaking = useSarahStore((s) => s.avatarSpeaking);
  // ✅ prevent TS/build crash if your store doesn’t define avatarListening yet
  const avatarListening = useSarahStore((s) => (s as any).avatarListening ?? false);

  const bootstrapData = useSarahStore((s) => s.bootstrapData);

  const [avatarState, setAvatarState] = useState<LocalAvatarState>({
    mode: "avatar_2d",
    expression: "neutral",
    speaking: false,
    listening: false,
    spec: { renderMode: "procedural_holo" },
  });

  const [webcamVisible, setWebcamVisible] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);

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

  const is3DMode = mediaState.screenMode === "avatar_3d";
  const isDesktopMirror = mediaState.screenMode === "desktop_mirror";

  // Best API base (works on pythonanywhere + local)
  const apiBase = bootstrapData?.env?.api_base || "";

  // Poll backend avatar state
  useEffect(() => {
    let alive = true;

    const fetchState = async () => {
      try {
        const state = await api.avatar.getState();
        if (!alive) return;

        setAvatarState((prev) => ({
          ...prev,
          ...state,
          spec: prev.spec,
        }));
      } catch {
        // keep UI alive
      }
    };

    fetchState();
    const interval = setInterval(fetchState, 1500);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);

  // Keep backend in sync with speaking/listening (store-driven)
  useEffect(() => {
    api.avatar.setSpeaking(!!avatarSpeaking).catch(() => {});
  }, [avatarSpeaking]);

  useEffect(() => {
    api.avatar.setListening(!!avatarListening).catch(() => {});
  }, [avatarListening]);

  const handleModeChange = async (mode: LocalAvatarState["mode"]) => {
    setScreenMode(mode as any);
    setAvatarState((p) => ({ ...p, mode }));

    try {
      await api.avatar.setMode(mode);
    } catch {
      // ok
    }
  };

  const openFullscreen = () => toast.info("Wire fullscreen to your layout manager when ready.");
  const openExternal = () => toast.info("Wire external view routing when ready.");

  return (
    <div className="relative aspect-video bg-gradient-to-b from-background/80 to-background border-b border-sidebar-border overflow-hidden">
      <AvatarBackground />

      {/* ✅ Desktop Mirror layer (renders above background, below controls) */}
      <DesktopMirrorView apiBase={apiBase} enabled={isDesktopMirror} />

      {/* Avatar display (hidden when desktop mirror is active) */}
      {!isDesktopMirror && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          {is3DMode ? (
            <Suspense
              fallback={
                <div className="flex items-center justify-center h-full text-muted-foreground">Loading 3D Avatar…</div>
              }
            >
              <Avatar3D
                speaking={avatarState.speaking}
                listening={avatarState.listening}
                expression={avatarState.expression}
                spec={avatarState.spec}
              />
            </Suspense>
          ) : (
            <picture>
              <source srcSet={sarahAvatarWebp} type="image/webp" />
              <img
                src={sarahAvatarPng}
                alt="Sarah AI Avatar"
                fetchPriority="high"
                width={640}
                height={360}
                className={cn(
                  "w-full h-full object-cover opacity-90 transition-all duration-300",
                  isAnimating && "scale-105",
                  avatarState.speaking && "animate-pulse",
                )}
              />
            </picture>
          )}
        </div>
      )}

      {/* Webcam overlay (optional: you might want to auto-hide this in desktop mirror mode) */}
      <WebcamOverlay
        enabled={mediaState.webcamEnabled}
        visible={webcamVisible && !isDesktopMirror} // ✅ avoid covering the mirrored desktop
        onToggleVisible={() => setWebcamVisible((v) => !v)}
        streamToBackend={false}
        maxFps={4}
      />

      {/* Controls */}
      <div className="absolute bottom-2 left-2 right-2 flex items-center gap-2 z-20">
        <Select value={mediaState.screenMode} onValueChange={(v) => handleModeChange(v as any)}>
          <SelectTrigger className="h-8 text-xs bg-background/80 backdrop-blur border-border flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {screenModes.map((mode) => (
              <SelectItem key={mode.value} value={mode.value} className="text-xs">
                {mode.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button variant="ghost" size="icon" className="h-8 w-8 bg-background/80 backdrop-blur" onClick={openFullscreen}>
          <Maximize2 className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8 bg-background/80 backdrop-blur" onClick={openExternal}>
          <ExternalLink className="h-4 w-4" />
        </Button>
      </div>

      {/* Status indicator */}
      <div className="absolute top-2 right-2 z-20 flex flex-col gap-1 items-end">
        <div className="px-2 py-1 bg-background/80 backdrop-blur rounded text-xs text-muted-foreground flex items-center gap-1.5">
          <span
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              avatarState.speaking
                ? "bg-status-warning animate-pulse"
                : avatarState.listening
                  ? "bg-status-info animate-pulse"
                  : "bg-status-online animate-pulse",
            )}
          />
          {avatarState.speaking ? "Speaking" : avatarState.listening ? "Listening" : "Ready"}
        </div>

        {mediaState.webcamEnabled && !webcamVisible && !isDesktopMirror && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs bg-background/80 backdrop-blur"
            onClick={() => setWebcamVisible(true)}
          >
            Show Webcam
          </Button>
        )}
      </div>
    </div>
  );
}

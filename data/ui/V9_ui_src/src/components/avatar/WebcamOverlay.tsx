import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Camera, EyeOff, Eye, AlertTriangle } from "lucide-react";

type VisionPolicy = {
  enabled?: boolean;
  accept_frontend_frames?: boolean;
  backend_controls_fps?: boolean;
  max_fps?: number;
  max_width?: number;
  max_height?: number;
  jpeg_quality?: number;
  frame_ttl_seconds?: number;
};

const DEFAULT_VISION_POLICY: Required<Pick<VisionPolicy, "enabled" | "accept_frontend_frames" | "backend_controls_fps" | "max_fps" | "max_width" | "max_height" | "jpeg_quality">> = {
  enabled: true,
  accept_frontend_frames: true,
  backend_controls_fps: true,
  max_fps: 2,
  max_width: 640,
  max_height: 360,
  jpeg_quality: 0.7,
};

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

/**
 * WebcamOverlay:
 * - Frontend authority: camera ON/OFF, show/hide preview, and user submit only.
 * - Backend authority: whether frames are accepted, max FPS, frame dimensions, learning policy.
 * - overlay layout: legacy corner camera overlay
 * - inline layout: full bottom-screen camera/vision pane for AvatarPanel dual-screen mode
 */
export function WebcamOverlay({
  enabled,
  visible,
  onToggleVisible,
  streamToBackend = true,
  maxFps = 4,
  layout = "overlay",
  className,
}: {
  enabled: boolean;
  visible: boolean;
  onToggleVisible: () => void;
  /** Legacy prop retained for compatibility. Backend policy is authoritative. */
  streamToBackend?: boolean;
  /** Legacy prop retained for compatibility. Backend policy max_fps is authoritative. */
  maxFps?: number;
  layout?: "overlay" | "inline";
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [policy, setPolicy] = useState<VisionPolicy>(DEFAULT_VISION_POLICY);

  const isInline = layout === "inline";
  const backendAcceptsFrames = policy.enabled !== false && policy.accept_frontend_frames !== false;
  const canStream = enabled && visible && backendAcceptsFrames && streamToBackend;
  const backendMaxFps = clampNumber(policy.max_fps, DEFAULT_VISION_POLICY.max_fps, 0.25, 10);
  const effectiveMaxFps = policy.backend_controls_fps === false ? clampNumber(maxFps, DEFAULT_VISION_POLICY.max_fps, 0.25, 10) : backendMaxFps;
  const targetWidth = Math.round(clampNumber(policy.max_width, DEFAULT_VISION_POLICY.max_width, 160, 1280));
  const targetHeight = Math.round(clampNumber(policy.max_height, DEFAULT_VISION_POLICY.max_height, 90, 720));
  const jpegQuality = clampNumber(policy.jpeg_quality, DEFAULT_VISION_POLICY.jpeg_quality, 0.35, 0.92);

  useEffect(() => {
    let cancelled = false;
    const loadPolicy = async () => {
      try {
        const res = await fetch("/api/vision/policy", { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        const next = data?.policy && typeof data.policy === "object" ? data.policy : data;
        if (!cancelled && next && typeof next === "object") {
          setPolicy({ ...DEFAULT_VISION_POLICY, ...next });
        }
      } catch {
        // Policy endpoint is optional during transition; local defaults remain safe.
      }
    };
    void loadPolicy();
    const timer = window.setInterval(loadPolicy, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    const start = async () => {
      if (!enabled) return;

      setError(null);
      setReady(false);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: targetWidth }, height: { ideal: targetHeight }, facingMode: "user" },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setReady(true);
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unable to access webcam";
        setError(msg);
      }
    };

    start();

    return () => {
      cancelled = true;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [enabled, targetWidth, targetHeight]);

  useEffect(() => {
    if (!canStream) return;

    let raf = 0;
    let lastSent = 0;
    let sending = false;

    const loop = () => {
      raf = requestAnimationFrame(loop);

      const now = performance.now();
      const minInterval = 1000 / Math.max(0.25, effectiveMaxFps);
      if (now - lastSent < minInterval) return;
      if (sending) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !ready) return;
      if (video.videoWidth <= 0 || video.videoHeight <= 0) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const sourceRatio = video.videoWidth / Math.max(1, video.videoHeight);
      const policyRatio = targetWidth / Math.max(1, targetHeight);
      let w = targetWidth;
      let h = targetHeight;
      if (Math.abs(sourceRatio - policyRatio) > 0.05) {
        h = Math.round(w / sourceRatio);
        if (h > targetHeight) {
          h = targetHeight;
          w = Math.round(h * sourceRatio);
        }
      }

      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);

      const dataUrl = canvas.toDataURL("image/jpeg", jpegQuality);
      lastSent = now;
      sending = true;

      void fetch("/api/vision/frame/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          imageBase64: dataUrl,
          data_url: dataUrl,
          source: "webcam_overlay",
          analyze: false,
          question: "VR HUD observation pass",
          width: w,
          height: h,
          mime: "image/jpeg",
          backendPolicy: {
            max_fps: effectiveMaxFps,
            max_width: targetWidth,
            max_height: targetHeight,
          },
          ts: Date.now(),
        }),
      })
        .catch(() => {
          // Vision endpoint is optional; never break the UI.
        })
        .finally(() => {
          sending = false;
        });
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [canStream, effectiveMaxFps, ready, targetWidth, targetHeight, jpegQuality]);

  if (!enabled) return null;

  if (isInline) {
    return (
      <div className={cn("relative h-full w-full overflow-hidden bg-slate-950", className)}>
        {visible ? (
          <div className="relative h-full w-full">
            {error ? (
              <div className="flex h-full w-full items-center justify-center p-3 text-xs text-muted-foreground">
                <div className="flex max-w-[80%] items-center gap-2 rounded-lg border border-red-500/20 bg-red-950/20 p-3">
                  <AlertTriangle className="h-4 w-4" />
                  {error}
                </div>
              </div>
            ) : (
              <>
                <video ref={videoRef} className="block h-full w-full object-cover" playsInline muted />
                <canvas ref={canvasRef} className="hidden" />
                {!ready && (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                    Starting camera…
                  </div>
                )}
                {ready && !backendAcceptsFrames && (
                  <div className="absolute bottom-2 left-2 rounded bg-background/70 px-2 py-1 text-[10px] text-muted-foreground backdrop-blur">
                    Backend frame ingest off
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-3 text-muted-foreground">
            <EyeOff className="h-8 w-8 text-cyan-400/50" />
            <div className="text-sm">Local vision hidden</div>
            <Button variant="outline" size="sm" onClick={onToggleVisible}>
              Show Webcam
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn("absolute left-2 top-2 z-30", className)}>
      <div
        className={cn(
          "overflow-hidden rounded-lg border border-border bg-background/70 shadow-sm backdrop-blur",
          !visible && "opacity-70",
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-2 py-1">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Camera className="h-3.5 w-3.5" />
            Vision
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onToggleVisible}
            title={visible ? "Hide webcam" : "Show webcam"}
          >
            {visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </Button>
        </div>

        {visible && (
          <div className="relative">
            {error ? (
              <div className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
                <AlertTriangle className="h-4 w-4" />
                {error}
              </div>
            ) : (
              <>
                <video ref={videoRef} className="block h-[135px] w-[240px] object-cover" playsInline muted />
                <canvas ref={canvasRef} className="hidden" />
                {!ready && (
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                    Starting camera…
                  </div>
                )}
                {ready && !backendAcceptsFrames && (
                  <div className="absolute bottom-1 left-1 rounded bg-background/70 px-1.5 py-0.5 text-[9px] text-muted-foreground backdrop-blur">
                    ingest off
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Camera, EyeOff, Eye, AlertTriangle } from "lucide-react";

/**
 * WebcamOverlay:
 * - overlay layout: legacy corner camera overlay
 * - inline layout: full bottom-screen camera/vision pane for AvatarPanel dual-screen mode
 * - streams low-FPS frames to /api/vision/frame when streamToBackend is enabled
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
  streamToBackend?: boolean;
  maxFps?: number;
  layout?: "overlay" | "inline";
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isInline = layout === "inline";
  const canStream = enabled && visible && streamToBackend;

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    const start = async () => {
      if (!enabled) return;

      setError(null);
      setReady(false);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: "user" },
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
  }, [enabled]);

  useEffect(() => {
    if (!canStream) return;

    let raf = 0;
    let lastSent = 0;
    let sending = false;

    const loop = () => {
      raf = requestAnimationFrame(loop);

      const now = performance.now();
      const minInterval = 1000 / Math.max(1, maxFps);
      if (now - lastSent < minInterval) return;
      if (sending) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !ready) return;
      if (video.videoWidth <= 0 || video.videoHeight <= 0) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = 320;
      const h = 180;
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      lastSent = now;
      sending = true;

      void fetch("/api/vision/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          imageBase64: dataUrl,
          source: "webcam_overlay",
          width: w,
          height: h,
          mime: "image/jpeg",
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
  }, [canStream, maxFps, ready]);

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
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

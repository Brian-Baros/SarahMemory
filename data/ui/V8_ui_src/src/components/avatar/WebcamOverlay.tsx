import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Camera, EyeOff, Eye, AlertTriangle } from "lucide-react";

/**
 * WebcamOverlay:
 * - Shows webcam feed in-corner
 * - Optionally streams frames to backend at low FPS for vision (wave detection etc)
 *
 * Backend optional endpoints you can add:
 * - api.avatar.pushVisionFrame({ imageBase64, ts })
 * OR a websocket "vision_frame" event.
 */
export function WebcamOverlay({
  enabled,
  visible,
  onToggleVisible,
  streamToBackend = true,
  maxFps = 4, // keep it safe by default
  className,
}: {
  enabled: boolean;
  visible: boolean;
  onToggleVisible: () => void;
  streamToBackend?: boolean;
  maxFps?: number;
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canStream = enabled && visible && streamToBackend;

  useEffect(() => {
    let stream: MediaStream | null = null;

    const start = async () => {
      if (!enabled) return;

      setError(null);
      setReady(false);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: "user" },
          audio: false,
        });

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
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [enabled]);

  // Throttled frame push - placeholder for future vision API
  useEffect(() => {
    if (!canStream) return;

    let raf = 0;
    let lastSent = 0;

    const loop = () => {
      raf = requestAnimationFrame(loop);

      const now = performance.now();
      const minInterval = 1000 / Math.max(1, maxFps);
      if (now - lastSent < minInterval) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      if (!ready) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Draw current frame into canvas
      const w = 320;
      const h = 180;
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);

      // Encode frame (ready for future backend vision endpoint)
      // const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      lastSent = now;

      // Note: pushVisionFrame endpoint not yet implemented in api.ts
      // When ready, uncomment and call: api.avatar.pushVisionFrame({ image: dataUrl, ts: Date.now() })
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [canStream, maxFps, ready]);

  if (!enabled) return null;

  return (
    <div className={cn("absolute top-2 left-2 z-30", className)}>
      <div
        className={cn(
          "rounded-lg overflow-hidden border border-border bg-background/70 backdrop-blur shadow-sm",
          !visible && "opacity-70",
        )}
      >
        <div className="flex items-center justify-between px-2 py-1 border-b border-border">
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
              <div className="p-3 text-xs text-muted-foreground flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                {error}
              </div>
            ) : (
              <>
                <video ref={videoRef} className="block w-[240px] h-[135px] object-cover" playsInline muted />
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

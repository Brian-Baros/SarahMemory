import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle, Camera, Crosshair, Maximize2, Pause, Play, Radar, Shield, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { api, type VisionHudPacket, type VisionHudTarget } from "@/lib/api";

const FRAME_SUBMIT_INTERVAL_MS = 500;
const ANALYZE_INTERVAL_MS = 1500;
const HUD_PACKET_POLL_MS = 750;

type HudMode = "idle" | "starting" | "running" | "paused" | "error";

function pct(value: unknown, fallback = 0): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(100, n * 100));
}

function confidenceText(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;
}

function compact(value: unknown, fallback = "--"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return fallback;
  }
}

function normalizeTarget(target: VisionHudTarget) {
  const box = Array.isArray(target.bbox) ? target.bbox : [0.42, 0.35, 0.58, 0.55];
  const [x1, y1, x2, y2] = box.map((v) => Math.max(0, Math.min(1, Number(v) || 0)));
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
  };
}

function TargetBracket({ target, index }: { target: VisionHudTarget; index: number }) {
  const b = normalizeTarget(target);
  const x = pct(b.x);
  const y = pct(b.y);
  const w = Math.max(4, pct(b.w));
  const h = Math.max(4, pct(b.h));
  const label = String(target.label || target.class || target.id || `TARGET_${index}`);
  const conf = confidenceText(target.confidence);
  const dx = compact(target.vectors?.dx);
  const dy = compact(target.vectors?.dy);
  const dz = compact(target.vectors?.dz_est);

  return (
    <div
      className="absolute pointer-events-none"
      style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` }}
    >
      <div className="absolute left-0 top-0 h-5 w-5 border-l-2 border-t-2 border-red-500/90" />
      <div className="absolute right-0 top-0 h-5 w-5 border-r-2 border-t-2 border-red-500/90" />
      <div className="absolute bottom-0 left-0 h-5 w-5 border-b-2 border-l-2 border-red-500/90" />
      <div className="absolute bottom-0 right-0 h-5 w-5 border-b-2 border-r-2 border-red-500/90" />
      <div className="absolute -right-2 top-0 translate-x-full min-w-40 rounded border border-red-500/60 bg-black/70 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-red-200 shadow-lg">
        <div className="text-red-400">{label}</div>
        <div>CONF {conf}</div>
        <div>DX {dx} DY {dy}</div>
        <div>DZ {dz}</div>
      </div>
    </div>
  );
}

function DataTape({ title, rows }: { title: string; rows: Array<[string, unknown]> }) {
  return (
    <div className="rounded border border-red-500/30 bg-black/55 p-3 font-mono text-[11px] text-red-200 backdrop-blur-sm">
      <div className="mb-2 flex items-center gap-2 text-red-500">
        <Activity className="h-3.5 w-3.5" />
        <span className="uppercase tracking-[0.22em]">{title}</span>
      </div>
      <div className="space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-red-500/10 pb-1">
            <span className="text-red-400/80">{k}</span>
            <span className="text-right text-red-100">{compact(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function VisionScreen() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const submitBusyRef = useRef(false);
  const [mode, setMode] = useState<HudMode>("idle");
  const [error, setError] = useState("");
  const [hudPacket, setHudPacket] = useState<VisionHudPacket | null>(null);
  const [frameStatus, setFrameStatus] = useState<any>(null);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
  const [analyzeEnabled, setAnalyzeEnabled] = useState(true);

  const targets = useMemo(() => hudPacket?.active_targets || [], [hudPacket]);
  const compute = hudPacket?.compute_integrity || {};
  const kinetic = hudPacket?.kinetic_integrity || {};
  const smget = hudPacket?.smget_state || {};
  const vision = hudPacket?.vision || {};

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setMode("idle");
  }, []);

  const startCamera = useCallback(async () => {
    setError("");
    setMode("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setMode("running");
    } catch (err: any) {
      setMode("error");
      setError(String(err?.message || err || "Camera start failed"));
    }
  }, []);

  const captureDataUrl = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 360;
    if (!width || !height) return null;
    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / width);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.62);
  }, []);

  const submitFrame = useCallback(async () => {
    if (submitBusyRef.current || mode !== "running") return;
    const dataUrl = captureDataUrl();
    if (!dataUrl) return;
    submitBusyRef.current = true;
    try {
      const result = await api.vision.submitFrame(dataUrl, {
        source: "vision_screen_live_submit",
        analyze: false,
        question: "VR HUD live frame submit",
      });
      if (result?.hud_packet) setHudPacket(result.hud_packet);
      if (result?.frame_status) setFrameStatus(result.frame_status);
      if (!result?.ok && result?.error) setError(result.error);
    } catch (err: any) {
      setError(String(err?.message || err || "Vision frame submit failed"));
    } finally {
      submitBusyRef.current = false;
    }
  }, [captureDataUrl, mode]);

  const analyzeFrame = useCallback(async () => {
    if (busyRef.current || mode !== "running" || !analyzeEnabled) return;
    const dataUrl = captureDataUrl();
    if (!dataUrl) return;
    busyRef.current = true;
    const started = performance.now();
    try {
      const result = await api.vision.analyzeFrame(dataUrl, "VR HUD observation pass", false);
      setLastLatencyMs(Math.round(performance.now() - started));
      if (result?.hud_packet) setHudPacket(result.hud_packet);
      if (!result?.ok && result?.error) setError(result.error);
      const status = await api.vision.frameStatus().catch(() => null);
      if (status) setFrameStatus(status);
    } catch (err: any) {
      setError(String(err?.message || err || "Vision analysis failed"));
    } finally {
      busyRef.current = false;
    }
  }, [analyzeEnabled, captureDataUrl, mode]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  useEffect(() => {
    const timer = window.setInterval(() => void submitFrame(), FRAME_SUBMIT_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [submitFrame]);

  useEffect(() => {
    const timer = window.setInterval(() => void analyzeFrame(), ANALYZE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [analyzeFrame]);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const [packet, status] = await Promise.all([
          api.vision.hudPacket().catch(() => null),
          api.vision.frameStatus().catch(() => null),
        ]);
        if (packet?.hud_packet) setHudPacket(packet.hud_packet);
        if (status) setFrameStatus(status);
      } catch {
        // HUD polling is advisory only.
      }
    }, HUD_PACKET_POLL_MS);
    return () => window.clearInterval(timer);
  }, []);

  const goFullscreen = async () => {
    try {
      await document.documentElement.requestFullscreen();
    } catch (err: any) {
      setError(String(err?.message || err || "Fullscreen request failed"));
    }
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black text-red-100">
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover opacity-85 contrast-125 grayscale"
        muted
        playsInline
      />
      <canvas ref={canvasRef} className="hidden" />
      <div className="absolute inset-0 bg-red-950/10 mix-blend-screen" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0,transparent_48%,rgba(0,0,0,0.65)_100%)]" />
      <div className="absolute inset-0 opacity-[0.08] [background-image:linear-gradient(rgba(255,0,0,.8)_1px,transparent_1px),linear-gradient(90deg,rgba(255,0,0,.8)_1px,transparent_1px)] [background-size:48px_48px]" />

      {/* Center reticle */}
      <div className="absolute left-1/2 top-1/2 h-28 w-28 -translate-x-1/2 -translate-y-1/2 rounded-full border border-red-500/30">
        <Crosshair className="absolute left-1/2 top-1/2 h-10 w-10 -translate-x-1/2 -translate-y-1/2 text-red-500/80" />
      </div>

      {targets.map((target, idx) => (
        <TargetBracket key={target.id || idx} target={target} index={idx} />
      ))}

      {/* Top bar */}
      <div className="absolute left-4 right-4 top-3 flex items-center justify-between rounded border border-red-500/30 bg-black/60 px-4 py-2 font-mono text-xs backdrop-blur-sm">
        <div className="flex items-center gap-3 uppercase tracking-[0.25em] text-red-500">
          <Radar className="h-4 w-4" />
          <span>SarahMemory VR Operator HUD</span>
        </div>
        <div className="flex gap-4 text-red-200">
          <span>MODE {hudPacket?.mode || "OBSERVE_ONLY"}</span>
          <span>FRAME {frameStatus?.frame_id || hudPacket?.frame?.frame_id || "NO_FRAME"}</span>
          <span>LAT {lastLatencyMs ?? "--"}MS</span>
          <span>TGT {targets.length}</span>
        </div>
      </div>

      {/* Data tapes */}
      <div className="absolute left-4 top-20 w-72 space-y-3">
        <DataTape
          title="Compute Integrity"
          rows={[
            ["HUD SCHEMA", hudPacket?.schema || "SMHUD_PACKET_V1"],
            ["THREADS", (compute as any)?.thread_state?.active_threads],
            ["MEM MB", (compute as any)?.memory_pool_mb],
            ["LATENCY", lastLatencyMs === null ? "--" : `${lastLatencyMs}ms`],
            ["ANALYSIS", analyzeEnabled ? "ACTIVE" : "PAUSED"],
          ]}
        />
        <DataTape
          title="Vision Feed"
          rows={[
            ["CAMERA", mode.toUpperCase()],
            ["FRAME", frameStatus?.has_frame ? "LOCKED" : "NO FRAME"],
            ["SOURCE", frameStatus?.source || hudPacket?.frame?.source],
            ["SIZE", frameStatus?.width && frameStatus?.height ? `${frameStatus.width}x${frameStatus.height}` : "--"],
            ["CONF", confidenceText((vision as any)?.confidence)],
          ]}
        />
      </div>

      <div className="absolute right-4 top-20 w-80 space-y-3">
        <DataTape
          title="Kinetic Integrity"
          rows={[
            ["BODY", (kinetic as any)?.body_state || "OBSERVE_ONLY"],
            ["MOVE LOCK", (kinetic as any)?.movement_lock === false ? "OPEN" : "LOCKED"],
            ["DEVICES", Array.isArray((kinetic as any)?.devices) ? (kinetic as any).devices.length : 0],
            ["MSDC", (kinetic as any)?.msdc?.ok === false ? "DEGRADED" : "READY"],
            ["FAULT", "NONE"],
          ]}
        />
        <DataTape
          title="SMGET Gate"
          rows={[
            ["STATE", (smget as any)?.state || "NO_ACTIVE_ACTION_CONTRACT"],
            ["DECISION", (smget as any)?.decision || "READ_ONLY_WITNESS"],
            ["ROLLBACK", (smget as any)?.rollback_ready === false ? "NO" : "READY"],
            ["HOW", (smget as any)?.six_question_loop?.HOW || "STANDBY"],
            ["AUTHORITY", "USER"],
          ]}
        />
      </div>

      {/* Controls */}
      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded border border-red-500/30 bg-black/70 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-red-500" />
          <Label className="font-mono text-xs uppercase tracking-[0.22em] text-red-200">
            OBSERVE_ONLY / MOVEMENT LOCKED / HUD CANNOT AUTHORIZE ACTIONS
          </Label>
        </div>
        <div className="flex flex-wrap gap-2">
          {mode === "running" ? (
            <Button variant="outline" size="sm" onClick={stopCamera} className="border-red-500/40 bg-black/40 text-red-100 hover:bg-red-950/40">
              <Pause className="mr-2 h-4 w-4" /> Stop Camera
            </Button>
          ) : (
            <Button variant="outline" size="sm" onClick={startCamera} className="border-red-500/40 bg-black/40 text-red-100 hover:bg-red-950/40">
              <Camera className="mr-2 h-4 w-4" /> Start Camera
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={() => setAnalyzeEnabled((v) => !v)} className="border-red-500/40 bg-black/40 text-red-100 hover:bg-red-950/40">
            {analyzeEnabled ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
            {analyzeEnabled ? "Pause Analysis" : "Resume Analysis"}
          </Button>
          <Button variant="outline" size="sm" onClick={() => void analyzeFrame()} className="border-red-500/40 bg-black/40 text-red-100 hover:bg-red-950/40">
            <Video className="mr-2 h-4 w-4" /> Analyze Now
          </Button>
          <Button variant="outline" size="sm" onClick={() => void goFullscreen()} className="border-red-500/40 bg-black/40 text-red-100 hover:bg-red-950/40">
            <Maximize2 className="mr-2 h-4 w-4" /> Fullscreen
          </Button>
        </div>
      </div>

      {error && (
        <div className="absolute bottom-24 left-1/2 flex -translate-x-1/2 items-start gap-2 rounded border border-yellow-500/50 bg-black/80 px-4 py-3 text-sm text-yellow-200">
          <AlertTriangle className="mt-0.5 h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {mode === "idle" && (
        <div className="absolute left-1/2 top-1/2 w-[min(680px,90vw)] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-red-500/30 bg-black/80 p-6 text-center shadow-2xl backdrop-blur-sm">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full border border-red-500/50">
            <Radar className="h-6 w-6 text-red-500" />
          </div>
          <h1 className="font-mono text-xl uppercase tracking-[0.28em] text-red-300">VR Operator HUD</h1>
          <p className="mt-3 text-sm text-red-100/80">
            Start the camera, move this browser window to the PSVR display, then press Fullscreen. This surface is read-only telemetry; it does not control movement.
          </p>
          <Button onClick={startCamera} className="mt-5 bg-red-700 text-white hover:bg-red-800">
            <Camera className="mr-2 h-4 w-4" /> Start Camera Feed
          </Button>
        </div>
      )}
    </div>
  );
}

export default VisionScreen;

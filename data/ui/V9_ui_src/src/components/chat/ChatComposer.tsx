import { useState, useRef, useEffect, useCallback, KeyboardEvent } from "react";
import { Send, Mic, Paperclip, Loader2, X, SlidersHorizontal, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { useIsMobile } from "@/hooks/use-mobile";

// Web Speech API types
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  onresult: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionErrorEvent) => void) | null;
  onend: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  start: () => void;
  stop: () => void;
}

declare global {
  interface Window {
    webkitSpeechRecognition?: new () => SpeechRecognitionInstance;
    SpeechRecognition?: new () => SpeechRecognitionInstance;
  }
}

type Props = {
  onSendText: (text: string, files?: File[], options?: { ingest?: boolean }) => Promise<void> | void;
  isSending?: boolean;
  onMicStateChange?: (listening: boolean, reason: string) => void;
};

export function ChatComposer({ onSendText, isSending: isSendingProp, onMicStateChange }: Props) {
  const isMobile = useIsMobile();

  const [message, setMessage] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [localSending, setLocalSending] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [ingestFiles, setIngestFiles] = useState(false);
  const [showMicOptions, setShowMicOptions] = useState(false);
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedMicId, setSelectedMicId] = useState(() => localStorage.getItem("sarah:microphone-device") || "default");
  const [speechLanguage, setSpeechLanguage] = useState(() => localStorage.getItem("sarah:speech-language") || "en-US");
  const [micPermission, setMicPermission] = useState<"unknown" | "granted" | "denied">("unknown");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const desiredListeningRef = useRef(false);
  const recognitionManualStopRef = useRef(false);
  const recognitionRestartTimerRef = useRef<number | null>(null);
  const composerRef = useRef<HTMLDivElement>(null);

  const isSending = Boolean(isSendingProp) || localSending;

  const emitMicState = useCallback(
    (listening: boolean, reason: string) => {
      setIsListening(listening);

      try {
        onMicStateChange?.(listening, reason);
      } catch {
        // non-critical parent bridge
      }

      try {
        window.dispatchEvent(
          new CustomEvent("sarah:chat-composer", {
            detail: {
              type: "mic_state",
              listening,
              reason,
              ts: Date.now(),
            },
          })
        );
      } catch {
        // non-critical browser event bridge
      }
    },
    [onMicStateChange]
  );

  // ---------------------------------------------------------------------------
  // Expose composer height to CSS for correct mobile scroll padding
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;

    const setH = () => {
      const h = Math.max(56, Math.ceil(el.getBoundingClientRect().height));
      document.documentElement.style.setProperty("--composer-h", `${h}px`);
    };

    setH();

    const ro = new ResizeObserver(() => setH());
    ro.observe(el);

    return () => {
      ro.disconnect();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Speech Recognition init
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionCtor) return;

    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = speechLanguage;

    recognition.onstart = () => {
      recognitionManualStopRef.current = false;
      emitMicState(true, "speech_recognition_start");
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let transcript = "";
      for (let i = event.results.length - 1; i >= 0; i--) {
        const result = event.results[i];
        transcript = result[0].transcript;
        if (result.isFinal) break;
      }
      if (transcript) {
        setMessage((prev) => {
          // Don't override typed text if user is actively typing
          if (prev.trim().length > 0 && !prev.endsWith(" ")) return prev;
          return transcript.trim();
        });
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.warn("[ChatComposer] Speech recognition error:", event.error);
      const err = event.error || "unknown";
      if (["no-speech", "network", "audio-capture"].includes(err) && desiredListeningRef.current) {
        emitMicState(true, `speech_recognition_retry:${err}`);
        return;
      }
      desiredListeningRef.current = false;
      emitMicState(false, `speech_recognition_error:${err}`);
      try {
        api.avatar.setListening(false).catch(() => {});
      } catch {
        // non-critical avatar sync
      }
    };

    recognition.onend = () => {
      if (recognitionRestartTimerRef.current) {
        window.clearTimeout(recognitionRestartTimerRef.current);
        recognitionRestartTimerRef.current = null;
      }
      if (desiredListeningRef.current && !recognitionManualStopRef.current) {
        emitMicState(true, "speech_recognition_relisten_pending");
        recognitionRestartTimerRef.current = window.setTimeout(() => {
          try {
            recognition.start();
            emitMicState(true, "speech_recognition_relisten_start");
            api.avatar.setListening(true).catch(() => {});
          } catch (e) {
            console.warn("[ChatComposer] Speech recognition relisten failed:", e);
            desiredListeningRef.current = false;
            emitMicState(false, "speech_recognition_relisten_failed");
            api.avatar.setListening(false).catch(() => {});
          }
        }, 250);
        return;
      }
      emitMicState(false, "speech_recognition_end");
      try {
        api.avatar.setListening(false).catch(() => {});
      } catch {
        // non-critical avatar sync
      }
    };

    recognitionRef.current = recognition;

    return () => {
      desiredListeningRef.current = false;
      recognitionManualStopRef.current = true;
      if (recognitionRestartTimerRef.current) {
        window.clearTimeout(recognitionRestartTimerRef.current);
        recognitionRestartTimerRef.current = null;
      }
      try {
        recognition.stop();
      } catch {}
      recognitionRef.current = null;
      emitMicState(false, "component_unmount");
      try {
        api.avatar.setListening(false).catch(() => {});
      } catch {
        // non-critical avatar sync
      }
    };
  }, [emitMicState, speechLanguage]);

  useEffect(() => {
    try {
      const pending = window.sessionStorage.getItem("sarah:chat-pending-draft");
      if (pending) {
        setMessage((current) => (current.trim() ? `${current}\n${pending}` : pending));
        window.sessionStorage.removeItem("sarah:chat-pending-draft");
      }
    } catch {
      // session storage may be unavailable in restricted browser contexts
    }
    const receiveDraft = (event: Event) => {
      const detail = (event as CustomEvent<{ draft?: string }>).detail;
      const draft = String(detail?.draft || "");
      if (!draft) return;
      setMessage((current) => (current.trim() ? `${current}\n${draft}` : draft));
      try { window.sessionStorage.removeItem("sarah:chat-pending-draft"); } catch { /* non-fatal */ }
    };
    window.addEventListener("sarah:chat-draft", receiveDraft);
    return () => window.removeEventListener("sarah:chat-draft", receiveDraft);
  }, []);

  const refreshMicrophones = useCallback(async (requestPermission = false) => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) throw new Error("Media devices are unavailable");
      if (requestPermission) {
        const constraints: MediaStreamConstraints = {
          audio: selectedMicId === "default" ? true : { deviceId: { exact: selectedMicId } },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        stream.getTracks().forEach((track) => track.stop());
        setMicPermission("granted");
      }
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "audioinput");
      setMicDevices(devices);
    } catch (error) {
      setMicPermission("denied");
      toast.error(error instanceof Error ? error.message : "Microphone permission denied");
    }
  }, [selectedMicId]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const toggleListening = async () => {
    const recognition = recognitionRef.current;
    if (!recognition) {
      toast.error("Speech recognition not supported in this browser");
      return;
    }

    try {
      if (isListening) {
        desiredListeningRef.current = false;
        recognitionManualStopRef.current = true;
        if (recognitionRestartTimerRef.current) {
          window.clearTimeout(recognitionRestartTimerRef.current);
          recognitionRestartTimerRef.current = null;
        }
        recognition.stop();
        emitMicState(false, "manual_stop");
        await api.avatar.setListening(false);
      } else {
        desiredListeningRef.current = true;
        recognitionManualStopRef.current = false;
        recognition.start();
        emitMicState(true, "manual_start");
        await api.avatar.setListening(true);
      }
    } catch (e) {
      console.warn("[ChatComposer] toggleListening failed:", e);
      desiredListeningRef.current = false;
      recognitionManualStopRef.current = true;
      emitMicState(false, "toggle_failed");
      try {
        await api.avatar.setListening(false);
      } catch {
        // non-critical avatar sync
      }
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setSelectedFiles((current) => [...current, ...Array.from(files)].slice(0, 20));
  };

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    const trimmed = message.trim();
    if (!trimmed && selectedFiles.length === 0) return;
    if (isSending) return;

    setLocalSending(true);

    // stop mic capture if active
    try {
      desiredListeningRef.current = false;
      recognitionManualStopRef.current = true;
      if (recognitionRestartTimerRef.current) {
        window.clearTimeout(recognitionRestartTimerRef.current);
        recognitionRestartTimerRef.current = null;
      }
      recognitionRef.current?.stop();
    } catch {}
    emitMicState(false, "submit_stop");

    const payloadText = trimmed || "(file upload)";

    try {
      setMessage("");
      setSelectedFiles([]);
      await onSendText(payloadText, selectedFiles, { ingest: ingestFiles });
      setIngestFiles(false);
    } catch (e) {
      console.error("[ChatComposer] send failed:", e);
      toast.error("Failed to send message");
      // restore on failure
      setMessage(payloadText === "(file upload)" ? "" : payloadText);
    } finally {
      setLocalSending(false);
      try {
        await api.avatar.setListening(false);
      } catch {}
      emitMicState(false, "submit_complete");
    }
  };

  return (
    <div
      ref={composerRef}
      className={cn(
        // Desktop/windowed: composer stays inside the chat panel at the bottom
        !isMobile && "sticky bottom-0 z-10",
        // Mobile shell: composer pinned above dock (dock height is --dock-h)
        isMobile &&
          "fixed left-0 right-0 z-50 bottom-[calc(var(--dock-h,56px)+env(safe-area-inset-bottom))]",
        "border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80"
      )}
    >
      <div className="max-w-3xl mx-auto px-3 sm:px-4 py-2 sm:py-3">
        {selectedFiles.length > 0 && (
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            {selectedFiles.map((file, index) => (
              <span key={`${file.name}-${file.size}-${index}`} className="inline-flex max-w-[220px] items-center gap-1 rounded-md border border-border bg-secondary/60 px-2 py-1 text-xs">
                <span className="truncate">{file.name}</span>
                <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setSelectedFiles((files) => files.filter((_, i) => i !== index))}>
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
            <Button type="button" size="sm" variant={ingestFiles ? "default" : "outline"} className="h-7 gap-1 text-xs" onClick={() => setIngestFiles((value) => !value)}>
              <Database className="h-3 w-3" /> {ingestFiles ? "Ingest locally" : "Upload only"}
            </Button>
          </div>
        )}
        {showMicOptions && (
          <div className="mb-2 grid gap-2 rounded-lg border border-border bg-card/90 p-2 text-xs sm:grid-cols-[1fr_140px_auto]">
            <label className="grid gap-1">
              <span className="text-muted-foreground">Microphone test device</span>
              <select className="h-8 rounded-md border border-border bg-background px-2" value={selectedMicId} onChange={(event) => { setSelectedMicId(event.target.value); localStorage.setItem("sarah:microphone-device", event.target.value); }}>
                <option value="default">System default</option>
                {micDevices.filter((device) => device.deviceId !== "default").map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Microphone ${index + 1}`}</option>)}
              </select>
            </label>
            <label className="grid gap-1">
              <span className="text-muted-foreground">Recognition language</span>
              <select className="h-8 rounded-md border border-border bg-background px-2" value={speechLanguage} onChange={(event) => { setSpeechLanguage(event.target.value); localStorage.setItem("sarah:speech-language", event.target.value); }}>
                <option value="en-US">English (US)</option><option value="en-GB">English (UK)</option><option value="es-US">Spanish</option><option value="fr-FR">French</option><option value="de-DE">German</option>
              </select>
            </label>
            <Button type="button" size="sm" variant="outline" className="self-end" onClick={() => void refreshMicrophones(true)}>
              Test / allow {micPermission === "granted" ? "✓" : ""}
            </Button>
            <p className="sm:col-span-3 text-[11px] text-muted-foreground">Browser speech recognition uses the browser/OS default input. Device selection above verifies permission and hardware availability.</p>
          </div>
        )}
        <div className="flex items-center gap-2">
          {/* Mic */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleListening}
            className={cn("shrink-0", isListening ? "text-primary" : "text-muted-foreground hover:text-foreground")}
            title={isListening ? "Stop listening" : "Start listening"}
            disabled={isSending}
          >
            <Mic className={cn("h-5 w-5", isListening && "animate-pulse")} />
          </Button>
          <Button type="button" variant="ghost" size="icon" onClick={() => { setShowMicOptions((value) => !value); void refreshMicrophones(false); }} className="shrink-0 text-muted-foreground" title="Microphone options">
            <SlidersHorizontal className="h-4 w-4" />
          </Button>

          {/* Input */}
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            className="flex-1 bg-secondary/30 border-border focus-visible:ring-primary/50"
            disabled={isSending}
          />

          {/* Attach */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => handleFileSelect(e.target.files)}
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={handleAttachClick}
            className="shrink-0 text-muted-foreground hover:text-foreground"
            title="Attach files"
            disabled={isSending}
          >
            <Paperclip className="h-5 w-5" />
          </Button>

          {/* Send */}
          <Button
            onClick={handleSubmit}
            disabled={(!message.trim() && selectedFiles.length === 0) || isSending}
            size="icon"
            className="shrink-0 bg-primary text-primary-foreground hover:bg-primary/90 h-9 w-9 sm:h-10 sm:w-10"
            title="Send message"
          >
            {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  );
}

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Send, Mic, Paperclip, Loader2 } from "lucide-react";
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
  onSendText: (text: string) => Promise<void> | void;
  isSending?: boolean;
};

export function ChatComposer({ onSendText, isSending: isSendingProp }: Props) {
  const isMobile = useIsMobile();

  const [message, setMessage] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [localSending, setLocalSending] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const composerRef = useRef<HTMLDivElement>(null);

  const isSending = Boolean(isSendingProp) || localSending;

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
    recognition.lang = "en-US";

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
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      try {
        recognition.stop();
      } catch {}
      recognitionRef.current = null;
    };
  }, []);

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
        recognition.stop();
        setIsListening(false);
        await api.avatar.setListening(false);
      } else {
        recognition.start();
        setIsListening(true);
        await api.avatar.setListening(true);
      }
    } catch (e) {
      console.warn("[ChatComposer] toggleListening failed:", e);
      setIsListening(false);
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setSelectedFiles(Array.from(files));
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
      recognitionRef.current?.stop();
    } catch {}
    setIsListening(false);

    const payloadText = trimmed || "(file upload)";

    try {
      setMessage("");
      setSelectedFiles([]);
      await onSendText(payloadText);
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

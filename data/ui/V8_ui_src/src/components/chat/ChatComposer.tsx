import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Send, Mic, Paperclip, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

import { useChatSend } from "@/hooks/useChatSend";

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
  start(): void;
  stop(): void;
  abort(): void;
}

export function ChatComposer() {
  const [message, setMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  const { send, stopAvatarSpeaking } = useChatSend();

  useEffect(() => {
    return () => {
      if (recognitionRef.current) recognitionRef.current.abort();
    };
    // Cleanup only - no dependencies needed
  }, []);

  const handleSubmit = async (overrideText?: string) => {
    const textToSend = (overrideText ?? message).trim();
    if (!textToSend && selectedFiles.length === 0) return;
    if (isSending) return;

    setIsSending(true);
    try {
      // NOTE: files are not wired yet, but keep UX intact
      if (selectedFiles.length > 0) {
        toast.info(`${selectedFiles.length} file(s) attached (upload pipeline not wired yet).`);
      }

      await send(textToSend);
      setMessage("");
      setSelectedFiles([]);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(files);
    if (files.length > 0) toast.info(`${files.length} file(s) selected. They will be analyzed when you send.`);
  };

  const toggleRecording = async () => {
    if (isRecording) {
      if (recognitionRef.current) recognitionRef.current.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast.error("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
      return;
    }

    try {
      const recognition: SpeechRecognitionInstance = new SpeechRecognition();
      recognitionRef.current = recognition;

      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onstart = () => {
        setIsRecording(true);
        toast.info("Listening... Speak now");
      };

      recognition.onresult = async (event: any) => {
        const transcript = event?.results?.[0]?.[0]?.transcript ?? "";
        const cleaned = String(transcript).trim();
        setIsRecording(false);
        if (!cleaned) return;

        setMessage(cleaned);

        setTimeout(async () => {
          await handleSubmit(cleaned);
        }, 200);
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error:", event.error);
        setIsRecording(false);

        if (event.error === "not-allowed") toast.error("Microphone access denied. Please allow microphone access.");
        else if (event.error === "no-speech") toast.info("No speech detected. Try again.");
        else toast.error(`Speech recognition error: ${event.error}`);
      };

      recognition.onend = () => setIsRecording(false);

      recognition.start();
    } catch (error) {
      console.error("Speech recognition error:", error);
      toast.error("Could not start speech recognition");
      setIsRecording(false);
    }
  };

  return (
    <div className="shrink-0 border-t border-border bg-card/50 backdrop-blur-sm p-2 sm:p-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:pb-3">
      <div className="max-w-3xl mx-auto">
        {selectedFiles.length > 0 && (
          <div className="flex gap-2 mb-3 flex-wrap">
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="px-3 py-1.5 bg-secondary rounded-full text-xs text-secondary-foreground flex items-center gap-2"
              >
                <Paperclip className="h-3 w-3" />
                {file.name}
                <button
                  onClick={() => setSelectedFiles((files) => files.filter((_, i) => i !== idx))}
                  className="hover:text-destructive"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleRecording}
            disabled={isSending}
            className={cn(
              "shrink-0 h-9 w-9 sm:h-10 sm:w-10",
              isRecording
                ? "text-destructive animate-pulse ring-2 ring-destructive bg-destructive/10"
                : "text-muted-foreground hover:text-foreground"
            )}
            title={isRecording ? "Stop recording" : "Start voice input"}
          >
            <Mic className="h-5 w-5" />
          </Button>

          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            disabled={isSending || isRecording}
            className="flex-1 bg-secondary border-border text-foreground placeholder:text-muted-foreground h-9 sm:h-10"
          />

          <input ref={fileInputRef} type="file" multiple className="hidden" onChange={handleFileSelect} />

          <Button
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={isSending}
            className="shrink-0 text-muted-foreground hover:text-foreground h-9 w-9 sm:h-10 sm:w-10"
            title="Attach files"
          >
            <Paperclip className="h-5 w-5" />
          </Button>

          <Button
            onClick={() => handleSubmit()}
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

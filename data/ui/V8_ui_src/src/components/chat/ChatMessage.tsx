import { useMemo, useState } from "react";
import {
  User,
  Download,
  X,
  Copy,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  CornerDownRight,
} from "lucide-react";
import sarahIcon from "@/assets/sarah-icon.ico";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { Message } from "@/types/sarah";
import { toast } from "sonner";
import { useSarahStore } from "@/stores/useSarahStore";

type Props = {
  message: Message;
  onSendFollowUp?: (text: string) => void;
};

// ------------------------------
// Helpers
// ------------------------------
function parseFollowUpSuggestions(content: string): { text: string; suggestions: string[] } {
  // Only match bracket tokens that are standalone:
  // avoids breaking markdown links like [docs](url)
  const suggestionRegex = /(^|\s)\[([^\]\n]+)\](?=\s|$)/g;

  const suggestions: string[] = [];
  let match: RegExpExecArray | null;

  while ((match = suggestionRegex.exec(content)) !== null) {
    suggestions.push(match[2].trim());
  }

  const text = content.replace(suggestionRegex, "$1").trim();
  return { text, suggestions };
}

function parseMedia(content: string): { text: string; images: string[]; videos: string[] } {
  const images: string[] = [];
  const videos: string[] = [];

  // Markdown images: ![alt](url)
  const mdImageRegex = /!\[[^\]]*?\]\(([^)]+)\)/gi;
  let m: RegExpExecArray | null;

  while ((m = mdImageRegex.exec(content)) !== null) {
    const url = (m[1] || "").trim();
    if (url && !images.includes(url)) images.push(url);
  }

  // Raw image URLs
  const rawImageRegex = /https?:\/\/[^\s)]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s)]*)?/gi;
  let im: RegExpExecArray | null;

  while ((im = rawImageRegex.exec(content)) !== null) {
    const url = (im[0] || "").trim();
    if (url && !images.includes(url)) images.push(url);
  }

  // Raw video URLs
  const rawVideoRegex = /https?:\/\/[^\s)]+\.(?:mp4|webm|mov|avi)(?:\?[^\s)]*)?/gi;
  let vm: RegExpExecArray | null;

  while ((vm = rawVideoRegex.exec(content)) !== null) {
    const url = (vm[0] || "").trim();
    if (url && !videos.includes(url)) videos.push(url);
  }

  // Clean text
  let text = content;

  // Remove markdown image tokens
  text = text.replace(mdImageRegex, "").trim();

  // Remove extracted raw URLs
  [...images, ...videos].forEach((u) => {
    const escaped = u.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    text = text.replace(new RegExp(escaped, "g"), "").trim();
  });

  // Normalize whitespace
  text = text.replace(/\n{3,}/g, "\n\n").replace(/[ \t]{2,}/g, " ").trim();

  return { text, images, videos };
}

function buildFollowUpPrompt(
  action: string,
  priorUserPrompt: string | null,
  priorAssistantAnswer: string | null
) {
  const a = (action || "").trim();

  const question = (priorUserPrompt || "").trim();
  const answer = (priorAssistantAnswer || "").trim();

  // If we have both, we can generate a “context-aware” follow-up that *actually* re-queries the prior exchange.
  if (question || answer) {
    switch (a.toLowerCase()) {
      case "explain simpler":
        return `Explain your previous answer in simpler terms for my question.\n\nQuestion: ${question || "(unknown)"}\n\nYour answer: ${answer || "(unknown)"}`;
      case "give steps":
        return `Rewrite your previous answer as clear step-by-step instructions for my question.\n\nQuestion: ${question || "(unknown)"}\n\nYour answer: ${answer || "(unknown)"}`;
      case "show example":
        return `Provide a concrete example based on your previous answer to my question.\n\nQuestion: ${question || "(unknown)"}\n\nYour answer: ${answer || "(unknown)"}`;
      case "summarize":
        return `Summarize your previous answer in 3-6 bullet points.\n\nQuestion: ${question || "(unknown)"}\n\nYour answer: ${answer || "(unknown)"}`;
      default:
        return `${a}\n\nQuestion: ${question || "(unknown)"}\n\nYour answer: ${answer || "(unknown)"}`;
    }
  }

  // Fallback if we cannot locate context
  return a;
}

// ------------------------------
// Component
// ------------------------------
export function ChatMessage({ message, onSendFollowUp }: Props) {
  const isUser = message.role === "user";
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // We need store messages so Regenerate + followups can locate the prompt/answer context
  const { messages } = useSarahStore();

  const formatTime = (ts: any) => {
    const date = ts instanceof Date ? ts : new Date(ts ?? Date.now());
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const { finalText, bracketSuggestions, images, videos } = useMemo(() => {
    const raw = message.content || "";
    const { text: withoutSuggestions, suggestions } = parseFollowUpSuggestions(raw);
    const { text: finalText, images, videos } = parseMedia(withoutSuggestions);
    return { finalText, bracketSuggestions: suggestions, images, videos };
  }, [message.content]);

  // Requested static follow ups (always available for assistant messages)
  const followUps = !isUser ? ["Explain simpler", "Give steps", "Show example", "Summarize"] : [];

  // Prefer bracket suggestions if present, otherwise fall back to static followUps
  const suggestionsToShow = !isUser && bracketSuggestions.length > 0 ? bracketSuggestions : followUps;

  const locateContext = () => {
    const idx = messages.findIndex((m: any) => m?.id === (message as any)?.id);

    // Find prior user prompt
    let priorUser: string | null = null;
    if (idx >= 0) {
      for (let i = idx - 1; i >= 0; i--) {
        const m: any = messages[i];
        if (m?.role === "user" && typeof m?.content === "string" && m.content.trim()) {
          priorUser = m.content.trim();
          break;
        }
      }
    }

    // Prior assistant answer = this message content (assistant)
    const priorAssistant = !isUser ? (message.content || "").trim() : null;

    return { priorUser, priorAssistant, idx };
  };

  const handleFollowUp = (action: string) => {
    if (isUser) return;
    if (!onSendFollowUp) return;

    const { priorUser, priorAssistant } = locateContext();
    const prompt = buildFollowUpPrompt(action, priorUser, priorAssistant);

    if (!prompt.trim()) return;
    onSendFollowUp(prompt);
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content || "");
      toast.success("Copied");
    } catch {
      toast.error("Copy failed");
    }
  };

  // Regenerate = re-send the user prompt immediately before this assistant message
  const handleRegenerate = () => {
    if (isUser) return;
    if (!onSendFollowUp) {
      toast.error("Regenerate not available");
      return;
    }

    const { priorUser } = locateContext();

    if (priorUser && priorUser.trim()) {
      toast.message("Regenerating…");
      onSendFollowUp(priorUser.trim());
      return;
    }

    toast.error("No prior user prompt found");
  };

  const handleFeedback = (kind: "like" | "dislike") => {
    // Kept for learning wiring
    if (kind === "like") toast.message("Liked");
    else toast.message("Disliked");
  };

  const handleDownload = async (url: string, filename?: string) => {
    try {
      const response = await fetch(url, { mode: "cors" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = filename || "download";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error("Download failed:", error);
      toast.error("Download failed");
    }
  };

  return (
    <div className={cn("flex gap-3 animate-fade-in", isUser ? "justify-end" : "justify-start")}>
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 mt-1 ring-2 ring-primary/40 shadow-[0_0_10px_rgba(var(--primary-rgb),0.3)]">
          <img src={sarahIcon} alt="Sarah AI" className="w-full h-full object-cover" />
        </div>
      )}

      <div className={cn("max-w-[85%] sm:max-w-[75%] space-y-2", isUser && "items-end")}>
        {/* Message Bubble */}
        <div className={cn("px-4 py-2.5 whitespace-pre-wrap", isUser ? "bubble-user" : "bubble-assistant")}>
          <p className="text-sm leading-relaxed break-words">{finalText || message.content}</p>

          {/* Assistant Toolbar + Follow-ups */}
          {!isUser && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
                onClick={copyToClipboard}
                title="Copy"
              >
                <Copy className="h-4 w-4" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
                onClick={handleRegenerate}
                title="Regenerate"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
                onClick={() => handleFeedback("like")}
                title="Like"
              >
                <ThumbsUp className="h-4 w-4" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-white/60 hover:text-white hover:bg-white/10"
                onClick={() => handleFeedback("dislike")}
                title="Dislike"
              >
                <ThumbsDown className="h-4 w-4" />
              </Button>

              {/* Follow-ups */}
              {suggestionsToShow.length > 0 && (
                <div className="flex flex-wrap gap-2 ml-auto">
                  {suggestionsToShow.slice(0, 6).map((t) => (
                    <Button
                      key={t}
                      variant="secondary"
                      size="sm"
                      className="h-8 bg-white/5 border border-white/10 text-white/80 hover:bg-white/10"
                      onClick={() => handleFollowUp(t)}
                      title={`Follow-up: ${t}`}
                    >
                      <CornerDownRight className="h-3.5 w-3.5 mr-2 opacity-70" />
                      {t}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Images */}
        {images.length > 0 && (
          <div
            className={cn(
              "grid gap-2",
              images.length === 1
                ? "grid-cols-1"
                : images.length === 2
                  ? "grid-cols-2"
                  : images.length <= 4
                    ? "grid-cols-2"
                    : "grid-cols-3"
            )}
          >
            {images.slice(0, 4).map((img, idx) => (
              <div
                key={`${img}-${idx}`}
                className="relative group rounded-lg overflow-hidden cursor-pointer bg-secondary"
                onClick={() => setSelectedImage(img)}
              >
                <img
                  src={img}
                  alt={`Generated image ${idx + 1}`}
                  className="w-full h-auto max-h-48 object-cover"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 text-white hover:bg-white/20"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(img, `image-${idx + 1}.png`);
                    }}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Videos */}
        {videos.length > 0 && (
          <div className="space-y-2">
            {videos.map((vid, idx) => (
              <div key={`${vid}-${idx}`} className="relative rounded-lg overflow-hidden bg-secondary">
                <video src={vid} className="w-full max-h-64" controls playsInline />
                <div className="absolute top-2 right-2 flex gap-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 bg-background/80 hover:bg-background"
                    onClick={() => handleDownload(vid, `video-${idx + 1}.mp4`)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <div className={cn("text-xs text-muted-foreground px-1", isUser ? "text-right" : "text-left")}>
          {formatTime((message as any).timestamp)}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-1">
          <User className="w-4 h-4 text-secondary-foreground" />
        </div>
      )}

      {/* Image Lightbox */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={selectedImage}
              alt="Full size"
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
            />
            <div className="absolute top-2 right-2 flex gap-2">
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 bg-background/80 hover:bg-background text-foreground"
                onClick={() => handleDownload(selectedImage, "image.png")}
              >
                <Download className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 bg-background/80 hover:bg-background text-foreground"
                onClick={() => setSelectedImage(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

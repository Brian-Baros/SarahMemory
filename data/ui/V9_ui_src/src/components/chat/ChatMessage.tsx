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
import { api, type ChatResponse, type DynamicChip } from "@/lib/api";

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

  if (isControlledRepairFollowUp(a)) {
    return normalizeControlledRepairFollowUp(a);
  }

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


const CONTROLLED_REPAIR_FOLLOWUPS = new Set([
  "accept fix",
  "accept",
  "approve fix",
  "reject fix",
  "reject",
  "yes",
  "yes satisfied",
  "no",
  "no rollback",
  "no - rollback",
  "rollback",
]);

function normalizeControlledRepairFollowUp(action: string): string {
  const normalized = (action || "")
    .trim()
    .replace(/^\[|\]$/g, "")
    .replace(/\s+/g, " ")
    .toLowerCase();

  switch (normalized) {
    case "accept":
    case "approve fix":
    case "accept fix":
      return "ACCEPT FIX";
    case "reject":
    case "reject fix":
      return "REJECT FIX";
    case "yes":
    case "yes satisfied":
      return "YES";
    case "no":
    case "no rollback":
    case "no - rollback":
    case "rollback":
      return "NO - ROLLBACK";
    default:
      return action.trim();
  }
}

function isControlledRepairFollowUp(action: string): boolean {
  const normalized = (action || "")
    .trim()
    .replace(/^\[|\]$/g, "")
    .replace(/\s+/g, " ")
    .toLowerCase();

  return CONTROLLED_REPAIR_FOLLOWUPS.has(normalized);
}

function followUpButtonClassName(action: string): string {
  const normalized = normalizeControlledRepairFollowUp(action).toLowerCase();

  if (normalized === "accept fix" || normalized === "yes") {
    return "h-8 bg-emerald-500/20 border border-emerald-400/40 text-emerald-100 hover:bg-emerald-500/30";
  }

  if (normalized === "reject fix" || normalized === "no - rollback") {
    return "h-8 bg-red-500/20 border border-red-400/40 text-red-100 hover:bg-red-500/30";
  }

  return "h-8 bg-white/5 border border-white/10 text-white/80 hover:bg-white/10";
}


// ------------------------------
// Component
// ------------------------------
export function ChatMessage({ message, onSendFollowUp }: Props) {
  const isUser = message.role === "user";
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // We need store messages so Regenerate + followups can locate the prompt/answer context
  const { messages, addMessage, enqueueUiActions } = useSarahStore();

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

  const operationalChips: DynamicChip[] = !isUser && Array.isArray((message as any).chips)
    ? ((message as any).chips as DynamicChip[]).filter((chip) => chip && chip.label && chip.action)
    : [];
  const responseType = String((message as any).response_type || (message as any).meta?.response_type || "");
  const task = (message as any).task && typeof (message as any).task === "object" ? (message as any).task : null;
  const pendingActionId = String((message as any).pending_action_id || (message as any).meta?.pending_action_id || "");

  // Requested static follow ups remain available for normal informational answers.
  const followUps = !isUser && operationalChips.length === 0 ? ["Explain simpler", "Give steps", "Show example", "Summarize"] : [];

  // Prefer bracket suggestions if present, otherwise fall back to static followUps
  const suggestionsToShow = !isUser && bracketSuggestions.length > 0 ? bracketSuggestions : followUps;

  const dispatchReturnedActions = (actions: any[]) => {
    if (!Array.isArray(actions) || actions.length === 0) return;
    try {
      enqueueUiActions(actions, "chat_action_chip");
    } catch (error) {
      console.warn("[ChatMessage] Failed to enqueue returned actions:", error);
    }
  };

  const appendCommandResponse = (response: ChatResponse) => {
    const content = response.content || response.reply || response.reason || "Command completed.";
    addMessage({
      role: "assistant",
      content,
      response_type: response.response_type,
      chips: response.chips,
      actions: response.actions,
      pending_action_id: response.pending_action_id,
      mission_id: response.mission_id,
      task_id: response.task_id,
      task: response.task,
      tasks: response.tasks,
      pending_actions: response.pending_actions,
      capability: response.capability,
      images: response.images,
      meta: response.meta,
    });
    dispatchReturnedActions(response.actions || []);
  };

  const handleOperationalChip = async (chip: DynamicChip) => {
    if (isUser) return;
    const label = String(chip.label || "Action");
    const action = String(chip.action || "");
    const chipPendingId = String(chip.pending_action_id || pendingActionId || "");

    try {
      if (action === "send_prompt" && chip.prompt && onSendFollowUp) {
        onSendFollowUp(String(chip.prompt));
        return;
      }
      if (action === "open_panel") {
        const actions = Array.isArray(chip.actions) ? chip.actions : Array.isArray((chip as any).payload?.actions) ? (chip as any).payload.actions : [];
        dispatchReturnedActions(actions);
        toast.success(label);
        return;
      }
      if (action === "approve_pending_action") {
        addMessage({ role: "user", content: label });
        appendCommandResponse(await api.actions.approve(chipPendingId));
        return;
      }
      if (action === "deny_pending_action") {
        addMessage({ role: "user", content: label });
        appendCommandResponse(await api.actions.deny(chipPendingId));
        return;
      }
      if (action === "show_pending_actions") {
        appendCommandResponse(await api.actions.resolve("show_pending_actions", { pending_action_id: chipPendingId }));
        return;
      }
      if (action === "show_task" && chip.task_id) {
        const result = await api.tasks.get(String(chip.task_id));
        const t = result?.task || {};
        const content = result?.ok
          ? `Task ${t.task_id || chip.task_id}\nStatus: ${t.status || "unknown"}\nCapability: ${t.capability || "unknown"}\nStep: ${t.current_step || "unknown"}`
          : `Task lookup failed: ${result?.error || "task_not_found"}`;
        addMessage({
          role: "assistant",
          content,
          response_type: result?.ok ? "task_registry_card" : "command_error_card",
          task: t,
          task_id: t.task_id || chip.task_id,
        });
        return;
      }
      if (onSendFollowUp) {
        onSendFollowUp(label);
      }
    } catch (error: any) {
      const msg = String(error?.message || error || "Action failed");
      toast.error(msg);
      addMessage({ role: "assistant", content: `Action failed: ${msg}`, response_type: "command_error_card" });
    }
  };

  const cardTitle = (() => {
    switch (responseType) {
      case "permission_card": return "Approval Required";
      case "creative_result_card": return "Creative Studios Result";
      case "mission_card": return "Mission";
      case "task_registry_card": return "Task Registry";
      case "pending_action_banner": return "Pending Approvals";
      case "clarification_card": return "Clarification Needed";
      case "command_error_card": return "Command Issue";
      case "command_result_card": return "Command Result";
      default: return "";
    }
  })();

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

          {!isUser && (cardTitle || task || operationalChips.length > 0) && (
            <div className="mt-3 rounded-md border border-white/10 bg-black/20 p-3 text-xs text-white/80">
              {cardTitle && (
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-white">{cardTitle}</span>
                  {pendingActionId && (
                    <span className="rounded bg-amber-400/10 px-2 py-0.5 text-amber-100">
                      {pendingActionId}
                    </span>
                  )}
                </div>
              )}
              {task && (
                <div className="mb-2 grid gap-1 sm:grid-cols-2">
                  <div>Status: <span className="text-white">{String(task.status || "unknown")}</span></div>
                  <div>Capability: <span className="text-white">{String(task.capability || "unknown")}</span></div>
                  <div>Target: <span className="text-white">{String(task.target || "unknown")}</span></div>
                  <div>Step: <span className="text-white">{String(task.current_step || "unknown")}</span></div>
                </div>
              )}
              {operationalChips.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {operationalChips.slice(0, 8).map((chip, idx) => (
                    <Button
                      key={`${chip.action}-${chip.label}-${idx}`}
                      variant="secondary"
                      size="sm"
                      className={cn(
                        "h-8 border text-xs",
                        chip.action === "approve_pending_action"
                          ? "border-emerald-400/40 bg-emerald-500/20 text-emerald-100 hover:bg-emerald-500/30"
                          : chip.action === "deny_pending_action"
                            ? "border-red-400/40 bg-red-500/20 text-red-100 hover:bg-red-500/30"
                            : "border-white/10 bg-white/5 text-white/80 hover:bg-white/10"
                      )}
                      onClick={() => handleOperationalChip(chip)}
                      title={String(chip.action || chip.label)}
                    >
                      <CornerDownRight className="mr-2 h-3.5 w-3.5 opacity-70" />
                      {chip.label}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}

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
                  {suggestionsToShow.slice(0, 6).map((t) => {
                    const label = isControlledRepairFollowUp(t) ? normalizeControlledRepairFollowUp(t) : t;

                    return (
                      <Button
                        key={t}
                        variant="secondary"
                        size="sm"
                        className={followUpButtonClassName(t)}
                        onClick={() => handleFollowUp(label)}
                        title={`Follow-up: ${label}`}
                      >
                        <CornerDownRight className="h-3.5 w-3.5 mr-2 opacity-70" />
                        {label}
                      </Button>
                    );
                  })}
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

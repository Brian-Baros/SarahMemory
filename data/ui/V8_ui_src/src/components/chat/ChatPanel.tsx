import { useRef, useEffect, useLayoutEffect, useCallback } from "react";
import { useSarahStore } from "@/stores/useSarahStore";
import { ChatMessage } from "./ChatMessage";
import { ChatComposer } from "./ChatComposer";
import { TypingIndicator } from "./TypingIndicator";
import { api, type ChatResponse } from "@/lib/api";
import { toast } from "sonner";
import { useIsMobile } from "@/hooks/use-mobile";

export function ChatPanel() {
  const {
    messages,
    isTyping,
    addMessage,
    setTyping,
    mediaState,
    settings,
    setSpeechCues,
    setAvatarSpeaking,
    setSpeechStartTime,
  } = useSarahStore();

  const isMobile = useIsMobile();

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track whether user is near bottom, so we don't fight manual scrolling
  const shouldAutoScrollRef = useRef(true);

  const computeNearBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    const threshold = 160; // px
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    return distance < threshold;
  }, []);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "auto") => {
    // Prefer sentinel for better reliability with dynamic heights (images, fonts, etc.)
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior, block: "end" });
      return;
    }
    const el = scrollContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  // Update auto-scroll intent on user scroll
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;

    const onScroll = () => {
      shouldAutoScrollRef.current = computeNearBottom();
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll as any);
  }, [computeNearBottom]);

  // Auto-scroll when messages / typing changes IF user is near bottom
  useLayoutEffect(() => {
    if (!shouldAutoScrollRef.current) return;

    // Two RAFs helps when layout changes (especially mobile + fixed composer + new text)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scrollToBottom("auto");
      });
    });
  }, [messages, isTyping, scrollToBottom]);

  // Cleanup speaking timeout on unmount
  useEffect(() => {
    return () => {
      if (speakingTimeoutRef.current) clearTimeout(speakingTimeoutRef.current);
    };
  }, []);

  const estimateSpeakingDuration = (text: string): number => {
    const wordCount = text.split(/\s+/).filter(Boolean).length;

    // Keep the 2D avatar mouth frames active for the full spoken response.
    // The previous 12s ceiling made the avatar stop while slower backend/OS TTS
    // was still talking. This is a fallback watchdog only; real audio/TTS
    // onended/onerror handlers still stop the animation precisely when available.
    const wordsPerSecond = 1.45;
    const estimatedMs = (wordCount / wordsPerSecond) * 1000;
    return Math.max(1600, Math.min(180000, estimatedMs + 4000));
  };

  const stopAvatarSpeaking = () => {
    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }
    setAvatarSpeaking(false);
    setSpeechStartTime(null);
    setSpeechCues([]);
    api.avatar.setSpeaking(false).catch(() => {});
  };

  const startAvatarSpeaking = (response: ChatResponse) => {
    if (speakingTimeoutRef.current) clearTimeout(speakingTimeoutRef.current);

    const avatarSpeech = (response as any)?.meta?.avatar_speech;

    if (avatarSpeech?.cues && Array.isArray(avatarSpeech.cues) && avatarSpeech.cues.length > 0) {
      setSpeechCues(avatarSpeech.cues);
    } else {
      setSpeechCues([]);
    }

    const backendDurationMs = Number(avatarSpeech?.duration_ms || 0);
    const durationMs = Math.max(backendDurationMs, estimateSpeakingDuration(response.content));

    setAvatarSpeaking(true);
    setSpeechStartTime(Date.now());
    api.avatar.setSpeaking(true).catch(() => {});

    // Fallback watchdog only. Browser/audio TTS end events below are authoritative.
    speakingTimeoutRef.current = setTimeout(() => {
      stopAvatarSpeaking();
    }, durationMs);
  };

  const useBrowserTTS = (text: string) => {
    if (!("speechSynthesis" in window)) return;

    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }
    setAvatarSpeaking(true);
    setSpeechStartTime(Date.now());
    api.avatar.setSpeaking(true).catch(() => {});

    const utterance = new SpeechSynthesisUtterance(text);

    const setVoiceAndSpeak = (voices: SpeechSynthesisVoice[]) => {
      const femaleKeywords = [
        "female",
        "samantha",
        "victoria",
        "karen",
        "zira",
        "susan",
        "hazel",
        "fiona",
        "moira",
        "tessa",
        "kate",
      ];

      const englishVoices = voices.filter((v) => v.lang?.startsWith("en"));

      let selectedVoice =
        englishVoices.find((v) => femaleKeywords.some((k) => v.name.toLowerCase().includes(k))) || null;

      if (!selectedVoice) {
        const maleKeywords = ["male", "david", "daniel", "james", "alex", "tom", "mark", "fred", "ralph"];
        selectedVoice =
          englishVoices.find((v) => !maleKeywords.some((k) => v.name.toLowerCase().includes(k))) || null;
      }

      if (!selectedVoice && englishVoices.length > 0) selectedVoice = englishVoices[0];

      if (selectedVoice) utterance.voice = selectedVoice;

      utterance.pitch = 1.1;
      utterance.rate = 0.95;

      utterance.onend = () => stopAvatarSpeaking();
      utterance.onerror = () => stopAvatarSpeaking();
      speechSynthesis.speak(utterance);

      // Emergency watchdog only; normal completion is utterance.onend.
      speakingTimeoutRef.current = setTimeout(() => {
        stopAvatarSpeaking();
      }, estimateSpeakingDuration(text) + 15000);
    };

    let voices = speechSynthesis.getVoices();
    if (!voices || voices.length === 0) {
      speechSynthesis.onvoiceschanged = () => {
        voices = speechSynthesis.getVoices();
        setVoiceAndSpeak(voices);
      };
    } else {
      setVoiceAndSpeak(voices);
    }
  };

  const speakResponse = async (text: string) => {
    try {
      const resp = await api.voice.speak(text, settings.selectedVoice);

      if (resp?.success && (resp.audio_url || resp.audio_base64)) {
        const audioSrc =
          resp.audio_url || (resp.audio_base64 ? `data:audio/mp3;base64,${resp.audio_base64}` : null);

        if (audioSrc) {
          const audio = new Audio(audioSrc);
          if (speakingTimeoutRef.current) {
            clearTimeout(speakingTimeoutRef.current);
            speakingTimeoutRef.current = null;
          }
          setAvatarSpeaking(true);
          setSpeechStartTime(Date.now());
          api.avatar.setSpeaking(true).catch(() => {});
          audio.onended = () => stopAvatarSpeaking();
          audio.onerror = () => stopAvatarSpeaking();
          speakingTimeoutRef.current = setTimeout(() => {
            stopAvatarSpeaking();
          }, estimateSpeakingDuration(text) + 15000);
          await audio.play();
          return;
        }
      }

      if (resp?.fallback || !resp?.success) {
        useBrowserTTS(text);
      }
    } catch (e) {
      console.warn("[ChatPanel] TTS failed, using browser fallback:", e);
      useBrowserTTS(text);
    }
  };

  // Unified send (used by composer + follow-ups + regenerate)
  const sendText = async (text: string, files?: File[]) => {
    const clean = (text || "").trim();
    if (!clean) return;
    if (isTyping) return;

    // If user is near bottom, keep it pinned there as new content comes in
    shouldAutoScrollRef.current = true;

    addMessage({ role: "user", content: clean, attachments: (files||[]).map((f:any, i:number)=>({ id: `${Date.now()}_${i}`, name: f.name, type: f.type || "application/octet-stream", size: f.size })) });
    setTyping(true);

// -------------------------------------------------------------------
// Local-only manual ingestion: "EAT THIS" + attachments
// - Will NOT send content to remote APIs.
// - Uses appsys.py endpoint: POST /api/ingest/eat_this
// -------------------------------------------------------------------
const cmd = (clean || "").trim().toLowerCase();
const hasFiles = Array.isArray(files) && files.length > 0;

if (hasFiles && cmd === "eat this") {
  try {
    const form = new FormData();
    form.append("text", "EAT THIS");
    for (const f of files || []) form.append("files", f);

    const resp = await fetch("/api/ingest/eat_this", {
      method: "POST",
      body: form,
    });

    const data = await resp.json().catch(() => ({} as any));
    setTyping(false);

    if (!resp.ok || !data?.ok) {
      const msg = data?.error || `Ingest failed (HTTP ${resp.status})`;
      toast.error(msg);
      addMessage({ role: "assistant", content: `❌ ${msg}` });
      return;
    }

    const learned = Number(data?.learned || 0);
    const count = Number(data?.files?.length || (files || []).length);
    addMessage({
      role: "assistant",
      content: `✅ Ingest complete. Learned ${learned} item(s) from ${count} file(s).`,
    });

    // optional UI toast action via bus
    try {
      window.dispatchEvent(
        new CustomEvent("sarah:ui", {
          detail: {
            source: "chat",
            ts: Date.now(),
            actions: [{ type: "toast", payload: { kind: "success", message: `Ingested ${count} file(s)` } }],
          },
        })
      );
    } catch {}

    return;
  } catch (e: any) {
    setTyping(false);
    const msg = e?.message ? String(e.message) : "Ingest failed";
    toast.error(msg);
    addMessage({ role: "assistant", content: `❌ ${msg}` });
    return;
  }
}



    try {
      await api.avatar.setListening(true);
    } catch {}

    try {
      const messageHistory = messages.map((m) => ({ role: m.role, content: m.content }));
      messageHistory.push({ role: "user" as const, content: clean });

      const response = await api.chat.sendMessage(messageHistory);

      setTyping(false);

      if (response?.error) {
        toast.error(response.error);
        addMessage({ role: "assistant", content: "I'm sorry, I encountered an error. Please try again." });
        return;
      }

      addMessage({ role: "assistant", content: response.content });

      startAvatarSpeaking(response);

      // IMPORTANT: this is now hit for *normal sends* and *follow-ups/regenerate*
      if (mediaState.voiceEnabled && settings.autoSpeak) {
        await speakResponse(response.content);
      }
    } catch (error) {
      console.error("Chat send error:", error);
      setTyping(false);
      toast.error("Failed to send message");
      addMessage({
        role: "assistant",
        content: "I'm having trouble connecting right now. Please try again in a moment.",
      });
    } finally {
      try {
        await api.avatar.setListening(false);
      } catch {}
    }
  };

  // Dynamic padding so bottom never hides behind composer/dock
  const messagesPaddingBottom = isMobile
    ? "calc(var(--composer-h,72px) + var(--dock-h,56px) + env(safe-area-inset-bottom) + 12px)"
    : "calc(var(--composer-h,72px) + 12px)";

  return (
    <div className="h-full w-full flex flex-col min-h-0 overflow-hidden">
      {/* Messages Area - ONLY scrollable element */}
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 overflow-y-auto px-3 sm:px-4 py-2 sm:py-4"
        style={{ paddingBottom: messagesPaddingBottom }}
      >
        <div className="max-w-3xl mx-auto space-y-3 sm:space-y-4">
          {messages.map((message) => (
            <ChatMessage key={(message as any).id} message={message} onSendFollowUp={sendText} />
          ))}
          {isTyping && <TypingIndicator />}

          {/* Sentinel for reliable scroll-to-bottom */}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Composer */}
      <ChatComposer onSendText={sendText} isSending={isTyping} />
    </div>
  );
}

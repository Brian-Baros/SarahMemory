import { useRef, useEffect } from "react";
import { useSarahStore } from "@/stores/useSarahStore";
import { ChatMessage } from "./ChatMessage";
import { ChatComposer } from "./ChatComposer";
import { TypingIndicator } from "./TypingIndicator";
import { api, type ChatResponse } from "@/lib/api";
import { toast } from "sonner";

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

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Cleanup speaking timeout on unmount
  useEffect(() => {
    return () => {
      if (speakingTimeoutRef.current) clearTimeout(speakingTimeoutRef.current);
    };
  }, []);

  /**
   * Estimate speaking duration from text using word count
   * Assumes ~2.2 words per second, clamped between 800ms and 12000ms
   */
  const estimateSpeakingDuration = (text: string): number => {
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const wordsPerSecond = 2.2;
    const estimatedMs = (wordCount / wordsPerSecond) * 1000;
    return Math.max(800, Math.min(12000, estimatedMs));
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

    const durationMs = avatarSpeech?.duration_ms || estimateSpeakingDuration(response.content);

    setAvatarSpeaking(true);
    setSpeechStartTime(Date.now());

    api.avatar.setSpeaking(true).catch(() => {});

    speakingTimeoutRef.current = setTimeout(() => {
      stopAvatarSpeaking();
    }, durationMs + 500);
  };

  const useBrowserTTS = (text: string) => {
    if (!("speechSynthesis" in window)) return;

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
      speechSynthesis.speak(utterance);
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
          audio.onended = () => stopAvatarSpeaking();
          audio.onerror = () => stopAvatarSpeaking();
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

  // Unified send (used by follow-up buttons)
  const sendText = async (text: string) => {
    const clean = (text || "").trim();
    if (!clean) return;

    addMessage({ role: "user", content: clean });
    setTyping(true);

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

      // Make follow-ups behave like normal replies: avatar speaks + optional TTS
      startAvatarSpeaking(response);

      if (mediaState.voiceEnabled && settings.autoSpeak) {
        await speakResponse(response.content);
      }
    } catch (error) {
      console.error("Follow-up error:", error);
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

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Messages Area - ONLY scrollable element */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-3 sm:px-4 py-2 sm:py-4">
        <div className="max-w-3xl mx-auto space-y-3 sm:space-y-4">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} onSendFollowUp={sendText} />
          ))}
          {isTyping && <TypingIndicator />}
        </div>
      </div>

      {/* Composer - Fixed at bottom, never scrolls */}
      <ChatComposer />
    </div>
  );
}

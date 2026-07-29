import { useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";

import { api, type ChatResponse } from "@/lib/api";
import { useSarahStore } from "@/stores/useSarahStore";

type SendOptions = {
  /** When true: try TTS audio (backend), fallback to browser TTS */
  speakIfEnabled?: boolean;
};

export function useChatSend() {
  const {
    messages,
    addMessage,
    setTyping,
    mediaState,
    settings,
    setSpeechCues,
    setAvatarSpeaking,
    setSpeechStartTime,
  } = useSarahStore();

  // Browser-safe timeout
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      if (speakingTimeoutRef.current) clearTimeout(speakingTimeoutRef.current);

      if (activeAudioRef.current) {
        try {
          activeAudioRef.current.pause();
          activeAudioRef.current.src = "";
        } catch {}
        activeAudioRef.current = null;
      }

      if ("speechSynthesis" in window) {
        try {
          window.speechSynthesis.cancel();
          window.speechSynthesis.onvoiceschanged = null;
        } catch {}
      }
    };
  }, []);

  const estimateSpeakingDuration = useCallback((text: string): number => {
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const wordsPerSecond = 2.2;
    const estimatedMs = (wordCount / wordsPerSecond) * 1000;
    return Math.max(800, Math.min(12000, estimatedMs));
  }, []);

  const stopAvatarSpeaking = useCallback(() => {
    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }

    if (activeAudioRef.current) {
      try {
        activeAudioRef.current.pause();
        activeAudioRef.current.src = "";
      } catch {}
      activeAudioRef.current = null;
    }

    if ("speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
        window.speechSynthesis.onvoiceschanged = null;
      } catch {}
    }

    setAvatarSpeaking(false);
    setSpeechStartTime(null);
    setSpeechCues([]);
    api.avatar.setSpeaking(false).catch(() => {});
  }, [setAvatarSpeaking, setSpeechStartTime, setSpeechCues]);

  const startAvatarSpeaking = useCallback((response: ChatResponse) => {
    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }

    const avatarSpeech = response.meta?.avatar_speech;

    if (avatarSpeech?.cues && avatarSpeech.cues.length > 0) {
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
  }, [setSpeechCues, setAvatarSpeaking, setSpeechStartTime, stopAvatarSpeaking, estimateSpeakingDuration]);

  const useBrowserTTS = useCallback((text: string) => {
    if (!("speechSynthesis" in window)) return;

    try {
      window.speechSynthesis.cancel();
      window.speechSynthesis.onvoiceschanged = null;
    } catch {}

    const utterance = new SpeechSynthesisUtterance(text);

    const speakWithVoices = (voices: SpeechSynthesisVoice[]) => {
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
        englishVoices.find((v) => femaleKeywords.some((kw) => v.name.toLowerCase().includes(kw))) || null;

      if (!selectedVoice) {
        const maleKeywords = ["male", "david", "daniel", "james", "alex", "tom", "mark", "fred", "ralph"];
        selectedVoice =
          englishVoices.find((v) => !maleKeywords.some((kw) => v.name.toLowerCase().includes(kw))) || null;
      }

      if (!selectedVoice && englishVoices.length > 0) selectedVoice = englishVoices[0];
      if (selectedVoice) utterance.voice = selectedVoice;

      utterance.pitch = 1.1;
      utterance.rate = 0.95;

      utterance.onend = () => stopAvatarSpeaking();
      utterance.onerror = () => stopAvatarSpeaking();

      window.speechSynthesis.speak(utterance);
    };

    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) {
      window.speechSynthesis.onvoiceschanged = () => {
        speakWithVoices(window.speechSynthesis.getVoices());
      };
    } else {
      speakWithVoices(voices);
    }
  }, [stopAvatarSpeaking]);

  const speakResponse = useCallback(async (text: string) => {
    try {
      const resp = await api.voice.speak(text, settings.selectedVoice);

      if (resp.success && (resp.audio_url || resp.audio_base64)) {
        const audioSrc =
          resp.audio_url || (resp.audio_base64 ? `data:audio/mp3;base64,${resp.audio_base64}` : null);

        if (audioSrc) {
          // stop any previous audio
          if (activeAudioRef.current) {
            try {
              activeAudioRef.current.pause();
              activeAudioRef.current.src = "";
            } catch {}
            activeAudioRef.current = null;
          }

          const audio = new Audio(audioSrc);
          activeAudioRef.current = audio;

          audio.onended = () => stopAvatarSpeaking();
          audio.onerror = () => stopAvatarSpeaking();

          await audio.play();
          return;
        }
      }

      if (resp.fallback || !resp.success) {
        useBrowserTTS(text);
      }
    } catch (e) {
      console.error("TTS error:", e);
      useBrowserTTS(text);
    }
  }, [settings.selectedVoice, stopAvatarSpeaking, useBrowserTTS]);

  const send = useCallback(async (text: string, opts: SendOptions = {}) => {
    const clean = (text || "").trim();
    if (!clean) return;

    // prevent overlap/stuck speaking
    stopAvatarSpeaking();

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

      if (response.error) {
        toast.error(response.error);
        addMessage({ role: "assistant", content: "I'm sorry, I encountered an error. Please try again." });
        return;
      }

      addMessage({ role: "assistant", content: response.content });

      // avatar always animates speaking
      startAvatarSpeaking(response);

      const shouldSpeak = (opts.speakIfEnabled ?? true) && mediaState.voiceEnabled && settings.autoSpeak;
      if (shouldSpeak) {
        await speakResponse(response.content);
      }
    } catch (err) {
      console.error("Chat send error:", err);
      setTyping(false);
      toast.error("Failed to send message. Please try again.");
      addMessage({
        role: "assistant",
        content: "I'm having trouble connecting right now. Please try again in a moment.",
      });
    } finally {
      try {
        await api.avatar.setListening(false);
      } catch {}
    }
  }, [messages, addMessage, setTyping, mediaState.voiceEnabled, settings.autoSpeak, stopAvatarSpeaking, startAvatarSpeaking, speakResponse]);

  return {
    send,
    stopAvatarSpeaking,
  };
}

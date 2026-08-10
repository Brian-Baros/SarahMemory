import { useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";

import { api, speakWithSarahBrowserVoice, type ChatResponse } from "@/lib/api";
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

  const useBrowserTTS = useCallback(async (text: string) => {
    const ok = await speakWithSarahBrowserVoice(text, settings.selectedVoice || "sarahvoice", stopAvatarSpeaking);
    if (!ok) stopAvatarSpeaking();
  }, [settings.selectedVoice, stopAvatarSpeaking]);


  const speakResponse = useCallback(async (text: string) => {
    try {
      const resp = await api.voice.speak(text, settings.selectedVoice || "sarahvoice");

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

      if (resp.server_tts_started) {
        const durationMs = Number(resp.estimated_duration_ms || resp.avatar_session?.estimated_duration_ms || estimateSpeakingDuration(text));
        setAvatarSpeaking(true);
        setSpeechStartTime(Date.now());
        api.avatar.setSpeaking(true).catch(() => {});
        if (speakingTimeoutRef.current) clearTimeout(speakingTimeoutRef.current);
        speakingTimeoutRef.current = setTimeout(() => stopAvatarSpeaking(), Math.max(1200, durationMs + 1000));
        return;
      }

      if (resp.browser_fallback_required || resp.fallback || !resp.success) {
        await useBrowserTTS(text);
      }
    } catch (e) {
      console.error("TTS error:", e);
      await useBrowserTTS(text);
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

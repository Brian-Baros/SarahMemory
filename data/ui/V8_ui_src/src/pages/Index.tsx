import { useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { SettingsModal } from "@/components/panels/SettingsModal";
import { useSarahStore } from "@/stores/useSarahStore";
import { api } from "@/lib/api";

// Phase 1 Mobile Shell
import { MobileShell } from "@/components/shell/MobileShell";
import { BottomNav } from "@/components/shell/BottomNav";

// Desktop Shell
import { DesktopShell } from "@/components/shell/DesktopShell";

/**
 * Orientation switch:
 *  - Portrait  => Mobile shell UI
 *  - Landscape => Desktop shell UI
 */
function useIsPortrait() {
  const get = () =>
    typeof window !== "undefined"
      ? window.matchMedia("(orientation: portrait)").matches
      : true;

  const [isPortrait, setIsPortrait] = useState(get);

  useEffect(() => {
    const mq = window.matchMedia("(orientation: portrait)");
    const onChange = () => setIsPortrait(mq.matches);

    try {
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    } catch {
      mq.addListener(onChange);
      return () => mq.removeListener(onChange);
    }
  }, []);

  return isPortrait;
}

const Index = () => {
  const {
    mediaState,
    hasPlayedWelcome,
    backendReady,
    setHasPlayedWelcome,
    setVoices,
  } = useSarahStore();

  const isPortrait = useIsPortrait();

  // Fetch available voices after backend is ready
  useEffect(() => {
    if (!backendReady) return;

    (async () => {
      try {
        const voices = await api.voice.listVoices();
        if (voices?.length) {
          setVoices(voices);
          console.log("[Index] Loaded voices:", voices.length);
        }
      } catch (error) {
        console.warn("[Index] Failed to load voices:", error);
      }
    })();
  }, [backendReady, setVoices]);

  // One-time TTS welcome intro when backend is ready and voice is enabled
  useEffect(() => {
    if (!backendReady || !mediaState.voiceEnabled || hasPlayedWelcome) return;

    (async () => {
      try {
        await api.voice.speak("SarahMemory is online and ready.");
      } catch (error) {
        console.warn("[Index] Welcome TTS failed:", error);
      } finally {
        setHasPlayedWelcome(true);
      }
    })();
  }, [backendReady, mediaState.voiceEnabled, hasPlayedWelcome, setHasPlayedWelcome]);

  return (
    <div className="min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background">
      {isPortrait ? (
        <div className="flex-1 min-h-0 flex flex-col">
          <Header />
          <div className="flex-1 min-h-0">
            <MobileShell />
          </div>
          <BottomNav />
        </div>
      ) : (
        <DesktopShell />
      )}

      <SettingsModal />
    </div>
  );
};

export default Index;

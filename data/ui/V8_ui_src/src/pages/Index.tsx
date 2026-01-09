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

    // modern + fallback
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

/**
 * Main Index page
 *
 * Portrait:
 *  - Header + MobileShell + BottomNav
 *
 * Landscape:
 *  - DesktopShell + BottomNav (acts as Dock in shell mode)
 */
const Index = () => {
  const {
    mediaState,
    hasPlayedWelcome,
    backendReady,
    setHasPlayedWelcome,
    setBackendReady,
    setVoices,
    setBootstrapData,
  } = useSarahStore();

  const isPortrait = useIsPortrait();

  // Bootstrap on mount - calls POST /api/session/bootstrap once
  useEffect(() => {
    const runBootstrap = async () => {
      try {
        console.log("[Index] Running bootstrap...");
        const bootstrapData = await api.bootstrap.init();

        if (bootstrapData.ok) {
          console.log("[Index] Bootstrap successful:", bootstrapData.version);
          setBootstrapData(bootstrapData);
          setBackendReady(true);
        } else {
          console.warn("[Index] Bootstrap returned ok:false");
          setBackendReady(true); // Still allow app to function
        }
      } catch (error) {
        console.warn("[Index] Bootstrap failed:", error);
        setBackendReady(true); // Allow app to function
      }
    };

    runBootstrap();
  }, [setBackendReady, setBootstrapData]);

  // Fetch available voices after backend is ready
  useEffect(() => {
    const fetchVoices = async () => {
      if (!backendReady) return;

      try {
        const voices = await api.voice.listVoices();
        if (voices.length > 0) {
          setVoices(voices);
          console.log("[Index] Loaded voices:", voices.length);
        }
      } catch (error) {
        console.warn("[Index] Failed to load voices:", error);
      }
    };

    fetchVoices();
  }, [backendReady, setVoices]);

  // One-time TTS welcome intro when backend is ready and voice is enabled
  useEffect(() => {
    const playWelcome = async () => {
      if (backendReady && mediaState.voiceEnabled && !hasPlayedWelcome) {
        try {
          await api.voice.speak("SarahMemory is online and ready.");
          setHasPlayedWelcome(true);
        } catch (error) {
          console.warn("[Index] Welcome TTS failed:", error);
          setHasPlayedWelcome(true);
        }
      }
    };

    playWelcome();
  }, [backendReady, mediaState.voiceEnabled, hasPlayedWelcome, setHasPlayedWelcome]);

  return (
    <div className="min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background">
      {/* Portrait = Mobile style */}
      {isPortrait ? (
        <div className="flex-1 min-h-0 flex flex-col">
          <Header />
          <div className="flex-1 min-h-0">
            <MobileShell />
          </div>
          <BottomNav />
        </div>
      ) : (
        /* Landscape = Desktop windowed shell */
        <DesktopShell />
      )}

      {/* Global Modal (keep available) */}
      <SettingsModal />
    </div>
  );
};

export default Index;

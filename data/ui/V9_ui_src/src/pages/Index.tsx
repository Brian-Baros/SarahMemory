import { useEffect } from "react";
import { Header } from "@/components/layout/Header";
import { MobileDrawers } from "@/components/layout/MobileDrawers";
import { SettingsModal } from "@/components/panels/SettingsModal";
import { useSarahStore } from "@/stores/useSarahStore";
import { api } from "@/lib/api";
import { useViewportProfile } from "@/hooks/use-mobile";

// Phase 1 Mobile Shell
import { MobileShell } from "@/components/shell/MobileShell";
import { BottomNav } from "@/components/shell/BottomNav";

// Desktop Shell
import { DesktopShell } from "@/components/shell/DesktopShell";

/**
 * SarahMemory viewport shell switch:
 *  - vertical phone/tablet posture => Mobile shell
 *  - horizontal posture / desktop viewport => Desktop shell
 *
 * The logic is centralized in useViewportProfile so child components do not
 * independently disagree about mobile vs desktop during orientation changes.
 */
const Index = () => {
  const {
    mediaState,
    hasPlayedWelcome,
    backendReady,
    setHasPlayedWelcome,
    setVoices,
  } = useSarahStore();

  const viewport = useViewportProfile();
  const isMobileShell = viewport.shellMode === "mobile";

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
    <div
      className="sarah-viewport-shell min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background"
      data-shell-mode={viewport.shellMode}
      data-orientation={viewport.isPortrait ? "portrait" : "landscape"}
      data-touch={viewport.isTouch ? "true" : "false"}
    >
      {isMobileShell ? (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          <Header />
          <div className="flex-1 min-h-0 overflow-hidden">
            <MobileShell />
          </div>
          <BottomNav />
          <MobileDrawers />
        </div>
      ) : (
        <DesktopShell />
      )}

      <SettingsModal />
    </div>
  );
};

export default Index;

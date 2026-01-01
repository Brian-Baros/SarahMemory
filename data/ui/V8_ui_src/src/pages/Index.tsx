import { useEffect } from 'react';
import { Header } from '@/components/layout/Header';
import { LeftSidebar } from '@/components/layout/LeftSidebar';
import { RightSidebar } from '@/components/layout/RightSidebar';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { StatusBar } from '@/components/StatusBar';
import { SettingsModal } from '@/components/panels/SettingsModal';
import { MobileDrawers } from '@/components/layout/MobileDrawers';
import { useSarahStore } from '@/stores/useSarahStore';
import { api } from '@/lib/api';

/**
 * Main Index page - Three-column responsive layout
 * 
 * Desktop (lg+): 
 *   - Left sidebar (collapsible) | Center chat | Right sidebar (collapsible)
 * 
 * Mobile (< lg):
 *   - Full-width center chat
 *   - Left drawer (slide from left) for chat history
 *   - Right drawer (slide from right) for tools
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

  // Bootstrap on mount - calls POST /api/session/bootstrap once
  useEffect(() => {
    const runBootstrap = async () => {
      try {
        console.log('[Index] Running bootstrap...');
        const bootstrapData = await api.bootstrap.init();
        
        if (bootstrapData.ok) {
          console.log('[Index] Bootstrap successful:', bootstrapData.version);
          setBootstrapData(bootstrapData);
          setBackendReady(true);
          
          // If bootstrap returns an api_base, it's already handled by config
        } else {
          console.warn('[Index] Bootstrap returned ok:false');
          setBackendReady(true); // Still allow app to function
        }
      } catch (error) {
        console.warn('[Index] Bootstrap failed:', error);
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
          console.log('[Index] Loaded voices:', voices.length);
        }
      } catch (error) {
        console.warn('[Index] Failed to load voices:', error);
      }
    };
    
    fetchVoices();
  }, [backendReady, setVoices]);

  // One-time TTS welcome intro when backend is ready and voice is enabled
  useEffect(() => {
    const playWelcome = async () => {
      if (backendReady && mediaState.voiceEnabled && !hasPlayedWelcome) {
        try {
          // Use the existing voice API speak function
          await api.voice.speak("SarahMemory is online and ready.");
          setHasPlayedWelcome(true);
        } catch (error) {
          console.warn('[Index] Welcome TTS failed:', error);
          // Still mark as played to avoid retrying
          setHasPlayedWelcome(true);
        }
      }
    };

    playWelcome();
  }, [backendReady, mediaState.voiceEnabled, hasPlayedWelcome, setHasPlayedWelcome]);

  return (
    <div className="min-h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-background">
      {/* Main Layout - Responsive three-column */}
      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar - Chat History (desktop only) */}
        <LeftSidebar />

        {/* Center - Chat Area (always visible, primary focus) */}
        <main className="flex-1 flex flex-col min-w-0 min-h-0">
          <Header />
          <ChatPanel />
        </main>

        {/* Right Sidebar - Preview & Controls (desktop only) */}
        <RightSidebar />
      </div>

      {/* Status Bar - Hidden on mobile to save space */}
      <div className="hidden sm:block">
        <StatusBar />
      </div>

      {/* Settings Modal */}
      <SettingsModal />

      {/* Mobile Drawers - Off-canvas panels for mobile */}
      <MobileDrawers />
    </div>
  );
};

export default Index;

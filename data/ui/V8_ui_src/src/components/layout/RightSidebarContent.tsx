import { Mic, Volume2, Eye, EyeOff, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useSarahStore } from '@/stores/useSarahStore';
import { cn } from '@/lib/utils';
import { ContactsPanel } from '@/components/panels/ContactsPanel';
import { RemindersPanel } from '@/components/panels/RemindersPanel';
import { DialerPanel } from '@/components/panels/DialerPanel';
import { CreativeToolsPanel } from '@/components/panels/CreativeToolsPanel';
import { AvatarPanel } from '@/components/avatar/AvatarPanel';
import { RightPanelTabs } from './RightPanelTabs';
import { config } from '@/lib/config';

/**
 * Right sidebar content - Tools, settings, and utilities
 * Organized into tabbed pages: Contacts, Keypad, Tools, Settings
 * Used by both desktop sidebar and mobile drawer
 */
export function RightSidebarContent() {
  const { 
    mediaState, 
    toggleWebcam, 
    toggleMicrophone, 
    toggleVoice,
    rightPanelPage,
    setSettingsOpen,
    setRightDrawerOpen,
  } = useSarahStore();

  const [isTogglingCam, setIsTogglingCam] = useState(false);
  const [isTogglingMic, setIsTogglingMic] = useState(false);
  const [isTogglingVoice, setIsTogglingVoice] = useState(false);

  // Toggle camera with backend call
  const handleToggleCam = async () => {
    const newState = !mediaState.webcamEnabled;
    setIsTogglingCam(true);
    
    try {
      await fetch(`${config.apiBaseUrl}/toggle_camera?state=${newState}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[RightSidebar] Camera toggle backend unavailable:', error);
    }
    
    toggleWebcam();
    setIsTogglingCam(false);
  };

  // Toggle microphone with backend call
  const handleToggleMic = async () => {
    const newState = !mediaState.microphoneEnabled;
    setIsTogglingMic(true);
    
    try {
      await fetch(`${config.apiBaseUrl}/toggle_microphone?state=${newState}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[RightSidebar] Mic toggle backend unavailable:', error);
    }
    
    toggleMicrophone();
    setIsTogglingMic(false);
  };

  // Toggle voice output with backend call
  const handleToggleVoice = async () => {
    const newState = !mediaState.voiceEnabled;
    setIsTogglingVoice(true);
    
    try {
      await fetch(`${config.apiBaseUrl}/toggle_voice_output?state=${newState}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[RightSidebar] Voice toggle backend unavailable:', error);
    }
    
    toggleVoice();
    setIsTogglingVoice(false);
  };

  const handleOpenSettings = () => {
    setSettingsOpen(true);
    setRightDrawerOpen(false); // Close drawer on mobile when opening settings
  };

  return (
    <div className="flex flex-col h-full">
      {/* Avatar Panel - Always visible at top */}
      <AvatarPanel />

      {/* Media Controls */}
      <div className="p-3 border-b border-sidebar-border">
        <div className="flex gap-2">
          <Button 
            variant={mediaState.webcamEnabled ? "default" : "outline"}
            size="sm"
            className={cn(
              "flex-1 text-xs",
              mediaState.webcamEnabled && "bg-primary text-primary-foreground"
            )}
            onClick={handleToggleCam}
            disabled={isTogglingCam}
          >
            {isTogglingCam ? (
              <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            ) : mediaState.webcamEnabled ? (
              <Eye className="h-3 w-3 mr-1.5" />
            ) : (
              <EyeOff className="h-3 w-3 mr-1.5" />
            )}
            Cam
          </Button>
          <Button 
            variant={mediaState.microphoneEnabled ? "default" : "outline"}
            size="sm"
            className={cn(
              "flex-1 text-xs",
              mediaState.microphoneEnabled && "bg-primary text-primary-foreground"
            )}
            onClick={handleToggleMic}
            disabled={isTogglingMic}
          >
            {isTogglingMic ? (
              <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            ) : (
              <Mic className="h-3 w-3 mr-1.5" />
            )}
            Mic
          </Button>
          <Button 
            variant={mediaState.voiceEnabled ? "default" : "outline"}
            size="sm"
            className={cn(
              "flex-1 text-xs",
              mediaState.voiceEnabled && "bg-primary text-primary-foreground"
            )}
            onClick={handleToggleVoice}
            disabled={isTogglingVoice}
          >
            {isTogglingVoice ? (
              <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            ) : (
              <Volume2 className="h-3 w-3 mr-1.5" />
            )}
            Voice
          </Button>
        </div>
      </div>

      {/* Tab Navigation for Pages */}
      <RightPanelTabs />

      {/* Page Content - Swipeable pages */}
      <div className="flex-1 overflow-y-auto">
        {/* Page 1: Contacts */}
        {rightPanelPage === 'contacts' && (
          <div className="animate-fade-in">
            <ContactsPanel />
          </div>
        )}

        {/* Page 2: Keypad / Dialer */}
        {rightPanelPage === 'keypad' && (
          <div className="animate-fade-in">
            <DialerPanel />
          </div>
        )}

        {/* Page 3: Tools / Creative */}
        {rightPanelPage === 'tools' && (
          <div className="animate-fade-in">
            <CreativeToolsPanel />
            <RemindersPanel />
          </div>
        )}

        {/* Page 4: Settings */}
        {rightPanelPage === 'settings' && (
          <div className="animate-fade-in p-4">
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleOpenSettings}
            >
              Open Full Settings
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

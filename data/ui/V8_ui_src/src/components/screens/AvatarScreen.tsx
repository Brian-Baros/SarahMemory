import { User, Mic, Volume2, Eye, EyeOff, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useSarahStore } from '@/stores/useSarahStore';
import { PreviewSurface } from '@/components/avatar/PreviewSurface';
import { config } from '@/lib/config';
import { cn } from '@/lib/utils';

/**
 * Avatar Screen - Dedicated avatar panel
 * Shows the avatar preview and speaking indicators
 * Never overlaps with chat or other content
 */
export function AvatarScreen() {
  const { 
    mediaState, 
    toggleWebcam, 
    toggleMicrophone, 
    toggleVoice,
    avatarSpeaking,
  } = useSarahStore();

  const [isTogglingCam, setIsTogglingCam] = useState(false);
  const [isTogglingMic, setIsTogglingMic] = useState(false);
  const [isTogglingVoice, setIsTogglingVoice] = useState(false);

  const handleToggleCam = async () => {
    setIsTogglingCam(true);
    try {
      await fetch(`${config.apiBaseUrl}/toggle_camera?state=${!mediaState.webcamEnabled}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[Avatar] Camera toggle unavailable:', error);
    }
    toggleWebcam();
    setIsTogglingCam(false);
  };

  const handleToggleMic = async () => {
    setIsTogglingMic(true);
    try {
      await fetch(`${config.apiBaseUrl}/toggle_microphone?state=${!mediaState.microphoneEnabled}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[Avatar] Mic toggle unavailable:', error);
    }
    toggleMicrophone();
    setIsTogglingMic(false);
  };

  const handleToggleVoice = async () => {
    setIsTogglingVoice(true);
    try {
      await fetch(`${config.apiBaseUrl}/toggle_voice_output?state=${!mediaState.voiceEnabled}`, {
        method: 'GET',
        credentials: 'include',
      });
    } catch (error) {
      console.warn('[Avatar] Voice toggle unavailable:', error);
    }
    toggleVoice();
    setIsTogglingVoice(false);
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <User className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Avatar</h1>
          {avatarSpeaking && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-primary animate-pulse">
              <Volume2 className="h-3.5 w-3.5" />
              Speaking
            </span>
          )}
        </div>
      </div>

      {/* Avatar Preview - Large display */}
      <div className="flex-1 min-h-0">
        <ScrollArea className="h-full">
          <div className="p-4">
            {/* Main Avatar Display */}
            <div className="aspect-[3/4] max-h-[60vh] rounded-xl overflow-hidden border border-border bg-card">
              <PreviewSurface />
            </div>

            {/* Media Controls */}
            <div className="mt-4 p-3 rounded-xl bg-card border border-border">
              <p className="text-xs text-muted-foreground mb-3">Media Controls</p>
              <div className="flex gap-2">
                <Button 
                  variant={mediaState.webcamEnabled ? "default" : "outline"}
                  size="sm"
                  className={cn(
                    "flex-1",
                    mediaState.webcamEnabled && "bg-primary text-primary-foreground"
                  )}
                  onClick={handleToggleCam}
                  disabled={isTogglingCam}
                >
                  {isTogglingCam ? (
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                  ) : mediaState.webcamEnabled ? (
                    <Eye className="h-4 w-4 mr-1.5" />
                  ) : (
                    <EyeOff className="h-4 w-4 mr-1.5" />
                  )}
                  Camera
                </Button>
                <Button 
                  variant={mediaState.microphoneEnabled ? "default" : "outline"}
                  size="sm"
                  className={cn(
                    "flex-1",
                    mediaState.microphoneEnabled && "bg-primary text-primary-foreground"
                  )}
                  onClick={handleToggleMic}
                  disabled={isTogglingMic}
                >
                  {isTogglingMic ? (
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                  ) : (
                    <Mic className="h-4 w-4 mr-1.5" />
                  )}
                  Mic
                </Button>
                <Button 
                  variant={mediaState.voiceEnabled ? "default" : "outline"}
                  size="sm"
                  className={cn(
                    "flex-1",
                    mediaState.voiceEnabled && "bg-primary text-primary-foreground"
                  )}
                  onClick={handleToggleVoice}
                  disabled={isTogglingVoice}
                >
                  {isTogglingVoice ? (
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                  ) : (
                    <Volume2 className="h-4 w-4 mr-1.5" />
                  )}
                  Voice
                </Button>
              </div>
            </div>

            {/* Status Info */}
            <div className="mt-4 p-3 rounded-xl bg-card border border-border">
              <p className="text-xs text-muted-foreground mb-2">Status</p>
              <div className="space-y-1.5 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Mode</span>
                  <span className="font-medium">{mediaState.screenMode.replace('_', ' ')}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Speaking</span>
                  <span className={cn(
                    "font-medium",
                    avatarSpeaking ? "text-primary" : "text-muted-foreground"
                  )}>
                    {avatarSpeaking ? "Active" : "Idle"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

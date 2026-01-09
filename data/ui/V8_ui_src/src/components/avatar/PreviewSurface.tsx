/**
 * PreviewSurface - The unified preview display area
 * Shows one active preview at a time based on the preview router state:
 * - Avatar (2D/3D)
 * - Image preview
 * - Audio player
 * - Video player
 * - Call (dual video feed)
 * - Desktop mirror
 */

import { Suspense, useRef, useEffect, useState } from 'react';
import { Image, Volume2, Play, Pause, X, PhoneOff, Mic, MicOff, Camera, CameraOff, Speaker } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { usePreviewStore } from '@/stores/usePreviewStore';
import { useSarahStore } from '@/stores/useSarahStore';
import { AvatarPanel } from './AvatarPanel';
import { config } from '@/lib/config';

// Normalize relative URLs to absolute
function normalizeMediaUrl(url?: string): string | undefined {
  if (!url) return undefined;
  if (url.startsWith('data:')) return url;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  // Relative path - prepend API base
  const base = config.apiBaseUrl || window.location.origin;
  return url.startsWith('/') ? `${base}${url}` : `${base}/${url}`;
}

// Audio Player Component
function AudioPreview({ url, base64, onClose }: { url?: string; base64?: string; onClose: () => void }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTapHelper, setShowTapHelper] = useState(false);

  const audioSrc = url ? normalizeMediaUrl(url) : base64 ? `data:audio/mpeg;base64,${base64}` : null;

  useEffect(() => {
    if (audioRef.current && audioSrc) {
      audioRef.current.src = audioSrc;
      audioRef.current.play()
        .then(() => setIsPlaying(true))
        .catch(() => {
          // Autoplay blocked - show tap helper once
          setShowTapHelper(true);
        });
    }
  }, [audioSrc]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(() => {});
    }
    setIsPlaying(!isPlaying);
    setShowTapHelper(false);
  };

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/90 z-20">
      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />
      <div className="flex flex-col items-center gap-4">
        <Volume2 className="h-12 w-12 text-primary animate-pulse" />
        <Button onClick={handlePlayPause} variant="outline" size="lg">
          {isPlaying ? <Pause className="h-5 w-5 mr-2" /> : <Play className="h-5 w-5 mr-2" />}
          {isPlaying ? 'Pause' : 'Play'}
        </Button>
        {showTapHelper && (
          <p className="text-xs text-muted-foreground">Tap to enable audio</p>
        )}
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2"
        onClick={onClose}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}

// Image Preview Component
function ImagePreview({ url, base64, onClose }: { url?: string; base64?: string; onClose: () => void }) {
  const imgSrc = url ? normalizeMediaUrl(url) : base64 ? `data:image/png;base64,${base64}` : null;

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-background/90 z-20">
      {imgSrc ? (
        <img src={imgSrc} alt="Preview" className="max-w-full max-h-full object-contain" />
      ) : (
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <Image className="h-12 w-12" />
          <span className="text-sm">No image available</span>
        </div>
      )}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 bg-background/80"
        onClick={onClose}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}

// Video Preview Component
function VideoPreview({ url, onClose }: { url?: string; onClose: () => void }) {
  const videoSrc = normalizeMediaUrl(url);

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-background z-20">
      {videoSrc ? (
        <video
          src={videoSrc}
          controls
          autoPlay
          className="max-w-full max-h-full"
        />
      ) : (
        <div className="text-muted-foreground">No video available</div>
      )}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 bg-background/80"
        onClick={onClose}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}

// Call View Component (dual video feed)
function CallPreview({ callId, onEndCall }: { callId?: string; onEndCall: () => void }) {
  const [micMuted, setMicMuted] = useState(false);
  const [camOff, setCamOff] = useState(false);

  return (
    <div className="absolute inset-0 flex flex-col bg-background z-20">
      {/* Remote video (main) */}
      <div className="flex-1 bg-muted flex items-center justify-center relative">
        <div className="text-muted-foreground text-sm">Remote Video</div>
        
        {/* Local video (PiP) */}
        <div className="absolute bottom-2 right-2 w-24 h-18 bg-sidebar rounded border border-border flex items-center justify-center">
          <span className="text-xs text-muted-foreground">You</span>
        </div>
      </div>

      {/* Call Controls */}
      <div className="flex items-center justify-center gap-3 p-3 bg-sidebar border-t border-border">
        <Button
          variant={micMuted ? 'destructive' : 'outline'}
          size="icon"
          onClick={() => setMicMuted(!micMuted)}
        >
          {micMuted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
        </Button>
        <Button
          variant={camOff ? 'destructive' : 'outline'}
          size="icon"
          onClick={() => setCamOff(!camOff)}
        >
          {camOff ? <CameraOff className="h-4 w-4" /> : <Camera className="h-4 w-4" />}
        </Button>
        <Button
          variant="outline"
          size="icon"
        >
          <Speaker className="h-4 w-4" />
        </Button>
        <Button
          variant="destructive"
          size="icon"
          onClick={onEndCall}
        >
          <PhoneOff className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export function PreviewSurface() {
  const { current, restorePrevious, endCall, resetToAvatar } = usePreviewStore();
  const addMessage = useSarahStore((s) => s.addMessage);

  const handleCloseMedia = () => {
    restorePrevious();
  };

  const handleEndCall = () => {
    addMessage({
      role: 'assistant',
      content: '[Call Ended] Video call disconnected',
    });
    endCall();
  };

  // Render based on current preview type
  return (
    <div className="relative aspect-video">
      {/* Base layer: Always render AvatarPanel as the foundation */}
      <AvatarPanel />

      {/* Overlay layers based on preview type */}
      {current.type === 'image' && (
        <ImagePreview
          url={current.mediaUrl}
          base64={current.mediaBase64}
          onClose={handleCloseMedia}
        />
      )}

      {current.type === 'audio' && (
        <AudioPreview
          url={current.mediaUrl}
          base64={current.mediaBase64}
          onClose={handleCloseMedia}
        />
      )}

      {current.type === 'video' && (
        <VideoPreview
          url={current.mediaUrl}
          onClose={handleCloseMedia}
        />
      )}

      {current.type === 'call' && (
        <CallPreview
          callId={current.callId}
          onEndCall={handleEndCall}
        />
      )}
    </div>
  );
}

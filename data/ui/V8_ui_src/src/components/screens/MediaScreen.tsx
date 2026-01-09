import { useState } from 'react';
import { 
  Play, 
  Pause, 
  SkipBack, 
  SkipForward, 
  Volume2,
  VolumeX,
  Loader2,
  Music,
  Video,
  Image,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Slider } from '@/components/ui/slider';
import { useCreativeCacheStore, type CachedItem } from '@/stores/useCreativeCacheStore';
import { cn } from '@/lib/utils';

/**
 * Media Screen - Media player for creative outputs
 * Plays audio/video from the creative cache
 */
export function MediaScreen() {
  const { items, downloadItem } = useCreativeCacheStore();
  const [currentItem, setCurrentItem] = useState<CachedItem | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState([75]);
  const [progress, setProgress] = useState([0]);

  // Get all playable media from cache
  const allMedia = [
    ...items.music,
    ...items.voice,
    ...items.video,
    ...items.image,
  ].sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

  const getMediaIcon = (type: CachedItem['type']) => {
    switch (type) {
      case 'music': 
      case 'voice': return Music;
      case 'video': return Video;
      case 'image': return Image;
      default: return Music;
    }
  };

  const handlePlay = (item: CachedItem) => {
    setCurrentItem(item);
    setIsPlaying(true);
    // TODO: Implement actual playback with audio/video element
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Play className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Media Player</h1>
        </div>
      </div>

      {/* Now Playing */}
      {currentItem && (
        <div className="p-4 border-b border-border bg-card/30">
          <div className="aspect-video max-h-40 rounded-lg bg-muted flex items-center justify-center mb-4">
            {currentItem.preview ? (
              <img 
                src={currentItem.preview} 
                alt={currentItem.prompt}
                className="w-full h-full object-cover rounded-lg"
              />
            ) : (
              <div className="text-center">
                {(() => {
                  const Icon = getMediaIcon(currentItem.type);
                  return <Icon className="h-12 w-12 text-muted-foreground/50 mx-auto" />;
                })()}
                <p className="text-xs text-muted-foreground mt-2 px-4 truncate max-w-full">
                  {currentItem.prompt}
                </p>
              </div>
            )}
          </div>

          {/* Progress */}
          <Slider
            value={progress}
            onValueChange={setProgress}
            max={100}
            step={1}
            className="mb-4"
          />

          {/* Controls */}
          <div className="flex items-center justify-center gap-2">
            <Button variant="ghost" size="icon" className="h-10 w-10">
              <SkipBack className="h-5 w-5" />
            </Button>
            <Button 
              variant="default" 
              size="icon" 
              className="h-12 w-12"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <Pause className="h-6 w-6" />
              ) : (
                <Play className="h-6 w-6" />
              )}
            </Button>
            <Button variant="ghost" size="icon" className="h-10 w-10">
              <SkipForward className="h-5 w-5" />
            </Button>
          </div>

          {/* Volume */}
          <div className="flex items-center gap-2 mt-4">
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8"
              onClick={() => setIsMuted(!isMuted)}
            >
              {isMuted ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            <Slider
              value={isMuted ? [0] : volume}
              onValueChange={setVolume}
              max={100}
              step={1}
              className="flex-1"
            />
          </div>
        </div>
      )}

      {/* Media Library */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          <p className="text-xs text-muted-foreground mb-3">Library ({allMedia.length} items)</p>
          
          {allMedia.length === 0 ? (
            <div className="text-center py-12">
              <Music className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">No media in session</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Generate content in Studios to see it here
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {allMedia.map((item) => {
                const Icon = getMediaIcon(item.type);
                const isActive = currentItem?.id === item.id;
                
                return (
                  <button
                    key={item.id}
                    onClick={() => handlePlay(item)}
                    className={cn(
                      "w-full text-left p-3 rounded-xl flex items-center gap-3 transition-all",
                      "bg-card border border-border hover:bg-card/80",
                      isActive && "border-primary/50 bg-primary/5"
                    )}
                  >
                    <div className={cn(
                      "w-10 h-10 rounded-lg flex items-center justify-center shrink-0",
                      isActive ? "bg-primary/20" : "bg-muted"
                    )}>
                      <Icon className={cn(
                        "h-5 w-5",
                        isActive ? "text-primary" : "text-muted-foreground"
                      )} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">{item.prompt}</p>
                      <p className="text-xs text-muted-foreground capitalize">
                        {item.type} • {item.createdAt.toLocaleTimeString()}
                      </p>
                    </div>
                    {isActive && isPlaying && (
                      <div className="flex gap-0.5">
                        <div className="w-0.5 h-3 bg-primary animate-pulse" />
                        <div className="w-0.5 h-4 bg-primary animate-pulse delay-75" />
                        <div className="w-0.5 h-2 bg-primary animate-pulse delay-150" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

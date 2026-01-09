import { useState } from 'react';
import { Video, Loader2, Sparkles, Download, Trash2, X, Play, Plus, Image, Music, Mic } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { useSarahStore } from '@/stores/useSarahStore';
import { useCreativeCacheStore, type CachedItem } from '@/stores/useCreativeCacheStore';
import { usePreviewStore } from '@/stores/usePreviewStore';
import { cn } from '@/lib/utils';

export function VideoStudioModule() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  
  const addMessage = useSarahStore((s) => s.addMessage);
  const { items, addItem, removeItem, clearModule, downloadItem } = useCreativeCacheStore();
  const { showVideo } = usePreviewStore();
  const videoItems = items.video;
  
  // Available items from other modules for stacking
  const imageItems = items.image;
  const musicItems = items.music;
  const voiceItems = items.voice;
  const allSourceItems = [...imageItems, ...musicItems, ...voiceItems];

  const toggleSelectItem = (id: string) => {
    setSelectedItems((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const getItemIcon = (type: string) => {
    switch (type) {
      case 'image': return Image;
      case 'music': return Music;
      case 'voice': return Mic;
      default: return Video;
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim() && selectedItems.length === 0) {
      toast.error('Please enter a prompt or select items to combine');
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    // Log to chat
    const stackInfo = selectedItems.length > 0 ? ` (stacking ${selectedItems.length} items)` : '';
    addMessage({
      role: 'user',
      content: `[Video Generation] ${prompt || 'Combining selected items'}${stackInfo}`,
    });

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 3, 90));
    }, 1000);

    try {
      const response = await api.proxy.call('/api/creative/video', {
        method: 'POST',
        body: {
          prompt: prompt.trim(),
          source_items: selectedItems,
        },
      });

      clearInterval(progressInterval);
      setProgress(100);

      const resultUrl = (response as any)?.url || (response as any)?.video_url;

      const itemId = addItem('video', {
        type: 'video',
        prompt: prompt.trim() || 'Stacked composition',
        url: resultUrl,
        metadata: { sourceItems: selectedItems },
      });

      // Show in preview surface
      if (resultUrl) {
        showVideo(itemId, resultUrl);
      }

      addMessage({
        role: 'assistant',
        content: `[Video Generated] Created video${stackInfo}`,
      });

      toast.success('Video generated!');
      setPrompt('');
      setSelectedItems([]);
    } catch (error) {
      clearInterval(progressInterval);
      
      // Demo placeholder
      addItem('video', {
        type: 'video',
        prompt: prompt.trim() || 'Stacked composition',
        metadata: { demo: true, sourceItems: selectedItems },
      });

      addMessage({
        role: 'assistant',
        content: `[Video Demo] Render job queued${stackInfo}`,
      });

      toast.info('Video generation demo - backend coming soon');
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  return (
    <div className="p-3 space-y-3">
      {/* Source Items Selector */}
      {allSourceItems.length > 0 && (
        <div className="space-y-1.5">
          <span className="text-xs text-muted-foreground">Stack items (optional)</span>
          <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
            {allSourceItems.slice(0, 8).map((item) => {
              const Icon = getItemIcon(item.type);
              const isSelected = selectedItems.includes(item.id);
              return (
                <button
                  key={item.id}
                  onClick={() => toggleSelectItem(item.id)}
                  className={cn(
                    "flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors",
                    isSelected
                      ? "bg-primary text-primary-foreground"
                      : "bg-sidebar-accent hover:bg-sidebar-accent/80"
                  )}
                >
                  <Icon className="h-3 w-3" />
                  <span className="truncate max-w-[60px]">{item.prompt.slice(0, 15)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Prompt */}
      <Textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe the video or leave empty to combine selected items..."
        className="min-h-[60px] text-sm bg-sidebar-accent border-sidebar-border resize-none"
      />

      {/* Generate Button */}
      <Button
        onClick={handleGenerate}
        disabled={isGenerating || (!prompt.trim() && selectedItems.length === 0)}
        className="w-full h-8 text-sm"
      >
        {isGenerating ? (
          <>
            <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            Rendering...
          </>
        ) : (
          <>
            <Sparkles className="h-3 w-3 mr-1.5" />
            Generate Video
          </>
        )}
      </Button>

      {/* Progress */}
      {isGenerating && (
        <div className="space-y-1">
          <Progress value={progress} className="h-1" />
          <p className="text-xs text-muted-foreground text-center">{progress}%</p>
        </div>
      )}

      {/* Render Jobs / Cached Videos */}
      {videoItems.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Render Jobs</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-destructive"
              onClick={() => clearModule('video')}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {videoItems.map((item) => (
              <div
                key={item.id}
                className="flex items-center gap-2 p-2 rounded bg-sidebar-accent/50 group"
              >
                <div className="w-8 h-8 rounded bg-secondary flex items-center justify-center shrink-0">
                  <Video className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs truncate">{item.prompt}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {(item.metadata as any)?.demo ? 'Demo' : 'Ready'}
                  </p>
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => downloadItem('video', item.id)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive"
                    onClick={() => removeItem('video', item.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

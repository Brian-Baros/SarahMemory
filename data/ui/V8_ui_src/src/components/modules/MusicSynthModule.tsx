import { useState } from 'react';
import { Music, Loader2, Sparkles, Download, Trash2, X, Play, Pause } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { useSarahStore } from '@/stores/useSarahStore';
import { useCreativeCacheStore } from '@/stores/useCreativeCacheStore';

export function MusicSynthModule() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [playingId, setPlayingId] = useState<string | null>(null);
  
  const addMessage = useSarahStore((s) => s.addMessage);
  const { items, addItem, removeItem, clearModule, downloadItem } = useCreativeCacheStore();
  const musicItems = items.music;

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    // Log to chat
    addMessage({
      role: 'user',
      content: `[Music Generation] ${prompt}`,
    });

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 5, 90));
    }, 800);

    try {
      const response = await api.proxy.call('/api/creative/music', {
        method: 'POST',
        body: { prompt: prompt.trim() },
      });

      clearInterval(progressInterval);
      setProgress(100);

      const resultUrl = (response as any)?.url || (response as any)?.audio_url;

      addItem('music', {
        type: 'music',
        prompt: prompt.trim(),
        url: resultUrl,
      });

      addMessage({
        role: 'assistant',
        content: `[Music Generated] Created track from prompt: "${prompt}"`,
      });

      toast.success('Music generated!');
      setPrompt('');
    } catch (error) {
      clearInterval(progressInterval);
      
      // Demo placeholder
      addItem('music', {
        type: 'music',
        prompt: prompt.trim(),
        metadata: { demo: true },
      });

      addMessage({
        role: 'assistant',
        content: `[Music Demo] Placeholder created for: "${prompt}"`,
      });

      toast.info('Music generation demo - backend coming soon');
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  const togglePlay = (id: string, url?: string) => {
    if (!url) {
      toast.info('Audio not available');
      return;
    }
    
    if (playingId === id) {
      setPlayingId(null);
    } else {
      setPlayingId(id);
      const audio = new Audio(url);
      audio.play();
      audio.onended = () => setPlayingId(null);
    }
  };

  return (
    <div className="p-3 space-y-3">
      {/* Prompt */}
      <Textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe the music you want to create (genre, mood, tempo)..."
        className="min-h-[60px] text-sm bg-sidebar-accent border-sidebar-border resize-none"
      />

      {/* Generate Button */}
      <Button
        onClick={handleGenerate}
        disabled={isGenerating || !prompt.trim()}
        className="w-full h-8 text-sm"
      >
        {isGenerating ? (
          <>
            <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Sparkles className="h-3 w-3 mr-1.5" />
            Generate Music
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

      {/* Cached Items */}
      {musicItems.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Session Cache</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-destructive"
              onClick={() => clearModule('music')}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {musicItems.slice(0, 5).map((item) => (
              <div
                key={item.id}
                className="flex items-center gap-2 p-2 rounded bg-sidebar-accent/50 group"
              >
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={() => togglePlay(item.id, item.url)}
                >
                  {playingId === item.id ? (
                    <Pause className="h-3 w-3" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                </Button>
                <div className="flex-1 min-w-0">
                  <p className="text-xs truncate">{item.prompt}</p>
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => downloadItem('music', item.id)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive"
                    onClick={() => removeItem('music', item.id)}
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

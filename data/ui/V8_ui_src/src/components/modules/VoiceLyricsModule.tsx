import { useState } from 'react';
import { Mic, Loader2, Sparkles, Download, Trash2, X, Play, Pause } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { useSarahStore } from '@/stores/useSarahStore';
import { useCreativeCacheStore } from '@/stores/useCreativeCacheStore';
import { usePreviewStore } from '@/stores/usePreviewStore';

type VoiceMode = 'tts' | 'lyrics';

export function VoiceLyricsModule() {
  const [mode, setMode] = useState<VoiceMode>('tts');
  const [text, setText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [playingId, setPlayingId] = useState<string | null>(null);
  
  const addMessage = useSarahStore((s) => s.addMessage);
  const voices = useSarahStore((s) => s.voices);
  const settings = useSarahStore((s) => s.settings);
  const { items, addItem, removeItem, clearModule, downloadItem } = useCreativeCacheStore();
  const { showAudio } = usePreviewStore();
  const voiceItems = items.voice;

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast.error('Please enter text');
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    const label = mode === 'tts' ? 'Voice TTS' : 'Lyrics to Song';

    // Log to chat
    addMessage({
      role: 'user',
      content: `[${label}] ${text.slice(0, 100)}${text.length > 100 ? '...' : ''}`,
    });

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 8, 90));
    }, 600);

    try {
      const endpoint = mode === 'tts' ? '/api/voice/speak' : '/api/creative/lyrics-to-song';
      const response = await api.proxy.call(endpoint, {
        method: 'POST',
        body: {
          text: text.trim(),
          voice: settings.selectedVoice,
        },
      });

      clearInterval(progressInterval);
      setProgress(100);

      const audioUrl = (response as any)?.audio_url || (response as any)?.url;

      const itemId = addItem('voice', {
        type: 'voice',
        prompt: text.trim(),
        url: audioUrl,
        metadata: { mode },
      });

      // Show in preview surface
      if (audioUrl) {
        showAudio(itemId, audioUrl);
      }

      addMessage({
        role: 'assistant',
        content: `[${label} Generated] Created audio from: "${text.slice(0, 50)}..."`,
      });

      toast.success(`${label} generated!`);
      setText('');
    } catch (error) {
      clearInterval(progressInterval);
      
      // Demo placeholder
      addItem('voice', {
        type: 'voice',
        prompt: text.trim(),
        metadata: { mode, demo: true },
      });

      addMessage({
        role: 'assistant',
        content: `[${label} Demo] Placeholder for: "${text.slice(0, 50)}..."`,
      });

      toast.info(`${label} demo - backend coming soon`);
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
      {/* Mode Selector */}
      <Select value={mode} onValueChange={(v) => setMode(v as VoiceMode)}>
        <SelectTrigger className="h-8 text-xs bg-sidebar-accent border-sidebar-border">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="tts" className="text-xs">Text to Speech (TTS)</SelectItem>
          <SelectItem value="lyrics" className="text-xs">Lyrics to Song</SelectItem>
        </SelectContent>
      </Select>

      {/* Text Input */}
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={mode === 'tts' ? 'Enter text to speak...' : 'Enter song lyrics...'}
        className="min-h-[80px] text-sm bg-sidebar-accent border-sidebar-border resize-none"
      />

      {/* Generate Button */}
      <Button
        onClick={handleGenerate}
        disabled={isGenerating || !text.trim()}
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
            Generate {mode === 'tts' ? 'Speech' : 'Song'}
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
      {voiceItems.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Session Cache</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-destructive"
              onClick={() => clearModule('voice')}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {voiceItems.slice(0, 5).map((item) => (
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
                  <p className="text-[10px] text-muted-foreground capitalize">
                    {(item.metadata as any)?.mode || 'voice'}
                  </p>
                </div>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => downloadItem('voice', item.id)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive"
                    onClick={() => removeItem('voice', item.id)}
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

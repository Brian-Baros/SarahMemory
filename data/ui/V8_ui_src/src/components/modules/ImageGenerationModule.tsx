import { useState, useRef } from 'react';
import { Image, Upload, Loader2, Sparkles, Download, Trash2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { useSarahStore } from '@/stores/useSarahStore';
import { useCreativeCacheStore } from '@/stores/useCreativeCacheStore';
import { usePreviewStore } from '@/stores/usePreviewStore';
import { cn } from '@/lib/utils';

export function ImageGenerationModule() {
  const [prompt, setPrompt] = useState('');
  const [seedImage, setSeedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const addMessage = useSarahStore((s) => s.addMessage);
  const { items, addItem, removeItem, clearModule, downloadItem } = useCreativeCacheStore();
  const { showImage } = usePreviewStore();
  const imageItems = items.image;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      toast.error('Please upload an image file');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
      setSeedImage(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSeedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

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
      content: `[Image Generation] ${prompt}${seedImage ? ' (with seed image)' : ''}`,
    });

    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 10, 90));
    }, 500);

    try {
      const response = await api.proxy.call('/api/creative/image', {
        method: 'POST',
        body: {
          prompt: prompt.trim(),
          seed_image: seedImage,
        },
      });

      clearInterval(progressInterval);
      setProgress(100);

      const resultUrl = (response as any)?.url || (response as any)?.result_url;
      const preview = (response as any)?.preview || (response as any)?.thumbnail || resultUrl;

      const itemId = addItem('image', {
        type: 'image',
        prompt: prompt.trim(),
        url: resultUrl,
        preview: preview,
      });

      // Show in preview surface
      if (resultUrl || preview) {
        showImage(itemId, resultUrl || preview);
      }

      addMessage({
        role: 'assistant',
        content: `[Image Generated] Created image from prompt: "${prompt}"`,
      });

      toast.success('Image generated!');
      setPrompt('');
      setSeedImage(null);
    } catch (error) {
      clearInterval(progressInterval);
      
      // Demo placeholder
      const itemId = addItem('image', {
        type: 'image',
        prompt: prompt.trim(),
        preview: 'https://via.placeholder.com/256x256?text=Generated+Image',
      });

      // Show demo in preview
      showImage(itemId, 'https://via.placeholder.com/256x256?text=Generated+Image');

      addMessage({
        role: 'assistant',
        content: `[Image Demo] Generated placeholder for: "${prompt}"`,
      });

      toast.info('Image generation demo - backend coming soon');
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  return (
    <div className="p-3 space-y-3">
      {/* Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className={cn(
          "border-2 border-dashed rounded-lg p-3 text-center transition-colors",
          "border-sidebar-border hover:border-primary/50 cursor-pointer"
        )}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileUpload}
        />
        {seedImage ? (
          <div className="relative inline-block">
            <img src={seedImage} alt="Seed" className="max-h-20 rounded" />
            <Button
              variant="destructive"
              size="icon"
              className="absolute -top-2 -right-2 h-5 w-5"
              onClick={(e) => {
                e.stopPropagation();
                setSeedImage(null);
              }}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1 text-muted-foreground">
            <Upload className="h-5 w-5" />
            <span className="text-xs">Drop seed image or click</span>
          </div>
        )}
      </div>

      {/* Prompt */}
      <Textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe the image you want to create..."
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
            Generate Image
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
      {imageItems.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Session Cache</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-destructive"
              onClick={() => clearModule('image')}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          </div>
          <div className="grid grid-cols-3 gap-2 max-h-32 overflow-y-auto">
            {imageItems.slice(0, 6).map((item) => (
              <div
                key={item.id}
                className="relative aspect-square rounded bg-secondary overflow-hidden group"
              >
                {item.preview ? (
                  <img src={item.preview} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Image className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
                <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => downloadItem('image', item.id)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive"
                    onClick={() => removeItem('image', item.id)}
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

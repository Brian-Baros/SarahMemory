import { useState } from 'react';
import { ChevronDown, ChevronUp, Palette, Image, Music, Video, Loader2, Sparkles, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import { Progress } from '@/components/ui/progress';

type CreativeMode = 'image' | 'music' | 'video';

interface GeneratedResult {
  type: CreativeMode;
  url?: string;
  preview?: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  error?: string;
}

export function CreativeToolsPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<CreativeMode>('image');
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<GeneratedResult[]>([]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + 10, 90));
    }, 500);

    try {
      // Call the appropriate backend endpoint via edge function
      const response = await api.proxy.call(`/api/creative/${activeTab}`, {
        method: 'POST',
        body: {
          prompt: prompt.trim(),
          mode: activeTab,
        },
      });

      clearInterval(progressInterval);
      setProgress(100);

      const result: GeneratedResult = {
        type: activeTab,
        url: (response as any)?.url || (response as any)?.result_url,
        preview: (response as any)?.preview || (response as any)?.thumbnail,
        status: 'completed',
      };

      setResults(prev => [result, ...prev]);
      toast.success(`${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} generated!`);
    } catch (error) {
      clearInterval(progressInterval);
      console.error('Generation error:', error);
      
      // Add placeholder result for demo
      const demoResult: GeneratedResult = {
        type: activeTab,
        status: 'completed',
        preview: activeTab === 'image' 
          ? 'https://via.placeholder.com/200x200?text=Generated+Image'
          : undefined,
      };
      setResults(prev => [demoResult, ...prev]);
      toast.info('Creative tools demo - backend integration coming soon');
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  const getTabIcon = (mode: CreativeMode) => {
    switch (mode) {
      case 'image': return Image;
      case 'music': return Music;
      case 'video': return Video;
    }
  };

  return (
    <div className="border-b border-sidebar-border">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-sidebar-accent transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-sidebar-foreground">
          <Palette className="h-4 w-4" />
          Creative Tools
          <span className="px-1.5 py-0.5 text-xs bg-primary/20 text-primary rounded-full">
            AI
          </span>
        </span>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-3 pt-0 space-y-3 animate-fade-in">
          {/* Mode Tabs */}
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as CreativeMode)}>
            <TabsList className="w-full grid grid-cols-3 h-8">
              {(['image', 'music', 'video'] as CreativeMode[]).map((mode) => {
                const Icon = getTabIcon(mode);
                return (
                  <TabsTrigger 
                    key={mode} 
                    value={mode} 
                    className="text-xs flex items-center gap-1.5"
                  >
                    <Icon className="h-3 w-3" />
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </TabsTrigger>
                );
              })}
            </TabsList>

            <TabsContent value={activeTab} className="mt-3 space-y-3">
              {/* Prompt Input */}
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={`Describe the ${activeTab} you want to create...`}
                className="min-h-[80px] text-sm bg-sidebar-accent border-sidebar-border resize-none"
              />

              {/* Generate Button */}
              <Button 
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className="w-full h-9"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
                  </>
                )}
              </Button>

              {/* Progress Bar */}
              {isGenerating && (
                <div className="space-y-1">
                  <Progress value={progress} className="h-1" />
                  <p className="text-xs text-muted-foreground text-center">
                    {progress}% complete
                  </p>
                </div>
              )}
            </TabsContent>
          </Tabs>

          {/* Results */}
          {results.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Recent Generations</div>
              <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
                {results.slice(0, 4).map((result, index) => (
                  <div 
                    key={index}
                    className="relative aspect-square rounded-lg bg-secondary overflow-hidden group"
                  >
                    {result.preview ? (
                      <img 
                        src={result.preview} 
                        alt="Generated content"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        {result.type === 'image' && <Image className="h-6 w-6 text-muted-foreground" />}
                        {result.type === 'music' && <Music className="h-6 w-6 text-muted-foreground" />}
                        {result.type === 'video' && <Video className="h-6 w-6 text-muted-foreground" />}
                      </div>
                    )}
                    
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>

                    {/* Type badge */}
                    <div className="absolute top-1 left-1">
                      <span className="px-1.5 py-0.5 text-[10px] bg-background/80 rounded capitalize">
                        {result.type}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

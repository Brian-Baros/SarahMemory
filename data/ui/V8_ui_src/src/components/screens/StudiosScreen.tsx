import { useEffect, useState } from 'react';
import { Palette, Image, Music, Video, Mic, Layers } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import { useSarahStore } from '@/stores/useSarahStore';

// Creative production modules only. Communication belongs in SarahNet.
import { ImageGenerationModule } from '@/components/modules/ImageGenerationModule';
import { MusicSynthModule } from '@/components/modules/MusicSynthModule';
import { VoiceLyricsModule } from '@/components/modules/VoiceLyricsModule';
import { VideoStudioModule } from '@/components/modules/VideoStudioModule';

/**
 * Studios Screen - Creative Production Suite
 *
 * Domain boundary:
 * - Studios creates/edits media: art, music, sound/voice, video, canvas/composition.
 * - SarahNet handles communication, contacts, calls, node presence, and transfer.
 */
export function StudiosScreen() {
  const { addMessage } = useSarahStore();
  const [activeTab, setActiveTab] = useState<'image' | 'music' | 'voice' | 'video' | 'canvas'>('image');

  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions) || actions.length === 0) return;
      for (const a of actions) {
        if (!a || !a.type) continue;
        try {
          if (a.type === 'studio_set_tab') {
            const tab = String(a.payload?.tab || '');
            if (['image', 'music', 'voice', 'video', 'canvas'].includes(tab)) setActiveTab(tab as any);
          }
        } catch (e) {
          console.warn('[StudiosScreen] UI action failed:', a, e);
        }
      }
    };
    window.addEventListener('sarah:ui', handler as any);
    return () => window.removeEventListener('sarah:ui', handler as any);
  }, []);

  useEffect(() => {
    addMessage({ role: 'user', content: '[Studios] Opened Creative Studios' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    addMessage({ role: 'user', content: `[Studios] Tab: ${activeTab}` });
  }, [activeTab, addMessage]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="shrink-0 border-b border-border bg-card/70 p-4 backdrop-blur-sm">
        <div className="flex flex-wrap items-start gap-3">
          <div className="rounded-xl border border-primary/20 bg-primary/10 p-2">
            <Palette className="h-5 w-5 text-primary" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-lg font-semibold leading-tight">Creative Studios</h1>
            <p className="mt-1 text-xs text-muted-foreground">
              Art, music, sound, video, and canvas composition for chat/media generation workflows.
            </p>
          </div>
          <div className="hidden rounded-full border border-border bg-background/60 px-3 py-1 text-[11px] text-muted-foreground sm:block">
          </div>
        </div>
      </div>

      {/* Studio Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="flex min-h-0 flex-1 flex-col">
        <div className="shrink-0 border-b border-border bg-background/95 px-2 py-2">
          <TabsList className="grid h-auto w-full grid-cols-5 gap-1 bg-transparent">
            <TabsTrigger value="image" className="gap-1.5 data-[state=active]:bg-primary/10">
              <Image className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">Art</span>
            </TabsTrigger>
            <TabsTrigger value="music" className="gap-1.5 data-[state=active]:bg-primary/10">
              <Music className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">Music</span>
            </TabsTrigger>
            <TabsTrigger value="voice" className="gap-1.5 data-[state=active]:bg-primary/10">
              <Mic className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">Sound</span>
            </TabsTrigger>
            <TabsTrigger value="video" className="gap-1.5 data-[state=active]:bg-primary/10">
              <Video className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">Video</span>
            </TabsTrigger>
            <TabsTrigger value="canvas" className="gap-1.5 data-[state=active]:bg-primary/10">
              <Layers className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">Canvas</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <TabsContent value="image" className="m-0 p-0"><ImageGenerationModule /></TabsContent>
          <TabsContent value="music" className="m-0 p-0"><MusicSynthModule /></TabsContent>
          <TabsContent value="voice" className="m-0 p-0"><VoiceLyricsModule /></TabsContent>
          <TabsContent value="video" className="m-0 p-0"><VideoStudioModule /></TabsContent>
          <TabsContent value="canvas" className="m-0 p-4">
            <div className="grid gap-4 lg:grid-cols-[0.8fr_1.2fr]">
              <div className="rounded-xl border border-border bg-card p-4">
                <p className="flex items-center gap-2 text-sm font-medium"><Layers className="h-4 w-4 text-primary" /> Canvas Studio</p>
                <p className="mt-2 text-xs text-muted-foreground">
                  Canvas is the compositor lane for generated images, audio, and video. It should collect creative outputs, stage layers/timelines, and export final media to Chat and Media Player.
                </p>
                <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg border border-border bg-background/50 p-3"><p className="text-muted-foreground">Backend</p><p className="font-semibold">CanvasStudio</p></div>
                  <div className="rounded-lg border border-border bg-background/50 p-3"><p className="text-muted-foreground">Mode</p><p className="font-semibold">Composition</p></div>
                  <div className="rounded-lg border border-border bg-background/50 p-3"><p className="text-muted-foreground">Output</p><p className="font-semibold">Chat / Media</p></div>
                  <div className="rounded-lg border border-border bg-background/50 p-3"><p className="text-muted-foreground">Authority</p><p className="font-semibold">Backend</p></div>
                </div>
              </div>
              <div className="rounded-xl border border-dashed border-border bg-card/60 p-6 text-center">
                <Layers className="mx-auto mb-3 h-12 w-12 text-muted-foreground/40" />
                <p className="text-sm font-medium">Composition workspace</p>
                <p className="mx-auto mt-2 max-w-md text-xs text-muted-foreground">
                  The UI surface is prepared for layer/timeline integration. No fake canvas output is generated here; backend capability must provide real project/session state.
                </p>
              </div>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

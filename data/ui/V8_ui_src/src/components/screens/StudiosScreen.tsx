import { useEffect, useState } from 'react';
import { Palette, Image, Music, Video, Mic, MessageSquare } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import { useSarahStore } from '@/stores/useSarahStore';

// Reuse existing creative modules
import { ImageGenerationModule } from '@/components/modules/ImageGenerationModule';
import { MusicSynthModule } from '@/components/modules/MusicSynthModule';
import { VoiceLyricsModule } from '@/components/modules/VoiceLyricsModule';
import { VideoStudioModule } from '@/components/modules/VideoStudioModule';
import { CommunicationModule } from '@/components/modules/CommunicationModule';

/**
 * Studios Screen - Creative Studios Suite
 * Mobile-first panel for image/music/video/voice/communication
 */
export function StudiosScreen() {
  const { addMessage } = useSarahStore();
  const [activeTab, setActiveTab] = useState<
    'image' | 'music' | 'voice' | 'video' | 'communication'
  >('image');

  useEffect(() => {
    addMessage({ role: 'user', content: '[Studios] Opened Creative Studios' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    addMessage({ role: 'user', content: `[Studios] Tab: ${activeTab}` });
  }, [activeTab, addMessage]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Palette className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Creative Studios</h1>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Generate images, music, voice, video, and communicate
        </p>
      </div>

      {/* Studio Tabs */}
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as any)}
        className="flex-1 flex flex-col min-h-0"
      >
        <div className="shrink-0 border-b border-border px-2">
          <TabsList className="w-full h-12 bg-transparent justify-start gap-1">
            <TabsTrigger value="image" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Image className="h-4 w-4" />
              <span className="text-xs">Image</span>
            </TabsTrigger>
            <TabsTrigger value="music" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Music className="h-4 w-4" />
              <span className="text-xs">Music</span>
            </TabsTrigger>
            <TabsTrigger value="voice" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Mic className="h-4 w-4" />
              <span className="text-xs">Voice</span>
            </TabsTrigger>
            <TabsTrigger value="video" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Video className="h-4 w-4" />
              <span className="text-xs">Video</span>
            </TabsTrigger>
            <TabsTrigger
              value="communication"
              className="flex-1 gap-1.5 data-[state=active]:bg-primary/10"
            >
              <MessageSquare className="h-4 w-4" />
              <span className="text-xs">Comm</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          <TabsContent value="image" className="m-0 p-0">
            <ImageGenerationModule />
          </TabsContent>
          <TabsContent value="music" className="m-0 p-0">
            <MusicSynthModule />
          </TabsContent>
          <TabsContent value="voice" className="m-0 p-0">
            <VoiceLyricsModule />
          </TabsContent>
          <TabsContent value="video" className="m-0 p-0">
            <VideoStudioModule />
          </TabsContent>
          <TabsContent value="communication" className="m-0 p-0">
            <CommunicationModule />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

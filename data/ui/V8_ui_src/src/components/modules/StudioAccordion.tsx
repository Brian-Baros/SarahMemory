import { useState } from 'react';
import { ChevronDown, MessageSquare, Image, Music, Mic, Video } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { CommunicationModule } from './CommunicationModule';
import { ImageGenerationModule } from './ImageGenerationModule';
import { MusicSynthModule } from './MusicSynthModule';
import { VoiceLyricsModule } from './VoiceLyricsModule';
import { VideoStudioModule } from './VideoStudioModule';

interface AccordionModule {
  id: string;
  label: string;
  icon: React.ElementType;
  component: React.ComponentType;
}

const modules: AccordionModule[] = [
  { id: 'communication', label: 'Communication', icon: MessageSquare, component: CommunicationModule },
  { id: 'image', label: 'Image Generation', icon: Image, component: ImageGenerationModule },
  { id: 'music', label: 'Music Synth', icon: Music, component: MusicSynthModule },
  { id: 'voice', label: 'Voice / Lyrics', icon: Mic, component: VoiceLyricsModule },
  { id: 'video', label: 'Video Studio', icon: Video, component: VideoStudioModule },
];

export function StudioAccordion() {
  const [openModules, setOpenModules] = useState<string[]>(['communication']);

  const toggleModule = (id: string) => {
    setOpenModules((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  return (
    <div className="flex-1 overflow-y-auto">
      {modules.map((mod) => {
        const Icon = mod.icon;
        const isOpen = openModules.includes(mod.id);
        const Component = mod.component;

        return (
          <Collapsible
            key={mod.id}
            open={isOpen}
            onOpenChange={() => toggleModule(mod.id)}
          >
            <CollapsibleTrigger className="w-full flex items-center justify-between p-3 hover:bg-sidebar-accent transition-colors border-b border-sidebar-border">
              <span className="flex items-center gap-2 text-sm font-medium text-sidebar-foreground">
                <Icon className="h-4 w-4" />
                {mod.label}
              </span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  isOpen && "rotate-180"
                )}
              />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="border-b border-sidebar-border">
                <Component />
              </div>
            </CollapsibleContent>
          </Collapsible>
        );
      })}
    </div>
  );
}

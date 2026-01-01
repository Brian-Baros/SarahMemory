import { MessageSquare, Image, Music, Mic, Video, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useSarahStore } from '@/stores/useSarahStore';
import { useCreativeCacheStore } from '@/stores/useCreativeCacheStore';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

export type StudioModuleId = 'communication' | 'image' | 'music' | 'voice' | 'video';

const modules: { id: StudioModuleId; label: string; icon: React.ElementType }[] = [
  { id: 'communication', label: 'Comm', icon: MessageSquare },
  { id: 'image', label: 'Image', icon: Image },
  { id: 'music', label: 'Music', icon: Music },
  { id: 'voice', label: 'Voice', icon: Mic },
  { id: 'video', label: 'Video', icon: Video },
];

interface StudioModuleTabsProps {
  activeModule: StudioModuleId;
  onModuleChange: (id: StudioModuleId) => void;
}

export function StudioModuleTabs({ activeModule, onModuleChange }: StudioModuleTabsProps) {
  const resetStack = useCreativeCacheStore((s) => s.resetStack);

  const handleResetStack = () => {
    resetStack();
    toast.success('Session cache cleared');
  };

  return (
    <div className="flex items-center border-b border-sidebar-border bg-sidebar-accent/30">
      <div className="flex flex-1 overflow-x-auto">
        {modules.map((mod) => {
          const Icon = mod.icon;
          const isActive = activeModule === mod.id;
          
          return (
            <button
              key={mod.id}
              onClick={() => onModuleChange(mod.id)}
              className={cn(
                "flex-1 flex flex-col items-center gap-0.5 py-2 px-2 text-[10px] transition-all min-w-[48px]",
                "border-b-2 -mb-[1px]",
                isActive 
                  ? "text-primary border-primary bg-primary/5" 
                  : "text-muted-foreground border-transparent hover:text-foreground hover:bg-sidebar-accent"
              )}
            >
              <Icon className="h-4 w-4" />
              <span className="hidden xs:inline">{mod.label}</span>
            </button>
          );
        })}
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 mx-1 shrink-0"
        onClick={handleResetStack}
        title="Reset Stack (clear all cached items)"
      >
        <RotateCcw className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

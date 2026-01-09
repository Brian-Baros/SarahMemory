import { 
  MessageCircle, 
  Folder, 
  Play, 
  Search, 
  Palette, 
  Cpu, 
  Network, 
  User, 
  Settings,
  Clock,
  Heart
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface DockItem {
  id: WindowId;
  label: string;
  icon: React.ReactNode;
}

const DOCK_ITEMS: DockItem[] = [
  { id: "chat", label: "Chat", icon: <MessageCircle className="h-5 w-5" /> },
  { id: "history", label: "History", icon: <Clock className="h-5 w-5" /> },
  { id: "files", label: "Files", icon: <Folder className="h-5 w-5" /> },
  { id: "research", label: "Research", icon: <Search className="h-5 w-5" /> },
  { id: "studio", label: "Studios", icon: <Palette className="h-5 w-5" /> },
  { id: "avatar", label: "Avatar", icon: <User className="h-5 w-5" /> },
  { id: "sarahnet", label: "SarahNet", icon: <Network className="h-5 w-5" /> },
  { id: "media", label: "Media", icon: <Play className="h-5 w-5" /> },
  { id: "dlengine", label: "DL Engine", icon: <Cpu className="h-5 w-5" /> },
  { id: "settings", label: "Settings", icon: <Settings className="h-5 w-5" /> },
];

export function Dock() {
  const { windows, openWindow, focusWindow, restoreWindow, focusedWindowId } = useWindowStore();

  const handleDockClick = (id: WindowId) => {
    const win = windows.find((w) => w.id === id);
    if (!win) {
      openWindow(id);
    } else if (win.isMinimized) {
      restoreWindow(id);
    } else {
      focusWindow(id);
    }
  };

  return (
    <div className="h-14 bg-card/80 backdrop-blur-md border-t border-border flex items-center justify-center px-4 gap-1">
      <TooltipProvider delayDuration={200}>
        <div className="flex items-center gap-1 px-3 py-1.5 bg-secondary/50 rounded-xl">
          {DOCK_ITEMS.map((item) => {
            const isOpen = windows.some((w) => w.id === item.id);
            const isFocused = focusedWindowId === item.id;
            const isMinimized = windows.find((w) => w.id === item.id)?.isMinimized;

            return (
              <Tooltip key={item.id}>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => handleDockClick(item.id)}
                    className={cn(
                      "relative p-2.5 rounded-lg transition-all duration-150",
                      "hover:bg-primary/20 hover:scale-110",
                      "active:scale-95",
                      isFocused && "bg-primary/30",
                      isMinimized && "opacity-60"
                    )}
                  >
                    <span className={cn(
                      "text-muted-foreground",
                      isFocused && "text-primary",
                      isOpen && !isFocused && "text-foreground"
                    )}>
                      {item.icon}
                    </span>
                    
                    {/* Open indicator dot */}
                    {isOpen && (
                      <span className={cn(
                        "absolute bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full",
                        isFocused ? "bg-primary" : "bg-muted-foreground"
                      )} />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs">
                  {item.label}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>

        {/* Donate Button - always visible */}
        <div className="ml-4 border-l border-border pl-4">
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                href="https://patreon.com/sarahmemory"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-pink-500/20 to-rose-500/20 hover:from-pink-500/30 hover:to-rose-500/30 transition-all text-pink-400 hover:text-pink-300"
              >
                <Heart className="h-4 w-4" />
                <span className="text-xs font-medium">Donate</span>
              </a>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              Support SarahMemory on Patreon
            </TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>
    </div>
  );
}

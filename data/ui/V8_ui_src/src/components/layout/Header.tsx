import { Settings, User, Menu, SlidersHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useSarahStore } from '@/stores/useSarahStore';
import { useBackendVersion } from '@/hooks/useBackendVersion';

/**
 * Main header with mobile hamburger menu and settings
 * Shows different controls on mobile vs desktop
 */
export function Header() {
  const { setSettingsOpen, setLeftDrawerOpen, setRightDrawerOpen } = useSarahStore();
  const { version } = useBackendVersion();

  return (
    <header className="shrink-0 h-11 sm:h-14 flex items-center justify-between px-2 sm:px-4 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-20">
      {/* Left side - Logo and mobile menu */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Mobile left drawer trigger */}
        <Button 
          variant="ghost" 
          size="icon" 
          className="lg:hidden"
          onClick={() => setLeftDrawerOpen(true)}
          aria-label="Open chat history"
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        <div className="flex items-center gap-2">
          <span className="font-semibold text-foreground text-sm sm:text-base whitespace-nowrap">
            <span className="hidden sm:inline">The SarahMemory Project AiOS</span>
            <span className="sm:hidden">Sarah AiOS</span>
            {version && <span className="text-muted-foreground ml-1">v{version}</span>}
          </span>
        </div>
      </div>
      
      {/* Right side - Actions */}
      <div className="flex items-center gap-1 sm:gap-2">
        <Button 
          variant="ghost" 
          size="icon" 
          className="text-muted-foreground hover:text-foreground hidden sm:flex"
        >
          <User className="h-5 w-5" />
        </Button>
        <Button 
          variant="ghost" 
          size="icon" 
          className="text-muted-foreground hover:text-foreground"
          onClick={() => setSettingsOpen(true)}
          aria-label="Open settings"
        >
          <Settings className="h-5 w-5" />
        </Button>
        
        {/* Mobile right drawer trigger */}
        <Button 
          variant="ghost" 
          size="icon" 
          className="lg:hidden"
          onClick={() => setRightDrawerOpen(true)}
          aria-label="Open tools"
        >
          <SlidersHorizontal className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}

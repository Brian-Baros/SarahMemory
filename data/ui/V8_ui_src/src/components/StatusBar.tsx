import { useState, useEffect } from 'react';
import { ExternalLink, Heart, Github, Zap, Database, Globe, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { config } from '@/lib/config';
import { api } from '@/lib/api';
import { useSarahStore } from '@/stores/useSarahStore';

interface SystemStatus {
  local: boolean;
  web: boolean;
  api: boolean;
  network: boolean;
}

// Mode display map
const MODE_LABELS: Record<string, { label: string; icon: typeof Zap }> = {
  any: { label: 'Any', icon: Zap },
  local: { label: 'Local', icon: Database },
  web: { label: 'Web', icon: Globe },
  api: { label: 'API', icon: Cpu },
};

export function StatusBar() {
  const { settings } = useSarahStore();
  const [status, setStatus] = useState<SystemStatus>({
    local: true,
    web: true,
    api: false,
    network: true,
  });

  // Check API health on mount via edge function
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.proxy.call('/api/health');
        setStatus(prev => ({ 
          ...prev, 
          api: response && !(response as any).fallback 
        }));
      } catch {
        setStatus(prev => ({ ...prev, api: false }));
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Get current mode info
  const currentMode = MODE_LABELS[settings.mode || 'any'] || MODE_LABELS.any;
  const ModeIcon = currentMode.icon;

  return (
    <div className="h-12 bg-card/95 backdrop-blur-sm border-t border-border flex items-center justify-between px-4 shrink-0">
      {/* Left side - Mode indicator and Backend source link */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-muted-foreground">MODE:</span>
          <ModeIcon className="h-3.5 w-3.5 text-primary" />
          <span className="text-sm font-medium text-foreground">
            {currentMode.label}
          </span>
        </div>
        
        <div className="h-4 w-px bg-border" />
        
        <a
          href={config.githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <Github className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Backend Source: SarahMemory on GitHub</span>
          <span className="sm:hidden">GitHub</span>
          <ExternalLink className="h-3 w-3" />
        </a>
        
        <div className="h-4 w-px bg-border hidden sm:block" />
        
        <span className="hidden sm:inline text-xs text-muted-foreground">
          The SarahMemory Project by Brian Lee Baros
        </span>
      </div>

      {/* Right side - Donate button and Status LEDs */}
      <div className="flex items-center gap-4">
        {/* Donate Button */}
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-3 text-xs border-primary/30 hover:border-primary hover:bg-primary/10"
          asChild
        >
          <a
            href={config.donateUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5"
          >
            <Heart className="h-3 w-3 text-destructive" />
            <span>Donate</span>
          </a>
        </Button>

        <div className="h-4 w-px bg-border" />

        {/* Status LEDs */}
        <div className="flex items-center gap-2">
          <div 
            className={cn(
              "led led-pulse",
              status.local ? "bg-status-online" : "bg-muted-foreground/50"
            )} 
            title="Local System" 
          />
          <div 
            className={cn(
              "led led-pulse",
              status.web ? "bg-status-warning" : "bg-muted-foreground/50"
            )} 
            title="Web Connected" 
          />
          <div 
            className={cn(
              "led",
              status.api ? "bg-status-online led-pulse" : "bg-status-error"
            )} 
            title="API Status" 
          />
          <div 
            className={cn(
              "led led-pulse",
              status.network ? "bg-status-info" : "bg-muted-foreground/50"
            )} 
            title="Network" 
          />
        </div>
        <span className="text-sm text-muted-foreground">
          {status.api ? 'Ready' : 'Connecting...'}
        </span>
      </div>
    </div>
  );
}

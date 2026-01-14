import {
  Settings as SettingsIcon,
  Palette,
  Volume2,
  Bell,
  Shield,
  Wrench,
  Heart,
  ExternalLink,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useSarahStore } from '@/stores/useSarahStore';
import { useNavigationStore } from '@/stores/useNavigationStore';

/**
 * Settings Screen - App configuration
 * Themes, voice, notifications, privacy controls
 */
export function SettingsScreen() {
  const {
    settings,
    updateSettings,
    themes,
    voices,
    setSettingsOpen,
  } = useSarahStore();

  const { connectionStatus } = useNavigationStore();

  const handleThemeChange = (themeId: string) => {
    updateSettings({ selectedTheme: themeId });

    // Apply theme
    const theme = themes.find(t => t.id === themeId);
    if (theme) {
      // Keep consistent with SettingsModal (so we don't end up with multiple theme <link> tags)
      document.documentElement.setAttribute('data-theme', themeId);

      const existingLink = document.getElementById('sarah-theme-css') as HTMLLinkElement | null;
      if (existingLink) existingLink.remove();

      const link = document.createElement('link');
      link.id = 'sarah-theme-css';
      link.rel = 'stylesheet';

      const apiBase = import.meta.env.VITE_API_BASE_URL || '';
      const isBuiltIn = ['default', 'light', 'matrix', 'tron', 'hal2000', 'skynet', 'vibrant'].includes(themeId);
      if (apiBase && !isBuiltIn) {
        link.href = `${apiBase}/api/data/mods/themes/${theme.filename}`;
      } else {
        link.href = `/themes/${theme.filename}`;
      }

      document.head.appendChild(link);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <SettingsIcon className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Settings</h1>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          {/* Connection Status */}
          <div className="p-3 rounded-xl bg-card border border-border">
            <div className="flex items-center justify-between">
              <span className="text-sm">Connection Status</span>
              <span
                className={`text-sm font-medium capitalize ${
                  connectionStatus === 'connected'
                    ? 'text-green-500'
                    : connectionStatus === 'degraded'
                      ? 'text-yellow-500'
                      : 'text-red-500'
                }`}
              >
                {connectionStatus}
              </span>
            </div>
          </div>

          {/* Theme */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Palette className="h-4 w-4 text-muted-foreground" />
              <Label className="text-sm font-medium">Theme</Label>
            </div>
            <Select value={settings.selectedTheme} onValueChange={handleThemeChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select theme" />
              </SelectTrigger>
              <SelectContent className="z-[100000]">
                {themes.map((theme) => (
                  <SelectItem key={theme.id} value={theme.id}>
                    {theme.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Voice */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Volume2 className="h-4 w-4 text-muted-foreground" />
              <Label className="text-sm font-medium">Voice</Label>
            </div>
            <Select value={settings.selectedVoice} onValueChange={(v) => updateSettings({ selectedVoice: v })}>
              <SelectTrigger>
                <SelectValue placeholder="Select voice" />
              </SelectTrigger>
              <SelectContent className="z-[100000]">
                {voices.map((voice) => (
                  <SelectItem key={voice.id} value={voice.id}>
                    {voice.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Toggles */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <Label className="text-sm font-medium">Preferences</Label>
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="auto-speak" className="text-sm">
                Auto-speak responses
              </Label>
              <Switch id="auto-speak" checked={settings.autoSpeak} onCheckedChange={(v) => updateSettings({ autoSpeak: v })} />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="sound-effects" className="text-sm">
                Sound effects
              </Label>
              <Switch id="sound-effects" checked={settings.soundEffects} onCheckedChange={(v) => updateSettings({ soundEffects: v })} />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="notifications" className="text-sm">
                Notifications
              </Label>
              <Switch id="notifications" checked={settings.notifications} onCheckedChange={(v) => updateSettings({ notifications: v })} />
            </div>
          </div>

          {/* Advanced */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Wrench className="h-4 w-4 text-muted-foreground" />
              <Label className="text-sm font-medium">Advanced</Label>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="studio-mode" className="text-sm">
                  Studio Layout (Option B)
                </Label>
                <p className="text-xs text-muted-foreground mt-0.5">Accordion mode for power users</p>
              </div>
              <Switch
                id="studio-mode"
                checked={settings.advancedStudioMode ?? false}
                onCheckedChange={(v) => updateSettings({ advancedStudioMode: v })}
              />
            </div>
          </div>

          {/* Privacy */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-muted-foreground" />
              <Label className="text-sm font-medium">Privacy</Label>
            </div>
            <p className="text-xs text-muted-foreground">
              Your data stays on your device and the SarahMemory server. No third-party analytics or tracking.
            </p>
          </div>

          {/* Full Settings Modal */}
          <Button variant="outline" className="w-full" onClick={() => setSettingsOpen(true)}>
            Open Full Settings
          </Button>

          {/* Donate */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/20">
            <div className="flex items-center gap-2 mb-2">
              <Heart className="h-4 w-4 text-primary" />
              <span className="font-medium text-sm">Support SarahMemory</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">Help keep SarahMemory free and open source</p>
            <a href="https://github.com/sponsors/Brian-Baros" target="_blank" rel="noopener noreferrer">
              <Button variant="default" size="sm" className="w-full gap-2">
                <Heart className="h-4 w-4" />
                Donate / Sponsor
                <ExternalLink className="h-3 w-3" />
              </Button>
            </a>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

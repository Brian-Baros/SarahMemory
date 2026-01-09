import { useEffect, useState } from 'react';
import { Volume2, Palette, Bell, Sparkles, Play, Loader2, Zap, Globe, Database, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useSarahStore } from '@/stores/useSarahStore';
import { api } from '@/lib/api';
import { toast } from 'sonner';

// Mode options for data source - matches app.py api_mode setting
const MODES = [
  { id: 'any', name: 'Any', description: 'Use all available sources', icon: Zap },
  { id: 'local', name: 'Local', description: 'Local knowledge only', icon: Database },
  { id: 'web', name: 'Web', description: 'Web search augmented', icon: Globe },
  { id: 'api', name: 'API', description: 'AI API fallback only', icon: Cpu },
] as const;

// Default themes that ship with the app
const DEFAULT_THEMES = [
  { id: 'dark', name: 'Default Dark', filename: 'Dark_Theme.css' },
  { id: 'light', name: 'Light', filename: 'Light_Theme.css' },
  { id: 'matrix', name: 'Matrix', filename: 'Matrix_Theme.css' },
  { id: 'tron', name: 'Tron', filename: 'Tron.css' },
  { id: 'hal2000', name: 'HAL 2000', filename: 'HAL2000_Theme.css' },
  { id: 'skynet', name: 'Skynet', filename: 'Skynet_Theme.css' },
  { id: 'vibrant', name: 'Vibrant', filename: 'Vibrant_Theme.css' },
];

export function SettingsModal() {
  const { 
    settingsOpen, 
    setSettingsOpen, 
    settings, 
    updateSettings,
    voices,
    themes,
    setVoices,
    setThemes,
  } = useSarahStore();

  const [isLoadingVoices, setIsLoadingVoices] = useState(false);
  const [isPreviewingVoice, setIsPreviewingVoice] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [pendingVoice, setPendingVoice] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState(settings.mode || 'any');

  // Load voices and settings when modal opens
  useEffect(() => {
    if (settingsOpen) {
      loadVoices();
      loadThemes();
      loadMode();
      // Sync local state with store
      setSelectedMode(settings.mode || 'any');
    }
  }, [settingsOpen]);

  const loadVoices = async () => {
    setIsLoadingVoices(true);
    try {
      const backendVoices = await api.voice.listVoices();
      if (backendVoices.length > 0) {
        setVoices(backendVoices);
      }
    } catch (error) {
      console.error('Failed to load voices from backend:', error);
    } finally {
      setIsLoadingVoices(false);
    }
  };

  const loadThemes = async () => {
    try {
      const backendThemes = await api.settings.getThemes();
      if (backendThemes.length > 0) {
        setThemes(backendThemes);
      } else {
        // Use default themes if backend doesn't return any
        setThemes(DEFAULT_THEMES);
      }
    } catch (error) {
      console.error('Failed to load themes from backend:', error);
      // Fallback to default themes
      setThemes(DEFAULT_THEMES);
    }
  };

  // Load saved theme on mount
  useEffect(() => {
    const loadSavedTheme = async () => {
      try {
        const savedTheme = await api.settings.getSetting('theme');
        if (savedTheme) {
          updateSettings({ selectedTheme: savedTheme });
          applyTheme(savedTheme);
        }
      } catch (error) {
        console.error('Failed to load saved theme:', error);
      }
    };
    loadSavedTheme();
  }, []);

  const loadMode = async () => {
    try {
      // Try api_mode first (as used by app.py settings.json)
      let mode = await api.settings.getSetting('api_mode');
      if (!mode) {
        // Fallback to 'mode' key
        mode = await api.settings.getSetting('mode');
      }
      if (mode && MODES.some(m => m.id === mode)) {
        setSelectedMode(mode);
        updateSettings({ mode });
      }
    } catch (error) {
      console.error('Failed to load mode from backend:', error);
    }
  };

  const handleVoiceChange = async (voiceId: string) => {
    setPendingVoice(voiceId);
    updateSettings({ selectedVoice: voiceId });
  };

  const handleModeChange = (modeId: string) => {
    setSelectedMode(modeId);
    // Immediately update store so StatusBar reflects the change
    updateSettings({ mode: modeId });
  };

  const handlePreviewVoice = async () => {
    const voiceId = pendingVoice || settings.selectedVoice;
    setIsPreviewingVoice(true);
    
    try {
      const response = await api.voice.previewVoice(voiceId);
      
      if (response.success && response.audio_url) {
        const audio = new Audio(response.audio_url);
        await audio.play();
      } else if (response.audio_base64) {
        const audio = new Audio(`data:audio/mp3;base64,${response.audio_base64}`);
        await audio.play();
      } else if (response.fallback) {
        // Use browser TTS as fallback for preview
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance("Hello! This is how I sound. I'm ready to assist you.");
          
          // Try to find a matching voice in browser
          const browserVoices = speechSynthesis.getVoices();
          const selectedVoice = voices.find(v => v.id === voiceId);
          
          if (selectedVoice) {
            const matchingBrowserVoice = browserVoices.find(v => 
              v.name.toLowerCase().includes(selectedVoice.name.toLowerCase()) ||
              (selectedVoice.gender === 'female' && v.name.toLowerCase().includes('female')) ||
              (selectedVoice.gender === 'male' && v.name.toLowerCase().includes('male'))
            );
            if (matchingBrowserVoice) {
              utterance.voice = matchingBrowserVoice;
            }
          }
          
          speechSynthesis.speak(utterance);
          toast.info('Using browser voice preview');
        }
      }
    } catch (error) {
      console.error('Failed to preview voice:', error);
      toast.error('Could not preview voice');
    } finally {
      setIsPreviewingVoice(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    
    try {
      // Save voice to backend
      if (pendingVoice) {
        await api.settings.setVoice(pendingVoice);
      }
      
      // Save theme to backend
      await api.settings.setTheme(settings.selectedTheme);
      
      // Save mode to backend using api_mode key (app.py convention)
      await api.settings.setSetting('api_mode', selectedMode);
      // Also save under 'mode' for compatibility
      await api.settings.setSetting('mode', selectedMode);
      
      // Update store with final mode
      updateSettings({ mode: selectedMode });
      
      // Apply theme CSS
      applyTheme(settings.selectedTheme);
      
      toast.success('Settings saved');
      setSettingsOpen(false);
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('Settings saved locally (backend unavailable)');
      setSettingsOpen(false);
    } finally {
      setIsSaving(false);
      setPendingVoice(null);
    }
  };

  const handleThemeChange = (themeId: string) => {
    updateSettings({ selectedTheme: themeId });
    // Apply theme immediately for preview
    applyTheme(themeId);
  };

  const applyTheme = (themeId: string) => {
    // Find theme info
    const availableThemes = themes.length > 0 ? themes : DEFAULT_THEMES;
    const theme = availableThemes.find(t => t.id === themeId);
    
    if (theme) {
      // Set data-theme attribute
      document.documentElement.setAttribute('data-theme', themeId);
      
      // Try to load the CSS file dynamically
      const existingLink = document.getElementById('sarah-theme-css');
      if (existingLink) {
        existingLink.remove();
      }
      
      // Create new link for theme CSS
      // Try backend path first, then local public folder
      const link = document.createElement('link');
      link.id = 'sarah-theme-css';
      link.rel = 'stylesheet';
      
      // Use the API base URL for backend themes, or local for defaults
      const apiBase = import.meta.env.VITE_API_BASE_URL || '';
      if (apiBase && !DEFAULT_THEMES.some(t => t.id === themeId)) {
        link.href = `${apiBase}/api/data/mods/themes/${theme.filename}`;
      } else {
        link.href = `/themes/${theme.filename}`;
      }
      
      document.head.appendChild(link);
    }
  };

  // Display themes - prefer loaded themes, fallback to defaults
  const displayThemes = themes.length > 0 ? themes : DEFAULT_THEMES;

  return (
    <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
      <DialogContent className="sm:max-w-md bg-card border-border" aria-describedby="settings-dialog-description">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Settings
          </DialogTitle>
        </DialogHeader>
        <p id="settings-dialog-description" className="sr-only">
          Configure application settings including voice, theme, and preferences
        </p>

        <div className="space-y-6 py-4">
          {/* Mode Selection */}
          <div className="space-y-3">
            <Label className="flex items-center gap-2 text-sm font-medium">
              <Zap className="h-4 w-4 text-muted-foreground" />
              Mode
            </Label>
            <div className="grid grid-cols-4 gap-2">
              {MODES.map((mode) => {
                const Icon = mode.icon;
                const isSelected = selectedMode === mode.id;
                return (
                  <Button
                    key={mode.id}
                    variant={isSelected ? "default" : "outline"}
                    size="sm"
                    className={`flex flex-col items-center gap-1 h-auto py-2 ${
                      isSelected ? "bg-primary text-primary-foreground" : ""
                    }`}
                    onClick={() => handleModeChange(mode.id)}
                    title={mode.description}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="text-xs">{mode.name}</span>
                  </Button>
                );
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              {MODES.find(m => m.id === selectedMode)?.description}
            </p>
          </div>

          {/* Voice Selection */}
          <div className="space-y-3">
            <Label className="flex items-center gap-2 text-sm font-medium">
              <Volume2 className="h-4 w-4 text-muted-foreground" />
              Voice
            </Label>
            <div className="flex gap-2">
              <Select 
                value={pendingVoice || settings.selectedVoice} 
                onValueChange={handleVoiceChange}
                disabled={isLoadingVoices}
              >
                <SelectTrigger className="bg-secondary border-border flex-1">
                  {isLoadingVoices ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading voices...
                    </div>
                  ) : (
                    <SelectValue placeholder="Select a voice" />
                  )}
                </SelectTrigger>
                <SelectContent>
                  {voices.map((voice) => (
                    <SelectItem key={voice.id} value={voice.id}>
                      <div className="flex items-center gap-2">
                        <span>{voice.name}</span>
                        {voice.language && (
                          <span className="text-xs text-muted-foreground">({voice.language})</span>
                        )}
                        {voice.gender && (
                          <span className="text-xs text-muted-foreground capitalize">• {voice.gender}</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Button 
                variant="outline" 
                size="icon"
                onClick={handlePreviewVoice}
                disabled={isPreviewingVoice}
                title="Preview voice"
              >
                {isPreviewingVoice ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              This voice will be used for all spoken responses when auto-speak is enabled.
            </p>
          </div>

          {/* Theme Selection */}
          <div className="space-y-3">
            <Label className="flex items-center gap-2 text-sm font-medium">
              <Palette className="h-4 w-4 text-muted-foreground" />
              Theme
            </Label>
            <Select 
              value={settings.selectedTheme} 
              onValueChange={handleThemeChange}
            >
              <SelectTrigger className="bg-secondary border-border">
                <SelectValue placeholder="Select a theme" />
              </SelectTrigger>
              <SelectContent>
                {displayThemes.map((theme) => (
                  <SelectItem key={theme.id} value={theme.id}>
                    {theme.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Toggle Settings */}
          <div className="space-y-4 pt-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="auto-speak" className="flex items-center gap-2 text-sm">
                <Volume2 className="h-4 w-4 text-muted-foreground" />
                Auto-speak responses
              </Label>
              <Switch
                id="auto-speak"
                checked={settings.autoSpeak}
                onCheckedChange={(checked) => updateSettings({ autoSpeak: checked })}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="sound-effects" className="flex items-center gap-2 text-sm">
                <Sparkles className="h-4 w-4 text-muted-foreground" />
                Sound effects
              </Label>
              <Switch
                id="sound-effects"
                checked={settings.soundEffects}
                onCheckedChange={(checked) => updateSettings({ soundEffects: checked })}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="notifications" className="flex items-center gap-2 text-sm">
                <Bell className="h-4 w-4 text-muted-foreground" />
                Notifications
              </Label>
              <Switch
                id="notifications"
                checked={settings.notifications}
                onCheckedChange={(checked) => updateSettings({ notifications: checked })}
              />
            </div>

            {/* Advanced Studio Mode - hidden on mobile */}
            <div className="hidden sm:flex items-center justify-between">
              <div>
                <Label htmlFor="advanced-studio" className="flex items-center gap-2 text-sm">
                  <Sparkles className="h-4 w-4 text-muted-foreground" />
                  Advanced Studio Mode
                </Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Show modules in accordion layout (desktop)
                </p>
              </div>
              <Switch
                id="advanced-studio"
                checked={settings.advancedStudioMode ?? false}
                onCheckedChange={(checked) => updateSettings({ advancedStudioMode: checked })}
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 pt-4 border-t border-border">
          <Button variant="outline" onClick={() => setSettingsOpen(false)}>
            Close
          </Button>
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              'Save Changes'
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

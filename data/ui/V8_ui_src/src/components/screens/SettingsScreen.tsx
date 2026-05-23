import { useEffect, useState } from 'react';
import { Volume2, Palette, Bell, Sparkles, Play, Loader2, Zap, Globe, Database, Cpu, ShieldCheck, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useSarahStore } from '@/stores/useSarahStore';
import { useNavigationStore } from "@/stores/useNavigationStore";
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
// NOTE: keep ids aligned with store defaults (useSarahStore.ts)
const DEFAULT_THEMES = [
  { id: 'default', name: 'Default Dark', filename: 'Dark_Theme.css' },
  { id: 'light', name: 'Light', filename: 'Light_Theme.css' },
  { id: 'matrix', name: 'Matrix', filename: 'Matrix_Theme.css' },
  { id: 'tron', name: 'Tron', filename: 'Tron.css' },
  { id: 'hal2000', name: 'HAL 2000', filename: 'HAL2000_Theme.css' },
  { id: 'skynet', name: 'Skynet', filename: 'Skynet_Theme.css' },
  { id: 'vibrant', name: 'Vibrant', filename: 'Vibrant_Theme.css' },
];

const MODEL_FRIENDLY_CATEGORIES = [
  { id: 'reasoning', label: 'General Thinking', hint: 'Everyday questions and reasoning' },
  { id: 'coder', label: 'Coding Help', hint: 'Programming, debugging, and patch work' },
  { id: 'embeddings', label: 'Memory Search', hint: 'Finding meaning across local memory' },
  { id: 'vision', label: 'Vision / Camera', hint: 'Camera, object detection, and visual tasks' },
  { id: 'image_generation', label: 'Image Creation', hint: 'Drawing and image generation models' },
  { id: 'tts', label: 'Voice / Speech', hint: 'Spoken output models' },
] as const;

const MODEL_DOMAIN_OPTIONS = [
  { id: 'general', label: 'General' },
  { id: 'medical', label: 'Medical' },
  { id: 'legal', label: 'Legal' },
  { id: 'engineering', label: 'Engineering' },
  { id: 'finance', label: 'Finance' },
  { id: 'robotics', label: 'Robotics' },
  { id: 'manufacturing', label: 'Manufacturing' },
  { id: 'education', label: 'Education' },
  { id: 'creative', label: 'Creative' },
  { id: 'custom', label: 'Custom' },
] as const;

const MODEL_LIVE_SCAN_INTERVAL_MS = 30_000;

type SettingsPanelProps = {
  /** When true, renders as a regular screen (no Dialog wrapper) */
  embedded?: boolean;
  /** Optional close handler (used by modal close button). */
  onRequestClose?: () => void;
};

function SettingsPanelBody({ embedded = false, onRequestClose }: SettingsPanelProps) {
  const {
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
  const [devBridgeStatus, setDevBridgeStatus] = useState<any>(null);
  const [devBridgeError, setDevBridgeError] = useState<string>("");

  const [modelStatus, setModelStatus] = useState<any>(null);
  const [modelError, setModelError] = useState<string>("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isModelBusy, setIsModelBusy] = useState(false);
  const [selectedModelCategory, setSelectedModelCategory] = useState("reasoning");
  const [advancedModelControls, setAdvancedModelControls] = useState(false);
  const [externalModelPath, setExternalModelPath] = useState("");
  const [customModelRepo, setCustomModelRepo] = useState("");
  const [selectedModelDomain, setSelectedModelDomain] = useState("general");
  const [lastModelScanAt, setLastModelScanAt] = useState<string>("");

  // Load voices/themes/mode/model status on mount. Modal remounts when opened.
  useEffect(() => {
    loadVoices();
    loadThemes();
    loadMode();
    loadDevBridgeStatus();
    loadModelStatus(true);
    setSelectedMode(settings.mode || 'any');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [embedded]);

  // Keep the local model dropdown live while Settings is open.
  // Backend owns discovery; this UI only asks for refreshed status every 30 seconds.
  useEffect(() => {
    const id = window.setInterval(() => {
      void loadModelStatus(true, true);
    }, MODEL_LIVE_SCAN_INTERVAL_MS);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [embedded]);

  // ✅ Keep local selectedMode in sync with global store mode (StatusBar cycles, etc.)
  useEffect(() => {
    const m = (settings.mode || 'any') as string;
    if (m !== selectedMode) {
      setSelectedMode(m);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.mode]);

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

      // If backend returns a partial list (common), always merge with shipped defaults
      // so the user can still pick built-in themes.
      const merged = [...(backendThemes || [])];

      for (const def of DEFAULT_THEMES) {
        const existing = merged.find((t: any) => t?.id === def.id);
        if (!existing) merged.push(def as any);
        // If backend theme exists but doesn't include filename/name, fill from default.
        if (existing && !existing.filename) (existing as any).filename = def.filename;
        if (existing && !existing.name) (existing as any).name = def.name;
      }

      // De-dupe by id (backend sometimes duplicates)
      const uniq: any[] = [];
      const seen = new Set<string>();
      for (const t of merged) {
        const id = String((t as any)?.id || '');
        if (!id || seen.has(id)) continue;
        seen.add(id);
        uniq.push(t);
      }

      setThemes(uniq.length > 0 ? (uniq as any) : (DEFAULT_THEMES as any));
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadDevBridgeStatus = async () => {
    try {
      const status = await api.devbridge.status();
      setDevBridgeStatus(status);
      setDevBridgeError("");
    } catch (error: any) {
      setDevBridgeStatus(null);
      setDevBridgeError(String(error?.message || error || "DevBridge unavailable"));
    }
  };

  const loadModelStatus = async (refresh = true, silent = false) => {
    if (!silent) setIsLoadingModels(true);
    try {
      const status = await api.models.status(refresh);
      setModelStatus(status);
      setLastModelScanAt(new Date().toISOString());
      setModelError(status?.ok ? "" : String(status?.error || "Model manager unavailable"));
    } catch (error: any) {
      if (!silent) setModelStatus(null);
      setModelError(String(error?.message || error || "Model manager unavailable"));
    } finally {
      if (!silent) setIsLoadingModels(false);
    }
  };

  const handleModelScan = async () => {
    setIsModelBusy(true);
    try {
      const status = await api.models.scan();
      setModelStatus(status);
      setModelError("");
      toast.success("Model folders scanned");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Scan failed"));
      toast.error("Could not scan model folders");
    } finally {
      setIsModelBusy(false);
    }
  };

  const handleModelSelect = async (modelId: string) => {
    if (!modelId || modelId === "__none") return;
    const model = (modelStatus?.models || []).find((m: any) => String(m?.id) === String(modelId));
    setIsModelBusy(true);
    try {
      if (model && String(model.category || model.detected_category || "unknown") === "unknown") {
        await api.models.classify(
          modelId,
          selectedModelCategory,
          selectedModelDomain,
          String(model.adapter_type || ""),
          String(model.display_name || model.simple_label || ""),
        );
      }
      const result = await api.models.select(selectedModelCategory, modelId);
      if (result?.status) setModelStatus(result.status);
      else await loadModelStatus(false);
      toast.success("Active model updated");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Model selection failed"));
      toast.error("Could not set active model");
    } finally {
      setIsModelBusy(false);
    }
  };

  const handleModelVerify = async () => {
    const activeId = String(modelStatus?.active_models?.[selectedModelCategory] || "");
    if (!activeId) {
      toast.info("Pick a model first");
      return;
    }
    setIsModelBusy(true);
    try {
      const result = await api.models.verify(activeId);
      if (result?.status) setModelStatus(result.status);
      else await loadModelStatus(false);
      toast.success(result?.verified ? "Model verified" : "Model needs review");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Verification failed"));
      toast.error("Could not verify model");
    } finally {
      setIsModelBusy(false);
    }
  };

  const handleModelReset = async () => {
    setIsModelBusy(true);
    try {
      const result = await api.models.reset(selectedModelCategory);
      if (result?.status) setModelStatus(result.status);
      else await loadModelStatus(false);
      toast.success("Reset to recommended installed model");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Reset failed"));
      toast.error("No installed recommended model found");
    } finally {
      setIsModelBusy(false);
    }
  };

  const handleAddExternalPath = async () => {
    if (!externalModelPath.trim()) {
      toast.info("Enter a folder path first");
      return;
    }
    setIsModelBusy(true);
    try {
      const result = await api.models.addExternalPath(externalModelPath.trim());
      if (result?.status) setModelStatus(result.status);
      else await loadModelStatus(true);
      setExternalModelPath("");
      setModelError("");
      toast.success("External model folder linked");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Could not link folder"));
      toast.error("Could not link that folder");
    } finally {
      setIsModelBusy(false);
    }
  };

  const handleDownloadCustomModel = async () => {
    if (!customModelRepo.trim()) {
      toast.info("Enter a Hugging Face model name first");
      return;
    }
    setIsModelBusy(true);
    try {
      const result = await api.models.download(selectedModelCategory, customModelRepo.trim());
      if (result?.status) setModelStatus(result.status);
      else await loadModelStatus(true);
      toast.success("Model download requested");
    } catch (error: any) {
      setModelError(String(error?.message || error || "Download failed"));
      toast.error("Could not download model");
    } finally {
      setIsModelBusy(false);
    }
  };

  const loadMode = async () => {
    try {
      // Try api_mode first (as used by app.py settings.json)
      let mode = await api.settings.getSetting('api_mode');
      if (!mode) {
        // Fallback to 'mode' key
        mode = await api.settings.getSetting('mode');
      }
      if (mode && MODES.some((m) => m.id === mode)) {
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
          const selectedVoice = voices.find((v) => v.id === voiceId);

          if (selectedVoice) {
            const matchingBrowserVoice = browserVoices.find(
              (v) =>
                v.name.toLowerCase().includes(selectedVoice.name.toLowerCase()) ||
                (selectedVoice.gender === 'female' && v.name.toLowerCase().includes('female')) ||
                (selectedVoice.gender === 'male' && v.name.toLowerCase().includes('male')),
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

  const handleThemeChange = (themeId: string) => {
    updateSettings({ selectedTheme: themeId });
    // Apply theme immediately for preview
    applyTheme(themeId);
  };

  const applyTheme = (themeId: string) => {
    // Find theme info
    const availableThemes = themes.length > 0 ? themes : DEFAULT_THEMES;
    const theme = availableThemes.find((t: any) => t.id === themeId);

    if (theme) {
      // Set data-theme attribute
      document.documentElement.setAttribute('data-theme', themeId);

      // Try to load the CSS file dynamically
      const existingLink = document.getElementById('sarah-theme-css');
      if (existingLink) {
        existingLink.remove();
      }

      const link = document.createElement('link');
      link.id = 'sarah-theme-css';
      link.rel = 'stylesheet';

      // Use the API base URL for backend themes, or local for defaults
      const apiBase = import.meta.env.VITE_API_BASE_URL || '';
      if (apiBase && !DEFAULT_THEMES.some((t) => t.id === themeId)) {
        link.href = `${apiBase}/api/data/mods/themes/${theme.filename}`;
      } else {
        link.href = `/themes/${theme.filename}`;
      }

      document.head.appendChild(link);
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

      // If in modal, close it; if embedded, do nothing.
      onRequestClose?.();
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('Settings saved locally (backend unavailable)');
      onRequestClose?.();
    } finally {
      setIsSaving(false);
      setPendingVoice(null);
    }
  };

  // Display themes - prefer loaded themes, fallback to defaults
  const displayThemes = themes.length > 0 ? themes : DEFAULT_THEMES;

  const modelCategories = (modelStatus?.categories?.length ? modelStatus.categories : MODEL_FRIENDLY_CATEGORIES) as any[];
  const allModels = Array.isArray(modelStatus?.models) ? modelStatus.models : [];
  const liveScanSeconds = Number(modelStatus?.recommended_poll_interval_sec || modelStatus?.live_scan_interval_sec || 30);
  const modelCount = Number(modelStatus?.model_count ?? allModels.length);
  const readyModelCount = Number(modelStatus?.ready_count ?? allModels.filter((m: any) => String(m?.status || "") === "ready").length);
  const unclassifiedModelCount = Number(modelStatus?.unclassified_count ?? allModels.filter((m: any) => String(m?.category || "unknown") === "unknown").length);
  const lastModelScanLabel =
    lastModelScanAt ? new Date(lastModelScanAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }) : "waiting";
  const activeModelId = String(modelStatus?.active_models?.[selectedModelCategory] || "");
  const activeModel =
    modelStatus?.active_records?.[selectedModelCategory] ||
    allModels.find((m: any) => String(m?.id) === activeModelId) ||
    null;
  const selectedCategoryHint =
    MODEL_FRIENDLY_CATEGORIES.find((c) => c.id === selectedModelCategory)?.hint ||
    modelCategories.find((c: any) => c.id === selectedModelCategory)?.label ||
    "Choose what this model should help SarahMemory do.";
  const visibleModelOptions = allModels.filter((m: any) => {
    const c = String(m?.category || m?.detected_category || "unknown");
    return c === selectedModelCategory || c === "unknown" || String(m?.id) === activeModelId;
  });

  return (
    <div className={embedded ? "p-4" : "space-y-6 py-4"}>
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
                variant={isSelected ? 'default' : 'outline'}
                size="sm"
                className={`flex flex-col items-center gap-1 h-auto py-2 ${isSelected ? 'bg-primary text-primary-foreground' : ''}`}
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
          {MODES.find((m) => m.id === selectedMode)?.description}
        </p>
      </div>

      {/* AI Model Manager */}
      <div className="rounded-xl border border-border bg-secondary/20 p-3 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Label className="flex items-center gap-2 text-sm font-medium">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              AI Models
            </Label>
            <p className="text-xs text-muted-foreground mt-1">
              Choose what local model SarahMemory uses for each job. The backend scans and verifies the folders.
            </p>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Live folder watch: every {liveScanSeconds}s · Last scan: {lastModelScanLabel}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => void handleModelScan()} disabled={isLoadingModels || isModelBusy}>
            {isLoadingModels || isModelBusy ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
            Scan Now
          </Button>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">What should SarahMemory be better at?</Label>
            <Select value={selectedModelCategory} onValueChange={setSelectedModelCategory}>
              <SelectTrigger className="bg-secondary border-border">
                <SelectValue placeholder="Choose a job" />
              </SelectTrigger>
              <SelectContent className="z-[100000]">
                {modelCategories.map((cat: any) => (
                  <SelectItem key={cat.id} value={cat.id}>
                    {cat.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{selectedCategoryHint}</p>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Active model</Label>
            <Select value={activeModelId || "__none"} onValueChange={(value) => void handleModelSelect(value)} disabled={isModelBusy || isLoadingModels}>
              <SelectTrigger className="bg-secondary border-border">
                <SelectValue placeholder="Pick a model" />
              </SelectTrigger>
              <SelectContent className="z-[100000]">
                <SelectItem value="__none">No model selected</SelectItem>
                {visibleModelOptions.map((model: any) => (
                  <SelectItem key={model.id} value={String(model.id)}>
                    {(model.simple_label || model.display_name || model.id)}
                    {model.status_label ? ` · ${model.status_label}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {activeModel
                ? `${activeModel.simple_label || activeModel.display_name || activeModel.id} · ${activeModel.status_label || activeModel.status || "Ready"}`
                : "No active model selected for this job yet."}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Models Found</div>
            <div className="font-medium">{modelCount}</div>
          </div>
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Ready</div>
            <div className="font-medium">{readyModelCount}</div>
          </div>
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Review</div>
            <div className={modelError || unclassifiedModelCount > 0 ? "font-medium text-yellow-500" : "font-medium"}>
              {modelError ? "Error" : unclassifiedModelCount > 0 ? unclassifiedModelCount : "Clear"}
            </div>
          </div>
        </div>

        {modelError ? (
          <div className="flex items-start gap-2 text-xs text-yellow-500">
            <AlertTriangle className="h-3.5 w-3.5 mt-0.5" />
            <span>{modelError}</span>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Unknown models can be assigned to a job before use. New folders added to data/models appear automatically while this panel is open.
          </p>
        )}

        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => void handleModelVerify()} disabled={isModelBusy || !activeModelId}>
            Verify Active Model
          </Button>
          <Button variant="outline" size="sm" onClick={() => void handleModelReset()} disabled={isModelBusy}>
            Reset to Recommended
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setAdvancedModelControls(!advancedModelControls)}>
            {advancedModelControls ? "Hide Advanced" : "Advanced"}
          </Button>
        </div>

        {advancedModelControls && (
          <div className="rounded-lg border border-border bg-background/40 p-3 space-y-3">
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Domain</Label>
                <Select value={selectedModelDomain} onValueChange={setSelectedModelDomain}>
                  <SelectTrigger className="bg-secondary border-border">
                    <SelectValue placeholder="Choose a domain" />
                  </SelectTrigger>
                  <SelectContent className="z-[100000]">
                    {MODEL_DOMAIN_OPTIONS.map((domain) => (
                      <SelectItem key={domain.id} value={domain.id}>
                        {domain.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Registry</Label>
                <div className="rounded-md border border-border bg-secondary px-3 py-2 text-xs truncate">
                  {modelStatus?.registry_path || "data/settings/model_registry.json"}
                </div>
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Link a model folder from another drive</Label>
              <div className="flex gap-2">
                <input
                  value={externalModelPath}
                  onChange={(e) => setExternalModelPath(e.target.value)}
                  placeholder="D:\\AIModels\\Gemma4"
                  className="flex-1 rounded-md border border-border bg-secondary px-3 py-2 text-sm outline-none"
                />
                <Button variant="outline" size="sm" onClick={() => void handleAddExternalPath()} disabled={isModelBusy}>
                  Link
                </Button>
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Download a Hugging Face model into data/models</Label>
              <div className="flex gap-2">
                <input
                  value={customModelRepo}
                  onChange={(e) => setCustomModelRepo(e.target.value)}
                  placeholder="google/gemma-2-2b-it"
                  className="flex-1 rounded-md border border-border bg-secondary px-3 py-2 text-sm outline-none"
                />
                <Button variant="outline" size="sm" onClick={() => void handleDownloadCustomModel()} disabled={isModelBusy}>
                  Download
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Large downloads may be blocked by backend storage policy or require local confirmation.
              </p>
            </div>

            {activeModel?.path && (
              <div className="rounded-md border border-border bg-secondary px-3 py-2 text-xs break-all">
                Active folder: {activeModel.path}
              </div>
            )}
          </div>
        )}
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
            <SelectContent className="z-[100000]">
              {voices.map((voice) => (
                <SelectItem key={voice.id} value={voice.id}>
                  <div className="flex items-center gap-2">
                    <span>{voice.name}</span>
                    {voice.language && <span className="text-xs text-muted-foreground">({voice.language})</span>}
                    {voice.gender && <span className="text-xs text-muted-foreground capitalize">• {voice.gender}</span>}
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
            {isPreviewingVoice ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
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
        <Select value={settings.selectedTheme} onValueChange={handleThemeChange}>
          <SelectTrigger className="bg-secondary border-border">
            <SelectValue placeholder="Select a theme" />
          </SelectTrigger>
          <SelectContent className="z-[100000]">
            {displayThemes.map((theme: any) => (
              <SelectItem key={theme.id} value={theme.id}>
                {theme.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>


      {/* DeveloperMode / DevBridge Visibility */}
      <div className="rounded-xl border border-border bg-secondary/20 p-3 space-y-2">
        <Label className="flex items-center gap-2 text-sm font-medium">
          <ShieldCheck className="h-4 w-4 text-muted-foreground" />
          Developer Mode / DevBridge
        </Label>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Backend</div>
            <div className={devBridgeStatus?.ok ? "text-green-500 font-medium" : "text-yellow-500 font-medium"}>
              {devBridgeStatus?.ok ? "Reachable" : "Unavailable"}
            </div>
          </div>
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Apply Gate</div>
            <div className="font-medium">
              {devBridgeStatus?.apply_gate?.developer_mode ? "Developer" : "Normal"}
              {devBridgeStatus?.apply_gate?.loopback ? " / Loopback" : ""}
            </div>
          </div>
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Cmd Tickets</div>
            <div className="font-medium">
              P {Number(devBridgeStatus?.cmd_tickets?.pending || 0)} / F {Number(devBridgeStatus?.cmd_tickets?.failed || 0)} / Done {Number(devBridgeStatus?.cmd_tickets?.processed || 0)}
            </div>
          </div>
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2">
            <div className="text-muted-foreground">Repair Backlog</div>
            <div className="font-medium">
              T {Number(devBridgeStatus?.repair_counts?.tickets || 0)} / B {Number(devBridgeStatus?.repair_counts?.batches || 0)}
            </div>
          </div>
        </div>
        {devBridgeError ? (
          <div className="flex items-start gap-2 text-xs text-yellow-500">
            <AlertTriangle className="h-3.5 w-3.5 mt-0.5" />
            <span>{devBridgeError}</span>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            This panel is read-only. It reports DeveloperMode/DevBridge gates without enabling unsafe behavior from Normal Mode.
          </p>
        )}
        <Button variant="outline" size="sm" onClick={() => void loadDevBridgeStatus()}>
          Refresh DevBridge Status
        </Button>
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
            <p className="text-xs text-muted-foreground mt-0.5">Show modules in accordion layout (desktop)</p>
          </div>
          <Switch
            id="advanced-studio"
            checked={settings.advancedStudioMode ?? false}
            onCheckedChange={(checked) => updateSettings({ advancedStudioMode: checked })}
          />
        </div>
      </div>

      {/* Actions */}
      <div className={embedded ? "flex justify-end gap-2 pt-4 border-t border-border mt-6" : "flex justify-end gap-2 pt-4 border-t border-border"}>
        {!embedded && (
          <Button variant="outline" onClick={() => onRequestClose?.()}>
            Close
          </Button>
        )}
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
    </div>
  );
}

/**
 * ✅ Desktop Window version (always visible, no Dialog wrapper)
 */
export function SettingsScreen() {
  const { setCurrentScreen } = useNavigationStore();

  // ---------------------------------------------------------------------------
  // SarahMemory UI Control Bus listener (Chat-driven automation)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions) || actions.length === 0) return;

      for (const a of actions) {
        if (!a || !a.type) continue;
        try {
          if (a.type === "navigate" || a.type === "set_screen") {
            const screen = a.payload?.screen || a.payload?.route;
            if (typeof screen === "string" && screen) {
              const s = screen.replace(/^\//, "");
              if (s) setCurrentScreen(s as any);
            }
          }
        } catch (e) {
          // eslint-disable-next-line no-console
          console.warn("[SettingsScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [setCurrentScreen]);

  return (
    <div className="h-full w-full overflow-auto">
      <div className="px-4 pt-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Settings</h2>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Configure mode, AI models, voice, themes, and preferences.
        </p>
      </div>

      <SettingsPanelBody embedded />
    </div>
  );
}
/**
 * ✅ Modal version (kept for mobile / legacy triggers)
 */
export function SettingsModal() {
  const {
    settingsOpen,
    setSettingsOpen,
  } = useSarahStore();

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

        {/* Re-mount body when modal opens so it refreshes */}
        <SettingsPanelBody
          key={settingsOpen ? "open" : "closed"}
          embedded={false}
          onRequestClose={() => setSettingsOpen(false)}
        />
      </DialogContent>
    </Dialog>
  );
}

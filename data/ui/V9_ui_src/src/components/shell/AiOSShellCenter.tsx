import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AppWindow,
  Bell,
  CheckCircle2,
  Cpu,
  Eye,
  Folder,
  Gauge,
  Image as ImageIcon,
  LayoutGrid,
  MessageCircle,
  MonitorCog,
  Palette,
  PlayCircle,
  Search,
  Shield,
  SlidersHorizontal,
  Terminal,
  User,
  WifiOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useSarahStore } from "@/stores/useSarahStore";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import sarahLogoUrl from "@/assets/smaios-logo.jpg";

const APPS: { id: WindowId; label: string; icon: any; mode: "simple" | "operator" | "engineer" }[] = [
  { id: "chat", label: "Sarah Chat", icon: MessageCircle, mode: "simple" },
  { id: "history", label: "Memory Trail", icon: Bell, mode: "simple" },
  { id: "files", label: "File Cortex", icon: Folder, mode: "simple" },
  { id: "research", label: "Evidence Lens", icon: Search, mode: "simple" },
  { id: "avatar", label: "Avatar Core", icon: User, mode: "simple" },
  { id: "sarahnet", label: "SarahNet", icon: Shield, mode: "operator" },
  { id: "media", label: "Media Deck", icon: PlayCircle, mode: "operator" },
  { id: "studio", label: "Creation Bay", icon: Palette, mode: "operator" },
  { id: "dlengine", label: "Model Forge", icon: Cpu, mode: "engineer" },
  { id: "terminal", label: "Operator Terminal", icon: Terminal, mode: "engineer" },
  { id: "addons", label: "Addons", icon: LayoutGrid, mode: "engineer" },
  { id: "settings", label: "System Tuning", icon: SlidersHorizontal, mode: "simple" },
];

const WORKSPACES = [
  { id: "chat", label: "Dialogue Field", windows: "Sarah Chat + Memory Trail + Avatar Core" },
  { id: "research", label: "Evidence Field", windows: "Evidence Lens + File Cortex + Sarah Chat" },
  { id: "operator", label: "Operator Field", windows: "Avatar Core + SarahNet + Media Deck + System Tuning" },
  { id: "engineer", label: "Builder Field", windows: "Operator Terminal + Model Forge + Addons + System Tuning" },
  { id: "media", label: "Media Field", windows: "Creation Bay + Media Deck + File Cortex" },
] as const;

function modeRank(mode: string) {
  if (mode === "engineer") return 3;
  if (mode === "operator") return 2;
  return 1;
}

function numberSetting(settings: any, key: string, fallback: number) {
  const value = Number(settings?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

export function AiOSShellCenter({ status }: { status?: any }) {
  const { settings, updateSettings, mediaState, toggleWebcam, toggleMicrophone, toggleVoice } = useSarahStore();
  const { windows, openWindow, applyWorkspacePreset } = useWindowStore();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [contracts, setContracts] = useState<any>(null);
  const [runtime, setRuntime] = useState<any>(null);

  const uiMode = settings.uiMode || "simple";
  const localOnly = Boolean(settings.localOnlyMode);
  const visibleApps = useMemo(() => {
    const q = query.trim().toLowerCase();
    return APPS.filter((app) => modeRank(uiMode) >= modeRank(app.mode)).filter((app) => !q || app.label.toLowerCase().includes(q));
  }, [query, uiMode]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const load = async () => {
      const [c, r] = await Promise.all([
        api.proxy.call("/api/ui/contracts"),
        api.proxy.call("/api/runtime/thrash/status"),
      ]);
      if (cancelled) return;
      setContracts(c);
      setRuntime(r);
    };
    void load();
    return () => { cancelled = true; };
  }, [open]);

  const setMode = (mode: "simple" | "operator" | "engineer") => updateSettings({ uiMode: mode });
  const setWallpaper = (wallpaperUrl: string) => updateSettings({ wallpaperUrl });
  const setNumericAppearance = (key: string, value: number[]) => {
    updateSettings({ [key]: value[0] } as any);
  };
  const applyWorkspace = (preset: any) => {
    updateSettings({ activeWorkspace: preset });
    applyWorkspacePreset(preset);
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="sarah-focus-ring h-9 px-2 gap-2 rounded-lg" title="SarahMemory Command Nexus">
          <img src={sarahLogoUrl} alt="" className="h-6 w-6 rounded-md object-cover" />
          <span className="hidden lg:inline text-xs font-semibold">Nexus</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[92vw] sm:w-[520px] p-0 overflow-hidden">
        <SheetHeader className="p-4 border-b border-border bg-card/70 sarah-desktop-beacon">
          <SheetTitle className="flex items-center gap-2"><Shield className="h-5 w-5 text-primary" /> SarahMemory Command Nexus</SheetTitle>
          <SheetDescription>Launch surfaces, inspect gate state, tune the shell, and switch work fields.</SheetDescription>
        </SheetHeader>
        <Tabs defaultValue="launcher" className="flex flex-col h-[calc(100dvh-92px)]">
          <TabsList className="grid grid-cols-5 rounded-none border-b border-border bg-background/80 h-11">
            <TabsTrigger value="launcher">Launch</TabsTrigger>
            <TabsTrigger value="activity">Pulse</TabsTrigger>
            <TabsTrigger value="permissions">Gates</TabsTrigger>
            <TabsTrigger value="workspaces">Fields</TabsTrigger>
            <TabsTrigger value="appearance">Skin</TabsTrigger>
          </TabsList>
          <ScrollArea className="flex-1">
            <TabsContent value="launcher" className="m-0 p-4 space-y-4">
              <div className="flex gap-2">
                {(["simple", "operator", "engineer"] as const).map((m) => (
                  <Button key={m} variant={uiMode === m ? "default" : "outline"} size="sm" onClick={() => setMode(m)} className="capitalize flex-1">
                    {m === "simple" ? "core" : m === "engineer" ? "builder" : m}
                  </Button>
                ))}
              </div>
              <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search SarahMemory surfaces..." />
              <div className="grid grid-cols-2 gap-2">
                {visibleApps.map((app) => {
                  const Icon = app.icon;
                  const isOpen = windows.some((w) => w.id === app.id);
                  return (
                    <button key={app.id} onClick={() => openWindow(app.id)} className={cn("text-left rounded-xl border border-border bg-card/70 p-3 hover:bg-primary/10 transition-colors", isOpen && "ring-1 ring-primary/40")}>
                      <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-primary" /><span className="font-medium text-sm">{app.label}</span></div>
                      <p className="text-xs text-muted-foreground mt-1 capitalize">
                        {app.mode === "simple" ? "core" : app.mode === "engineer" ? "builder" : app.mode} surface
                      </p>
                    </button>
                  );
                })}
              </div>
            </TabsContent>
            <TabsContent value="activity" className="m-0 p-4 space-y-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Bridge Pulse</div><div className="font-medium">{status?.api ? "Linked" : "Degraded"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Route Ledger</div><div className="font-medium">{contracts?.route_count ?? "Unknown"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Broker Lock</div><div className="font-medium">{contracts?.doctrine?.one_way_broker ? "Locked" : "Unknown"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Thrash Guard</div><div className="font-medium">{runtime?.ok ? "Online" : "Unknown"}</div></div>
              </div>
              <div className="rounded-xl border border-border p-3 bg-card/60">
                <div className="flex items-center gap-2 font-medium text-sm"><Activity className="h-4 w-4 text-primary" /> Active Surfaces</div>
                <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                  {windows.length ? windows.map((w) => <div key={w.id}>{w.title} — {w.isMinimized ? "minimized" : w.isMaximized ? "maximized" : "open"}</div>) : <div>No active windows.</div>}
                </div>
              </div>
              <div className="rounded-xl border border-border p-3 bg-card/60">
                <div className="flex items-center gap-2 font-medium text-sm"><Gauge className="h-4 w-4 text-primary" /> Runtime Anti-Thrash</div>
                <pre className="mt-2 whitespace-pre-wrap text-[11px] text-muted-foreground">{JSON.stringify(runtime?.profile || runtime || {}, null, 2).slice(0, 1400)}</pre>
              </div>
            </TabsContent>
            <TabsContent value="permissions" className="m-0 p-4 space-y-3">
              {[
                ["Camera Gate / Eyes", mediaState.webcamEnabled, toggleWebcam, Eye],
                ["Microphone Gate / Ears", mediaState.microphoneEnabled, toggleMicrophone, Activity],
                ["Voice Output", mediaState.voiceEnabled, toggleVoice, User],
                ["Local-First Lock", localOnly, () => updateSettings({ localOnlyMode: !localOnly }), WifiOff],
              ].map(([label, checked, fn, Icon]: any) => (
                <div key={label} className="flex items-center justify-between rounded-xl border p-3 bg-card/60">
                  <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-primary" /><span className="text-sm font-medium">{label}</span></div>
                  <Switch checked={Boolean(checked)} onCheckedChange={() => fn()} />
                </div>
              ))}
              <div className="rounded-xl border p-3 bg-card/60 text-xs text-muted-foreground">
                Physical actions, terminal execution, external tools, MCP/A2A, and robotics remain governed by SMGET, SecurityGovernor, AssuranceGate, OperatorCore, and MSDC. This panel reflects user-facing gate intent; it does not bypass backend authority.
              </div>
            </TabsContent>
            <TabsContent value="workspaces" className="m-0 p-4 space-y-2">
              {WORKSPACES.map((w) => (
                <Button key={w.id} variant={settings.activeWorkspace === w.id ? "default" : "outline"} className="w-full justify-start h-auto py-3" onClick={() => applyWorkspace(w.id)}>
                  <div className="text-left"><div className="font-medium">{w.label}</div><div className="text-xs opacity-70">{w.windows}</div></div>
                </Button>
              ))}
            </TabsContent>
            <TabsContent value="appearance" className="m-0 p-4 space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2"><ImageIcon className="h-4 w-4" /> Workspace Image URL</label>
                <Input value={settings.wallpaperUrl || ""} onChange={(e) => setWallpaper(e.target.value)} placeholder="/assets/sarah-hero.jpeg or https://..." />
                <Button variant="outline" size="sm" onClick={() => setWallpaper("")}>Clear workspace image</Button>
              </div>
              <div className="rounded-xl border border-border bg-card/60 p-3 space-y-4">
                {[
                  ["Background brightness", "backgroundBrightness", 82, 20, 125, "%"],
                  ["Background overlay", "backgroundOverlay", 42, 0, 85, "%"],
                  ["Background blur", "backgroundBlur", 0, 0, 18, "px"],
                  ["Panel opacity", "panelOpacity", 82, 45, 100, "%"],
                ].map(([label, key, fallback, min, max, suffix]) => {
                  const value = numberSetting(settings, String(key), Number(fallback));
                  return (
                    <div key={String(key)} className="space-y-2">
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="font-medium">{String(label)}</span>
                        <span className="font-mono text-xs text-muted-foreground">{value}{String(suffix)}</span>
                      </div>
                      <Slider
                        value={[value]}
                        min={Number(min)}
                        max={Number(max)}
                        step={1}
                        onValueChange={(next) => setNumericAppearance(String(key), next)}
                      />
                    </div>
                  );
                })}
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Shell Density</label>
                <div className="grid grid-cols-3 gap-2">
                  {(["compact", "comfortable", "operator"] as const).map((density) => (
                    <Button key={density} size="sm" variant={settings.shellDensity === density ? "default" : "outline"} onClick={() => updateSettings({ shellDensity: density })} className="capitalize">{density}</Button>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border p-3 bg-card/60 text-xs text-muted-foreground">
                Themes remain loaded through the existing theme registry. This pass adds shell-level material tuning without replacing the current theme system.
              </div>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AppWindow,
  Bell,
  CheckCircle2,
  Cpu,
  Eye,
  Gauge,
  Image as ImageIcon,
  LayoutGrid,
  Lock,
  MonitorCog,
  Palette,
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
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useSarahStore } from "@/stores/useSarahStore";
import { useWindowStore, type WindowId } from "@/stores/useWindowStore";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const APPS: { id: WindowId; label: string; icon: any; mode: "simple" | "operator" | "engineer" }[] = [
  { id: "chat", label: "Chat", icon: AppWindow, mode: "simple" },
  { id: "history", label: "History", icon: Bell, mode: "simple" },
  { id: "files", label: "Files", icon: MonitorCog, mode: "simple" },
  { id: "research", label: "Research", icon: Search, mode: "simple" },
  { id: "avatar", label: "Avatar", icon: User, mode: "simple" },
  { id: "sarahnet", label: "SarahNet", icon: Shield, mode: "operator" },
  { id: "media", label: "Media", icon: Eye, mode: "operator" },
  { id: "studio", label: "Studios", icon: Palette, mode: "operator" },
  { id: "dlengine", label: "DL Engine", icon: Cpu, mode: "engineer" },
  { id: "terminal", label: "Terminal", icon: Terminal, mode: "engineer" },
  { id: "addons", label: "Addons", icon: LayoutGrid, mode: "engineer" },
  { id: "settings", label: "Settings", icon: SlidersHorizontal, mode: "simple" },
];

const WORKSPACES = [
  { id: "chat", label: "Chat Workspace", windows: "Chat + History + Avatar" },
  { id: "research", label: "Research Workspace", windows: "Research + Files + Chat" },
  { id: "operator", label: "Operator Workspace", windows: "Avatar + SarahNet + Media + Settings" },
  { id: "engineer", label: "Engineer Workspace", windows: "Terminal + DL Engine + Addons + Settings" },
  { id: "media", label: "Media Workspace", windows: "Studios + Media + Files" },
] as const;

function modeRank(mode: string) {
  if (mode === "engineer") return 3;
  if (mode === "operator") return 2;
  return 1;
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
  const applyWorkspace = (preset: any) => {
    updateSettings({ activeWorkspace: preset });
    applyWorkspacePreset(preset);
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 px-2 gap-1.5" title="SarahMemory AiOS Shell Center">
          <LayoutGrid className="h-4 w-4 text-primary" />
          <span className="hidden lg:inline text-xs">AiOS</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[92vw] sm:w-[520px] p-0 overflow-hidden">
        <SheetHeader className="p-4 border-b border-border bg-card/70">
          <SheetTitle className="flex items-center gap-2"><Shield className="h-5 w-5 text-primary" /> SarahMemory AiOS Center</SheetTitle>
          <SheetDescription>Launcher, permissions, activity, workspace, and runtime status.</SheetDescription>
        </SheetHeader>
        <Tabs defaultValue="launcher" className="flex flex-col h-[calc(100dvh-92px)]">
          <TabsList className="grid grid-cols-5 rounded-none border-b border-border bg-background/80 h-11">
            <TabsTrigger value="launcher">Apps</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
            <TabsTrigger value="permissions">Perms</TabsTrigger>
            <TabsTrigger value="workspaces">Spaces</TabsTrigger>
            <TabsTrigger value="appearance">Look</TabsTrigger>
          </TabsList>
          <ScrollArea className="flex-1">
            <TabsContent value="launcher" className="m-0 p-4 space-y-4">
              <div className="flex gap-2">
                {(["simple", "operator", "engineer"] as const).map((m) => (
                  <Button key={m} variant={uiMode === m ? "default" : "outline"} size="sm" onClick={() => setMode(m)} className="capitalize flex-1">{m}</Button>
                ))}
              </div>
              <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search apps and panels..." />
              <div className="grid grid-cols-2 gap-2">
                {visibleApps.map((app) => {
                  const Icon = app.icon;
                  const isOpen = windows.some((w) => w.id === app.id);
                  return (
                    <button key={app.id} onClick={() => openWindow(app.id)} className={cn("text-left rounded-xl border border-border bg-card/70 p-3 hover:bg-primary/10 transition-colors", isOpen && "ring-1 ring-primary/40")}>
                      <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-primary" /><span className="font-medium text-sm">{app.label}</span></div>
                      <p className="text-xs text-muted-foreground mt-1 capitalize">{app.mode} surface</p>
                    </button>
                  );
                })}
              </div>
            </TabsContent>
            <TabsContent value="activity" className="m-0 p-4 space-y-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Backend</div><div className="font-medium">{status?.api ? "Connected" : "Degraded"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Routes</div><div className="font-medium">{contracts?.route_count ?? "Unknown"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">One-way Broker</div><div className="font-medium">{contracts?.doctrine?.one_way_broker ? "Locked" : "Unknown"}</div></div>
                <div className="rounded-lg border p-3"><div className="text-muted-foreground">Runtime Guard</div><div className="font-medium">{runtime?.ok ? "Online" : "Unknown"}</div></div>
              </div>
              <div className="rounded-xl border border-border p-3 bg-card/60">
                <div className="flex items-center gap-2 font-medium text-sm"><Activity className="h-4 w-4 text-primary" /> Active Windows</div>
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
                ["Camera / Eyes", mediaState.webcamEnabled, toggleWebcam, Eye],
                ["Microphone / Ears", mediaState.microphoneEnabled, toggleMicrophone, Activity],
                ["Voice Output", mediaState.voiceEnabled, toggleVoice, User],
                ["Local Only / Airgap", localOnly, () => updateSettings({ localOnlyMode: !localOnly }), WifiOff],
              ].map(([label, checked, fn, Icon]: any) => (
                <div key={label} className="flex items-center justify-between rounded-xl border p-3 bg-card/60">
                  <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-primary" /><span className="text-sm font-medium">{label}</span></div>
                  <Switch checked={Boolean(checked)} onCheckedChange={() => fn()} />
                </div>
              ))}
              <div className="rounded-xl border p-3 bg-card/60 text-xs text-muted-foreground">
                Physical actions, terminal execution, external tools, MCP/A2A, and robotics remain governed by SMGET, SecurityGovernor, AssuranceGate, OperatorCore, and MSDC. This panel reflects user-facing permission intent; it does not bypass backend authority.
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
                <label className="text-sm font-medium flex items-center gap-2"><ImageIcon className="h-4 w-4" /> Wallpaper URL</label>
                <Input value={settings.wallpaperUrl || ""} onChange={(e) => setWallpaper(e.target.value)} placeholder="/assets/sarah-hero.jpeg or https://..." />
                <Button variant="outline" size="sm" onClick={() => setWallpaper("")}>Clear wallpaper</Button>
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
                Themes remain loaded through the existing theme registry. This pass adds shell-level wallpaper and density support without replacing the current theme system.
              </div>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

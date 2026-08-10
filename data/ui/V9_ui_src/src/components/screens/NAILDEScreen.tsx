import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Brain,
  CheckCircle2,
  Code2,
  Command,
  Cpu,
  Database,
  FileCode2,
  FileText,
  FolderTree,
  Gauge,
  GitCompareArrows,
  Hammer,
  Layers,
  Loader2,
  Lock,
  Maximize2,
  Minimize2,
  MonitorCog,
  Orbit,
  PackageCheck,
  PanelRightOpen,
  Play,
  RefreshCw,
  RotateCw,
  Save,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Terminal,
  Usb,
  Wand2,
  Workflow,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

interface FloatingWindowState {
  id: string;
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  open: boolean;
  minimized: boolean;
  maximized: boolean;
  dock?: string;
}

interface DragState {
  type: "move" | "resize";
  id: string;
  startX: number;
  startY: number;
  startWindow: FloatingWindowState;
}

type WindowsMap = Record<string, FloatingWindowState>;
type LogEntry = { ts: string; level: string; text: string };
type FileRow = { path: string; size?: number; mtime?: number; text_candidate?: boolean };

type MenuItem = {
  id?: string;
  label?: string;
  command?: string;
  shortcut?: string;
  requires?: string[];
};

type MenuGroup = { id?: string; label?: string; items?: MenuItem[] };

type ToolboxItem = {
  id?: string;
  category?: string;
  label?: string;
  kind?: string;
  icon?: string;
  description?: string;
  code_snippet?: string;
  tsx_snippet?: string;
  form_object?: Record<string, unknown>;
  block_object?: Record<string, unknown>;
  sandbox_only?: boolean;
  execution_authority?: boolean;
};

const TOOLBOX_MIME = "application/x-nailde-toolbox-item";

const DEFAULT_TOP_PROMPT = "Build a Pacman style game";
const DEFAULT_DETAILS_PROMPT =
  "Use keyboard and a game controller if one is found. Build in the sandbox, validate it, and prepare it for Addons after the user approves.";

const ACTIVITY_TO_WINDOW: Record<string, string> = {
  explorer: "explorer",
  search: "search",
  source: "diff",
  run_debug: "run_debug",
  extensions: "sdk",
  toolbox: "toolbox",
  database: "database_builder",
  agents: "agents",
  models: "model_bay",
  blocks: "blockforge",
  forms: "form_designer",
  devices: "device_bay",
  terminal: "terminal",
  filesystem: "filesystem",
  problems: "problems",
  settings: "settings",
  github: "github",
  holoforge: "holoforge",
  governance: "governance",
};

const PANEL_ICONS: Record<string, any> = {
  battle_plan: ShieldCheck,
  explorer: FolderTree,
  prompt: Wand2,
  editor: Code2,
  output: Terminal,
  sdk: Database,
  agents: Bot,
  weightlab: SlidersHorizontal,
  toolbox: Hammer,
  device_bay: Usb,
  holoforge: Orbit,
  validation: CheckCircle2,
  terminal: Terminal,
  search: Search,
  diff: GitCompareArrows,
  graph: Workflow,
  blockforge: Workflow,
  form_designer: Layers,
  model_bay: Cpu,
  governance: Lock,
  receipts: FileText,
  properties: PanelRightOpen,
  simulation: Gauge,
  run_debug: Play,
  database_builder: Database,
  filesystem: FolderTree,
  problems: CheckCircle2,
  settings: Lock,
  github: GitCompareArrows,
};

const DEFAULT_WINDOWS: WindowsMap = {
  battle_plan: { id: "battle_plan", title: "Battle Plan", x: 72, y: 82, w: 340, h: 480, z: 1, open: true, minimized: false, maximized: false, dock: "float" },
  explorer: { id: "explorer", title: "Sandbox Explorer", x: 88, y: 585, w: 360, h: 350, z: 2, open: true, minimized: false, maximized: false, dock: "float" },
  prompt: { id: "prompt", title: "Natural Language Build Prompt", x: 452, y: 82, w: 610, h: 268, z: 3, open: true, minimized: false, maximized: false, dock: "float" },
  editor: { id: "editor", title: "Code Editor", x: 1088, y: 82, w: 650, h: 590, z: 4, open: true, minimized: false, maximized: false, dock: "float" },
  output: { id: "output", title: "Output / Evidence", x: 452, y: 372, w: 610, h: 300, z: 5, open: true, minimized: false, maximized: false, dock: "float" },
  sdk: { id: "sdk", title: "SDK Library", x: 72, y: 958, w: 430, h: 300, z: 6, open: true, minimized: false, maximized: false, dock: "float" },
  agents: { id: "agents", title: "Agent Mission Bay", x: 526, y: 708, w: 440, h: 270, z: 7, open: true, minimized: false, maximized: false, dock: "float" },
  weightlab: { id: "weightlab", title: "WeightLab", x: 990, y: 708, w: 450, h: 270, z: 8, open: true, minimized: false, maximized: false, dock: "float" },
  toolbox: { id: "toolbox", title: "Visual Object Toolbox", x: 1462, y: 708, w: 360, h: 270, z: 9, open: true, minimized: false, maximized: false, dock: "float" },
  device_bay: { id: "device_bay", title: "Device Bay", x: 1462, y: 372, w: 360, h: 310, z: 10, open: true, minimized: false, maximized: false, dock: "float" },
  holoforge: { id: "holoforge", title: "HoloForge / XR", x: 990, y: 372, w: 450, h: 310, z: 11, open: true, minimized: false, maximized: false, dock: "float" },
  validation: { id: "validation", title: "Validation", x: 526, y: 1008, w: 640, h: 240, z: 12, open: true, minimized: false, maximized: false, dock: "float" },
  terminal: { id: "terminal", title: "Governed Terminal Output", x: 1188, y: 1008, w: 634, h: 240, z: 13, open: true, minimized: false, maximized: false, dock: "float" },
  search: { id: "search", title: "Workspace Search", x: 188, y: 174, w: 450, h: 300, z: 14, open: false, minimized: false, maximized: false, dock: "float" },
  diff: { id: "diff", title: "Diff Viewer", x: 740, y: 184, w: 600, h: 330, z: 15, open: false, minimized: false, maximized: false, dock: "float" },
  graph: { id: "graph", title: "Flow Graph", x: 500, y: 520, w: 560, h: 360, z: 16, open: false, minimized: false, maximized: false, dock: "float" },
  blockforge: { id: "blockforge", title: "BlockForge", x: 540, y: 550, w: 560, h: 360, z: 17, open: false, minimized: false, maximized: false, dock: "float" },
  form_designer: { id: "form_designer", title: "Form Designer", x: 580, y: 580, w: 560, h: 360, z: 18, open: false, minimized: false, maximized: false, dock: "float" },
  model_bay: { id: "model_bay", title: "Model Bay", x: 760, y: 230, w: 480, h: 300, z: 19, open: false, minimized: false, maximized: false, dock: "float" },
  governance: { id: "governance", title: "Governance Gates", x: 260, y: 230, w: 520, h: 340, z: 20, open: false, minimized: false, maximized: false, dock: "float" },
  receipts: { id: "receipts", title: "Ledger Receipts", x: 820, y: 280, w: 520, h: 320, z: 21, open: false, minimized: false, maximized: false, dock: "float" },
  properties: { id: "properties", title: "Properties", x: 1330, y: 190, w: 360, h: 330, z: 22, open: false, minimized: false, maximized: false, dock: "float" },
  simulation: { id: "simulation", title: "Simulation Media", x: 500, y: 250, w: 620, h: 340, z: 23, open: false, minimized: false, maximized: false, dock: "float" },
  run_debug: { id: "run_debug", title: "Run and Debug", x: 620, y: 250, w: 560, h: 330, z: 24, open: false, minimized: false, maximized: false, dock: "float" },
  database_builder: { id: "database_builder", title: "Access-Style Database Builder", x: 1180, y: 520, w: 470, h: 360, z: 25, open: false, minimized: false, maximized: false, dock: "float" },
  filesystem: { id: "filesystem", title: "Filesystem Map", x: 70, y: 220, w: 500, h: 420, z: 26, open: false, minimized: false, maximized: false, dock: "float" },
  problems: { id: "problems", title: "Problems / Bugs / Tasks", x: 560, y: 925, w: 640, h: 300, z: 27, open: false, minimized: false, maximized: false, dock: "float" },
  settings: { id: "settings", title: "NAILDE Settings", x: 1220, y: 190, w: 540, h: 420, z: 28, open: false, minimized: false, maximized: false, dock: "float" },
  github: { id: "github", title: "GitHub Sandbox Bridge", x: 1220, y: 640, w: 540, h: 360, z: 29, open: false, minimized: false, maximized: false, dock: "float" },
};

function nowIso() {
  return new Date().toISOString();
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value ?? "");
  }
}

function short(value: unknown, limit = 96): string {
  const s = String(value ?? "");
  return s.length > limit ? `${s.slice(0, limit)}…` : s;
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function buildLayoutPayload(windows: WindowsMap) {
  return {
    schema: "SarahMemory.nailde.floating_layout.v1",
    updated_at: nowIso(),
    windows,
    execution_authority: false,
  };
}

function StatCard({ icon: Icon, label, value, detail }: { icon: any; label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-lg border border-border bg-background/70 p-2 min-w-0">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        <span>{label}</span>
      </div>
      <div className="mt-1 truncate text-sm font-semibold">{value}</div>
      {detail ? <div className="mt-1 truncate text-[10px] text-muted-foreground">{detail}</div> : null}
    </div>
  );
}

function MiniButton({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <Button size="sm" variant="outline" className="h-7 px-2 text-xs" onClick={onClick} disabled={disabled}>
      {children}
    </Button>
  );
}

function LineNumberedEditor({
  value,
  onChange,
  onDrop,
  textareaRef,
}: {
  value: string;
  onChange: (value: string) => void;
  onDrop: (event: React.DragEvent<HTMLTextAreaElement>) => void;
  textareaRef: React.MutableRefObject<HTMLTextAreaElement | null>;
}) {
  const lines = Math.max(1, value.split("\n").length);
  const numbers = Array.from({ length: lines }, (_, index) => index + 1).join("\n");
  return (
    <div className="min-h-0 flex-1 overflow-hidden rounded-md border border-primary/30 bg-background/95">
      <div className="grid h-full grid-cols-[3.5rem_1fr]">
        <pre className="select-none overflow-hidden border-r border-border bg-muted/40 px-2 py-2 text-right font-mono text-xs leading-5 text-muted-foreground">
          {numbers}
        </pre>
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }}
          onDrop={onDrop}
          className="h-full min-h-0 resize-none rounded-none border-0 bg-[linear-gradient(to_right,transparent_0,transparent_calc(2ch-1px),hsl(var(--border)/0.22)_calc(2ch-1px),hsl(var(--border)/0.22)_2ch)] bg-[length:4ch_100%] font-mono text-xs leading-5 focus-visible:ring-0"
          spellCheck={false}
        />
      </div>
    </div>
  );
}

function WindowFrame({
  win,
  active,
  children,
  onFocus,
  onClose,
  onMinimize,
  onMaximize,
  onStartDrag,
  onStartResize,
}: {
  win: FloatingWindowState;
  active: boolean;
  children: React.ReactNode;
  onFocus: (id: string) => void;
  onClose: (id: string) => void;
  onMinimize: (id: string) => void;
  onMaximize: (id: string) => void;
  onStartDrag: (event: React.PointerEvent, id: string) => void;
  onStartResize: (event: React.PointerEvent, id: string) => void;
}) {
  if (!win.open || win.minimized) return null;
  const Icon = PANEL_ICONS[win.id] || PanelRightOpen;
  const style = win.maximized
    ? { left: 56, top: 40, width: "calc(100% - 72px)", height: "calc(100% - 72px)", zIndex: win.z }
    : { left: win.x, top: win.y, width: win.w, height: win.h, zIndex: win.z };
  return (
    <div
      className={cn(
        "absolute overflow-hidden rounded-xl border bg-card shadow-2xl backdrop-blur",
        active ? "border-primary/70 ring-1 ring-primary/30" : "border-border/80",
      )}
      style={style}
      onPointerDown={() => onFocus(win.id)}
    >
      <div
        className="flex h-9 cursor-move select-none items-center justify-between border-b border-border bg-muted/60 px-2"
        onPointerDown={(event) => onStartDrag(event, win.id)}
      >
        <div className="flex min-w-0 items-center gap-2">
          <Icon className="h-4 w-4 text-primary" />
          <span className="truncate text-xs font-semibold uppercase tracking-wide">{win.title}</span>
          <Badge variant="outline" className="h-5 px-1.5 text-[10px]">FLOAT</Badge>
        </div>
        <div className="flex items-center gap-1">
          <Button size="icon" variant="ghost" className="h-6 w-6" onPointerDown={(e) => e.stopPropagation()} onClick={() => onMinimize(win.id)}>
            <Minimize2 className="h-3.5 w-3.5" />
          </Button>
          <Button size="icon" variant="ghost" className="h-6 w-6" onPointerDown={(e) => e.stopPropagation()} onClick={() => onMaximize(win.id)}>
            <Maximize2 className="h-3.5 w-3.5" />
          </Button>
          <Button size="icon" variant="ghost" className="h-6 w-6" onPointerDown={(e) => e.stopPropagation()} onClick={() => onClose(win.id)}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
      <div className="h-[calc(100%-2.25rem)] overflow-hidden p-3">{children}</div>
      {!win.maximized ? (
        <div
          className="absolute bottom-0 right-0 h-5 w-5 cursor-se-resize border-b-2 border-r-2 border-primary/60"
          onPointerDown={(event) => onStartResize(event, win.id)}
        />
      ) : null}
    </div>
  );
}

export default function NAILDEScreen() {
  const [status, setStatus] = useState<any>(null);
  const [sdk, setSdk] = useState<any>(null);
  const [environment, setEnvironment] = useState<any>(null);
  const [toolbox, setToolbox] = useState<any>(null);
  const [toolboxSearch, setToolboxSearch] = useState("");
  const [workspaceId, setWorkspaceId] = useState("");
  const [goal, setGoal] = useState(DEFAULT_TOP_PROMPT);
  const [prompt, setPrompt] = useState(DEFAULT_DETAILS_PROMPT);
  const [filePath, setFilePath] = useState("sandbox/README.md");
  const [editorText, setEditorText] = useState("# NAILDE Sandbox\n\nCreate or load a sandbox file to edit here.\n");
  const [battlePlan, setBattlePlan] = useState("Requirements:\n- Natural language coding\n- AI-agent code proposals\n- Floating adjustable panels\n- Sandbox-only writes\n- Validation before staging\n- No production tensor/global weight edits\n");
  const [terminalText, setTerminalText] = useState("NAILDE governed terminal-output panel. Shell authority is false.\n");
  const [outputText, setOutputText] = useState("NAILDE output stream ready.\n");
  const [validationText, setValidationText] = useState("Validation not run.\n");
  const [filesystemText, setFilesystemText] = useState("Filesystem map not loaded.\n");
  const [problemsText, setProblemsText] = useState("No problems collected yet.\n");
  const [settingsText, setSettingsText] = useState("NAILDE settings not loaded.\n");
  const [githubRepo, setGithubRepo] = useState("");
  const [githubBranch, setGithubBranch] = useState("main");
  const [githubOperation, setGithubOperation] = useState("status");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchText, setSearchText] = useState("Search not run.\n");
  const [diffText, setDiffText] = useState("Diff not generated.\n");
  const [propertiesText, setPropertiesText] = useState("Select a panel/file/block to view editable properties.\n");
  const [files, setFiles] = useState<FileRow[]>([]);
  const [windows, setWindows] = useState<WindowsMap>(DEFAULT_WINDOWS);
  const [activeWindow, setActiveWindow] = useState("editor");
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [commandFilter, setCommandFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([{ ts: nowIso(), level: "info", text: "NAILDE Extreme Workbench initialized." }]);
  const [projectSeedHash, setProjectSeedHash] = useState("");
  const [lastTopPrompt, setLastTopPrompt] = useState(DEFAULT_TOP_PROMPT);
  const [workspaceDecision, setWorkspaceDecision] = useState<any | null>(null);
  const [postBuildPopup, setPostBuildPopup] = useState<any | null>(null);
  const [recoveryPopup, setRecoveryPopup] = useState<any | null>(null);
  const [noviceMode, setNoviceMode] = useState(true);
  const dragRef = useRef<DragState | null>(null);
  const editorRef = useRef<HTMLTextAreaElement | null>(null);

  const menus: MenuGroup[] = useMemo(() => (Array.isArray(environment?.menus) ? environment.menus : []), [environment]);
  const commands: any[] = useMemo(() => (Array.isArray(environment?.commands) ? environment.commands : []), [environment]);
  const filteredCommands = useMemo(() => {
    const q = commandFilter.trim().toLowerCase();
    if (!q) return commands.slice(0, 80);
    return commands.filter((item) => `${item.label || ""} ${item.command || ""} ${item.menu || ""}`.toLowerCase().includes(q)).slice(0, 80);
  }, [commands, commandFilter]);

  const toolboxItems = useMemo<ToolboxItem[]>(() => {
    const items: ToolboxItem[] = [];
    const direct = Array.isArray(toolbox?.items) ? toolbox.items : [];
    for (const item of direct) {
      if (item && typeof item === "object") items.push(item as ToolboxItem);
    }
    if (!items.length && Array.isArray(toolbox?.categories)) {
      for (const category of toolbox.categories) {
        const categoryId = String(category?.id || category?.label || "toolbox");
        for (const raw of category?.items || []) {
          if (typeof raw === "string") {
            items.push({ id: `${categoryId}.${raw}`, category: categoryId, label: raw, kind: "toolbox_item", code_snippet: `/* ${raw} */`, sandbox_only: true, execution_authority: false });
          } else if (raw && typeof raw === "object") {
            items.push({ category: categoryId, ...(raw as ToolboxItem) });
          }
        }
      }
    }
    const q = toolboxSearch.trim().toLowerCase();
    return items.filter((item) => {
      if (!q) return true;
      return `${item.label || ""} ${item.category || ""} ${item.kind || ""} ${item.description || ""}`.toLowerCase().includes(q);
    });
  }, [toolbox, toolboxSearch]);

  const addLog = useCallback((level: string, text: string) => {
    setLogs((prev) => [{ ts: nowIso(), level, text }, ...prev].slice(0, 200));
    setTerminalText((prev) => `[${new Date().toLocaleTimeString()}] ${level.toUpperCase()}: ${text}\n${prev}`.slice(0, 24000));
  }, []);

  const snippetForToolboxItem = useCallback((item: ToolboxItem): string => {
    const snippet = item.tsx_snippet || item.code_snippet;
    if (snippet) return String(snippet);
    if (item.form_object) return `// NAILDE form object\n${pretty(item.form_object)}`;
    if (item.block_object) return `// NAILDE block object\n${pretty(item.block_object)}`;
    return `/* NAILDE Toolbox Item: ${item.label || item.id || "unnamed"} */`;
  }, []);

  const insertSnippetAtCursor = useCallback((snippet: string, label = "Toolbox Item") => {
    const textarea = editorRef.current;
    const current = editorText;
    const insertion = `\n\n{/* NAILDE ${label} — sandbox-only insert */}\n${snippet}\n`;
    if (!textarea) {
      setEditorText((prev) => `${prev}${insertion}`);
      addLog("info", `Inserted ${label} at end of editor.`);
      return;
    }
    const start = textarea.selectionStart ?? current.length;
    const end = textarea.selectionEnd ?? current.length;
    const next = `${current.slice(0, start)}${insertion}${current.slice(end)}`;
    setEditorText(next);
    requestAnimationFrame(() => {
      try {
        textarea.focus();
        textarea.selectionStart = textarea.selectionEnd = start + insertion.length;
      } catch {
        // focus restoration is best-effort only
      }
    });
    addLog("info", `Dragged ${label} into sandbox editor.`);
  }, [addLog, editorText]);

  const handleToolboxDragStart = useCallback((event: React.DragEvent, item: ToolboxItem) => {
    const serialized = JSON.stringify(item);
    event.dataTransfer.setData(TOOLBOX_MIME, serialized);
    event.dataTransfer.setData("text/plain", snippetForToolboxItem(item));
    event.dataTransfer.effectAllowed = "copy";
  }, [snippetForToolboxItem]);

  const handleEditorDrop = useCallback((event: React.DragEvent<HTMLTextAreaElement>) => {
    const serialized = event.dataTransfer.getData(TOOLBOX_MIME);
    if (!serialized) return;
    event.preventDefault();
    try {
      const item = JSON.parse(serialized) as ToolboxItem;
      insertSnippetAtCursor(snippetForToolboxItem(item), String(item.label || item.id || "Toolbox Item"));
    } catch {
      insertSnippetAtCursor(event.dataTransfer.getData("text/plain") || "", "Toolbox Item");
    }
  }, [insertSnippetAtCursor, snippetForToolboxItem]);

  const focusWindow = useCallback((id: string) => {
    setActiveWindow(id);
    setWindows((prev) => {
      const maxZ = Math.max(1, ...Object.values(prev).map((w) => w.z || 1));
      const current = prev[id] || DEFAULT_WINDOWS[id];
      if (!current) return prev;
      return { ...prev, [id]: { ...current, open: true, minimized: false, z: maxZ + 1 } };
    });
  }, []);

  const openWindow = useCallback((id: string) => focusWindow(id), [focusWindow]);

  const closeWindow = useCallback((id: string) => {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], open: false } }));
  }, []);

  const minimizeWindow = useCallback((id: string) => {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], minimized: true } }));
  }, []);

  const maximizeWindow = useCallback((id: string) => {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], maximized: !prev[id]?.maximized, open: true, minimized: false } }));
  }, []);

  const resetLayout = useCallback(async () => {
    setWindows(DEFAULT_WINDOWS);
    try {
      await api.nailde.layout({ workspace_id: workspaceId || "__global__", action: "reset" });
    } catch {
      // local reset still works
    }
    addLog("info", "Floating layout reset.");
  }, [addLog, workspaceId]);

  const persistLayout = useCallback(async () => {
    try {
      await api.nailde.layout({ workspace_id: workspaceId || "__global__", action: "save", layout: buildLayoutPayload(windows) });
      addLog("info", "Floating layout saved to NAILDE UI state.");
    } catch (err) {
      addLog("warn", `Layout save failed: ${String(err)}`);
    }
  }, [addLog, windows, workspaceId]);

  const startDrag = useCallback((event: React.PointerEvent, id: string) => {
    const win = windows[id];
    if (!win) return;
    event.preventDefault();
    event.stopPropagation();
    focusWindow(id);
    dragRef.current = { type: "move", id, startX: event.clientX, startY: event.clientY, startWindow: { ...win } };
  }, [focusWindow, windows]);

  const startResize = useCallback((event: React.PointerEvent, id: string) => {
    const win = windows[id];
    if (!win) return;
    event.preventDefault();
    event.stopPropagation();
    focusWindow(id);
    dragRef.current = { type: "resize", id, startX: event.clientX, startY: event.clientY, startWindow: { ...win } };
  }, [focusWindow, windows]);

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      setWindows((prev) => {
        const current = prev[drag.id];
        if (!current) return prev;
        if (drag.type === "move") {
          return { ...prev, [drag.id]: { ...current, x: clamp(drag.startWindow.x + dx, 0, 3600), y: clamp(drag.startWindow.y + dy, 40, 2600) } };
        }
        return { ...prev, [drag.id]: { ...current, w: clamp(drag.startWindow.w + dx, 240, 2600), h: clamp(drag.startWindow.h + dy, 180, 2000) } };
      });
    };
    const onUp = () => { dragRef.current = null; };
    globalThis.addEventListener("pointermove", onMove);
    globalThis.addEventListener("pointerup", onUp);
    return () => {
      globalThis.removeEventListener("pointermove", onMove);
      globalThis.removeEventListener("pointerup", onUp);
    };
  }, []);

  const refreshCore = useCallback(async () => {
    setBusy(true);
    try {
      const [statusPacket, sdkPacket, envPacket, toolboxPacket, fsPacket, settingsPacket] = await Promise.all([
        api.nailde.status(),
        api.nailde.sdk(),
        api.nailde.environment(),
        api.nailde.toolbox(),
        api.nailde.filesystemStatus(),
        api.nailde.settings({ action: "load" }),
      ]);
      setStatus(statusPacket);
      setSdk(sdkPacket);
      setEnvironment(envPacket);
      setToolbox(toolboxPacket);
      setFilesystemText(pretty(fsPacket).slice(0, 24000));
      setSettingsText(pretty(settingsPacket).slice(0, 24000));
      const gh = (settingsPacket as any)?.settings?.github || {};
      if (gh.repo_url) setGithubRepo(String(gh.repo_url));
      if (gh.branch) setGithubBranch(String(gh.branch));
      setOutputText(pretty({ status: statusPacket, environment: envPacket }).slice(0, 24000));
      addLog("info", "NAILDE status, SDK, environment, and toolbox refreshed.");
    } catch (err) {
      addLog("error", `Refresh failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }, [addLog]);

  useEffect(() => {
    refreshCore();
  }, [refreshCore]);

  const loadLayout = useCallback(async (nextWorkspaceId?: string) => {
    try {
      const packet = await api.nailde.layout({ workspace_id: nextWorkspaceId || workspaceId || "__global__", action: "load" });
      const loaded = (packet as any)?.layout?.windows;
      if (loaded && typeof loaded === "object") {
        setWindows((prev) => ({ ...prev, ...loaded }));
        addLog("info", "Floating layout loaded from backend state.");
      }
    } catch (err) {
      addLog("warn", `Layout load failed: ${String(err)}`);
    }
  }, [addLog, workspaceId]);

  useEffect(() => {
    loadLayout("__global__");
  }, [loadLayout]);

  const refreshFiles = useCallback(async (id?: string) => {
    const wid = id || workspaceId;
    if (!wid) return;
    try {
      const packet = await api.nailde.files({ action: "list", workspace_id: wid });
      setFiles(Array.isArray((packet as any).files) ? (packet as any).files : []);
      addLog("info", `Workspace files refreshed: ${wid}`);
    } catch (err) {
      addLog("error", `File refresh failed: ${String(err)}`);
    }
  }, [addLog, workspaceId]);

  const postNailde = useCallback(async (path: string, body: Record<string, unknown>) => {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({ ok: false, error: "invalid_json_response" }));
    return data;
  }, []);

  const checkRecovery = useCallback(async () => {
    try {
      const response = await fetch('/api/nailde/workspace/recovery?action=latest');
      const packet = await response.json();
      if (packet?.restore_available && !workspaceId) {
        setRecoveryPopup(packet.latest || packet);
        addLog("info", "Recoverable NAILDE workspace found.");
      }
    } catch (err) {
      addLog("warn", `Recovery check failed: ${String(err)}`);
    }
  }, [addLog, workspaceId]);

  const restoreWorkspace = useCallback(async () => {
    try {
      const packet = await postNailde('/api/nailde/workspace/recovery', { action: 'restore' });
      const snapshot = packet?.snapshot || {};
      const wid = String(packet?.workspace_id || snapshot.workspace_id || "");
      if (wid) setWorkspaceId(wid);
      if (snapshot.top_prompt) setGoal(String(snapshot.top_prompt));
      if (snapshot.details_prompt) setPrompt(String(snapshot.details_prompt));
      if (snapshot.project_seed_hash) setProjectSeedHash(String(snapshot.project_seed_hash));
      if (snapshot.file_path) setFilePath(String(snapshot.file_path));
      if (snapshot.editor_text) setEditorText(String(snapshot.editor_text));
      if (snapshot.battle_plan) setBattlePlan(String(snapshot.battle_plan));
      setRecoveryPopup(null);
      setOutputText(pretty(packet).slice(0, 24000));
      addLog("info", `Restored NAILDE workspace: ${wid}`);
      await refreshFiles(wid);
    } catch (err) {
      addLog("error", `Restore failed: ${String(err)}`);
    }
  }, [addLog, postNailde, refreshFiles]);

  useEffect(() => {
    if (!workspaceId) void checkRecovery();
  }, [checkRecovery, workspaceId]);

  useEffect(() => {
    if (!workspaceId) return;
    const timer = globalThis.setTimeout(() => {
      void postNailde('/api/nailde/workspace/autosave', {
        workspace_id: workspaceId,
        top_prompt: goal,
        details_prompt: prompt,
        project_seed_hash: projectSeedHash,
        file_path: filePath,
        editor_text: editorText,
        battle_plan: battlePlan,
        dirty: true,
        status: 'ui_autosave',
      }).catch(() => undefined);
    }, 1800);
    return () => globalThis.clearTimeout(timer);
  }, [battlePlan, editorText, filePath, goal, postNailde, projectSeedHash, prompt, workspaceId]);

  const createWorkspace = useCallback(async () => {
    setBusy(true);
    try {
      const packet = await api.nailde.createWorkspace({ goal, mode: "PLAN" });
      const wid = String((packet as any)?.workspace?.workspace_id || "");
      setWorkspaceId(wid);
      setOutputText(pretty(packet));
      addLog("info", `Workspace created: ${wid}`);
      await refreshFiles(wid);
      await loadLayout(wid);
    } catch (err) {
      addLog("error", `Workspace create failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }, [addLog, goal, loadLayout, refreshFiles]);


  const runAutoBuild = useCallback(async (promptChangeDecision?: string) => {
    setBusy(true);
    try {
      const packet = await postNailde("/api/nailde/auto-build", {
        top_prompt: goal,
        details_prompt: prompt,
        current_workspace_id: workspaceId,
        current_project_seed_hash: projectSeedHash,
        previous_top_prompt: lastTopPrompt,
        previous_details_prompt: prompt,
        prompt_change_decision: promptChangeDecision || "",
        editor_text: editorText,
        file_path: filePath,
        novice_mode: noviceMode,
      });
      if (packet?.requires_workspace_decision) {
        setWorkspaceDecision(packet);
        addLog("warn", "Top prompt changed. Waiting for workspace save/discard decision before creating a new sandbox.");
        return;
      }
      if (!packet?.ok) {
        setOutputText(pretty(packet).slice(0, 30000));
        addLog("warn", `Auto-build blocked: ${String(packet?.error || packet?.message || "unknown")}`);
        return;
      }
      const wid = String(packet.workspace_id || "");
      setWorkspaceId(wid);
      setProjectSeedHash(String(packet.project_seed_hash || ""));
      setLastTopPrompt(goal);
      setBattlePlan(String(packet.battle_plan || battlePlan));
      const firstFile = Array.isArray(packet.files) ? packet.files.find((f: any) => String(f?.path || "").endsWith("README.md")) || packet.files[0] : null;
      if (firstFile?.path) setFilePath(String(firstFile.path));
      setOutputText(pretty(packet).slice(0, 30000));
      setValidationText(pretty(packet.validation || {}));
      setProblemsText(pretty({ problems: packet.validation?.problems || [], tasks: packet.validation?.tasks || [] }));
      setPostBuildPopup(packet.post_test_popup || null);
      addLog("info", `Auto-build completed from Natural Language Prompt: ${goal}`);
      await refreshFiles(wid);
      openWindow("battle_plan");
      openWindow("editor");
      openWindow("validation");
      openWindow("output");
    } catch (err) {
      addLog("error", `Auto-build failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }, [addLog, battlePlan, editorText, filePath, goal, lastTopPrompt, noviceMode, openWindow, postNailde, projectSeedHash, prompt, refreshFiles, workspaceId]);

  const resolveWorkspacePromptChange = useCallback(async (decision: string) => {
    setWorkspaceDecision(null);
    await runAutoBuild(decision);
  }, [runAutoBuild]);

  const draftFromLanguage = useCallback(async () => {
    setBusy(true);
    try {
      const packet = await api.nailde.codeDraft({ workspace_id: workspaceId, prompt, target: "extreme_workbench_addon" });
      const wid = String((packet as any)?.workspace_id || workspaceId);
      if (wid) setWorkspaceId(wid);
      const firstFile = Array.isArray((packet as any)?.files) ? (packet as any).files[0] : null;
      if (firstFile?.path) {
        setFilePath(String(firstFile.path));
        setEditorText(String(firstFile.content || ""));
      }
      setOutputText(pretty(packet).slice(0, 30000));
      setValidationText(pretty((packet as any)?.validation || {}));
      addLog("info", "Natural-language code draft generated into sandbox.");
      await refreshFiles(wid);
      openWindow("editor");
    } catch (err) {
      addLog("error", `Draft failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }, [addLog, openWindow, prompt, refreshFiles, workspaceId]);

  const scaffoldExtreme = useCallback(async () => {
    setBusy(true);
    try {
      const packet = await api.nailde.scaffold({ workspace_id: workspaceId, goal: prompt, target: "extreme_workbench_addon" });
      const wid = String((packet as any)?.workspace_id || workspaceId);
      if (wid) setWorkspaceId(wid);
      setOutputText(pretty(packet).slice(0, 30000));
      setValidationText(pretty((packet as any)?.validation || {}));
      addLog("info", "Extreme multi-surface scaffold created.");
      await refreshFiles(wid);
      openWindow("graph");
      openWindow("blockforge");
      openWindow("form_designer");
    } catch (err) {
      addLog("error", `Scaffold failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }, [addLog, openWindow, prompt, refreshFiles, workspaceId]);

  const saveEditor = useCallback(async () => {
    if (!workspaceId) {
      addLog("warn", "Create a workspace before saving sandbox files.");
      return;
    }
    try {
      const packet = await api.nailde.files({ action: "save", workspace_id: workspaceId, path: filePath, content: editorText });
      setOutputText(pretty(packet));
      addLog("info", `Saved sandbox file: ${filePath}`);
      await refreshFiles(workspaceId);
    } catch (err) {
      addLog("error", `Save failed: ${String(err)}`);
    }
  }, [addLog, editorText, filePath, refreshFiles, workspaceId]);

  const openFile = useCallback(async (path: string) => {
    if (!workspaceId) return;
    try {
      const packet = await api.nailde.files({ action: "read", workspace_id: workspaceId, path });
      if ((packet as any)?.ok) {
        setFilePath(path);
        setEditorText(String((packet as any).content || ""));
        setPropertiesText(pretty({ path, size: String((packet as any).content || "").length, sha256: (packet as any).sha256 }));
        openWindow("editor");
        openWindow("properties");
        addLog("info", `Opened sandbox file: ${path}`);
      } else {
        setOutputText(pretty(packet));
      }
    } catch (err) {
      addLog("error", `Open file failed: ${String(err)}`);
    }
  }, [addLog, openWindow, workspaceId]);

  const runThought = useCallback(async () => {
    try {
      const packet = await api.nailde.thoughtLoop({ workspace_id: workspaceId, problem: prompt, selected_object: filePath, max_ideas: 8 });
      setOutputText(pretty(packet).slice(0, 30000));
      addLog("info", "Thought Loop generated sandbox ideas.");
      openWindow("output");
    } catch (err) {
      addLog("error", `Thought Loop failed: ${String(err)}`);
    }
  }, [addLog, filePath, openWindow, prompt, workspaceId]);

  const runWeightLab = useCallback(async () => {
    try {
      const packet = await api.nailde.weightLabSimulate({ workspace_id: workspaceId, problem: prompt, category: "coder", max_candidates: 8 });
      setOutputText(pretty(packet).slice(0, 30000));
      addLog("info", "WeightLab simulated sandbox-only learning weights.");
      openWindow("weightlab");
    } catch (err) {
      addLog("error", `WeightLab failed: ${String(err)}`);
    }
  }, [addLog, openWindow, prompt, workspaceId]);

  const validateEditor = useCallback(async () => {
    try {
      const packet = await api.nailde.editorValidate({ workspace_id: workspaceId, path: filePath, content: editorText });
      setValidationText(pretty(packet));
      setProblemsText(pretty({ problems: (packet as any)?.problems || [], tasks: (packet as any)?.tasks || [] }));
      addLog((packet as any)?.ok ? "info" : "warn", "Editor syntax/indent diagnostics completed.");
      openWindow("validation");
      if (!((packet as any)?.ok)) openWindow("problems");
    } catch (err) {
      addLog("error", `Validation failed: ${String(err)}`);
    }
  }, [addLog, editorText, filePath, openWindow, workspaceId]);

  const createApplicationFromEditor = useCallback(async () => {
    try {
      const packet = await api.nailde.editorCreateApplication({ workspace_id: workspaceId, path: filePath, content: editorText, goal, prompt });
      const wid = String((packet as any)?.workspace_id || workspaceId);
      if (wid) setWorkspaceId(wid);
      setOutputText(pretty(packet).slice(0, 30000));
      setValidationText(pretty((packet as any)?.validation || {}));
      setFilesystemText(pretty((packet as any)?.filesystem || {}));
      setProblemsText(pretty({ problems: (packet as any)?.validation?.problems || [], tasks: (packet as any)?.validation?.tasks || [] }));
      addLog("info", "Sandbox application draft created from Code Editor.");
      await refreshFiles(wid);
      openWindow("output");
      openWindow("filesystem");
    } catch (err) {
      addLog("error", `Create application failed: ${String(err)}`);
    }
  }, [addLog, editorText, filePath, goal, openWindow, prompt, refreshFiles, workspaceId]);

  const installAddonFromSandbox = useCallback(async () => {
    try {
      const packet = await fetch('/api/nailde/addons/install-authorized', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace_id: workspaceId,
          confirm: true,
          confirmed: true,
          user_confirmed: true,
        }),
      }).then((response) => response.json());
      setOutputText(pretty(packet).slice(0, 30000));
      addLog(packet?.ok ? 'info' : 'warn', packet?.ok ? 'Installed sandbox addon icon into Addons folder.' : 'Addon install was blocked or failed.');
      openWindow('output');
      openWindow('filesystem');
    } catch (err) {
      addLog('error', `Install addon failed: ${String(err)}`);
    }
  }, [addLog, openWindow, workspaceId]);

  const handlePostBuildDecision = useCallback(async (action: string) => {
    if (action === "add_to_addons") {
      setPostBuildPopup(null);
      await installAddonFromSandbox();
      return;
    }
    try {
      const body: Record<string, unknown> = {
        action,
        workspace_id: workspaceId,
        top_prompt: goal,
        details_prompt: prompt,
        project_seed_hash: projectSeedHash,
        file_path: filePath,
        editor_text: editorText,
        battle_plan: battlePlan,
      };
      if (action === "save_as") {
        const suffix = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 12);
        body.new_workspace_id = `${workspaceId || "NAILDEWorkspace"}_copy_${suffix}`;
      }
      const packet = await postNailde('/api/nailde/workspace/decision', body);
      if (packet?.workspace_id && action === "save_as") setWorkspaceId(String(packet.workspace_id));
      setOutputText(pretty(packet).slice(0, 24000));
      addLog(packet?.ok ? "info" : "warn", `Post-build decision: ${action}`);
      setPostBuildPopup(null);
    } catch (err) {
      addLog("error", `Post-build decision failed: ${String(err)}`);
    }
  }, [addLog, battlePlan, editorText, filePath, goal, installAddonFromSandbox, postNailde, projectSeedHash, prompt, workspaceId]);

  const loadFilesystemMap = useCallback(async () => {
    try {
      const packet = await api.nailde.filesystemMap({ workspace_id: workspaceId, include_checksums: false, max_files: 250 });
      setFilesystemText(pretty(packet).slice(0, 30000));
      addLog("info", "Filesystem map refreshed for NAILDE sandbox.");
      openWindow("filesystem");
    } catch (err) {
      addLog("error", `Filesystem map failed: ${String(err)}`);
    }
  }, [addLog, openWindow, workspaceId]);

  const saveSettings = useCallback(async () => {
    try {
      const packet = await api.nailde.settings({
        action: "save",
        settings: {
          editor: { line_numbers: true, indent_grid: true, indent_size: 4, show_problems: true },
          filesystem: { show_sandbox_map: true, checksum_on_demand: true, live_files_read_only: true },
          github: { enabled: Boolean(githubRepo), repo_url: githubRepo, branch: githubBranch, auth_mode: "user_managed", store_tokens_here: false },
        },
      });
      setSettingsText(pretty(packet).slice(0, 24000));
      addLog("info", "NAILDE settings saved without secrets.");
      openWindow("settings");
    } catch (err) {
      addLog("error", `Settings save failed: ${String(err)}`);
    }
  }, [addLog, githubBranch, githubRepo, openWindow]);

  const planGithubOperation = useCallback(async () => {
    try {
      const packet = await api.nailde.githubPlan({ workspace_id: workspaceId, repo_url: githubRepo, branch: githubBranch, operation: githubOperation });
      setSettingsText(pretty(packet).slice(0, 30000));
      setOutputText(pretty(packet).slice(0, 30000));
      addLog("info", `GitHub ${githubOperation} plan prepared; no network execution.`);
      openWindow("github");
      await refreshFiles(String((packet as any)?.plan?.workspace_id || workspaceId));
    } catch (err) {
      addLog("error", `GitHub plan failed: ${String(err)}`);
    }
  }, [addLog, githubBranch, githubOperation, githubRepo, openWindow, refreshFiles, workspaceId]);

  const reconcileEditor = useCallback(async () => {
    try {
      const packet = await api.nailde.reconcile({ path: filePath, original: "", edited: editorText });
      setDiffText(String((packet as any)?.artifact?.diff || pretty(packet)));
      addLog("info", "Human edit reconciliation artifact generated.");
      openWindow("diff");
    } catch (err) {
      addLog("error", `Reconcile failed: ${String(err)}`);
    }
  }, [addLog, editorText, filePath, openWindow]);

  const prepareAgentMission = useCallback(async () => {
    try {
      const packet = await api.nailde.agentMission({ workspace_id: workspaceId, goal: prompt, target_files: [filePath], mission_type: "code_generation_review" });
      setOutputText(pretty(packet));
      addLog("info", "Agent mission prepared; not launched.");
      openWindow("agents");
    } catch (err) {
      addLog("error", `Agent mission failed: ${String(err)}`);
    }
  }, [addLog, filePath, openWindow, prompt, workspaceId]);

  const searchWorkspace = useCallback(async () => {
    if (!workspaceId || !searchQuery.trim()) {
      setSearchText("Create a workspace and enter a search query.\n");
      openWindow("search");
      return;
    }
    try {
      const packet = await api.nailde.search({ workspace_id: workspaceId, query: searchQuery });
      setSearchText(pretty(packet));
      addLog("info", `Search complete: ${searchQuery}`);
      openWindow("search");
    } catch (err) {
      addLog("error", `Search failed: ${String(err)}`);
    }
  }, [addLog, openWindow, searchQuery, workspaceId]);

  const sendAvatarMessage = useCallback(async () => {
    try {
      const packet = await api.nailde.avatarMessage({ workspace_id: workspaceId, message: "NAILDE is operating in sandbox-only development mode. Live apply is locked.", speak: true, level: "info" });
      setOutputText(pretty(packet));
      addLog("info", "NAILDE avatar channel message requested.");
    } catch (err) {
      addLog("error", `Avatar message failed: ${String(err)}`);
    }
  }, [addLog, workspaceId]);

  const runCommand = useCallback(async (item: MenuItem | any) => {
    const command = String(item?.command || item?.id || "");
    setActiveMenu(null);
    setCommandPaletteOpen(false);
    if (!command) return;
    const openMap: Record<string, string> = {
      show_explorer: "explorer",
      show_search: "search",
      show_flow_graph: "graph",
      show_blockforge: "blockforge",
      show_form_designer: "form_designer",
      show_device_bay: "device_bay",
      show_holoforge: "holoforge",
      show_output: "output",
      show_doctrine: "governance",
      show_shortcuts: "governance",
      show_boundaries: "governance",
    };
    if (openMap[command]) {
      openWindow(openMap[command]);
      return;
    }
    if (command === "create_workspace") return createWorkspace();
    if (command === "save_workspace_file") return saveEditor();
    if (command === "natural_language_code_draft") return draftFromLanguage();
    if (command === "thought_loop") return runThought();
    if (command === "weightlab_simulate") return runWeightLab();
    if (command === "reconcile_edits") return reconcileEditor();
    if (command === "prepare_agent_mission") return prepareAgentMission();
    try {
      const packet = await api.nailde.command({ command, workspace_id: workspaceId, args: { workspace_id: workspaceId, goal, prompt, path: filePath, content: editorText } });
      setOutputText(pretty(packet));
      addLog((packet as any)?.ok ? "info" : "warn", `Command result: ${command}`);
    } catch (err) {
      addLog("error", `Command failed ${command}: ${String(err)}`);
    }
  }, [addLog, createWorkspace, draftFromLanguage, editorText, filePath, goal, openWindow, prepareAgentMission, prompt, reconcileEditor, runAutoBuild, runThought, runWeightLab, saveEditor, workspaceId]);

  const panelContent = useCallback((id: string) => {
    switch (id) {
      case "battle_plan":
        return <Textarea value={battlePlan} onChange={(e) => setBattlePlan(e.target.value)} className="h-full resize-none font-mono text-xs" />;
      case "explorer":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="flex gap-2">
              <MiniButton onClick={() => refreshFiles()}>Refresh</MiniButton>
              <MiniButton onClick={createWorkspace}>New Workspace</MiniButton>
            </div>
            <div className="text-xs text-muted-foreground">Workspace: {workspaceId || "none"}</div>
            <ScrollArea className="min-h-0 flex-1 rounded-md border border-border bg-background/60 p-2">
              {files.length ? files.map((file) => (
                <button key={file.path} className="block w-full truncate rounded px-2 py-1 text-left text-xs hover:bg-muted" onClick={() => openFile(file.path)}>
                  <FileCode2 className="mr-1 inline h-3 w-3" />{file.path}
                </button>
              )) : <div className="p-2 text-xs text-muted-foreground">No sandbox files loaded.</div>}
            </ScrollArea>
          </div>
        );
      case "prompt":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="rounded-lg border border-primary/30 bg-primary/5 p-2">
              <div className="mb-1 flex items-center justify-between gap-2">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-primary">Top Natural Language Prompt — project seed</div>
                <label className="flex items-center gap-1 text-[10px] text-muted-foreground">
                  <input type="checkbox" checked={noviceMode} onChange={(event) => setNoviceMode(event.target.checked)} />
                  child-simple mode
                </label>
              </div>
              <Input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Example: Build a Pacman style game" className="h-9 text-sm" />
              <div className="mt-1 text-[10px] text-muted-foreground">Changing this top prompt creates a new workspace only after NAILDE asks what to do with the previous one.</div>
            </div>
            <div className="min-h-0 flex-1 rounded-lg border border-border bg-background/60 p-2">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Extra instructions for this workspace</div>
              <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} className="h-[calc(100%-1.25rem)] resize-none font-mono text-xs" placeholder="Example: Use keyboard and game controller if found; test it in the sandbox." />
            </div>
            <div className="flex flex-wrap gap-2">
              <MiniButton onClick={() => runAutoBuild()} disabled={busy}><Sparkles className="mr-1 h-3 w-3" />Build Automatically</MiniButton>
              <MiniButton onClick={draftFromLanguage} disabled={busy}><Wand2 className="mr-1 h-3 w-3" />Advanced Draft</MiniButton>
              <MiniButton onClick={scaffoldExtreme} disabled={busy}>Enterprise Scaffold</MiniButton>
              <MiniButton onClick={runThought}><Brain className="mr-1 h-3 w-3" />Think</MiniButton>
              <MiniButton onClick={runWeightLab}><SlidersHorizontal className="mr-1 h-3 w-3" />WeightLab</MiniButton>
              <MiniButton onClick={prepareAgentMission}><Bot className="mr-1 h-3 w-3" />Agent</MiniButton>
            </div>
          </div>
        );
      case "editor":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="flex gap-2">
              <Input value={filePath} onChange={(e) => setFilePath(e.target.value)} className="h-8 font-mono text-xs" />
              <MiniButton onClick={saveEditor}><Save className="mr-1 h-3 w-3" />Save</MiniButton>
              <MiniButton onClick={validateEditor}>Validate</MiniButton>
              <MiniButton onClick={reconcileEditor}>Reconcile</MiniButton>
              <MiniButton onClick={createApplicationFromEditor}>Create App</MiniButton>
              <MiniButton onClick={installAddonFromSandbox}><PackageCheck className="mr-1 h-3 w-3" />Install Addon</MiniButton>
              <MiniButton onClick={loadFilesystemMap}>FS Map</MiniButton>
            </div>
            <LineNumberedEditor
              value={editorText}
              onChange={setEditorText}
              onDrop={handleEditorDrop}
              textareaRef={editorRef}
            />
            <div className="rounded-md border border-dashed border-primary/40 bg-primary/5 px-2 py-1 text-[11px] text-muted-foreground">Drag Visual Toolbox controls here. Inserts sandbox snippets only; live files and global weights remain locked.</div>
          </div>
        );
      case "output":
        return <Textarea value={outputText} onChange={(e) => setOutputText(e.target.value)} className="h-full resize-none font-mono text-xs" spellCheck={false} />;
      case "terminal":
        return <Textarea value={terminalText} onChange={(e) => setTerminalText(e.target.value)} className="h-full resize-none font-mono text-xs" spellCheck={false} />;
      case "validation":
        return <Textarea value={validationText} onChange={(e) => setValidationText(e.target.value)} className="h-full resize-none font-mono text-xs" spellCheck={false} />;
      case "search":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="flex gap-2">
              <Input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Search sandbox text" className="h-8 text-xs" />
              <MiniButton onClick={searchWorkspace}>Search</MiniButton>
            </div>
            <Textarea value={searchText} onChange={(e) => setSearchText(e.target.value)} className="min-h-0 flex-1 resize-none font-mono text-xs" />
          </div>
        );
      case "diff":
        return <Textarea value={diffText} onChange={(e) => setDiffText(e.target.value)} className="h-full resize-none font-mono text-xs" />;
      case "sdk":
        return <Textarea value={pretty(sdk)} readOnly className="h-full resize-none font-mono text-xs" />;
      case "toolbox":
        return (
          <div className="flex h-full flex-col gap-2">
            <Input value={toolboxSearch} onChange={(e) => setToolboxSearch(e.target.value)} placeholder="Search toolbox controls" className="h-8 text-xs" />
            <div className="grid grid-cols-2 gap-1 text-[10px] text-muted-foreground">
              <Badge variant="outline" className="justify-center">VB-style controls</Badge>
              <Badge variant="outline" className="justify-center">Access-style data</Badge>
            </div>
            <ScrollArea className="min-h-0 flex-1 rounded-md border border-border bg-background/60 p-2">
              <div className="space-y-2">
                {toolboxItems.map((item) => (
                  <button
                    key={`${item.category}-${item.id}-${item.label}`}
                    draggable
                    onDragStart={(event) => handleToolboxDragStart(event, item)}
                    onDoubleClick={() => insertSnippetAtCursor(snippetForToolboxItem(item), String(item.label || item.id || "Toolbox Item"))}
                    className="block w-full rounded-md border border-border bg-card/80 p-2 text-left hover:border-primary/50 hover:bg-primary/5"
                    title="Drag into the Code Editor or double-click to insert"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold">{item.label || item.id}</span>
                      <Badge variant="secondary" className="h-5 px-1 text-[9px]">{item.category}</Badge>
                    </div>
                    <div className="mt-1 line-clamp-2 text-[10px] text-muted-foreground">{item.description || item.kind || "Sandbox toolbox template"}</div>
                  </button>
                ))}
                {!toolboxItems.length ? <div className="p-2 text-xs text-muted-foreground">No toolbox items loaded.</div> : null}
              </div>
            </ScrollArea>
            <div className="rounded-md border border-dashed border-border p-2 text-[10px] text-muted-foreground">Drag/drop inserts templates into the active sandbox editor only. It does not apply to live files.</div>
          </div>
        );
      case "filesystem":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="flex flex-wrap gap-2">
              <MiniButton onClick={loadFilesystemMap}>Refresh Map</MiniButton>
              <MiniButton onClick={() => openWindow("settings")}>Settings</MiniButton>
            </div>
            <Textarea value={filesystemText} onChange={(e) => setFilesystemText(e.target.value)} className="min-h-0 flex-1 resize-none font-mono text-xs" spellCheck={false} />
          </div>
        );
      case "problems":
        return <Textarea value={problemsText} onChange={(e) => setProblemsText(e.target.value)} className="h-full resize-none font-mono text-xs" spellCheck={false} />;
      case "settings":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="mb-1 text-[11px] text-muted-foreground">GitHub repository URL</div>
                <Input value={githubRepo} onChange={(e) => setGithubRepo(e.target.value)} placeholder="https://github.com/OWNER/REPO.git" className="h-8 text-xs" />
              </div>
              <div>
                <div className="mb-1 text-[11px] text-muted-foreground">Branch</div>
                <Input value={githubBranch} onChange={(e) => setGithubBranch(e.target.value)} placeholder="main" className="h-8 text-xs" />
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <MiniButton onClick={saveSettings}>Save Settings</MiniButton>
              <MiniButton onClick={() => openWindow("github")}>GitHub Bridge</MiniButton>
              <MiniButton onClick={loadFilesystemMap}>Filesystem Map</MiniButton>
            </div>
            <div className="rounded-md border border-dashed border-border p-2 text-[11px] text-muted-foreground">
              Settings store repo metadata only. Tokens/passwords/client secrets are not persisted by NAILDE.
            </div>
            <Textarea value={settingsText} onChange={(e) => setSettingsText(e.target.value)} className="min-h-0 flex-1 resize-none font-mono text-xs" spellCheck={false} />
          </div>
        );
      case "github":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="grid grid-cols-[1fr_9rem_8rem] gap-2">
              <Input value={githubRepo} onChange={(e) => setGithubRepo(e.target.value)} placeholder="Repository URL" className="h-8 text-xs" />
              <Input value={githubBranch} onChange={(e) => setGithubBranch(e.target.value)} placeholder="Branch" className="h-8 text-xs" />
              <select value={githubOperation} onChange={(e) => setGithubOperation(e.target.value)} className="h-8 rounded-md border border-border bg-background px-2 text-xs">
                <option value="status">status</option>
                <option value="clone">clone</option>
                <option value="fetch">fetch</option>
                <option value="pull">pull</option>
                <option value="push">push</option>
              </select>
            </div>
            <div className="flex flex-wrap gap-2">
              <MiniButton onClick={saveSettings}>Save Repo Metadata</MiniButton>
              <MiniButton onClick={planGithubOperation}>Prepare Push/Pull Plan</MiniButton>
            </div>
            <div className="rounded-md border border-dashed border-primary/40 bg-primary/5 p-2 text-[11px] text-muted-foreground">
              GitHub actions are plan-only in NAILDE. Actual clone/fetch/pull/push requires governed Terminal/DevBridge, credentials managed by the user, and explicit approval.
            </div>
            <Textarea value={settingsText} onChange={(e) => setSettingsText(e.target.value)} className="min-h-0 flex-1 resize-none font-mono text-xs" spellCheck={false} />
          </div>
        );
      case "agents":
        return <Textarea value={"Agent missions are prepared, not launched.\n\n" + pretty({ workspace_id: workspaceId, current_file: filePath, denied: ["shell", "live_file_write", "device_control", "self_approval"] })} readOnly className="h-full resize-none font-mono text-xs" />;
      case "weightlab":
        return <Textarea value={pretty(status?.weight_isolation || { sandbox_only: true, raw_tensor_edit: false, global_dlpanel_write: false })} readOnly className="h-full resize-none font-mono text-xs" />;
      case "device_bay":
        return <Textarea value={"READ_ONLY_DISCOVERY\n\nBlocked: upload_firmware, write_plc_logic, toggle_outputs, start_motors, arbitrary_serial_commands.\n\n" + pretty(status?.desktop_status || {})} readOnly className="h-full resize-none font-mono text-xs" />;
      case "database_builder":
        return (
          <div className="flex h-full flex-col gap-2">
            <div className="grid grid-cols-2 gap-2">
              <MiniButton onClick={() => insertSnippetAtCursor("CREATE TABLE IF NOT EXISTS items (\n  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  name TEXT NOT NULL,\n  notes TEXT,\n  created_ts TEXT\n);", "SQLite Local Table")}>Insert Table</MiniButton>
              <MiniButton onClick={() => insertSnippetAtCursor("// Bound form: items.name -> txtName\nconst binding = { table: 'items', field: 'name', control: 'txtName', sandboxOnly: true };", "Bound Form")}>Insert Binding</MiniButton>
            </div>
            <Textarea value={"Access-style builder:\n- Tables\n- Fields\n- Relationships\n- Forms\n- Queries\n- Reports\n\nDrag Data / Access toolbox items into the code editor. Generated database artifacts remain addon-local and sandbox-only."} readOnly className="min-h-0 flex-1 resize-none font-mono text-xs" />
          </div>
        );
      case "run_debug":
        return <Textarea value={"Run and Debug\n\nF5 = sandbox validation only.\nNo shell authority. No live apply. No hardware writes.\n\nAvailable runs:\n- Validate current text artifact\n- Thought Loop\n- WeightLab Simulation\n- Prepare Agent Mission\n- Compare/Assurance request packet"} readOnly className="h-full resize-none font-mono text-xs" />;
      case "holoforge":
        return <Textarea value={"HoloForge / XR Spatial Coding\n\nVR/XR object edits update the NAILDE sandbox graph/model only. No live file/device/model action authority.\n\nObjects: modules, routes, React panels, tables, agents, passports, models, device nodes, validation gates."} readOnly className="h-full resize-none font-mono text-xs" />;
      case "graph":
        return <Textarea value={"flowchart LR\n  UserPrompt --> NAILDEPlanner\n  NAILDEPlanner --> AgentMission\n  AgentMission --> SandboxCode\n  SandboxCode --> Validation\n  Validation --> Compare\n  Compare --> Ledger\n  Ledger --> UserApproval\n"} onChange={() => undefined} className="h-full resize-none font-mono text-xs" />;
      case "blockforge":
        return <Textarea value={"[WHEN Build Sandbox clicked]\n  -> [VALIDATE visible editor content]\n  -> [RUN Compare]\n  -> [RECORD Ledger receipt]\n  -> [REQUIRE user approval]\n\nBlocks are editable model objects; live code is not changed directly."} onChange={() => undefined} className="h-full resize-none font-mono text-xs" />;
      case "form_designer":
        return <Textarea value={"Object: txtGoal\nType: TextArea\nBound Field: project.goal\nRequired: true\n\nObject: btnBuild\nType: Button\nEvent: onClick -> Build Sandbox\n\nObject: gridFiles\nType: Table/Grid\nSource: workspace_files"} onChange={() => undefined} className="h-full resize-none font-mono text-xs" />;
      case "model_bay":
        return <Textarea value={pretty(status?.dl_status || {})} readOnly className="h-full resize-none font-mono text-xs" />;
      case "governance":
        return <Textarea value={"NAILDE Governance Gates\n\nPASS required before live apply:\n- Sandbox validation\n- Compare\n- Assurance/Security where relevant\n- Ledger receipt\n- Backup ZIP\n- User confirmation #1\n- User confirmation #2\n\nCurrent hard boundary:\n" + pretty(status?.denied_actions || [])} readOnly className="h-full resize-none font-mono text-xs" />;
      case "receipts":
        return <Textarea value={logs.map((l) => `${l.ts} ${l.level.toUpperCase()} ${l.text}`).join("\n")} readOnly className="h-full resize-none font-mono text-xs" />;
      case "properties":
        return <Textarea value={propertiesText} onChange={(e) => setPropertiesText(e.target.value)} className="h-full resize-none font-mono text-xs" />;
      case "simulation":
        return <Textarea value={"Simulation Media Panel\n\nSimulation is advisory and sandbox-only. Runtime validation remains NOT VERIFIED until tested. Hardware write is LOCKED. Live apply requires user approval."} readOnly className="h-full resize-none font-mono text-xs" />;
      default:
        return <Textarea value={`Panel ${id}`} readOnly className="h-full resize-none font-mono text-xs" />;
    }
  }, [battlePlan, busy, createWorkspace, diffText, draftFromLanguage, editorText, filePath, files, goal, handleEditorDrop, handleToolboxDragStart, insertSnippetAtCursor, logs, openFile, outputText, prepareAgentMission, prompt, propertiesText, reconcileEditor, refreshFiles, runAutoBuild, runThought, runWeightLab, saveEditor, scaffoldExtreme, sdk, searchQuery, searchText, searchWorkspace, snippetForToolboxItem, status, toolbox, toolboxItems, toolboxSearch, terminalText, validateEditor, validationText, workspaceId, createApplicationFromEditor, installAddonFromSandbox, filesystemText, githubBranch, githubOperation, githubRepo, loadFilesystemMap, planGithubOperation, problemsText, saveSettings, settingsText, noviceMode]);

  return (
    <div className="relative h-full min-h-[720px] overflow-hidden bg-background text-foreground">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,hsl(var(--primary)/0.18),transparent_28%),radial-gradient(circle_at_bottom_right,hsl(var(--accent)/0.14),transparent_30%)]" />

      <div className="relative z-[10000] flex h-10 items-center gap-2 border-b border-border bg-card/95 px-2 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 pr-3">
          <div className="grid h-7 w-7 place-items-center rounded bg-primary text-primary-foreground"><Code2 className="h-4 w-4" /></div>
          <div className="text-xs font-bold uppercase tracking-widest">NAILDE Extreme</div>
        </div>
        <div className="flex h-full items-center">
          {menus.map((menu) => (
            <div key={menu.id || menu.label} className="relative h-full">
              <button className="h-full px-3 text-xs hover:bg-muted" onClick={() => setActiveMenu(activeMenu === menu.id ? null : String(menu.id || ""))}>
                {menu.label}
              </button>
              {activeMenu === menu.id ? (
                <div className="absolute left-0 top-10 z-[12000] w-72 rounded-md border border-border bg-popover p-1 shadow-xl">
                  {(menu.items || []).map((item) => (
                    <button key={item.id || item.label} className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs hover:bg-muted" onClick={() => runCommand(item)}>
                      <span>{item.label}</span>
                      {item.shortcut ? <span className="text-[10px] text-muted-foreground">{item.shortcut}</span> : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => setCommandPaletteOpen(true)}><Command className="mr-1 h-3 w-3" />Command</Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={persistLayout}>Save Layout</Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={resetLayout}>Reset</Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={refreshCore} disabled={busy}>{busy ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <RefreshCw className="mr-1 h-3 w-3" />}Refresh</Button>
        </div>
      </div>

      <div className="absolute left-0 top-10 z-[9999] flex h-[calc(100%-4.5rem)] w-12 flex-col items-center gap-2 border-r border-border bg-card/80 py-2 backdrop-blur">
        {Object.entries(ACTIVITY_TO_WINDOW).map(([activity, panel]) => {
          const Icon = PANEL_ICONS[panel] || Workflow;
          const isOpen = Boolean(windows[panel]?.open && !windows[panel]?.minimized);
          return (
            <button key={activity} className={cn("grid h-9 w-9 place-items-center rounded-md hover:bg-muted", isOpen && "bg-primary/15 text-primary")} onClick={() => openWindow(panel)} title={activity}>
              <Icon className="h-4 w-4" />
            </button>
          );
        })}
        <div className="mt-auto flex flex-col gap-2">
          <button className="grid h-9 w-9 place-items-center rounded-md hover:bg-muted" onClick={() => openWindow("receipts")} title="Receipts"><FileText className="h-4 w-4" /></button>
          <button className="grid h-9 w-9 place-items-center rounded-md hover:bg-muted" onClick={sendAvatarMessage} title="Avatar"><MonitorCog className="h-4 w-4" /></button>
        </div>
      </div>

      <div className="absolute left-12 right-0 top-10 bottom-8 overflow-auto">
        <div className="relative h-[1300px] min-w-[1850px]">
          <div className="absolute left-6 top-4 right-6 grid grid-cols-6 gap-2">
            <StatCard icon={ShieldCheck} label="Sandbox" value={status?.sandbox_first ? "ON" : "UNKNOWN"} detail="live files read-only" />
            <StatCard icon={Lock} label="Execution" value={status?.execution_authority ? "AUTH" : "FALSE"} detail="NAILDE cannot self-apply" />
            <StatCard icon={SlidersHorizontal} label="Weights" value="ISOLATED" detail="sandbox only" />
            <StatCard icon={Database} label="SDK" value={String(status?.sdk_count ?? "—")} detail="capability adapters" />
            <StatCard icon={FolderTree} label="Workspace" value={workspaceId || "none"} detail="sandbox write zone" />
            <StatCard icon={Cpu} label="DL" value={status?.dl_status?.ok ? "READY" : "UNKNOWN"} detail="no raw tensor edit" />
          </div>
          {Object.values(windows).map((win) => (
            <WindowFrame
              key={win.id}
              win={win}
              active={activeWindow === win.id}
              onFocus={focusWindow}
              onClose={closeWindow}
              onMinimize={minimizeWindow}
              onMaximize={maximizeWindow}
              onStartDrag={startDrag}
              onStartResize={startResize}
            >
              {panelContent(win.id)}
            </WindowFrame>
          ))}
        </div>
      </div>

      <div className="absolute left-12 right-0 bottom-0 z-[10000] flex h-8 items-center gap-3 border-t border-border bg-card/95 px-3 text-[11px] text-muted-foreground backdrop-blur">
        <span>Workspace: {workspaceId || "none"}</span>
        <span>Sandbox: ON</span>
        <span>Live Write: BLOCKED</span>
        <span>Shell: FALSE</span>
        <span>Device Write: BLOCKED</span>
        <span>Production Tensor Edit: FALSE</span>
        <span>Global DLScreen Write: USER ONLY</span>
        <span className="ml-auto">{busy ? "Working…" : "Ready"}</span>
      </div>

      {Object.values(windows).filter((win) => win.open && win.minimized).length ? (
        <div className="absolute bottom-9 left-16 z-[12000] flex flex-wrap gap-1 rounded-md border border-border bg-card/95 p-1 shadow-xl">
          {Object.values(windows).filter((win) => win.open && win.minimized).map((win) => (
            <button key={win.id} className="rounded border border-border px-2 py-1 text-[11px] hover:bg-muted" onClick={() => focusWindow(win.id)}>{win.title}</button>
          ))}
        </div>
      ) : null}

      {commandPaletteOpen ? (
        <div className="absolute inset-0 z-[15000] bg-background/60 backdrop-blur-sm" onClick={() => setCommandPaletteOpen(false)}>
          <div className="mx-auto mt-20 w-[720px] rounded-xl border border-border bg-popover p-3 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="mb-2 flex items-center gap-2">
              <Command className="h-4 w-4 text-primary" />
              <Input value={commandFilter} onChange={(e) => setCommandFilter(e.target.value)} placeholder="Type a NAILDE command…" className="h-9" autoFocus />
            </div>
            <ScrollArea className="h-[420px]">
              {filteredCommands.map((item) => (
                <button key={`${item.id}-${item.command}`} className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm hover:bg-muted" onClick={() => runCommand(item)}>
                  <span>{item.label}</span>
                  <span className="text-xs text-muted-foreground">{item.menu} · {item.command}</span>
                </button>
              ))}
            </ScrollArea>
          </div>
        </div>
      ) : null}
      {workspaceDecision ? (
        <div className="absolute inset-0 z-[16000] grid place-items-center bg-background/70 backdrop-blur-sm">
          <div className="w-[620px] rounded-xl border border-border bg-popover p-5 shadow-2xl">
            <div className="mb-3 flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-primary" />
              <h2 className="text-base font-semibold">Project prompt changed</h2>
            </div>
            <p className="text-sm text-muted-foreground">NAILDE will create a new sandbox only after you decide what to do with the current workspace.</p>
            <div className="mt-3 grid gap-2 rounded-lg border border-border bg-background/70 p-3 text-xs">
              <div><span className="font-semibold">Previous:</span> {workspaceDecision.previous_prompt || lastTopPrompt || "current workspace"}</div>
              <div><span className="font-semibold">Next:</span> {workspaceDecision.next_prompt || goal}</div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              <Button onClick={() => void resolveWorkspacePromptChange("keep_save_current")}>Keep and Save Current</Button>
              <Button variant="outline" onClick={() => void resolveWorkspacePromptChange("save_as_current")}>Save Current As</Button>
              <Button variant="outline" onClick={() => void resolveWorkspacePromptChange("discard_current")}>Discard Current</Button>
              <Button variant="secondary" onClick={() => setWorkspaceDecision(null)}>Cancel Prompt Change</Button>
            </div>
          </div>
        </div>
      ) : null}

      {postBuildPopup ? (
        <div className="absolute inset-0 z-[15500] grid place-items-center bg-background/70 backdrop-blur-sm">
          <div className="w-[620px] rounded-xl border border-border bg-popover p-5 shadow-2xl">
            <div className="mb-3 flex items-center gap-2">
              <PackageCheck className="h-5 w-5 text-primary" />
              <h2 className="text-base font-semibold">{postBuildPopup.title || "Sandbox build completed"}</h2>
            </div>
            <p className="text-sm text-muted-foreground">{postBuildPopup.message || "Choose what to do next."}</p>
            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div className="rounded border border-border p-2"><div className="text-muted-foreground">Application</div><div className="font-semibold">{postBuildPopup.application_name || "NAILDE App"}</div></div>
              <div className="rounded border border-border p-2"><div className="text-muted-foreground">Validation</div><div className="font-semibold">{postBuildPopup.validation || "UNKNOWN"}</div></div>
              <div className="rounded border border-border p-2"><div className="text-muted-foreground">Install</div><div className="font-semibold">{postBuildPopup.install_readiness || "UNKNOWN"}</div></div>
            </div>
            <div className="mt-4 grid grid-cols-4 gap-2">
              <Button onClick={() => void handlePostBuildDecision("add_to_addons")} disabled={!Array.isArray(postBuildPopup.options) || !postBuildPopup.options.find((o: any) => o.id === "add_to_addons")?.enabled}>Add to Addons</Button>
              <Button variant="outline" onClick={() => void handlePostBuildDecision("save")}>Save</Button>
              <Button variant="outline" onClick={() => void handlePostBuildDecision("save_as")}>Save As</Button>
              <Button variant="secondary" onClick={() => void handlePostBuildDecision("cancel")}>Cancel</Button>
            </div>
            <p className="mt-3 text-[11px] text-muted-foreground">Cancel keeps the current session active. It does not delete the autosave recovery files.</p>
          </div>
        </div>
      ) : null}

      {recoveryPopup ? (
        <div className="absolute inset-0 z-[15400] grid place-items-center bg-background/70 backdrop-blur-sm">
          <div className="w-[560px] rounded-xl border border-border bg-popover p-5 shadow-2xl">
            <div className="mb-3 flex items-center gap-2">
              <RotateCw className="h-5 w-5 text-primary" />
              <h2 className="text-base font-semibold">Restore unfinished NAILDE workspace?</h2>
            </div>
            <p className="text-sm text-muted-foreground">Power-loss/session recovery found a sandbox workspace that can be restored.</p>
            <pre className="mt-3 max-h-48 overflow-auto rounded border border-border bg-background p-3 text-xs">{pretty(recoveryPopup).slice(0, 3000)}</pre>
            <div className="mt-4 grid grid-cols-3 gap-2">
              <Button onClick={() => void restoreWorkspace()}>Restore</Button>
              <Button variant="outline" onClick={() => setRecoveryPopup(null)}>Not Now</Button>
              <Button variant="secondary" onClick={() => setRecoveryPopup(null)}>Cancel</Button>
            </div>
          </div>
        </div>
      ) : null}

    </div>
  );
}

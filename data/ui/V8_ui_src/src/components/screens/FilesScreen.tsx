import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Folder,
  File as FileIcon,
  HardDrive,
  RefreshCcw,
  ChevronRight,
  MoreVertical,
  ArrowUp,
  Trash2,
  Scissors,
  Copy,
  ClipboardPaste,
  FolderPlus,
  Pencil,
  Download,
  Search,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

type FileItem = {
  name: string;
  path: string;
  type: "folder" | "file";
  size?: number;
  modified?: number;
};

type DriveItem = {
  name: string;
  path: string;
  kind?: string;
};

type Capabilities = {
  provider: string;
  os: string;
  canBrowse: boolean;
  canListDrives: boolean;
  canMkdir: boolean;
  canRename: boolean;
  canDelete: boolean;
  canMove: boolean;
  canCopy: boolean;
  canDownload: boolean;
  canTrash: boolean;
  canUnmount: boolean;
  canFormat: boolean;
};

type ClipboardState =
  | { mode: "copy" | "cut"; items: FileItem[]; fromDir: string }
  | null;

function fmtBytes(n?: number) {
  const v = Number(n || 0);
  if (!v) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let x = v;
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i++;
  }
  return `${x.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function fmtDate(ts?: number) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

async function apiGet<T>(url: string): Promise<T> {
  const res = await fetch(url, { credentials: "include" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data?.ok === false) {
    throw new Error(data?.error || `Request failed: ${res.status}`);
  }
  return data as T;
}

async function apiPost<T>(url: string, body: any): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data?.ok === false) {
    throw new Error(data?.error || `Request failed: ${res.status}`);
  }
  return data as T;
}

export function FilesScreen() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [capsNote, setCapsNote] = useState<string>("");
  const [drives, setDrives] = useState<DriveItem[]>([]);
  const [cwd, setCwd] = useState<string>("/");
  const [items, setItems] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(false);

  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [clipboard, setClipboard] = useState<ClipboardState>(null);

  // Context menu state (mouse + touch)
  const [ctxOpen, setCtxOpen] = useState(false);
  const [ctxXY, setCtxXY] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [ctxItem, setCtxItem] = useState<FileItem | null>(null);
  const longPressTimer = useRef<number | null>(null);

  const providerLabel = useMemo(() => {
    if (!caps) return "unknown";
    return `${caps.provider || "unknown"} • OS: ${caps.os || "unknown"}`;
  }, [caps]);

  const selectedItems = useMemo(() => {
    const set = selected;
    return items.filter((it) => !!set[it.path]);
  }, [items, selected]);

  const filteredItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((it) => (it.name || "").toLowerCase().includes(q));
  }, [items, query]);

  function clearSelection() {
    setSelected({});
  }

  function selectAll() {
    const m: Record<string, boolean> = {};
    for (const it of filteredItems) m[it.path] = true;
    setSelected(m);
  }

  function isSelected(path: string) {
    return !!selected[path];
  }

  function toggleSelect(path: string, multi: boolean) {
    setSelected((prev) => {
      if (!multi) {
        // single select
        if (prev[path] && Object.keys(prev).length === 1) return {};
        return { [path]: true };
      }
      // multi toggle
      const next = { ...prev };
      if (next[path]) delete next[path];
      else next[path] = true;
      return next;
    });
  }

  function closeContext() {
    setCtxOpen(false);
    setCtxItem(null);
  }

  function openContextFor(item: FileItem | null, x: number, y: number) {
    setCtxItem(item);
    setCtxXY({ x, y });
    setCtxOpen(true);
  }

  function startLongPress(item: FileItem | null, e: React.PointerEvent) {
    // Only for touch/pen; mouse uses right-click
    if (e.pointerType === "mouse") return;
    if (longPressTimer.current) window.clearTimeout(longPressTimer.current);
    const x = e.clientX;
    const y = e.clientY;
    longPressTimer.current = window.setTimeout(() => {
      openContextFor(item, x, y);
    }, 520);
  }

  function cancelLongPress() {
    if (longPressTimer.current) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
  }

  async function loadCapabilities() {
    try {
      const data = await apiGet<{ ok: boolean; capabilities: Capabilities; note?: string }>("/api/files/capabilities");
      setCaps(data.capabilities);
      setCapsNote(data.note || "");
      return data.capabilities;
    } catch (e: any) {
      setCaps(null);
      setCapsNote(String(e?.message || e));
      return null;
    }
  }

  async function loadDrivesIfSupported(c: Capabilities | null) {
    if (!c?.canListDrives) {
      setDrives([]);
      return;
    }
    try {
      const data = await apiGet<{ ok: boolean; drives: DriveItem[] }>("/api/files/drives");
      setDrives(data.drives || []);
    } catch (e: any) {
      setDrives([]);
    }
  }

  async function list(path: string) {
    setLoading(true);
    try {
      const data = await apiPost<{ ok: boolean; path: string; items: FileItem[] }>("/api/files/list", { path });
      setCwd(path);
      setItems(data.items || []);
      clearSelection();
    } catch (e: any) {
      toast.error(e?.message || "Failed to list directory");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  async function refreshAll() {
    const c = await loadCapabilities();
    await loadDrivesIfSupported(c);
    if (c?.canBrowse) {
      await list(cwd || "/");
    }
  }

  useEffect(() => {
    (async () => {
      const c = await loadCapabilities();
      await loadDrivesIfSupported(c);
      if (c?.canBrowse) {
        await list("/");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function goUp() {
    if (!caps?.canBrowse) return;
    // Windows paths: C:\ -> This PC
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";
    if (osName.startsWith("win")) {
      // "C:\" -> up to "/"
      if (/^[A-Za-z]:\\?$/.test(cur.replace(/\\+$/, "\\"))) {
        list("/");
        return;
      }
      const norm = cur.replace(/\//g, "\\");
      const idx = norm.lastIndexOf("\\");
      if (idx <= 2) {
        list(norm.slice(0, 3));
      } else {
        list(norm.slice(0, idx));
      }
      return;
    }
    // Unix
    if (cur === "/" || !cur) return;
    const p = cur.replace(/\/+$/, "");
    const idx = p.lastIndexOf("/");
    if (idx <= 0) list("/");
    else list(p.slice(0, idx));
  }

  async function openItem(it: FileItem) {
    if (it.type === "folder") {
      await list(it.path);
      return;
    }
    if (!caps?.canDownload) return;
    try {
      const data = await apiPost<{ ok: boolean; url: string }>("/api/files/download", { path: it.path });
      window.open(data.url, "_blank");
    } catch (e: any) {
      toast.error(e?.message || "Download failed");
    }
  }

  async function actionNewFolder() {
    if (!caps?.canMkdir) return;
    const name = prompt("New folder name:", "New Folder");
    if (!name) return;
    try {
      await apiPost("/api/files/mkdir", { path: cwd, name });
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Create folder failed");
    }
  }

  async function actionRename(target?: FileItem) {
    const it = target || (selectedItems.length === 1 ? selectedItems[0] : null);
    if (!it || !caps?.canRename) return;
    const newName = prompt("Rename to:", it.name);
    if (!newName || newName === it.name) return;
    try {
      await apiPost("/api/files/rename", { path: it.path, new_name: newName });
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Rename failed");
    }
  }

  async function actionDelete() {
    if (!caps?.canDelete) return;
    const targets = selectedItems.length ? selectedItems : (ctxItem ? [ctxItem] : []);
    if (!targets.length) return;

    const ok = confirm(`Delete ${targets.length} item(s) permanently?`);
    if (!ok) return;

    try {
      for (const t of targets) {
        await apiPost("/api/files/delete", { path: t.path, mode: "permanent" });
      }
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Delete failed");
    }
  }

  function actionCopy() {
    const targets = selectedItems.length ? selectedItems : (ctxItem ? [ctxItem] : []);
    if (!targets.length) return;
    setClipboard({ mode: "copy", items: targets, fromDir: cwd });
    toast.success(`Copied ${targets.length} item(s)`);
  }

  function actionCut() {
    const targets = selectedItems.length ? selectedItems : (ctxItem ? [ctxItem] : []);
    if (!targets.length) return;
    setClipboard({ mode: "cut", items: targets, fromDir: cwd });
    toast.success(`Cut ${targets.length} item(s)`);
  }

  async function actionPaste() {
    if (!clipboard) return;
    if (!(caps?.canMove || caps?.canCopy)) return;

    const targets = clipboard.items;
    if (!targets.length) return;

    try {
      for (const t of targets) {
        const dst = String(PathJoin(cwd, t.name, caps?.os));
        if (clipboard.mode === "copy") {
          await apiPost("/api/files/copy", { src: t.path, dst });
        } else {
          await apiPost("/api/files/move", { src: t.path, dst });
        }
      }
      // clear cut after paste
      if (clipboard.mode === "cut") setClipboard(null);
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Paste failed");
    }
  }

  function PathJoin(dir: string, name: string, osName?: string) {
    const osLower = (osName || "").toLowerCase();
    if (osLower.startsWith("win")) {
      const d = dir.replace(/\//g, "\\");
      const sep = d.endsWith("\\") ? "" : "\\";
      return `${d}${sep}${name}`;
    }
    const d = dir.endsWith("/") ? dir.slice(0, -1) : dir;
    return `${d}/${name}`;
  }

  // Breadcrumb segments (simple)
  const crumbs = useMemo(() => {
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";
    if (osName.startsWith("win")) {
      // C:\Users\Bob -> ["C:\", "Users", "Bob"]
      const norm = cur.replace(/\//g, "\\");
      const parts = norm.split("\\").filter(Boolean);
      if (/^[A-Za-z]:$/.test(parts[0] || "")) {
        // keep drive as "C:\"
        const drive = parts[0] + "\\";
        return [drive, ...parts.slice(1)];
      }
      if (/^[A-Za-z]:\\$/.test(norm)) return [norm];
      // if cur is "/" (This PC)
      if (cur === "/" || !cur) return ["This PC"];
      return parts.length ? parts : ["This PC"];
    }
    // Unix
    if (cur === "/" || !cur) return ["/"];
    const parts = cur.split("/").filter(Boolean);
    return ["/", ...parts];
  }, [cwd, caps?.os]);

  function crumbToPath(i: number) {
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";
    if (osName.startsWith("win")) {
      if (cur === "/" || !cur) return "/";
      // rebuild from crumbs
      const c = crumbs;
      if (c[0] === "This PC") return "/";
      const drive = c[0]; // "C:\"
      if (i === 0) return drive;
      const rest = c.slice(1, i + 1).join("\\");
      return drive + rest;
    }
    // Unix
    if (crumbs[0] === "/" && i === 0) return "/";
    const parts = crumbs.slice(1, i + 1);
    return "/" + parts.join("/");
  }

  const actionsDisabled = !caps?.canBrowse;

  return (
    <div className="h-full w-full flex bg-background text-foreground">
      {/* Sidebar */}
      <div className="w-64 shrink-0 border-r border-border/60 p-3">
        <div className="flex items-center gap-2 font-semibold mb-3">
          <Folder className="h-5 w-5" />
          <div className="flex flex-col leading-tight">
            <span>Files</span>
            <span className="text-xs text-muted-foreground">{providerLabel}</span>
          </div>
        </div>

        <div className="space-y-2">
          <SidebarItem
            icon={<HardDrive className="h-4 w-4" />}
            label="This PC"
            active={cwd === "/"}
            onClick={() => caps?.canBrowse && list("/")}
          />
          <SidebarItem
            icon={<Trash2 className="h-4 w-4" />}
            label="Trash"
            muted={!caps?.canTrash}
            rightText={!caps?.canTrash ? "Not supported" : ""}
            onClick={() => toast.message("Trash not implemented yet")}
          />

          <div className="mt-4">
            <div className="text-xs text-muted-foreground mb-2">Drives</div>
            {drives.length === 0 ? (
              <div className="text-xs text-muted-foreground">Drive listing not available.</div>
            ) : (
              <div className="space-y-1">
                {drives.map((d) => (
                  <SidebarItem
                    key={d.path}
                    icon={<HardDrive className="h-4 w-4" />}
                    label={d.name}
                    onClick={() => list(d.path)}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="mt-4">
            <div className="text-xs text-muted-foreground mb-2">Quick Access</div>
            <SidebarItem
              icon={<Folder className="h-4 w-4" />}
              label="Root"
              active={cwd === "/"}
              onClick={() => caps?.canBrowse && list("/")}
            />
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0" onClick={() => ctxOpen && closeContext()}>
        {/* Toolbar */}
        <div className="border-b border-border/60 p-3 flex flex-wrap items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={() => refreshAll()}
            title="Refresh"
          >
            <RefreshCcw className={cn("h-4 w-4", loading && "animate-spin")} />
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={goUp}
            disabled={actionsDisabled}
            title="Up"
          >
            <ArrowUp className="h-4 w-4" />
          </Button>

          {/* Breadcrumb */}
          <div className="flex items-center gap-1 px-2 py-1 rounded-md border border-border/60 bg-muted/20 min-w-[260px] max-w-[520px] overflow-hidden">
            {crumbs.map((c, i) => (
              <React.Fragment key={i}>
                <button
                  className="text-sm hover:underline truncate max-w-[160px]"
                  onClick={() => {
                    const p = crumbToPath(i);
                    list(p);
                  }}
                >
                  {c}
                </button>
                {i < crumbs.length - 1 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
              </React.Fragment>
            ))}
          </div>

          {/* Search */}
          <div className="flex items-center gap-2 flex-1 min-w-[200px] max-w-[520px]">
            <div className="relative w-full">
              <Search className="h-4 w-4 text-muted-foreground absolute left-2 top-1/2 -translate-y-1/2" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search"
                className="h-8 pl-8"
              />
            </div>
          </div>

          {/* Primary action */}
          <Button
            variant="outline"
            className="h-8"
            onClick={actionNewFolder}
            disabled={!caps?.canMkdir}
            title="New Folder"
          >
            <FolderPlus className="h-4 w-4 mr-2" />
            New
          </Button>

          {/* Actions dropdown (replaces giant button row) */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-8" disabled={actionsDisabled}>
                <MoreVertical className="h-4 w-4 mr-2" />
                Actions
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52">
              <DropdownMenuItem onClick={() => selectAll()}>
                Select All
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => clearSelection()}>
                Clear Selection
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => actionCopy()} disabled={!selectedItems.length}>
                <Copy className="h-4 w-4 mr-2" /> Copy
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => actionCut()} disabled={!selectedItems.length}>
                <Scissors className="h-4 w-4 mr-2" /> Cut
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => actionPaste()} disabled={!clipboard}>
                <ClipboardPaste className="h-4 w-4 mr-2" /> Paste
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => actionRename()} disabled={selectedItems.length !== 1}>
                <Pencil className="h-4 w-4 mr-2" /> Rename
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => actionDelete()} disabled={!selectedItems.length}>
                <Trash2 className="h-4 w-4 mr-2" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Selection mini-controls */}
          <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            <span>{selectedItems.length} selected</span>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 min-h-0 p-3">
          {!caps?.canBrowse ? (
            <div className="border border-border/60 rounded-lg p-4 bg-muted/10">
              <div className="font-semibold mb-1">Not available (server configuration)</div>
              <div className="text-sm text-muted-foreground">
                {capsNote || "File browsing requires local mode support. If you are on ai.sarahmemory.com, full browsing requires a local agent bridge."}
              </div>
            </div>
          ) : (
            <div className="border border-border/60 rounded-lg overflow-hidden">
              {/* Table header */}
              <div className="grid grid-cols-[1fr_120px_140px_200px] gap-2 px-3 py-2 bg-muted/30 text-xs font-semibold border-b border-border/60">
                <div>Name</div>
                <div>Type</div>
                <div className="text-right">Size</div>
                <div>Modified</div>
              </div>

              {/* Rows */}
              <div className="max-h-[calc(100vh-220px)] overflow-auto">
                {loading ? (
                  <div className="p-6 text-sm text-muted-foreground">Loading…</div>
                ) : filteredItems.length === 0 ? (
                  <div className="p-6 text-sm text-muted-foreground">No items</div>
                ) : (
                  filteredItems.map((it) => (
                    <div
                      key={it.path}
                      className={cn(
                        "grid grid-cols-[1fr_120px_140px_200px] gap-2 px-3 py-2 text-sm border-b border-border/40 cursor-default",
                        isSelected(it.path) ? "bg-muted/40" : "hover:bg-muted/20"
                      )}
                      onClick={(e) => toggleSelect(it.path, e.ctrlKey || e.metaKey)}
                      onDoubleClick={() => openItem(it)}
                      onContextMenu={(e) => {
                        e.preventDefault();
                        openContextFor(it, e.clientX, e.clientY);
                      }}
                      onPointerDown={(e) => startLongPress(it, e)}
                      onPointerUp={cancelLongPress}
                      onPointerCancel={cancelLongPress}
                      onPointerMove={cancelLongPress}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        {it.type === "folder" ? (
                          <Folder className="h-4 w-4 shrink-0" />
                        ) : (
                          <FileIcon className="h-4 w-4 shrink-0" />
                        )}
                        <span className="truncate">{it.name}</span>
                      </div>
                      <div className="text-muted-foreground">{it.type}</div>
                      <div className="text-right text-muted-foreground">{fmtBytes(it.size)}</div>
                      <div className="text-muted-foreground">{fmtDate(it.modified)}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border/60 p-2 text-xs text-muted-foreground flex items-center justify-between">
          <span>{items.length} item(s)</span>
          <span>{clipboard ? `Clipboard: ${clipboard.mode} • ${clipboard.items.length} item(s)` : "Clipboard: empty"}</span>
        </div>

        {/* Context Menu Overlay */}
        {ctxOpen && (
          <div
            className="fixed z-[9999] min-w-[200px] rounded-md border border-border bg-popover shadow-lg p-1"
            style={{ left: ctxXY.x, top: ctxXY.y }}
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <CtxItem
              icon={<Download className="h-4 w-4" />}
              label="Download"
              disabled={!caps?.canDownload || !ctxItem || ctxItem.type !== "file"}
              onClick={() => {
                if (ctxItem) openItem(ctxItem);
                closeContext();
              }}
            />
            <CtxItem
              icon={<Copy className="h-4 w-4" />}
              label="Copy"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) setSelected({ [ctxItem.path]: true });
                actionCopy();
                closeContext();
              }}
            />
            <CtxItem
              icon={<Scissors className="h-4 w-4" />}
              label="Cut"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) setSelected({ [ctxItem.path]: true });
                actionCut();
                closeContext();
              }}
            />
            <CtxItem
              icon={<ClipboardPaste className="h-4 w-4" />}
              label="Paste"
              disabled={!clipboard}
              onClick={() => {
                actionPaste();
                closeContext();
              }}
            />
            <div className="my-1 border-t border-border/60" />
            <CtxItem
              icon={<Pencil className="h-4 w-4" />}
              label="Rename"
              disabled={!caps?.canRename || (!ctxItem && selectedItems.length !== 1)}
              onClick={() => {
                actionRename(ctxItem || undefined);
                closeContext();
              }}
            />
            <CtxItem
              icon={<Trash2 className="h-4 w-4" />}
              label="Delete"
              danger
              disabled={!caps?.canDelete || (!ctxItem && !selectedItems.length)}
              onClick={() => {
                if (ctxItem) setSelected({ [ctxItem.path]: true });
                actionDelete();
                closeContext();
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function SidebarItem(props: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  muted?: boolean;
  rightText?: string;
  onClick?: () => void;
}) {
  const { icon, label, active, muted, rightText, onClick } = props;
  return (
    <button
      className={cn(
        "w-full flex items-center justify-between gap-2 px-2 py-2 rounded-md text-sm text-left",
        active ? "bg-muted/40" : "hover:bg-muted/20",
        muted && "opacity-60 cursor-not-allowed"
      )}
      onClick={(e) => {
        e.preventDefault();
        if (!muted) onClick?.();
      }}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span className="shrink-0">{icon}</span>
        <span className="truncate">{label}</span>
      </div>
      {rightText ? <span className="text-[11px] text-muted-foreground">{rightText}</span> : null}
    </button>
  );
}

function CtxItem(props: {
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
  danger?: boolean;
  onClick?: () => void;
}) {
  const { icon, label, disabled, danger, onClick } = props;
  return (
    <button
      className={cn(
        "w-full flex items-center gap-2 px-2 py-2 rounded-sm text-sm",
        disabled ? "opacity-50 cursor-not-allowed" : "hover:bg-muted/30",
        danger && !disabled && "text-red-500"
      )}
      onClick={(e) => {
        e.preventDefault();
        if (!disabled) onClick?.();
      }}
    >
      <span className="shrink-0">{icon}</span>
      <span>{label}</span>
    </button>
  );
}

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
  Eye,
  LayoutGrid,
  List as ListIcon,
  Table2,
  X,
  Info,
  UploadCloud,
  FileText,
  FileCode2,
  Image as ImageIcon,
  Music,
  Video,
  Archive,
  FileSpreadsheet,
  FileJson,
  Terminal,
  ShieldAlert,
  CheckSquare,
  Square,
  ClipboardCopy,
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
  modified?: number; // epoch seconds
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

  // optional / may come later
  canTrash?: boolean;
  canUpload?: boolean;
  canUnmount?: boolean;
  canFormat?: boolean;
};

type ClipboardState =
  | { mode: "copy" | "cut"; items: FileItem[]; fromDir: string }
  | null;

type SortKey = "name" | "type" | "size" | "modified";
type SortDir = "asc" | "desc";
type ViewMode = "details" | "list" | "icons";

type PreviewKind = "image" | "text" | "pdf" | "unknown";

type PreviewState = {
  open: boolean;
  item: FileItem | null;
  kind: PreviewKind;
  title: string;
  url?: string;
  text?: string;
  loading?: boolean;
  error?: string;
};

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

function lower(s: any) {
  return String(s || "").toLowerCase();
}

function extOf(name: string) {
  const i = name.lastIndexOf(".");
  if (i <= 0) return "";
  return name.slice(i + 1).toLowerCase();
}

function isTextExt(ext: string) {
  return [
    "txt",
    "md",
    "log",
    "json",
    "yaml",
    "yml",
    "ini",
    "cfg",
    "conf",
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "css",
    "html",
    "htm",
    "xml",
    "csv",
    "sql",
    "bat",
    "ps1",
    "sh",
    "c",
    "cpp",
    "h",
    "hpp",
    "java",
    "cs",
    "go",
    "rs",
    "toml",
  ].includes(ext);
}

function isImageExt(ext: string) {
  return ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"].includes(ext);
}

function iconForItem(it: FileItem) {
  if (it.type === "folder") return Folder;
  const ext = extOf(it.name);
  if (!ext) return FileIcon;

  if (["txt", "md", "log"].includes(ext)) return FileText;
  if (["py", "js", "ts", "tsx", "jsx", "css", "html", "htm", "sql", "c", "cpp", "h", "hpp", "java", "cs", "go", "rs", "sh", "ps1", "bat"].includes(ext))
    return FileCode2;

  if (["json"].includes(ext)) return FileJson;
  if (["csv", "xlsx", "xls"].includes(ext)) return FileSpreadsheet;

  if (isImageExt(ext)) return ImageIcon;
  if (["mp3", "wav", "flac", "aac", "m4a", "ogg"].includes(ext)) return Music;
  if (["mp4", "mov", "mkv", "webm", "avi"].includes(ext)) return Video;

  if (["zip", "rar", "7z", "tar", "gz"].includes(ext)) return Archive;

  if (["exe", "msi"].includes(ext)) return Terminal;

  return FileIcon;
}

async function apiGet<T>(url: string): Promise<T> {
  const res = await fetch(url, { credentials: "include" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || (data as any)?.ok === false) {
    throw new Error((data as any)?.error || `Request failed: ${res.status}`);
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
  if (!res.ok || (data as any)?.ok === false) {
    throw new Error((data as any)?.error || `Request failed: ${res.status}`);
  }
  return data as T;
}

async function apiUpload<T>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || (data as any)?.ok === false) {
    throw new Error((data as any)?.error || `Upload failed: ${res.status}`);
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

  // Multi-select support
  const lastSelectedPathRef = useRef<string | null>(null);

  // Sorting + view
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [viewMode, setViewMode] = useState<ViewMode>("details");
  const [showCheckboxes, setShowCheckboxes] = useState<boolean>(false);

  // Context menu (mouse + touch long-press)
  const [ctxOpen, setCtxOpen] = useState(false);
  const [ctxXY, setCtxXY] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [ctxItem, setCtxItem] = useState<FileItem | null>(null);
  const longPressTimer = useRef<number | null>(null);

  // Preview (Open)
  const [preview, setPreview] = useState<PreviewState>({
    open: false,
    item: null,
    kind: "unknown",
    title: "",
  });

  // Properties drawer
  const [propsOpen, setPropsOpen] = useState(false);
  const [propsItem, setPropsItem] = useState<FileItem | null>(null);

  // Inline rename
  const [renamingPath, setRenamingPath] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState<string>("");
  const renameInputRef = useRef<HTMLInputElement | null>(null);

  // Drag-drop upload
  const [dragOver, setDragOver] = useState(false);
  const filePickRef = useRef<HTMLInputElement | null>(null);

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

  const sortedItems = useMemo(() => {
    const arr = [...filteredItems];

    // Keep folders first, then apply sort within type groups (Explorer-like)
    const folders = arr.filter((x) => x.type === "folder");
    const files = arr.filter((x) => x.type === "file");

    const cmp = (a: FileItem, b: FileItem) => {
      const dir = sortDir === "asc" ? 1 : -1;

      if (sortKey === "name") {
        return dir * lower(a.name).localeCompare(lower(b.name));
      }
      if (sortKey === "type") {
        const at = a.type === "folder" ? "folder" : extOf(a.name);
        const bt = b.type === "folder" ? "folder" : extOf(b.name);
        const r = lower(at).localeCompare(lower(bt));
        if (r !== 0) return dir * r;
        return dir * lower(a.name).localeCompare(lower(b.name));
      }
      if (sortKey === "size") {
        const as = Number(a.size || 0);
        const bs = Number(b.size || 0);
        if (as !== bs) return dir * (as - bs);
        return dir * lower(a.name).localeCompare(lower(b.name));
      }
      // modified
      const am = Number(a.modified || 0);
      const bm = Number(b.modified || 0);
      if (am !== bm) return dir * (am - bm);
      return dir * lower(a.name).localeCompare(lower(b.name));
    };

    folders.sort(cmp);
    files.sort(cmp);
    return [...folders, ...files];
  }, [filteredItems, sortKey, sortDir]);

  function clearSelection() {
    setSelected({});
    lastSelectedPathRef.current = null;
  }

  function selectAll() {
    const m: Record<string, boolean> = {};
    for (const it of sortedItems) m[it.path] = true;
    setSelected(m);
    if (sortedItems.length) lastSelectedPathRef.current = sortedItems[sortedItems.length - 1].path;
  }

  function isSelected(path: string) {
    return !!selected[path];
  }

  function setSingleSelect(path: string) {
    setSelected({ [path]: true });
    lastSelectedPathRef.current = path;
  }

  function toggleSelect(path: string) {
    setSelected((prev) => {
      const next = { ...prev };
      if (next[path]) delete next[path];
      else next[path] = true;
      return next;
    });
    lastSelectedPathRef.current = path;
  }

  function rangeSelect(toPath: string) {
    const from = lastSelectedPathRef.current;
    const list = sortedItems.map((x) => x.path);
    if (!from || !list.includes(from) || !list.includes(toPath)) {
      setSingleSelect(toPath);
      return;
    }

    const a = list.indexOf(from);
    const b = list.indexOf(toPath);
    const [start, end] = a < b ? [a, b] : [b, a];

    setSelected(() => {
      const next: Record<string, boolean> = {};
      for (let i = start; i <= end; i++) next[list[i]] = true;
      return next;
    });

    lastSelectedPathRef.current = toPath;
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
    } catch {
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
      setRenamingPath(null);
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

  function setSort(nextKey: SortKey) {
    setSortKey((prevKey) => {
      if (prevKey !== nextKey) {
        setSortDir("asc");
        return nextKey;
      }
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return prevKey;
    });
  }

  function goUp() {
    if (!caps?.canBrowse) return;
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";

    if (osName.startsWith("win")) {
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

    if (cur === "/" || !cur) return;
    const p = cur.replace(/\/+$/, "");
    const idx = p.lastIndexOf("/");
    if (idx <= 0) list("/");
    else list(p.slice(0, idx));
  }

  function pathJoin(dir: string, name: string, osName?: string) {
    const osLower = (osName || "").toLowerCase();
    if (osLower.startsWith("win")) {
      const d = dir.replace(/\//g, "\\");
      const sep = d.endsWith("\\") ? "" : "\\";
      return `${d}${sep}${name}`;
    }
    const d = dir.endsWith("/") ? dir.slice(0, -1) : dir;
    return `${d}/${name}`;
  }

  // Breadcrumb segments
  const crumbs = useMemo(() => {
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";
    if (osName.startsWith("win")) {
      const norm = cur.replace(/\//g, "\\");
      const parts = norm.split("\\").filter(Boolean);
      if (/^[A-Za-z]:$/.test(parts[0] || "")) {
        const drive = parts[0] + "\\";
        return [drive, ...parts.slice(1)];
      }
      if (/^[A-Za-z]:\\$/.test(norm)) return [norm];
      if (cur === "/" || !cur) return ["This PC"];
      return parts.length ? parts : ["This PC"];
    }
    if (cur === "/" || !cur) return ["/"];
    const parts = cur.split("/").filter(Boolean);
    return ["/", ...parts];
  }, [cwd, caps?.os]);

  function crumbToPath(i: number) {
    const osName = (caps?.os || "").toLowerCase();
    const cur = cwd || "/";
    if (osName.startsWith("win")) {
      if (cur === "/" || !cur) return "/";
      const c = crumbs;
      if (c[0] === "This PC") return "/";
      const drive = c[0]; // "C:\"
      if (i === 0) return drive;
      const rest = c.slice(1, i + 1).join("\\");
      return drive + rest;
    }
    if (crumbs[0] === "/" && i === 0) return "/";
    const parts = crumbs.slice(1, i + 1);
    return "/" + parts.join("/");
  }

  const actionsDisabled = !caps?.canBrowse;

  async function getDownloadUrl(path: string) {
    const data = await apiPost<{ ok: boolean; url: string; expires_in?: number }>("/api/files/download", { path });
    return data.url;
  }

  async function openFilePreview(it: FileItem) {
    if (!caps?.canDownload) {
      toast.error("Download/Open not supported");
      return;
    }

    const name = it.name || "File";
    const ext = extOf(name);

    const kind: PreviewKind =
      isImageExt(ext) ? "image" : ext === "pdf" ? "pdf" : isTextExt(ext) ? "text" : "unknown";

    setPreview({
      open: true,
      item: it,
      kind,
      title: name,
      loading: true,
    });

    try {
      const url = await getDownloadUrl(it.path);

      if (kind === "image" || kind === "pdf" || kind === "unknown") {
        setPreview((p) => ({
          ...p,
          url,
          loading: false,
        }));
        return;
      }

      // text
      const res = await fetch(url, { credentials: "include" });
      if (!res.ok) throw new Error(`Open failed: ${res.status}`);
      const text = await res.text();

      const MAX = 1024 * 512; // 512KB
      const safeText = text.length > MAX ? text.slice(0, MAX) + "\n\n--- Truncated (preview limit) ---" : text;

      setPreview((p) => ({
        ...p,
        url,
        text: safeText,
        loading: false,
      }));
    } catch (e: any) {
      setPreview((p) => ({
        ...p,
        loading: false,
        error: e?.message || "Open failed",
      }));
    }
  }

  async function openItem(it: FileItem) {
    if (it.type === "folder") {
      await list(it.path);
      return;
    }
    await openFilePreview(it);
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

  function beginInlineRename(it: FileItem) {
    if (!caps?.canRename) return;
    setRenamingPath(it.path);
    setRenameValue(it.name);
    // focus next tick
    window.setTimeout(() => renameInputRef.current?.focus(), 0);
    window.setTimeout(() => renameInputRef.current?.select(), 0);
  }

  async function commitInlineRename() {
    if (!renamingPath) return;
    const it = items.find((x) => x.path === renamingPath);
    const newName = (renameValue || "").trim();
    if (!it) {
      setRenamingPath(null);
      return;
    }
    if (!newName || newName === it.name) {
      setRenamingPath(null);
      return;
    }
    try {
      await apiPost("/api/files/rename", { path: it.path, new_name: newName });
      setRenamingPath(null);
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Rename failed");
    }
  }

  function cancelInlineRename() {
    setRenamingPath(null);
    setRenameValue("");
  }

  async function actionRename(target?: FileItem) {
    const it = target || (selectedItems.length === 1 ? selectedItems[0] : null);
    if (!it || !caps?.canRename) return;
    // Use inline rename instead of prompt
    beginInlineRename(it);
  }

  async function actionDelete(mode: "permanent" | "trash" = "permanent") {
    if (!caps?.canDelete) return;
    const targets = selectedItems.length ? selectedItems : ctxItem ? [ctxItem] : [];
    if (!targets.length) return;

    // Safety: permanent confirm, trash confirm lightly
    if (mode === "permanent") {
      const ok = confirm(`Delete ${targets.length} item(s) permanently?`);
      if (!ok) return;
    }

    try {
      for (const t of targets) {
        await apiPost("/api/files/delete", { path: t.path, mode });
      }
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Delete failed");
    }
  }

  function actionCopy() {
    const targets = selectedItems.length ? selectedItems : ctxItem ? [ctxItem] : [];
    if (!targets.length) return;
    setClipboard({ mode: "copy", items: targets, fromDir: cwd });
    toast.success(`Copied ${targets.length} item(s)`);
  }

  function actionCut() {
    const targets = selectedItems.length ? selectedItems : ctxItem ? [ctxItem] : [];
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
        const dst = String(pathJoin(cwd, t.name, caps?.os));
        if (clipboard.mode === "copy") {
          await apiPost("/api/files/copy", { src: t.path, dst });
        } else {
          await apiPost("/api/files/move", { src: t.path, dst });
        }
      }
      if (clipboard.mode === "cut") setClipboard(null);
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Paste failed");
    }
  }

  async function actionDownload(target?: FileItem) {
    const it = target || (selectedItems.length === 1 ? selectedItems[0] : null);
    if (!it || it.type !== "file" || !caps?.canDownload) return;
    try {
      const url = await getDownloadUrl(it.path);
      window.open(url, "_blank");
    } catch (e: any) {
      toast.error(e?.message || "Download failed");
    }
  }

  async function actionOpen() {
    if (selectedItems.length !== 1) return;
    await openItem(selectedItems[0]);
  }

  function copyToClipboard(text: string, label: string) {
    try {
      navigator.clipboard?.writeText(text);
      toast.success(`${label} copied`);
    } catch {
      toast.error("Clipboard not available");
    }
  }

  function actionCopyName() {
    const it = selectedItems.length === 1 ? selectedItems[0] : ctxItem;
    if (!it) return;
    copyToClipboard(it.name, "Name");
  }

  function actionCopyPath() {
    const it = selectedItems.length === 1 ? selectedItems[0] : ctxItem;
    if (!it) return;
    copyToClipboard(it.path, "Path");
  }

  function openProperties(it?: FileItem) {
    const target = it || (selectedItems.length === 1 ? selectedItems[0] : ctxItem);
    if (!target) return;
    setPropsItem(target);
    setPropsOpen(true);
  }

  function closeProperties() {
    setPropsOpen(false);
    setPropsItem(null);
  }

  function onRowClick(it: FileItem, e: React.MouseEvent) {
    // Ignore clicks when inline renaming input is active
    if (renamingPath === it.path) return;

    const shift = (e as any).shiftKey;
    const multi = (e as any).ctrlKey || (e as any).metaKey;

    if (shift) {
      rangeSelect(it.path);
      return;
    }
    if (multi) {
      toggleSelect(it.path);
      return;
    }
    setSingleSelect(it.path);
  }

  function ensureSelectedContext(it: FileItem | null) {
    if (!it) return;
    if (!isSelected(it.path)) setSingleSelect(it.path);
  }

  function closePreview() {
    setPreview({ open: false, item: null, kind: "unknown", title: "" });
  }

  // Upload: drag/drop + picker
  async function uploadFiles(fileList: FileList | null) {
    if (!fileList || !fileList.length) return;
    if (!caps?.canBrowse) return;

    // If backend doesn't support it yet, we still attempt; failure will toast.
    const files = Array.from(fileList);
    toast.message(`Uploading ${files.length} file(s)…`);

    try {
      for (const f of files) {
        const form = new FormData();
        form.append("file", f);
        form.append("target_dir", cwd || "/");
        await apiUpload("/api/files/upload", form);
      }
      toast.success("Upload complete");
      await list(cwd);
    } catch (e: any) {
      toast.error(e?.message || "Upload failed (endpoint not supported yet)");
    }
  }

  function onDragOver(e: React.DragEvent) {
    if (!caps?.canBrowse) return;
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }
  function onDrop(e: React.DragEvent) {
    if (!caps?.canBrowse) return;
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (e.dataTransfer?.files?.length) {
      uploadFiles(e.dataTransfer.files);
    }
  }

  function triggerFilePick() {
    filePickRef.current?.click();
  }

  // Keyboard shortcuts (desktop)
  useEffect(() => {
    const onKeyDown = (ev: KeyboardEvent) => {
      if (!caps?.canBrowse) return;

      // Preview open: escape closes
      if (preview.open && ev.key === "Escape") {
        ev.preventDefault();
        closePreview();
        return;
      }

      // Properties open: escape closes
      if (propsOpen && ev.key === "Escape") {
        ev.preventDefault();
        closeProperties();
        return;
      }

      // If inline rename active: Enter commits, Escape cancels
      if (renamingPath) {
        if (ev.key === "Enter") {
          ev.preventDefault();
          commitInlineRename();
          return;
        }
        if (ev.key === "Escape") {
          ev.preventDefault();
          cancelInlineRename();
          return;
        }
      }

      // ctrl+a
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "a") {
        ev.preventDefault();
        selectAll();
        return;
      }

      // delete
      if (ev.key === "Delete") {
        if (selectedItems.length) {
          ev.preventDefault();
          // Shift+Delete => permanent
          if ((ev as any).shiftKey) actionDelete("permanent");
          else {
            // Try trash if supported; else permanent confirm
            const canTrash = !!caps?.canTrash;
            if (canTrash) actionDelete("trash");
            else actionDelete("permanent");
          }
        }
        return;
      }

      // Enter: open (1 selected)
      if (ev.key === "Enter") {
        if (selectedItems.length === 1) {
          ev.preventDefault();
          openItem(selectedItems[0]);
        }
        return;
      }

      // F2: rename
      if (ev.key === "F2") {
        if (selectedItems.length === 1 && caps?.canRename) {
          ev.preventDefault();
          beginInlineRename(selectedItems[0]);
        }
        return;
      }

      // ctrl+c / ctrl+x / ctrl+v
      if (ev.ctrlKey || ev.metaKey) {
        const k = ev.key.toLowerCase();
        if (k === "c") {
          if (selectedItems.length) {
            ev.preventDefault();
            actionCopy();
          }
        } else if (k === "x") {
          if (selectedItems.length) {
            ev.preventDefault();
            actionCut();
          }
        } else if (k === "v") {
          if (clipboard) {
            ev.preventDefault();
            actionPaste();
          }
        } else if (k === "l") {
          // Ctrl+L focuses search
          const el = document.getElementById("sm-files-search") as HTMLInputElement | null;
          if (el) {
            ev.preventDefault();
            el.focus();
            el.select();
          }
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caps?.canBrowse, caps?.canTrash, preview.open, propsOpen, selectedItems, clipboard, renamingPath, renameValue]);

  const canTrash = !!caps?.canTrash;

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
            muted={!canTrash}
            rightText={!canTrash ? "Not supported" : ""}
            onClick={() => toast.message("Trash view needs backend trash endpoints")}
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
                    active={cwd === d.path}
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
      <div
        className="flex-1 flex flex-col min-w-0 relative"
        onClick={() => {
          if (ctxOpen) closeContext();
        }}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {/* Drag overlay */}
        {dragOver && caps?.canBrowse && (
          <div className="absolute inset-0 z-[9998] bg-black/30 backdrop-blur-[1px] flex items-center justify-center">
            <div className="rounded-xl border border-border bg-background/90 shadow-xl px-6 py-5 flex items-center gap-3">
              <UploadCloud className="h-6 w-6" />
              <div>
                <div className="font-semibold">Drop files to upload</div>
                <div className="text-xs text-muted-foreground">Uploads to the current folder (backend may redirect to downloads)</div>
              </div>
            </div>
          </div>
        )}

        {/* Toolbar */}
        <div className="border-b border-border/60 p-3 flex flex-wrap items-center gap-2">
          <Button variant="outline" size="icon" className="h-8 w-8" onClick={refreshAll} title="Refresh">
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
                  onClick={() => list(crumbToPath(i))}
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
                id="sm-files-search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search"
                className="h-8 pl-8"
              />
            </div>
          </div>

          {/* New Folder */}
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

          {/* Upload */}
          <Button
            variant="outline"
            className="h-8"
            onClick={triggerFilePick}
            disabled={!caps?.canBrowse}
            title="Upload"
          >
            <UploadCloud className="h-4 w-4 mr-2" />
            Upload
          </Button>
          <input
            ref={filePickRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              uploadFiles(e.target.files);
              if (e.target) e.target.value = "";
            }}
          />

          {/* View dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-8" disabled={actionsDisabled} title="View">
                {viewMode === "details" ? (
                  <Table2 className="h-4 w-4 mr-2" />
                ) : viewMode === "list" ? (
                  <ListIcon className="h-4 w-4 mr-2" />
                ) : (
                  <LayoutGrid className="h-4 w-4 mr-2" />
                )}
                View
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52">
              <DropdownMenuItem onClick={() => setViewMode("details")}>
                <Table2 className="h-4 w-4 mr-2" /> Details
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setViewMode("list")}>
                <ListIcon className="h-4 w-4 mr-2" /> List
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setViewMode("icons")}>
                <LayoutGrid className="h-4 w-4 mr-2" /> Icons
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setShowCheckboxes((v) => !v)}>
                {showCheckboxes ? (
                  <CheckSquare className="h-4 w-4 mr-2" />
                ) : (
                  <Square className="h-4 w-4 mr-2" />
                )}
                {showCheckboxes ? "Hide" : "Show"} checkboxes (touch)
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Actions dropdown (smaller + professional) */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-8" disabled={actionsDisabled}>
                <MoreVertical className="h-4 w-4 mr-2" />
                Actions
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-64">
              <DropdownMenuItem onClick={selectAll}>Select All</DropdownMenuItem>
              <DropdownMenuItem onClick={clearSelection}>Clear Selection</DropdownMenuItem>
              <DropdownMenuSeparator />

              <DropdownMenuItem onClick={actionOpen} disabled={selectedItems.length !== 1}>
                <Eye className="h-4 w-4 mr-2" /> Open
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => actionDownload()} disabled={selectedItems.length !== 1}>
                <Download className="h-4 w-4 mr-2" /> Download
              </DropdownMenuItem>

              <DropdownMenuSeparator />

              <DropdownMenuItem onClick={actionCopy} disabled={!selectedItems.length}>
                <Copy className="h-4 w-4 mr-2" /> Copy
              </DropdownMenuItem>
              <DropdownMenuItem onClick={actionCut} disabled={!selectedItems.length}>
                <Scissors className="h-4 w-4 mr-2" /> Cut
              </DropdownMenuItem>
              <DropdownMenuItem onClick={actionPaste} disabled={!clipboard}>
                <ClipboardPaste className="h-4 w-4 mr-2" /> Paste
              </DropdownMenuItem>

              <DropdownMenuSeparator />

              <DropdownMenuItem onClick={() => actionRename()} disabled={selectedItems.length !== 1}>
                <Pencil className="h-4 w-4 mr-2" /> Rename (F2)
              </DropdownMenuItem>

              <DropdownMenuItem onClick={actionCopyName} disabled={selectedItems.length !== 1 && !ctxItem}>
                <ClipboardCopy className="h-4 w-4 mr-2" /> Copy Name
              </DropdownMenuItem>
              <DropdownMenuItem onClick={actionCopyPath} disabled={selectedItems.length !== 1 && !ctxItem}>
                <ClipboardCopy className="h-4 w-4 mr-2" /> Copy Path
              </DropdownMenuItem>

              <DropdownMenuItem onClick={() => openProperties()} disabled={selectedItems.length !== 1 && !ctxItem}>
                <Info className="h-4 w-4 mr-2" /> Properties
              </DropdownMenuItem>

              <DropdownMenuSeparator />

              <DropdownMenuItem
                onClick={() => {
                  if (canTrash) actionDelete("trash");
                  else actionDelete("permanent");
                }}
                disabled={!selectedItems.length}
              >
                <Trash2 className="h-4 w-4 mr-2 text-red-500" />{" "}
                <span className="text-red-500">{canTrash ? "Move to Trash" : "Delete"}</span>
              </DropdownMenuItem>

              {!canTrash && (
                <div className="px-2 py-1.5 text-[11px] text-muted-foreground">
                  Trash requires backend dumpster endpoints.
                </div>
              )}
            </DropdownMenuContent>
          </DropdownMenu>

          <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
            <span>{selectedItems.length} selected</span>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 min-h-0 p-3 flex gap-3">
          {/* File list panel */}
          <div className="flex-1 min-w-0">
            {!caps?.canBrowse ? (
              <div className="border border-border/60 rounded-lg p-4 bg-muted/10">
                <div className="font-semibold mb-1">Not available (server configuration)</div>
                <div className="text-sm text-muted-foreground">
                  {capsNote ||
                    "File browsing requires local mode support. If you are on ai.sarahmemory.com, full browsing requires a local agent bridge."}
                </div>
              </div>
            ) : (
              <div className="border border-border/60 rounded-lg overflow-hidden">
                {viewMode === "details" && (
                  <>
                    <div className="grid grid-cols-[min-content_1fr_120px_140px_200px] gap-2 px-3 py-2 bg-muted/30 text-xs font-semibold border-b border-border/60">
                      <div className="w-8" />
                      <HeaderCell label="Name" active={sortKey === "name"} dir={sortDir} onClick={() => setSort("name")} />
                      <HeaderCell label="Type" active={sortKey === "type"} dir={sortDir} onClick={() => setSort("type")} />
                      <HeaderCellRight
                        label="Size"
                        active={sortKey === "size"}
                        dir={sortDir}
                        onClick={() => setSort("size")}
                      />
                      <HeaderCell
                        label="Modified"
                        active={sortKey === "modified"}
                        dir={sortDir}
                        onClick={() => setSort("modified")}
                      />
                    </div>

                    <div className="max-h-[calc(100vh-220px)] overflow-auto">
                      {loading ? (
                        <div className="p-6 text-sm text-muted-foreground">Loading…</div>
                      ) : sortedItems.length === 0 ? (
                        <div className="p-6 text-sm text-muted-foreground">No items</div>
                      ) : (
                        sortedItems.map((it) => {
                          const Icon = iconForItem(it);
                          const renaming = renamingPath === it.path;

                          return (
                            <div
                              key={it.path}
                              className={cn(
                                "grid grid-cols-[min-content_1fr_120px_140px_200px] gap-2 px-3 py-2 text-sm border-b border-border/40 cursor-default select-none",
                                isSelected(it.path) ? "bg-muted/40" : "hover:bg-muted/20"
                              )}
                              onClick={(e) => onRowClick(it, e)}
                              onDoubleClick={() => openItem(it)}
                              onContextMenu={(e) => {
                                e.preventDefault();
                                ensureSelectedContext(it);
                                openContextFor(it, e.clientX, e.clientY);
                              }}
                              onPointerDown={(e) => startLongPress(it, e)}
                              onPointerUp={cancelLongPress}
                              onPointerCancel={cancelLongPress}
                              onPointerMove={cancelLongPress}
                            >
                              <div className="flex items-center justify-center w-8">
                                {showCheckboxes && (
                                  <input
                                    type="checkbox"
                                    checked={isSelected(it.path)}
                                    onChange={(e) => {
                                      e.stopPropagation();
                                      toggleSelect(it.path);
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                    className="h-4 w-4"
                                  />
                                )}
                              </div>

                              <div className="flex items-center gap-2 min-w-0">
                                <Icon className="h-4 w-4 shrink-0" />
                                {renaming ? (
                                  <input
                                    ref={(el) => (renameInputRef.current = el)}
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        commitInlineRename();
                                      } else if (e.key === "Escape") {
                                        e.preventDefault();
                                        cancelInlineRename();
                                      }
                                    }}
                                    onBlur={() => commitInlineRename()}
                                    className="h-7 px-2 rounded-md border border-border bg-background w-full min-w-0"
                                  />
                                ) : (
                                  <span className="truncate">{it.name}</span>
                                )}
                              </div>

                              <div className="text-muted-foreground">
                                {it.type === "folder" ? "folder" : extOf(it.name) || "file"}
                              </div>

                              <div className="text-right text-muted-foreground">{it.type === "folder" ? "" : fmtBytes(it.size)}</div>

                              <div className="text-muted-foreground">{fmtDate(it.modified)}</div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </>
                )}

                {viewMode === "list" && (
                  <div className="max-h-[calc(100vh-180px)] overflow-auto">
                    {loading ? (
                      <div className="p-6 text-sm text-muted-foreground">Loading…</div>
                    ) : sortedItems.length === 0 ? (
                      <div className="p-6 text-sm text-muted-foreground">No items</div>
                    ) : (
                      <div className="divide-y divide-border/40">
                        {sortedItems.map((it) => {
                          const Icon = iconForItem(it);
                          const renaming = renamingPath === it.path;

                          return (
                            <div
                              key={it.path}
                              className={cn(
                                "flex items-center gap-2 px-3 py-2 text-sm cursor-default select-none",
                                isSelected(it.path) ? "bg-muted/40" : "hover:bg-muted/20"
                              )}
                              onClick={(e) => onRowClick(it, e)}
                              onDoubleClick={() => openItem(it)}
                              onContextMenu={(e) => {
                                e.preventDefault();
                                ensureSelectedContext(it);
                                openContextFor(it, e.clientX, e.clientY);
                              }}
                              onPointerDown={(e) => startLongPress(it, e)}
                              onPointerUp={cancelLongPress}
                              onPointerCancel={cancelLongPress}
                              onPointerMove={cancelLongPress}
                            >
                              {showCheckboxes && (
                                <input
                                  type="checkbox"
                                  checked={isSelected(it.path)}
                                  onChange={(e) => {
                                    e.stopPropagation();
                                    toggleSelect(it.path);
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                  className="h-4 w-4"
                                />
                              )}

                              <Icon className="h-4 w-4 shrink-0" />

                              <div className="truncate flex-1">
                                {renaming ? (
                                  <input
                                    ref={(el) => (renameInputRef.current = el)}
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        commitInlineRename();
                                      } else if (e.key === "Escape") {
                                        e.preventDefault();
                                        cancelInlineRename();
                                      }
                                    }}
                                    onBlur={() => commitInlineRename()}
                                    className="h-7 px-2 rounded-md border border-border bg-background w-full"
                                  />
                                ) : (
                                  it.name
                                )}
                              </div>

                              <div className="text-xs text-muted-foreground w-28 text-right">
                                {it.type === "folder" ? "" : fmtBytes(it.size)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {viewMode === "icons" && (
                  <div className="p-3 max-h-[calc(100vh-180px)] overflow-auto">
                    {loading ? (
                      <div className="p-6 text-sm text-muted-foreground">Loading…</div>
                    ) : sortedItems.length === 0 ? (
                      <div className="p-6 text-sm text-muted-foreground">No items</div>
                    ) : (
                      <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                        {sortedItems.map((it) => {
                          const Icon = iconForItem(it);
                          const renaming = renamingPath === it.path;

                          return (
                            <div
                              key={it.path}
                              className={cn(
                                "rounded-lg border border-border/50 p-3 cursor-default select-none hover:bg-muted/20",
                                isSelected(it.path) && "bg-muted/40 border-border"
                              )}
                              onClick={(e) => onRowClick(it, e)}
                              onDoubleClick={() => openItem(it)}
                              onContextMenu={(e) => {
                                e.preventDefault();
                                ensureSelectedContext(it);
                                openContextFor(it, e.clientX, e.clientY);
                              }}
                              onPointerDown={(e) => startLongPress(it, e)}
                              onPointerUp={cancelLongPress}
                              onPointerCancel={cancelLongPress}
                              onPointerMove={cancelLongPress}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex items-center gap-2">
                                  <Icon className="h-8 w-8 shrink-0" />
                                  <div className="text-xs text-muted-foreground">
                                    {it.type === "folder" ? "folder" : extOf(it.name) || "file"}
                                  </div>
                                </div>
                                {showCheckboxes && (
                                  <input
                                    type="checkbox"
                                    checked={isSelected(it.path)}
                                    onChange={(e) => {
                                      e.stopPropagation();
                                      toggleSelect(it.path);
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                    className="h-4 w-4 mt-1"
                                  />
                                )}
                              </div>

                              <div className="mt-2 text-sm leading-tight break-words line-clamp-3">
                                {renaming ? (
                                  <input
                                    ref={(el) => (renameInputRef.current = el)}
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        commitInlineRename();
                                      } else if (e.key === "Escape") {
                                        e.preventDefault();
                                        cancelInlineRename();
                                      }
                                    }}
                                    onBlur={() => commitInlineRename()}
                                    className="h-7 px-2 rounded-md border border-border bg-background w-full"
                                  />
                                ) : (
                                  it.name
                                )}
                              </div>

                              <div className="mt-2 text-xs text-muted-foreground flex items-center justify-between">
                                <span>{it.type === "folder" ? "" : fmtBytes(it.size)}</span>
                                <span>{it.modified ? new Date(it.modified * 1000).toLocaleDateString() : ""}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Properties panel (right) */}
          {propsOpen && propsItem && (
            <div className="w-[340px] shrink-0 border border-border/60 rounded-lg overflow-hidden bg-muted/10">
              <div className="p-3 border-b border-border/60 flex items-center justify-between bg-muted/20">
                <div className="flex items-center gap-2 min-w-0">
                  <Info className="h-4 w-4" />
                  <div className="font-semibold truncate">Properties</div>
                </div>
                <Button variant="outline" size="icon" className="h-8 w-8" onClick={closeProperties}>
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <div className="p-3 space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  {React.createElement(iconForItem(propsItem), { className: "h-6 w-6 shrink-0" })}
                  <div className="min-w-0">
                    <div className="font-semibold break-words">{propsItem.name}</div>
                    <div className="text-xs text-muted-foreground break-all">{propsItem.path}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 rounded-md border border-border/60 bg-background/60">
                    <div className="text-muted-foreground">Type</div>
                    <div className="mt-1 font-medium">
                      {propsItem.type === "folder" ? "folder" : extOf(propsItem.name) || "file"}
                    </div>
                  </div>
                  <div className="p-2 rounded-md border border-border/60 bg-background/60">
                    <div className="text-muted-foreground">Size</div>
                    <div className="mt-1 font-medium">{propsItem.type === "folder" ? "—" : fmtBytes(propsItem.size)}</div>
                  </div>
                  <div className="p-2 rounded-md border border-border/60 bg-background/60 col-span-2">
                    <div className="text-muted-foreground">Modified</div>
                    <div className="mt-1 font-medium">{fmtDate(propsItem.modified) || "—"}</div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" className="h-8" onClick={() => copyToClipboard(propsItem.name, "Name")}>
                    <ClipboardCopy className="h-4 w-4 mr-2" />
                    Copy name
                  </Button>
                  <Button variant="outline" className="h-8" onClick={() => copyToClipboard(propsItem.path, "Path")}>
                    <ClipboardCopy className="h-4 w-4 mr-2" />
                    Copy path
                  </Button>
                  <Button
                    variant="outline"
                    className="h-8"
                    onClick={() => actionDownload(propsItem)}
                    disabled={propsItem.type !== "file" || !caps?.canDownload}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>

                <div className="border-t border-border/60 pt-3 space-y-2">
                  <div className="text-xs text-muted-foreground">Actions</div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      className="h-8"
                      onClick={() => beginInlineRename(propsItem)}
                      disabled={!caps?.canRename}
                    >
                      <Pencil className="h-4 w-4 mr-2" />
                      Rename
                    </Button>
                    <Button
                      variant="outline"
                      className="h-8"
                      onClick={() => openItem(propsItem)}
                      disabled={propsItem.type === "folder" ? !caps?.canBrowse : !caps?.canDownload}
                    >
                      <Eye className="h-4 w-4 mr-2" />
                      Open
                    </Button>
                    <Button
                      variant="outline"
                      className="h-8"
                      onClick={() => {
                        if (canTrash) actionDelete("trash");
                        else actionDelete("permanent");
                      }}
                      disabled={!caps?.canDelete}
                    >
                      <Trash2 className="h-4 w-4 mr-2 text-red-500" />
                      <span className="text-red-500">{canTrash ? "Trash" : "Delete"}</span>
                    </Button>
                  </div>

                  {!canTrash && (
                    <div className="text-[11px] text-muted-foreground flex items-start gap-2">
                      <ShieldAlert className="h-4 w-4 mt-0.5" />
                      <div>
                        Trash is disabled until backend “dumpster” endpoints exist.
                        Delete currently uses permanent mode (with confirm).
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border/60 p-2 text-xs text-muted-foreground flex items-center justify-between">
          <span>
            {items.length} item(s) • {sortedItems.length !== items.length ? `${sortedItems.length} shown` : "shown"} •{" "}
            Mode: {viewMode} • Sort: {sortKey} ({sortDir})
          </span>
          <span>{clipboard ? `Clipboard: ${clipboard.mode} • ${clipboard.items.length} item(s)` : "Clipboard: empty"}</span>
        </div>

        {/* Context Menu */}
        {ctxOpen && (
          <div
            className="fixed z-[9999] min-w-[240px] rounded-md border border-border bg-popover shadow-lg p-1"
            style={{ left: ctxXY.x, top: ctxXY.y }}
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <CtxItem
              icon={<Eye className="h-4 w-4" />}
              label="Open"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) openItem(ctxItem);
                closeContext();
              }}
            />
            <CtxItem
              icon={<Info className="h-4 w-4" />}
              label="Properties"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) openProperties(ctxItem);
                closeContext();
              }}
            />

            <div className="my-1 border-t border-border/60" />

            <CtxItem
              icon={<Download className="h-4 w-4" />}
              label="Download"
              disabled={!caps?.canDownload || !ctxItem || ctxItem.type !== "file"}
              onClick={() => {
                if (ctxItem) actionDownload(ctxItem);
                closeContext();
              }}
            />

            <CtxItem
              icon={<ClipboardCopy className="h-4 w-4" />}
              label="Copy Name"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) copyToClipboard(ctxItem.name, "Name");
                closeContext();
              }}
            />
            <CtxItem
              icon={<ClipboardCopy className="h-4 w-4" />}
              label="Copy Path"
              disabled={!ctxItem}
              onClick={() => {
                if (ctxItem) copyToClipboard(ctxItem.path, "Path");
                closeContext();
              }}
            />

            <div className="my-1 border-t border-border/60" />

            <CtxItem
              icon={<Copy className="h-4 w-4" />}
              label="Copy"
              disabled={!ctxItem && !selectedItems.length}
              onClick={() => {
                closeContext();
                actionCopy();
              }}
            />
            <CtxItem
              icon={<Scissors className="h-4 w-4" />}
              label="Cut"
              disabled={!ctxItem && !selectedItems.length}
              onClick={() => {
                closeContext();
                actionCut();
              }}
            />
            <CtxItem
              icon={<ClipboardPaste className="h-4 w-4" />}
              label="Paste"
              disabled={!clipboard}
              onClick={() => {
                closeContext();
                actionPaste();
              }}
            />

            <div className="my-1 border-t border-border/60" />

            <CtxItem
              icon={<Pencil className="h-4 w-4" />}
              label="Rename"
              disabled={!caps?.canRename || selectedItems.length !== 1}
              onClick={() => {
                closeContext();
                actionRename();
              }}
            />

            <CtxItem
              icon={<Trash2 className="h-4 w-4" />}
              label={canTrash ? "Move to Trash" : "Delete"}
              danger
              disabled={!caps?.canDelete || !selectedItems.length}
              onClick={() => {
                closeContext();
                if (canTrash) actionDelete("trash");
                else actionDelete("permanent");
              }}
            />

            {!canTrash && (
              <div className="px-2 py-1.5 text-[11px] text-muted-foreground">
                Trash needs backend “dumpster” endpoints. Delete uses permanent mode.
              </div>
            )}
          </div>
        )}

        {/* Preview Modal */}
        {preview.open && (
          <div
            className="fixed inset-0 z-[10000] bg-black/50 flex items-center justify-center p-4"
            onClick={closePreview}
          >
            <div
              className="w-full max-w-5xl max-h-[85vh] rounded-xl border border-border bg-background shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-2 p-3 border-b border-border/60 bg-muted/20">
                <div className="min-w-0">
                  <div className="font-semibold truncate">{preview.title}</div>
                  <div className="text-xs text-muted-foreground truncate">{preview.item?.path || ""}</div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    className="h-8"
                    onClick={() => preview.item && actionDownload(preview.item)}
                    disabled={!preview.item || preview.item.type !== "file"}
                    title="Download"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                  <Button variant="outline" size="icon" className="h-8 w-8" onClick={closePreview} title="Close">
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div className="p-3 overflow-auto max-h-[calc(85vh-56px)]">
                {preview.loading && <div className="text-sm text-muted-foreground p-6">Opening…</div>}
                {preview.error && <div className="text-sm text-red-500 p-6">{preview.error}</div>}

                {!preview.loading && !preview.error && preview.kind === "image" && preview.url && (
                  <div className="flex justify-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={preview.url}
                      alt={preview.title}
                      className="max-h-[70vh] max-w-full rounded-lg border border-border/50"
                    />
                  </div>
                )}

                {!preview.loading && !preview.error && preview.kind === "pdf" && preview.url && (
                  <iframe src={preview.url} className="w-full h-[70vh] rounded-lg border border-border/50" title={preview.title} />
                )}

                {!preview.loading && !preview.error && preview.kind === "text" && (
                  <pre className="text-xs leading-relaxed p-3 rounded-lg border border-border/50 bg-muted/20 overflow-auto">
                    {preview.text || ""}
                  </pre>
                )}

                {!preview.loading && !preview.error && preview.kind === "unknown" && preview.url && (
                  <div className="p-6 text-sm text-muted-foreground">Preview not supported for this file type. Use Download.</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function HeaderCell(props: { label: string; active: boolean; dir: "asc" | "desc"; onClick: () => void }) {
  const { label, active, dir, onClick } = props;
  return (
    <button
      className={cn("text-left text-xs font-semibold flex items-center gap-2 hover:underline", active && "text-foreground")}
      onClick={onClick}
    >
      <span>{label}</span>
      {active && <span className="text-[10px] text-muted-foreground">{dir === "asc" ? "▲" : "▼"}</span>}
    </button>
  );
}

function HeaderCellRight(props: { label: string; active: boolean; dir: "asc" | "desc"; onClick: () => void }) {
  const { label, active, dir, onClick } = props;
  return (
    <button
      className={cn(
        "text-right text-xs font-semibold flex items-center justify-end gap-2 hover:underline",
        active && "text-foreground"
      )}
      onClick={onClick}
    >
      <span>{label}</span>
      {active && <span className="text-[10px] text-muted-foreground">{dir === "asc" ? "▲" : "▼"}</span>}
    </button>
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

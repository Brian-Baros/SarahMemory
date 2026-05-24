import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Loader2,
  ExternalLink,
  ArrowLeft,
  ArrowRight,
  RefreshCw,
  Globe,
  Search,
  BookOpen,
  MessageSquare,
  Copy,
  ClipboardPaste,
  FileCode,
  ShieldCheck,
  Send,
} from "lucide-react";
import { toast } from "sonner";
import { useNavigationStore } from "@/stores/useNavigationStore";
import { apiFetch } from "@/lib/config";

type FetchBundle = {
  ok: boolean;
  url?: string;
  title?: string;
  clean_html?: string;
  text?: string;
  links?: Array<{ text: string; url: string }>;
  content_type?: string;
  error?: string;
  detail?: string;
};

type ResearchAction = { type: string; payload?: any };

type DevBridgeHealthResponse = {
  ok?: boolean;
  enabled?: boolean;
  url?: string;
  session_target?: string;
  apply_gate?: {
    developer_mode?: boolean;
    env_allow_apply?: boolean;
    loopback?: boolean;
  };
  error?: string;
};

type DevBridgeImportResponse = {
  ok?: boolean;
  response_id?: string;
  stage_id?: string;
  staged?: boolean;
  staged_files?: number;
  message?: string;
  error?: string;
};

type DevBridgeCmdTicketCounts = {
  pending?: number;
  processed?: number;
  failed?: number;
};

type DevBridgeCmdTicketItem = {
  file?: string;
  path?: string;
  category?: string;
  ticket_type?: string;
  command?: string;
  ticket_id?: string;
  summary?: string;
  paths?: string[];
  status?: string;
  error?: string;
  result_error?: string;
  result_ok?: boolean;
  processed_at?: string;
  processed_ts?: number;
  mtime?: number;
  mtime_iso?: string;
  diagnostics?: {
    valid_json?: boolean;
    error?: string;
    used_tolerant_parse?: boolean;
    size_bytes?: number;
    sha256?: string;
  };
};

type DevBridgeCmdTicketsResponse = {
  ok?: boolean;
  counts?: DevBridgeCmdTicketCounts;
  extended_counts?: DevBridgeCmdTicketCounts & { archived_failed?: number; total_inventory?: number };
  pending?: DevBridgeCmdTicketItem[];
  pending_count?: number;
  failed?: DevBridgeCmdTicketItem[];
  failed_count?: number;
  processed_recent?: DevBridgeCmdTicketItem[];
  processed_recent_count?: number;
  archived_failed_recent?: DevBridgeCmdTicketItem[];
  archived_failed_count?: number;
  latest_failed?: DevBridgeCmdTicketItem | null;
  latest_processed?: DevBridgeCmdTicketItem | null;
  invalid_count?: number;
  requested_limit?: number;
  detail_limit?: number;
  max_per_process?: number;
  retention_days?: number;
  inventory_note?: string;
  generated_at?: string;
  updated_at?: string;
  error?: string;
};

type DevBridgeCmdTicketProcessResponse = {
  ok?: boolean;
  dry_run?: boolean;
  requested_limit?: number;
  discovered_count?: number;
  processed_count?: number;
  failed_count?: number;
  invalid_json_count?: number;
  counts?: DevBridgeCmdTicketCounts;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  results?: Array<{ ok?: boolean; file?: string; ticket_type?: string; command?: string; summary?: string; error?: string; detail?: string }>;
  error?: string;
};

type DevBridgeRuntimeProbe = {
  ok: boolean;
  path: string;
  data?: unknown;
  error?: string;
};

type DevBridgeRuntimeSnapshot = {
  generated_at: string;
  probes: DevBridgeRuntimeProbe[];
};

const RESEARCH_PENDING_KEY = "sarahmemory:panel:pending:research";
const DEVBRIDGE_RESPONSE_KEY = "sarahmemory:devbridge:last_response";
const DEFAULT_CHATGPT_PROJECT_URL =
  "https://chatgpt.com/g/g-p-67e0408d2a048191ae36592e6e45ae89-the-sarahmemory-project-v8-0-0-aios/c/69fde7eb-0204-83e8-8b9e-65636d3ece39";
const DEVBRIDGE_PREPARE_COOLDOWN_MS = 5000;

function simpleHash(value: string): string {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(8, "0");
}

function normalizeUrl(raw: string): string {
  const u = (raw || "").trim();
  if (!u) return "";
  if (u.startsWith("http://") || u.startsWith("https://")) return u;
  if (/^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$/i.test(u)) return `https://${u}`;
  return u;
}

function isProbablyUrl(raw: string): boolean {
  const u = (raw || "").trim();
  if (!u) return false;
  if (u.startsWith("http://") || u.startsWith("https://")) return true;
  return /^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$/i.test(u);
}

function searchUrl(query: string): string {
  return `https://duckduckgo.com/?q=${encodeURIComponent((query || "").trim())}`;
}

function actionType(action: ResearchAction): string {
  return String(action?.type || "").trim().toLowerCase().replace(/\./g, "_");
}

async function postJson<T>(url: string, body: any): Promise<T> {
  const data = await apiFetch<any>(url, {
    method: "POST",
    body: JSON.stringify(body ?? {}),
  });
  if (data?.ok === false) {
    const msg = data?.error || "Request failed";
    const detail = data?.detail ? ` — ${data.detail}` : "";
    throw new Error(`${msg}${detail}`);
  }
  return data as T;
}

async function getJson<T>(url: string): Promise<T> {
  const data = await apiFetch<any>(url, { method: "GET" });
  if (data?.ok === false) throw new Error(data?.error || "Request failed");
  return data as T;
}


function asDevBridgeRecord(value: unknown): Record<string, any> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, any>) : {};
}

function countItems(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function summarizeDevBridgeLatest(raw: unknown): Record<string, any> {
  const root = asDevBridgeRecord(raw);
  const latest = asDevBridgeRecord(root.latest || root.latest_response);
  const latestJson = asDevBridgeRecord(latest.parsed_json);
  const latestPacket = asDevBridgeRecord(root.latest_packet);
  const latestPacketPage = asDevBridgeRecord(latestPacket.current_page);
  const latestPacketPanel = asDevBridgeRecord(latestPacket.panel);
  const latestPacketRuntime = asDevBridgeRecord(latestPacket.devbridge_runtime);
  const latestPacketProbes = Array.isArray(latestPacketRuntime.probes) ? latestPacketRuntime.probes : [];
  const latestStage = asDevBridgeRecord(root.latest_stage);
  const stageValidation = asDevBridgeRecord(latestStage.validation);
  const stageApply = asDevBridgeRecord(latestStage.apply_result);

  return {
    ok: Boolean(root.ok),
    service: root.service || "devbridge",
    version: root.version || "8.0.0",
    ts: root.ts || null,
    bridge_root: root.bridge_root || "",
    counts: root.counts || {},
    latest_response: latest.response_id
      ? {
          response_id: latest.response_id,
          imported_at: latest.imported_at || "",
          source: latest.source || "",
          sha256: latest.sha256 || "",
          stage_id: latest.stage_id || "",
          staged: Boolean(latest.staged),
          packet_type: latestJson.packet_type || "",
          status: latestJson.status || "",
          summary: latestJson.summary || "",
          patched_files_count: countItems(latestJson.patched_files),
        }
      : null,
    latest_packet: latestPacket._path
      ? {
          path: latestPacket._path || "",
          created_at: latestPacket.created_at || "",
          source: latestPacket.source || "",
          task: latestPacket.task || "",
          panel: {
            name: latestPacketPanel.name || "",
            current_url: latestPacketPanel.current_url || "",
            reader_loaded: Boolean(latestPacketPanel.reader_loaded),
          },
          current_page: {
            ok: Boolean(latestPacketPage.ok),
            url: latestPacketPage.url || "",
            title: latestPacketPage.title || "",
            content_type: latestPacketPage.content_type || "",
            text_excerpt_chars: String(latestPacketPage.text_excerpt || "").length,
            links_count: countItems(latestPacketPage.links),
          },
          devbridge_probe_summary: latestPacketProbes.map((probe: any) => ({
            ok: Boolean(probe?.ok),
            path: String(probe?.path || ""),
          })),
        }
      : null,
    latest_stage: latestStage.stage_id
      ? {
          stage_id: latestStage.stage_id,
          status: latestStage.status || "",
          created_at: latestStage.created_at || "",
          files_count: countItems(latestStage.files),
          validation_ok: Boolean(stageValidation.ok),
          applied_ok: Boolean(stageApply.ok),
        }
      : null,
    compacted: true,
    omitted_large_fields: ["response_text", "copy_text", "latest_record", "nested_latest_payloads"],
  };
}


function formatCmdTicketTime(value?: string | number): string {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function safeCount(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function describeCmdTicket(ticket: DevBridgeCmdTicketItem): string {
  return ticket.summary || ticket.error || ticket.result_error || ticket.command || ticket.ticket_id || "No summary provided.";
}

async function getDevBridgeRuntimeSnapshot(): Promise<DevBridgeRuntimeSnapshot> {
  const paths = [
    "/api/devbridge/health",
    "/api/devbridge/status",
    "/api/devbridge/latest",
  ];

  const probes = await Promise.all(
    paths.map(async (path): Promise<DevBridgeRuntimeProbe> => {
      try {
        const data = await getJson<unknown>(path);
        const compactData = path === "/api/devbridge/latest" ? summarizeDevBridgeLatest(data) : data;
        return { ok: true, path, data: compactData };
      } catch (e: any) {
        return { ok: false, path, error: String(e?.message || e || "request_failed") };
      }
    })
  );

  return {
    generated_at: new Date().toISOString(),
    probes,
  };
}

export function ResearchScreen() {
  const { setCurrentScreen } = useNavigationStore();
  const [address, setAddress] = useState("https://duckduckgo.com/");
  const [loading, setLoading] = useState(false);
  const [bundle, setBundle] = useState<FetchBundle | null>(null);
  const [hist, setHist] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState<number>(-1);
  const [bridgeOpen, setBridgeOpen] = useState(false);
  const [bridgeTask, setBridgeTask] = useState(
    "Connect SarahMemory ResearchPanel to the current ChatGPT project session and help repair FrontEnd/backend hooks."
  );
  const [bridgePacket, setBridgePacket] = useState("");
  const [chatGptMessageDraft, setChatGptMessageDraft] = useState("");
  const [lastPreparedAt, setLastPreparedAt] = useState<number>(0);
  const [lastPreparedHash, setLastPreparedHash] = useState("");
  const [bridgeResponse, setBridgeResponse] = useState("");
  const [bridgeStatus, setBridgeStatus] = useState("");
  const [bridgeTargetUrl, setBridgeTargetUrl] = useState(DEFAULT_CHATGPT_PROJECT_URL);
  const [bridgeBusy, setBridgeBusy] = useState(false);
  const [cmdTicketBusy, setCmdTicketBusy] = useState(false);
  const [cmdTicketStatus, setCmdTicketStatus] = useState("");
  const [cmdTickets, setCmdTickets] = useState<DevBridgeCmdTicketsResponse | null>(null);
  const [lastCmdTicketBatch, setLastCmdTicketBatch] = useState<DevBridgeCmdTicketProcessResponse | null>(null);
  const [devBridgeDeveloperMode, setDevBridgeDeveloperMode] = useState(false);
  const [devBridgeStatusKnown, setDevBridgeStatusKnown] = useState(false);
  const htmlRef = useRef<HTMLDivElement | null>(null);

  const bundleRef = useRef<FetchBundle | null>(null);
  const histRef = useRef<string[]>([]);
  const histIdxRef = useRef<number>(-1);
  const loadingRef = useRef(false);

  useEffect(() => {
    let alive = true;

    getJson<DevBridgeHealthResponse>("/api/devbridge/health")
      .then((data) => {
        if (!alive) return;

        const gate = data?.apply_gate || {};
        const developerMode = Boolean(gate.developer_mode || gate.env_allow_apply);
        const url = data?.url || data?.session_target;

        setDevBridgeDeveloperMode(developerMode);
        setDevBridgeStatusKnown(true);

        if (typeof url === "string" && url.trim()) {
          setBridgeTargetUrl(url.trim());
        }

        if (!developerMode) {
          setBridgeOpen(false);
        }
      })
      .catch(() => {
        if (!alive) return;
        setDevBridgeDeveloperMode(false);
        setDevBridgeStatusKnown(true);
        setBridgeOpen(false);
      });

    return () => {
      alive = false;
    };
  }, []);

  const canBack = histIdx > 0;
  const canFwd = histIdx >= 0 && histIdx < hist.length - 1;

  const currentUrl = useMemo(() => {
    if (histIdx >= 0 && histIdx < hist.length) return hist[histIdx];
    return bundle?.url || "";
  }, [bundle?.url, hist, histIdx]);

  const cmdTicketCounts = cmdTickets?.counts || {};
  const cmdTicketGeneratedAt = cmdTickets?.generated_at || cmdTickets?.updated_at || "";
  const cmdTicketPendingCount = safeCount(cmdTickets?.pending_count ?? cmdTicketCounts.pending);
  const cmdTicketFailedCount = safeCount(cmdTickets?.failed_count ?? cmdTicketCounts.failed);
  const cmdTicketProcessedCount = safeCount(cmdTicketCounts.processed);
  const cmdTicketArchivedFailedCount = safeCount(cmdTickets?.archived_failed_count ?? cmdTickets?.extended_counts?.archived_failed);
  const cmdTicketProcessDisabled = cmdTicketBusy || cmdTicketPendingCount <= 0;

  useEffect(() => {
    bundleRef.current = bundle;
  }, [bundle]);

  useEffect(() => {
    histRef.current = hist;
  }, [hist]);

  useEffect(() => {
    histIdxRef.current = histIdx;
  }, [histIdx]);

  const syncBrowserState = useCallback(async (nextBundle: FetchBundle | null, source = "research_panel") => {
    if (!nextBundle) return;
    try {
      await postJson("/api/browser/state", {
        source,
        surface: "research_panel",
        url: nextBundle.url || "",
        title: nextBundle.title || nextBundle.url || "",
        text: nextBundle.text || "",
        clean_html: nextBundle.clean_html || "",
        links: Array.isArray(nextBundle.links) ? nextBundle.links.slice(0, 50) : [],
        content_type: nextBundle.content_type || "",
        ok: Boolean(nextBundle.ok),
        ts: Date.now(),
      });
    } catch {
      // Browser state sync is non-critical; the reader still works.
    }
  }, []);

  const fetchReader = useCallback(
    async (rawUrl: string, historyMode: "push" | "replaceCurrent" | "silent" = "push") => {
      const u = normalizeUrl(rawUrl);
      if (!u) return;

      if (loadingRef.current) return;
      loadingRef.current = true;
      setLoading(true);

      try {
        const data = await postJson<FetchBundle>("/api/browser/fetch", { url: u });
        const nextUrl = data.url || u;
        setBundle(data);
        setAddress(nextUrl);
        void syncBrowserState(data, "fetch_reader");

        if (historyMode === "push") {
          const idx = histIdxRef.current;
          const base = histRef.current.slice(0, idx + 1);
          const nextHist = base[base.length - 1] === nextUrl ? base : [...base, nextUrl];
          histRef.current = nextHist;
          histIdxRef.current = nextHist.length - 1;
          setHist(nextHist);
          setHistIdx(nextHist.length - 1);
        } else if (historyMode === "replaceCurrent") {
          const idx = Math.max(0, histIdxRef.current);
          const prev = histRef.current.length > 0 ? [...histRef.current] : [nextUrl];
          prev[idx] = nextUrl;
          histRef.current = prev;
          histIdxRef.current = idx;
          setHist(prev);
          setHistIdx(idx);
        }
      } catch (e: any) {
        const failBundle: FetchBundle = {
          ok: false,
          error: e?.message || "Fetch failed",
          url: u,
          title: u,
          clean_html: `<pre>${String(e?.message || "Fetch failed")}</pre>`,
          text: String(e?.message || "Fetch failed"),
          links: [],
        };
        toast.error(e?.message || "Fetch failed");
        setBundle(failBundle);
        setAddress(u);
        void syncBrowserState(failBundle, "fetch_error");
      } finally {
        loadingRef.current = false;
        setLoading(false);
      }
    },
    [syncBrowserState]
  );

  const readCurrentPage = useCallback(async () => {
    const b = bundleRef.current;
    if (!b?.url) {
      toast.info("No Research Browser page is loaded yet.");
      return;
    }
    await syncBrowserState(b, "manual_read_current");
    toast.success("Research Browser page state sent to SarahMemory.");
  }, [syncBrowserState]);

  const applyResearchAction = useCallback(
    (action: ResearchAction) => {
      const t = actionType(action);
      const p = action?.payload || {};

      if (t === "navigate" || t === "set_screen" || t === "nav_set_screen") {
        const screen = p?.screen || p?.route || p?.app;
        if (typeof screen === "string" && screen.trim()) setCurrentScreen(screen.replace(/^\//, "") as any);
        return;
      }

      if (t === "research_open" || t === "browser_open") {
        const url = p?.url || p?.href || p?.address || p?.value;
        if (typeof url === "string" && url.trim()) void fetchReader(url.trim(), "push");
        return;
      }

      if (t === "research_search" || t === "browser_search") {
        const query = p?.query || p?.q || p?.text || p?.value;
        if (typeof query === "string" && query.trim()) void fetchReader(searchUrl(query), "push");
        return;
      }

      if (t === "research_back" || t === "browser_back") {
        const idx = histIdxRef.current;
        const h = histRef.current;
        if (idx > 0) {
          const nextIdx = idx - 1;
          histIdxRef.current = nextIdx;
          setHistIdx(nextIdx);
          void fetchReader(h[nextIdx], "silent");
        }
        return;
      }

      if (t === "research_forward" || t === "browser_forward") {
        const idx = histIdxRef.current;
        const h = histRef.current;
        if (idx >= 0 && idx < h.length - 1) {
          const nextIdx = idx + 1;
          histIdxRef.current = nextIdx;
          setHistIdx(nextIdx);
          void fetchReader(h[nextIdx], "silent");
        }
        return;
      }

      if (t === "research_reload" || t === "browser_reload") {
        const u = histRef.current[histIdxRef.current] || bundleRef.current?.url || address;
        if (u) void fetchReader(u, "replaceCurrent");
        return;
      }

      if (t === "research_read_current" || t === "browser_read_current") {
        void readCurrentPage();
      }
    },
    [address, fetchReader, readCurrentPage, setCurrentScreen]
  );

  const flushPendingResearchActions = useCallback(() => {
    try {
      const raw = window.sessionStorage.getItem(RESEARCH_PENDING_KEY);
      if (!raw) return;
      window.sessionStorage.removeItem(RESEARCH_PENDING_KEY);
      const actions = JSON.parse(raw);
      if (Array.isArray(actions)) actions.forEach((a) => applyResearchAction(a));
    } catch {
      // ignore bad pending payloads
    }
  }, [applyResearchAction]);

  useEffect(() => {
    const u = normalizeUrl(address);
    if (u && isProbablyUrl(u)) {
      const initial = [u];
      histRef.current = initial;
      histIdxRef.current = 0;
      setHist(initial);
      setHistIdx(0);
      void fetchReader(u, "replaceCurrent");
    }
    queueMicrotask(flushPendingResearchActions);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions)) return;
      actions.forEach((a) => applyResearchAction(a));
    };

    window.addEventListener("sarah:ui", handler as any);
    window.addEventListener("sarah:research", handler as any);
    window.addEventListener("focus", flushPendingResearchActions);
    flushPendingResearchActions();

    return () => {
      window.removeEventListener("sarah:ui", handler as any);
      window.removeEventListener("sarah:research", handler as any);
      window.removeEventListener("focus", flushPendingResearchActions);
    };
  }, [applyResearchAction, flushPendingResearchActions]);

  async function handleGo() {
    const raw = address.trim();
    if (!raw) return;
    await fetchReader(isProbablyUrl(raw) ? raw : searchUrl(raw), "push");
  }

  async function handleBack() {
    if (!canBack) return;
    const nextIdx = histIdx - 1;
    const url = hist[nextIdx];
    histIdxRef.current = nextIdx;
    setHistIdx(nextIdx);
    await fetchReader(url, "silent");
  }

  async function handleForward() {
    if (!canFwd) return;
    const nextIdx = histIdx + 1;
    const url = hist[nextIdx];
    histIdxRef.current = nextIdx;
    setHistIdx(nextIdx);
    await fetchReader(url, "silent");
  }

  async function handleReload() {
    const u = currentUrl || address;
    if (!u) return;
    await fetchReader(u, "replaceCurrent");
  }

  function openInLiveBrowser(url: string) {
    const u = normalizeUrl(url);
    if (!u) return;

    const anyWin = window as any;
    if (anyWin?.chrome?.webview?.postMessage) {
      try {
        anyWin.chrome.webview.postMessage(JSON.stringify({ type: "NAVIGATE_BROWSER", url: u }));
        toast.success("Routed to Live Browser");
        return;
      } catch {
        // fall through
      }
    }

    postJson("/api/browser/open", { url: u })
      .then(() => toast.success("Opened native browser"))
      .catch(() => {
        window.open(u, "_blank", "noopener,noreferrer");
      });
  }

  function buildBridgePacketObject(runtimeSnapshot?: DevBridgeRuntimeSnapshot | null) {
    const b = bundleRef.current;
    const pageText = String(b?.text || "");
    return {
      packet_type: "sarahmemory.devbridge.request.v1",
      created_at: new Date().toISOString(),
      source: "ResearchScreen.tsx",
      target_session_url: bridgeTargetUrl,
      task: bridgeTask.trim(),
      panel: {
        name: "ResearchScreen",
        current_url: currentUrl || address,
        reader_loaded: Boolean(b?.url),
        loading: loadingRef.current,
      },
      devbridge_visibility: {
        developer_mode: devBridgeDeveloperMode,
        status_known: devBridgeStatusKnown,
        gated_by: "/api/devbridge/health.apply_gate.developer_mode",
      },
      current_page: b
        ? {
            ok: Boolean(b.ok),
            url: b.url || "",
            title: b.title || "",
            content_type: b.content_type || "",
            text_excerpt: pageText.slice(0, 6000),
            links: Array.isArray(b.links) ? b.links.slice(0, 25) : [],
            error: b.error || "",
            detail: b.detail || "",
          }
        : null,
      devbridge_runtime: runtimeSnapshot || {
        generated_at: new Date().toISOString(),
        probes: [],
        note: "Runtime probes are added when Generate Packet or Copy Packet runs through the DevBridge async path.",
      },
      frontend_contract: {
        component: "data/ui/V8_ui_src/src/components/screens/ResearchScreen.tsx",
        api_helper: "data/ui/V8_ui_src/src/lib/config.ts::apiFetch",
        browser_routes: ["POST /api/browser/fetch", "POST /api/browser/state", "POST /api/browser/open"],
        devbridge_routes: [
          "GET /api/devbridge/health",
          "GET /api/devbridge/session-target",
          "GET /api/devbridge/status",
          "GET /api/devbridge/latest",
          "GET /api/devbridge/cmd-tickets",
          "POST /api/devbridge/cmd-tickets/process",
          "POST /api/devbridge/export-packet",
          "POST /api/devbridge/import-response",
          "POST /api/devbridge/stage-patch",
          "POST /api/devbridge/validate",
          "POST /api/devbridge/apply-approved",
        ],
      },
      instructions_for_chatgpt: [
        "Review the packet and respond with clear repair guidance.",
        "If code is required, return full-file patches only.",
        "For backend staging, JSON may include patched_files: [{ path, content }].",
        "Do not suggest autonomous apply; user approval is required.",
      ],
    };
  }

  function buildChatGptMessage(packetText: string): string {
    return [
      "SarahMemory DevBridge request package.",
      "",
      "Task:",
      bridgeTask.trim() || "Review this SarahMemory DevBridge packet and provide repair guidance.",
      "",
      "Operating rules:",
      "- Review the package and describe what the task does.",
      "- If a code change is required, return full-file patches only.",
      "- Use patched_files only when the user should stage a patch in SarahMemory.",
      "- Do not instruct SarahMemory to auto-apply anything; the user manually approves, imports, stages, validates, and applies.",
      "",
      "DevBridge packet JSON:",
      packetText,
    ].join("\n");
  }

  function ensureDevBridgeAllowed(): boolean {
    if (devBridgeDeveloperMode) return true;
    setBridgeOpen(false);
    toast.error("DevBridge is hidden because Developer Mode is disabled.");
    return false;
  }

  async function refreshCmdTickets(silent = false) {
    if (!ensureDevBridgeAllowed()) return;
    setCmdTicketBusy(true);
    try {
      const data = await getJson<DevBridgeCmdTicketsResponse>("/api/devbridge/cmd-tickets?limit=50&detail_limit=25");
      setCmdTickets(data);
      const counts = data?.counts || {};
      const pending = safeCount(data?.pending_count ?? counts.pending);
      const failed = safeCount(data?.failed_count ?? counts.failed);
      const processed = safeCount(counts.processed);
      const status = `Cmd ticket inventory refreshed: pending ${pending}, processed archive ${processed}, active failed ${failed}, archived failed ${data?.archived_failed_count ?? 0}. Invalid pending JSON: ${data?.invalid_count ?? 0}.`;
      setCmdTicketStatus(status);
      if (!silent) toast.success("Cmd tickets refreshed");
    } catch (e: any) {
      setCmdTicketStatus(`Cmd ticket refresh failed: ${e?.message || e}`);
      if (!silent) toast.error("Cmd ticket refresh failed");
    } finally {
      setCmdTicketBusy(false);
    }
  }

  async function processCmdTickets(dryRun = false) {
    if (!ensureDevBridgeAllowed()) return;
    setCmdTicketBusy(true);
    try {
      const data = await postJson<DevBridgeCmdTicketProcessResponse>("/api/devbridge/cmd-tickets/process", {
        limit: 50,
        dry_run: dryRun,
        retention_days: 3,
      });
      setLastCmdTicketBatch(data);
      const counts = data?.counts || {};
      setCmdTicketStatus(
        `${dryRun ? "Dry-run" : "Processed"} batch: discovered ${data?.discovered_count ?? 0}, ok ${data?.processed_count ?? 0}, failed ${data?.failed_count ?? 0}, invalid JSON ${data?.invalid_json_count ?? 0}. Pending now ${counts.pending ?? 0}. Completed ${data?.completed_at ? formatCmdTicketTime(data.completed_at) : "now"}.`
      );
      toast.success(dryRun ? "Cmd ticket dry-run complete" : "Cmd ticket batch processed");
      await refreshCmdTickets(true);
    } catch (e: any) {
      setCmdTicketStatus(`Cmd ticket process failed: ${e?.message || e}`);
      toast.error("Cmd ticket process failed");
    } finally {
      setCmdTicketBusy(false);
    }
  }

  async function archiveFailedCmdTickets(all = true, file?: string) {
    if (!ensureDevBridgeAllowed()) return;
    setCmdTicketBusy(true);
    try {
      const data = await postJson<any>("/api/devbridge/cmd-tickets/failed/archive", {
        all,
        file,
      });
      setCmdTicketStatus(`Archived ${data?.archived_count ?? 0} active failed cmd ticket(s). Active failed now ${data?.counts?.failed ?? 0}.`);
      toast.success("Failed cmd-ticket inventory archived");
      await refreshCmdTickets(true);
    } catch (e: any) {
      setCmdTicketStatus(`Failed-ticket archive failed: ${e?.message || e}`);
      toast.error("Failed-ticket archive failed");
    } finally {
      setCmdTicketBusy(false);
    }
  }

  async function requeueFailedCmdTicket(file: string) {
    if (!ensureDevBridgeAllowed()) return;
    if (!file) return;
    setCmdTicketBusy(true);
    try {
      const data = await postJson<any>("/api/devbridge/cmd-tickets/failed/requeue", { file });
      setCmdTicketStatus(`Requeued ${data?.requeued_count ?? 0} failed cmd ticket(s). Pending now ${data?.counts?.pending ?? 0}.`);
      toast.success("Failed cmd ticket requeued");
      await refreshCmdTickets(true);
    } catch (e: any) {
      setCmdTicketStatus(`Failed-ticket requeue failed: ${e?.message || e}`);
      toast.error("Failed-ticket requeue failed");
    } finally {
      setCmdTicketBusy(false);
    }
  }

  async function preparePacketForChatGpt() {
    if (!ensureDevBridgeAllowed()) return;
    const now = Date.now();
    if (bridgeBusy) return;

    if (chatGptMessageDraft && now - lastPreparedAt < DEVBRIDGE_PREPARE_COOLDOWN_MS) {
      try {
        await navigator.clipboard.writeText(chatGptMessageDraft);
        setBridgeStatus(
          `Reused the last prepared ChatGPT message (${lastPreparedHash}). Copied to clipboard. Paste it into the ChatGPT session and submit manually.`
        );
        toast.success("Last ChatGPT message copied");
        openChatGptSession();
      } catch {
        toast.error("Clipboard copy failed");
      }
      return;
    }

    setBridgeOpen(true);
    setBridgeBusy(true);
    setBridgeStatus("Preparing one-click ChatGPT package with backend probes...");

    try {
      const runtimeSnapshot = await getDevBridgeRuntimeSnapshot();
      const packet = buildBridgePacketObject(runtimeSnapshot);
      const packetText = JSON.stringify(packet, null, 2);
      const message = buildChatGptMessage(packetText);
      const hash = simpleHash(message);

      setBridgePacket(packetText);
      setChatGptMessageDraft(message);
      setLastPreparedAt(now);
      setLastPreparedHash(hash);

      try {
        const saved = await postJson<any>("/api/devbridge/export-packet", packet);
        if (saved?.packet_id) {
          setBridgeStatus(
            `Prepared ChatGPT package ${hash}, saved packet as ${saved.packet_id}, and copied it to clipboard. Paste it into the ChatGPT session and submit manually.`
          );
        } else {
          setBridgeStatus(
            `Prepared ChatGPT package ${hash} and copied it to clipboard. Paste it into the ChatGPT session and submit manually.`
          );
        }
      } catch (e: any) {
        setBridgeStatus(
          `Prepared ChatGPT package ${hash} locally and copied it to clipboard. Backend export did not complete: ${e?.message || e}`
        );
      }

      try {
        await navigator.clipboard.writeText(message);
        toast.success("ChatGPT package prepared and copied");
      } catch {
        toast.error("Prepared package, but clipboard copy failed");
      }

      openChatGptSession();
    } finally {
      setBridgeBusy(false);
    }
  }

  async function generateBridgePacket() {
    if (!ensureDevBridgeAllowed()) return;
    setBridgeBusy(true);
    setBridgeStatus("Generating packet and probing DevBridge backend status...");
    try {
      const runtimeSnapshot = await getDevBridgeRuntimeSnapshot();
      const packet = buildBridgePacketObject(runtimeSnapshot);
      const pretty = JSON.stringify(packet, null, 2);
      setBridgePacket(pretty);
      setBridgeStatus("Packet generated locally with DevBridge backend probes.");
      try {
        const saved = await postJson<any>("/api/devbridge/export-packet", packet);
        if (saved?.packet_id) setBridgeStatus(`Packet generated with backend probes and saved as ${saved.packet_id}.`);
      } catch (e: any) {
        setBridgeStatus(`Packet generated locally with backend probes. Backend save not available: ${e?.message || e}`);
      }
    } finally {
      setBridgeBusy(false);
    }
  }

  async function copyBridgePacket() {
    if (!ensureDevBridgeAllowed()) return;
    let text = bridgePacket;
    if (!text) {
      const runtimeSnapshot = await getDevBridgeRuntimeSnapshot();
      text = JSON.stringify(buildBridgePacketObject(runtimeSnapshot), null, 2);
      setBridgePacket(text);
    }
    try {
      await navigator.clipboard.writeText(text);
      toast.success("DevBridge packet copied");
      setBridgeStatus("Packet copied. Paste it into the ChatGPT project session.");
    } catch {
      toast.error("Clipboard copy failed");
    }
  }

  async function pasteBridgeResponse() {
    if (!ensureDevBridgeAllowed()) return;
    try {
      const text = await navigator.clipboard.readText();
      setBridgeResponse(text || "");
      setBridgeStatus(text ? "Clipboard response pasted." : "Clipboard was empty.");
    } catch {
      toast.error("Clipboard paste failed");
    }
  }

  async function stageBridgeResponse() {
    if (!ensureDevBridgeAllowed()) return;
    const text = bridgeResponse.trim();
    if (!text) {
      toast.error("Paste the ChatGPT response first.");
      return;
    }
    setBridgeBusy(true);
    try {
      const data = await postJson<DevBridgeImportResponse>("/api/devbridge/import-response", {
        source: "ResearchScreen.tsx",
        packet_text: bridgePacket,
        response_text: text,
        current_url: currentUrl || address,
        ts: Date.now(),
      });
      const msg = data?.message || "Response staged to DevBridge.";
      setBridgeStatus(
        `${msg}${data?.response_id ? ` Response ID: ${data.response_id}.` : ""}${
          data?.stage_id ? ` Stage ID: ${data.stage_id}.` : ""
        }`
      );
      toast.success(data?.staged ? "Response imported and patch staged" : "Response imported");
    } catch (e: any) {
      try {
        window.localStorage.setItem(DEVBRIDGE_RESPONSE_KEY, text);
      } catch {
        // ignore local fallback storage failure
      }
      setBridgeStatus(`Backend staging failed. Response saved locally only: ${e?.message || e}`);
      toast.error("Backend staging failed; saved locally");
    } finally {
      setBridgeBusy(false);
    }
  }

  useEffect(() => {
    if (!devBridgeDeveloperMode || !bridgeOpen) return;
    void refreshCmdTickets(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [devBridgeDeveloperMode, bridgeOpen]);

  function openChatGptSession() {
    if (!ensureDevBridgeAllowed()) return;
    window.open(bridgeTargetUrl || DEFAULT_CHATGPT_PROJECT_URL, "_blank", "noopener,noreferrer");
  }

  function onReaderClickCapture(e: React.MouseEvent) {
    const target = e.target as HTMLElement | null;
    if (!target) return;

    const a = target.closest("a") as HTMLAnchorElement | null;
    if (!a) return;

    const href = (a.getAttribute("href") || "").trim();
    if (!href) return;

    e.preventDefault();
    e.stopPropagation();
    void fetchReader(href, "push");
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 p-3">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="icon" disabled={!canBack || loading} onClick={handleBack}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={!canFwd || loading} onClick={handleForward}>
          <ArrowRight className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={loading} onClick={handleReload}>
          <RefreshCw className="h-4 w-4" />
        </Button>

        <div className="flex-1 flex items-center gap-2">
          <Input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleGo();
            }}
            placeholder="Enter URL or search query"
          />
          <Button onClick={() => void handleGo()} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            <span className="ml-2">Go</span>
          </Button>
        </div>

        <Button variant="secondary" onClick={() => openInLiveBrowser(currentUrl || address)} disabled={loading}>
          <Globe className="h-4 w-4" />
          <span className="ml-2">Live Browser</span>
        </Button>

        <Button variant="outline" onClick={() => void readCurrentPage()} disabled={loading || !bundle?.url}>
          <BookOpen className="h-4 w-4" />
          <span className="ml-2">Send Page State</span>
        </Button>

        {devBridgeDeveloperMode && (
          <Button variant={bridgeOpen ? "default" : "outline"} onClick={() => setBridgeOpen((v) => !v)}>
            <MessageSquare className="h-4 w-4" />
            <span className="ml-2">ChatGPT Bridge</span>
          </Button>
        )}

        <Button variant="outline" onClick={() => window.open(normalizeUrl(currentUrl || address) || "about:blank", "_blank")}>
          <ExternalLink className="h-4 w-4" />
        </Button>
      </div>

      {devBridgeDeveloperMode && bridgeOpen && (
        <div className="rounded-lg border bg-muted/20 p-3 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="font-semibold flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                ChatGPT DevBridge Mode
              </div>
              <div className="text-xs opacity-75">Manual packet bridge. No OpenAI backend API call is made by SarahMemory.</div>
              <div className="text-xs opacity-75">Visible only when Developer Mode is enabled by the backend gate.</div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="secondary" onClick={openChatGptSession}>
                <ExternalLink className="h-4 w-4" />
                <span className="ml-2">Open Session</span>
              </Button>
              <Button size="sm" onClick={() => void preparePacketForChatGpt()} disabled={bridgeBusy}>
                {bridgeBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                <span className="ml-2">Prepare for ChatGPT</span>
              </Button>
              <Button size="sm" variant="outline" onClick={() => void generateBridgePacket()}>
                <FileCode className="h-4 w-4" />
                <span className="ml-2">Generate Packet</span>
              </Button>
              <Button size="sm" variant="outline" onClick={() => void copyBridgePacket()}>
                <Copy className="h-4 w-4" />
                <span className="ml-2">Copy Packet</span>
              </Button>
              <Button size="sm" variant="outline" onClick={() => void pasteBridgeResponse()}>
                <ClipboardPaste className="h-4 w-4" />
                <span className="ml-2">Paste Response</span>
              </Button>
              <Button size="sm" onClick={() => void stageBridgeResponse()} disabled={bridgeBusy}>
                {bridgeBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
                <span className="ml-2">Stage to Backend</span>
              </Button>
              <Button size="sm" variant="outline" onClick={() => void refreshCmdTickets()} disabled={cmdTicketBusy}>
                {cmdTicketBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                <span className="ml-2">Refresh Cmd Tickets</span>
              </Button>
              <Button size="sm" variant="outline" onClick={() => void processCmdTickets(true)} disabled={cmdTicketProcessDisabled}>
                <FileCode className="h-4 w-4" />
                <span className="ml-2">Dry Run Cmd</span>
              </Button>
              <Button size="sm" variant="secondary" onClick={() => void processCmdTickets(false)} disabled={cmdTicketProcessDisabled}>
                <ShieldCheck className="h-4 w-4" />
                <span className="ml-2">Process Cmd Batch</span>
              </Button>
            </div>
          </div>

          <div className="rounded-lg border bg-background/70 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <FileCode className="h-4 w-4" />
                  Cmd Ticket Inventory
                </div>
                <div className="mt-1 text-xs opacity-75">
                  Filesystem inventory lane only. Runtime files are not patched by cmd tickets. Counts move only when JSON files move between pending, processed, failed, and archive folders.
                </div>
                <div className="mt-1 text-[11px] opacity-60">
                  Last inventory refresh: {cmdTicketGeneratedAt ? formatCmdTicketTime(cmdTicketGeneratedAt) : "not loaded"}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-center text-xs md:grid-cols-4">
                <div className="rounded-md border px-3 py-2">
                  <div className="opacity-60">Pending Queue</div>
                  <div className="font-semibold">{cmdTicketPendingCount}</div>
                  <div className="mt-1 text-[10px] opacity-50">processable now</div>
                </div>
                <div className="rounded-md border px-3 py-2">
                  <div className="opacity-60">Processed Archive</div>
                  <div className="font-semibold">{cmdTicketProcessedCount}</div>
                  <div className="mt-1 text-[10px] opacity-50">historical success</div>
                </div>
                <div className="rounded-md border px-3 py-2">
                  <div className="opacity-60">Active Failed</div>
                  <div className={cmdTicketFailedCount > 0 ? "font-semibold text-red-500" : "font-semibold"}>{cmdTicketFailedCount}</div>
                  <div className="mt-1 text-[10px] opacity-50">needs review</div>
                </div>
                <div className="rounded-md border px-3 py-2">
                  <div className="opacity-60">Failed Archive</div>
                  <div className="font-semibold">{cmdTicketArchivedFailedCount}</div>
                  <div className="mt-1 text-[10px] opacity-50">audit evidence</div>
                </div>
              </div>
            </div>

            {!!cmdTicketStatus && <div className="mt-2 rounded border bg-muted/20 p-2 text-xs">{cmdTicketStatus}</div>}

            {cmdTicketPendingCount <= 0 && (
              <div className="mt-3 rounded border border-dashed p-3 text-xs opacity-80">
                No pending cmd tickets are queued. Dry Run Cmd and Process Cmd Batch will not change counts until new .json tickets are placed in the pending folder.
              </div>
            )}

            {cmdTicketPendingCount > 0 && (
              <div className="mt-3 max-h-52 overflow-auto rounded border">
                {(cmdTickets?.pending || []).slice(0, 50).map((ticket, idx) => (
                  <div key={`${ticket.file || "ticket"}-${idx}`} className="border-b p-2 text-xs last:border-b-0">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-mono font-semibold">{ticket.file || "ticket.json"}</span>
                      <span className="rounded bg-muted px-2 py-0.5 uppercase">{ticket.ticket_type || "unknown"}</span>
                    </div>
                    <div className="mt-1 truncate opacity-75">{describeCmdTicket(ticket)}</div>
                    {!!ticket.paths?.length && (
                      <div className="mt-1 truncate font-mono opacity-70">{ticket.paths.join(", ")}</div>
                    )}
                    {ticket.diagnostics?.valid_json === false && (
                      <div className="mt-1 text-red-500">Invalid JSON: {ticket.diagnostics?.error || "parse_failed"}</div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              <div className="rounded border bg-muted/10 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold">Active Failed Cmd Tickets</div>
                    <div className="mt-1 text-[11px] opacity-70">Failed inventory is not a live processing error. It is the content of the failed_tickets folder.</div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void archiveFailedCmdTickets(true)}
                    disabled={cmdTicketBusy || cmdTicketFailedCount <= 0}
                  >
                    Archive Failed
                  </Button>
                </div>

                {cmdTicketFailedCount <= 0 && (
                  <div className="mt-3 rounded border border-dashed p-3 text-xs opacity-75">Active failed inventory is clean.</div>
                )}

                {cmdTicketFailedCount > 0 && (
                  <div className="mt-3 max-h-52 overflow-auto rounded border">
                    {(cmdTickets?.failed || []).slice(0, 25).map((ticket, idx) => (
                      <div key={`${ticket.file || "failed"}-${idx}`} className="border-b p-2 text-xs last:border-b-0">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-mono font-semibold">{ticket.file || "failed_ticket.json"}</span>
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => void requeueFailedCmdTicket(ticket.file || "")}
                            disabled={cmdTicketBusy || !ticket.file}
                          >
                            Requeue
                          </Button>
                        </div>
                        <div className="mt-1 text-red-500">{ticket.error || ticket.result_error || "No error summary available."}</div>
                        <div className="mt-1 opacity-60">Processed: {formatCmdTicketTime(ticket.processed_at || ticket.mtime_iso || ticket.mtime)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="rounded border bg-muted/10 p-3">
                <div className="text-xs font-semibold">Last Process Result</div>
                {!lastCmdTicketBatch && (
                  <div className="mt-3 rounded border border-dashed p-3 text-xs opacity-75">No process action has run from this panel in the current UI session.</div>
                )}
                {!!lastCmdTicketBatch && (
                  <div className="mt-2 space-y-2 text-xs">
                    <div className="rounded border bg-background/60 p-2">
                      Mode: {lastCmdTicketBatch.dry_run ? "dry-run" : "process"} / discovered {lastCmdTicketBatch.discovered_count ?? 0} / ok {lastCmdTicketBatch.processed_count ?? 0} / failed {lastCmdTicketBatch.failed_count ?? 0} / invalid JSON {lastCmdTicketBatch.invalid_json_count ?? 0}
                    </div>
                    <div className="text-[11px] opacity-70">Completed: {formatCmdTicketTime(lastCmdTicketBatch.completed_at)}</div>
                    {!!lastCmdTicketBatch.results?.length && (
                      <div className="max-h-40 overflow-auto rounded border">
                        {lastCmdTicketBatch.results.slice(0, 12).map((item, idx) => (
                          <div key={`${item.file || "result"}-${idx}`} className="border-b p-2 last:border-b-0">
                            <span className={item.ok ? "font-semibold" : "font-semibold text-red-500"}>{item.ok ? "OK" : "FAIL"}</span>
                            <span className="ml-2 font-mono">{item.file || "ticket.json"}</span>
                            <div className="mt-1 opacity-70">{item.summary || item.error || item.detail || item.ticket_type || "No detail."}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <label className="grid gap-1 text-sm">
              <span className="font-medium">Task for ChatGPT</span>
              <textarea
                className="min-h-24 rounded-md border bg-background p-2 font-mono text-xs"
                value={bridgeTask}
                onChange={(e) => setBridgeTask(e.target.value)}
              />
            </label>
            <label className="grid gap-1 text-sm">
              <span className="font-medium">ChatGPT Response Import</span>
              <textarea
                className="min-h-24 rounded-md border bg-background p-2 font-mono text-xs"
                value={bridgeResponse}
                onChange={(e) => setBridgeResponse(e.target.value)}
                placeholder="Paste ChatGPT response or JSON patch packet here."
              />
            </label>
          </div>

          <label className="grid gap-1 text-sm">
            <span className="font-medium">Prepared ChatGPT Message</span>
            <textarea
              className="min-h-28 rounded-md border bg-background p-2 font-mono text-xs"
              value={chatGptMessageDraft}
              onChange={(e) => setChatGptMessageDraft(e.target.value)}
              placeholder="Click Prepare for ChatGPT to generate, copy, and open the project session. Paste this message into ChatGPT and submit manually."
            />
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-medium">Generated Packet</span>
            <textarea
              className="min-h-32 rounded-md border bg-background p-2 font-mono text-xs"
              value={bridgePacket}
              onChange={(e) => setBridgePacket(e.target.value)}
              placeholder="Generate a packet, then copy it into the ChatGPT project session."
            />
          </label>

          <div className="text-xs opacity-80">
            <span className="font-medium">Session:</span> {bridgeTargetUrl}
          </div>
          <div className="text-xs opacity-80">
            <span className="font-medium">Prepared hash:</span> {lastPreparedHash || "—"}
          </div>
          {!!bridgeStatus && <div className="text-xs rounded border bg-background p-2">{bridgeStatus}</div>}
        </div>
      )}

      <div className="flex items-center justify-between gap-2 text-sm opacity-80">
        <div className="truncate">
          <span className="font-medium">Reader:</span>{" "}
          <span className="truncate">{bundle?.title || currentUrl || "—"}</span>
        </div>
        <div className="shrink-0">{loading ? "Loading…" : bundle?.content_type ? bundle.content_type : ""}</div>
      </div>

      <div className="flex-1 rounded-lg border bg-background overflow-auto">
        <div
          ref={htmlRef}
          className="prose prose-invert max-w-none p-4"
          onClickCapture={onReaderClickCapture}
          dangerouslySetInnerHTML={{ __html: bundle?.clean_html || "<p>Enter a URL or query, then press Go.</p>" }}
        />
      </div>

      {!!bundle?.links?.length && (
        <div className="rounded-lg border p-3">
          <div className="text-sm font-medium mb-2">Links</div>
          <div className="grid gap-2">
            {bundle.links.slice(0, 12).map((l, idx) => (
              <div key={`${l.url}-${idx}`} className="flex items-center justify-between gap-2">
                <button
                  className="text-left text-sm underline truncate"
                  onClick={() => void fetchReader(l.url, "push")}
                  title={l.url}
                >
                  {l.text || l.url}
                </button>
                <Button variant="outline" size="sm" onClick={() => openInLiveBrowser(l.url)}>
                  Live
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

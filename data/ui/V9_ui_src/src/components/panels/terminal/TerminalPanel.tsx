import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { config } from "@/lib/config";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useIsMobile } from "@/hooks/use-mobile";
import { useSarahStore } from "@/stores/useSarahStore";
import {
  Terminal,
  Shield,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Play,
  Eraser,
  Bot,
} from "lucide-react";

type TerminalEngineMode = "auto" | "windows" | "bash" | "powershell";
type TerminalRoute = "shell" | "ai";

type TerminalExecuteResponse = {
  ok: boolean;
  blocked: boolean;
  reason: string | null;
  session_id: string;
  engine: string | null;
  cwd: string | null;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
  ts: string;
};

type TerminalStatusResponse = {
  ok?: boolean;
  available?: boolean;
  developers_mode?: boolean;
  reason?: string | null;
  session_id?: string;
  cwd?: string | null;
  default_workdir?: string | null;
  base_dir?: string | null;
  prompt?: string | null;
  platform?: string | null;
  ts?: string;
};

type AgentVisibilitySnapshot = {
  ok?: boolean;
  schema?: string;
  summary?: {
    outbound?: number;
    internal?: number;
    incoming?: number;
    quarantined?: number;
    blocked?: number;
    review_required?: number;
    active_passports?: number;
    active_external_ai_processes?: number;
    active_external_ai_connections?: number;
    observed_external_ai_processes?: number;
    observed_external_ai_network_connections?: number;
    sarahmemory_facing_external_ai_processes?: number;
  };
  risk?: {
    low?: number;
    medium?: number;
    high?: number;
    critical?: number;
  };
  execution_authority?: boolean;
};

type TerminalAgentCounterResponse = {
  ok?: boolean;
  schema?: string;
  summary?: Record<string, number>;
  agent_visibility?: AgentVisibilitySnapshot;
  reason?: string;
  execution_authority?: boolean;
};

type AgentScoreboardSection = "overview" | "observed" | "outbound" | "returned" | "quarantined" | "blocked" | "review" | "passports";

type AgentScoreboardResponse = {
  ok?: boolean;
  schema?: string;
  section?: AgentScoreboardSection | string;
  summary?: Record<string, number>;
  agent_visibility?: AgentVisibilitySnapshot;
  details?: Record<string, Array<Record<string, any>>>;
  reason?: string;
  execution_authority?: boolean;
};

type TerminalAIResponse = {
  ok?: boolean;
  blocked?: boolean;
  reason?: string | null;
  reply?: unknown;
  response?: unknown;
  message?: unknown;
  output?: unknown;
  result?: unknown;
  content?: unknown;
  text?: unknown;
  actions?: any[];
  links?: unknown;
  stdout?: unknown;
  stderr?: unknown;
  session_id?: unknown;
  cwd?: unknown;
  agent_status?: Record<string, any> | null;
  meta?: Record<string, any> | null;
};

type TerminalLine = {
  id: string;
  ts: string;
  kind: "input" | "stdout" | "stderr" | "meta";
  text: string;
};

const TERMINAL_PANEL_LINES_KEY = "sarahmemory:terminalpanel:lines";
const TERMINAL_PANEL_HISTORY_KEY = "sarahmemory:terminalpanel:history";
const TERMINAL_MAX_LINES = 500;
const TERMINAL_MAX_HISTORY = 120;
const SARAH_PROMPT = "Sarah:\\>";

function uid() {
  return `t_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function prettyTime(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString();
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function saveJson<T>(key: string, value: T) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // non-fatal
  }
}

function normalizeText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function shorten(value: string, max = 96): string {
  const text = String(value || "").trim();
  if (!text) return "";
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

function dispatchUiActions(actions: any[]) {
  try {
    window.dispatchEvent(
      new CustomEvent("sarah:ui", {
        detail: {
          source: "TerminalPanel",
          ts: Date.now(),
          actions,
        },
      })
    );
  } catch {
    // non-fatal
  }
}

async function requestJSON<T>(url: string, init?: RequestInit): Promise<{ status: number; data: T }> {
  const target = url.startsWith("http") ? url : `${config.apiBaseUrl}${url}`;
  const res = await fetch(target, {
    credentials: "include",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  const txt = await res.text();
  let json: any = null;

  try {
    json = txt ? JSON.parse(txt) : {};
  } catch {
    json = {
      ok: false,
      blocked: false,
      reason: "Non-JSON response",
      stderr: txt,
    };
  }

  return { status: res.status, data: json as T };
}

function looksLikeShellCommand(cmd: string): boolean {
  const text = String(cmd || "").trim();
  if (!text) return false;

  if (text.startsWith("/") || text.startsWith("!")) return true;
  if (/[|><;&]{1,2}/.test(text)) return true;

  const shellish = [
    /^(cd|dir|ls|pwd|echo|type|cat|cls|clear)(\s|$)/i,
    /^(python|py|pip|pytest|uvicorn|flask)(\s|$)/i,
    /^(git|npm|pnpm|yarn|node|npx)(\s|$)/i,
    /^(mkdir|rmdir|rm|del|copy|cp|move|mv|ren|rename)(\s|$)/i,
    /^(where|which|find|grep|ripgrep|rg)(\s|$)/i,
    /^(cmd|bash|powershell|pwsh)(\s|$)/i,
    /^(cargo|rustc|go|java|javac|dotnet)(\s|$)/i,
  ];

  return shellish.some((pattern) => pattern.test(text));
}

function buildAiReply(resp: TerminalAIResponse | null | undefined): string {
  if (!resp) return "";

  return (
    normalizeText(resp.reply) ||
    normalizeText(resp.stdout) ||
    normalizeText(resp.response) ||
    normalizeText(resp.message) ||
    normalizeText(resp.output) ||
    normalizeText(resp.result) ||
    normalizeText(resp.content) ||
    normalizeText(resp.text) ||
    normalizeText(resp.reason) ||
    normalizeText(resp.stderr) ||
    normalizeText(resp.error) ||
    ""
  );
}

function agentRadarRiskLabel(risk?: AgentVisibilitySnapshot["risk"]): string {
  const r = risk || {};
  if (Number(r.critical || 0) > 0) return "CRITICAL";
  if (Number(r.high || 0) > 0) return "HIGH";
  if (Number(r.medium || 0) > 0) return "MEDIUM";
  return "LOW";
}

function formatAgentRadarLine(status: TerminalAgentCounterResponse | null): string {
  const visibility = status?.agent_visibility;
  const summary = visibility?.summary || {};
  if (!status || status.ok === false || !visibility) return "AgentRadar unavailable";
  const base = [
    `Agents: OUT ${Number(summary.outbound || 0)}`,
    `INT ${Number(summary.internal || 0)}`,
    `IN ${Number(summary.incoming || 0)}`,
    `Q ${Number(summary.quarantined || 0)}`,
    `BLOCK ${Number(summary.blocked || 0)}`,
    `Risk: ${agentRadarRiskLabel(visibility.risk)}`,
  ];
  const proc = Number(summary.active_external_ai_processes || 0);
  const conn = Number(summary.active_external_ai_connections || 0);
  const observed = Number(summary.observed_external_ai_processes || 0);
  if (observed > 0) base.push(`Observed AI processes ${observed}`);
  if (proc > 0) base.push(`SarahMemory-facing AI processes ${proc}`);
  if (conn > 0) base.push(`SarahMemory AI connections ${conn}`);
  return base.join(" | ");
}

function formatAgentScoreboardLine(section: AgentScoreboardSection, status: AgentScoreboardResponse | null): string {
  if (!status || status.ok === false) {
    return `Agent Scoreboard ${section.toUpperCase()} unavailable${status?.reason ? `: ${status.reason}` : ""}`;
  }
  const rows = status.details?.[section] || [];
  const header = `Agent Scoreboard ${section.toUpperCase()}: ${rows.length} visible row${rows.length === 1 ? "" : "s"}`;
  if (rows.length === 0) return `${header}\nNo sanitized rows are currently available for this section.`;
  const body = rows.slice(0, 8).map((row, index) => {
    const id =
      row.passport_id ||
      row.task_id ||
      row.artifact_id ||
      row.receipt_id ||
      row.payload_sha256 ||
      "unidentified";
    const statusText = row.status || row.verdict || row.quarantine_status || row.event_type || row.current_stage || "observed";
    const detail = row.details || row.objective || row.who || row.agent_id || row.containment_state || row.risk || "";
    const where = row.where || row.relation_to_sarahmemory || "";
    return `${index + 1}. ${String(id).slice(0, 80)} | ${String(statusText).slice(0, 80)}${detail ? ` | ${String(detail).slice(0, 160)}` : ""}${where ? ` | ${String(where).slice(0, 140)}` : ""}`;
  });
  return [header, ...body].join("\n");
}

export function TerminalPanel() {
  const isMobile = useIsMobile();
  const store = useSarahStore();

  const frontendDevMode = Boolean(
    (store as any)?.developersMode ??
      (store as any)?.DEVELOPERSMODE ??
      (store as any)?.settings?.developersMode ??
      false
  );

  const desktopAllowed = !isMobile;
  // Frontend Developer Mode is only a presentation hint. The backend
  // SarahMemoryTerminal.py / DEVELOPERSMODE gate remains the authority.
  const enabled = desktopAllowed;

  const [mode, setMode] = useState<TerminalEngineMode>("auto");
  const [sessionId, setSessionId] = useState<string>(() => `webui_${Date.now()}`);
  const [cwd, setCwd] = useState<string | null>(null);
  const [input, setInput] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [backendReason, setBackendReason] = useState<string>("");
  const [agentRadar, setAgentRadar] = useState<TerminalAgentCounterResponse | null>(null);
  const [agentRadarUnavailable, setAgentRadarUnavailable] = useState<boolean>(false);
  const [agentRadarBusy, setAgentRadarBusy] = useState<boolean>(false);
  const [agentScoreboardBusy, setAgentScoreboardBusy] = useState<boolean>(false);
  const [agentScoreboardSection, setAgentScoreboardSection] = useState<AgentScoreboardSection>("overview");
  const [historyIndex, setHistoryIndex] = useState<number>(-1);

  const [history, setHistory] = useState<string[]>(() =>
    loadJson<string[]>(TERMINAL_PANEL_HISTORY_KEY, [])
  );

  const [lines, setLines] = useState<TerminalLine[]>(() =>
    loadJson<TerminalLine[]>(TERMINAL_PANEL_LINES_KEY, [
      {
        id: uid(),
        ts: new Date().toISOString(),
        kind: "meta",
        text:
          "Sarah developer terminal initialized. Use /run for raw shell commands, /ai for governed AI tasks, /screen <name> to switch panels.",
      },
    ])
  );

  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const agentRadarBusyRef = useRef<boolean>(false);
  const agentScoreboardBusyRef = useRef<boolean>(false);

  const prompt = useMemo(() => SARAH_PROMPT, []);

  const persistLines = useCallback((next: TerminalLine[]) => {
    saveJson(TERMINAL_PANEL_LINES_KEY, next);
  }, []);

  const append = useCallback(
    (kind: TerminalLine["kind"], text: string) => {
      const safeText = String(text ?? "");
      setLines((prev) => {
        const next = [
          ...prev,
          {
            id: uid(),
            ts: new Date().toISOString(),
            kind,
            text: safeText,
          },
        ].slice(-TERMINAL_MAX_LINES);

        persistLines(next);
        return next;
      });
    },
    [persistLines]
  );

  const resetConsole = useCallback(
    (message = "Console cleared.") => {
      const next = [
        {
          id: uid(),
          ts: new Date().toISOString(),
          kind: "meta" as const,
          text: message,
        },
      ];
      setLines(next);
      persistLines(next);
    },
    [persistLines]
  );

  const pushHistory = useCallback((cmd: string) => {
    const trimmed = String(cmd || "").trim();
    if (!trimmed) return;

    setHistory((prev) => {
      const next = [trimmed, ...prev.filter((x) => x !== trimmed)].slice(0, TERMINAL_MAX_HISTORY);
      saveJson(TERMINAL_PANEL_HISTORY_KEY, next);
      return next;
    });
    setHistoryIndex(-1);
  }, []);

  const focusInput = useCallback(() => {
    window.requestAnimationFrame(() => {
      try {
        inputRef.current?.focus();
      } catch {
        // ignore
      }
    });
  }, []);

  const emitDlTrace = useCallback(
    (payload: Record<string, any>) => {
      dispatchUiActions([
        {
          type: "dlengine_add_trace",
          payload,
        },
      ]);
    },
    []
  );

  const emitDlSubject = useCallback(
    (payload: Record<string, any>) => {
      dispatchUiActions([
        {
          type: "dlengine_add_subject",
          payload,
        },
      ]);
    },
    []
  );

  const checkAgentRadar = useCallback(
    async (emitLine = false) => {
      if (!desktopAllowed || agentRadarBusyRef.current) return;
      agentRadarBusyRef.current = true;
      setAgentRadarBusy(true);
      try {
        const resp = await requestJSON<TerminalAgentCounterResponse>("/api/terminal/agents/status", {
          method: "GET",
        });
        const data = resp.data || {};
        if (data?.execution_authority !== false) {
          data.execution_authority = false;
        }
        setAgentRadar(data);
        setAgentRadarUnavailable(resp.status >= 500 || data.ok === false || !data.agent_visibility);
        if (emitLine) append("meta", formatAgentRadarLine(data));
      } catch {
        setAgentRadar(null);
        setAgentRadarUnavailable(true);
        if (emitLine) append("meta", "AgentRadar unavailable");
      } finally {
        agentRadarBusyRef.current = false;
        setAgentRadarBusy(false);
      }
    },
    [append, desktopAllowed]
  );

  const inspectAgentScoreboard = useCallback(
    async (section: AgentScoreboardSection = "overview", emitLine = true) => {
      if (!desktopAllowed || agentScoreboardBusyRef.current) return;
      agentScoreboardBusyRef.current = true;
      setAgentScoreboardBusy(true);
      setAgentScoreboardSection(section);
      try {
        const resp = await requestJSON<AgentScoreboardResponse>(
          `/api/terminal/agents/scoreboard?section=${encodeURIComponent(section)}&limit=24`,
          { method: "GET" }
        );
        const data = resp.data || {};
        if (data?.execution_authority !== false) {
          data.execution_authority = false;
        }
        if (emitLine) append(resp.status >= 500 || data.ok === false ? "stderr" : "meta", formatAgentScoreboardLine(section, data));
      } catch (e: any) {
        if (emitLine) append("stderr", `Agent Scoreboard ${section.toUpperCase()} unavailable: ${String(e?.message || e || "request failed")}`);
      } finally {
        agentScoreboardBusyRef.current = false;
        setAgentScoreboardBusy(false);
      }
    },
    [append, desktopAllowed]
  );

  const handleLocalDirective = useCallback(
    async (raw: string): Promise<boolean> => {
      const text = String(raw || "").trim();
      if (!text) return false;

      const lower = text.toLowerCase();

      if (lower === "help" || lower === "/help") {
        append(
          "meta",
          [
            "Sarah terminal help:",
            "  /run <command>   Execute a raw shell command in the governed terminal backend.",
            "  /ai <task>       Route a natural-language task through Sarah AI.",
            "  /task <task>     Alias for /ai.",
            "  /agent <task>    Governed inspect/propose agent lane; no autonomous execution.",
            "  /smugcc status   Show SMUGCC contract status.",
            "  /smugcc validate <json>  Validate a SMUGCC envelope without execution.",
            "  /smugcc build mission <objective>  Build a governed mission envelope draft.",
            "  /agents          Show AgentRadar counters.",
            "  /agent status    Show AgentRadar counters.",
            "  /agents observed Inspect observed local/external AI surfaces.",
            "  /agents out      Inspect outbound/passported agents.",
            "  /agents returned Inspect returned/captured agent evidence.",
            "  /agents q        Inspect quarantined agent artifacts.",
            "  /agents blocked  Inspect blocked agent tasks.",
            "  /screen <name>   Switch UI panel, for example /screen dlengine.",
            "  /agent passport help   Show AI-agent passport commands.",
            "  clear | cls      Clear the terminal surface.",
            "",
            "Examples:",
            "  /run python SarahMemoryMain.py",
            "  /run dir",
            "  /ai Create a website landing page for SarahMemory and explain the file plan.",
            "  /agent passport list",
            "  /smugcc trace <task_id-or-json>",
            "  /agent Inspect the AI-agent governance lane.",
          ].join("\n")
        );
        return true;
      }

      const scoreboardMatch = text.match(/^\/agents\s+(observed|observe|obs|out|outbound|returned|return|q|quarantine|quarantined|blocked|block|review|passports|passport)$/i);
      if (scoreboardMatch) {
        const rawSection = String(scoreboardMatch[1] || "").toLowerCase();
        const section: AgentScoreboardSection =
          rawSection === "observed" || rawSection === "observe" || rawSection === "obs"
            ? "observed"
            : rawSection === "out" || rawSection === "outbound"
            ? "outbound"
            : rawSection === "return" || rawSection === "returned"
              ? "returned"
              : rawSection === "q" || rawSection === "quarantine" || rawSection === "quarantined"
                ? "quarantined"
                : rawSection === "block" || rawSection === "blocked"
                  ? "blocked"
                  : rawSection === "passport" || rawSection === "passports"
                    ? "passports"
                    : "review";
        append("input", `${prompt} ${text}`);
        await inspectAgentScoreboard(section, true);
        return true;
      }

      if (
        lower === "/agents" ||
        lower === "/agent status" ||
        lower === "/agent counter" ||
        lower === "/firewall agents" ||
        lower === "/roach status"
      ) {
        append("input", `${prompt} ${text}`);
        await checkAgentRadar(true);
        return true;
      }

      if (lower === "clear" || lower === "cls" || lower === "/clear") {
        resetConsole();
        setInput("");
        focusInput();
        return true;
      }

      const screenMatch = text.match(/^\/screen\s+(.+)$/i);
      if (screenMatch) {
        const screen = String(screenMatch[1] || "").trim();
        if (screen) {
          dispatchUiActions([{ type: "set_screen", payload: { screen } }]);
          append("meta", `UI switched to screen: ${screen}`);
        }
        return true;
      }

      return false;
    },
    [append, checkAgentRadar, focusInput, inspectAgentScoreboard, prompt, resetConsole]
  );

  useEffect(() => {
    if (!scrollerRef.current) return;
    scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
  }, [lines]);

  useEffect(() => {
    if (enabled) {
      focusInput();
    }
  }, [enabled, focusInput]);

  const checkTerminalBackend = useCallback(async () => {
    if (!desktopAllowed) {
      setBackendAvailable(false);
      setBackendReason("Terminal is desktop-only.");
      return;
    }

    try {
      const statusResp = await requestJSON<TerminalStatusResponse>("/api/terminal/status", {
        method: "GET",
      });

      const status = statusResp.data || {};
      const available =
        Boolean(status?.available ?? status?.ok ?? true) && status?.developers_mode !== false;

      setBackendAvailable(available);
      setBackendReason(String(status?.reason || ""));

      if (status?.session_id) setSessionId(String(status.session_id));
      if (status?.cwd) setCwd(String(status.cwd));

      append(
        "meta",
        available
          ? `Terminal backend available.${status?.cwd ? ` CWD=${status.cwd}` : ""}`
          : `Terminal backend unavailable.${status?.reason ? ` ${status.reason}` : ""}`
      );
    } catch (e: any) {
      try {
        const healthResp = await requestJSON<Record<string, any>>("/api/health", { method: "GET" });
        if (healthResp.status >= 200 && healthResp.status < 500) {
          setBackendAvailable(true);
          setBackendReason("Terminal status endpoint unavailable; API health endpoint responded.");
          append("meta", "API health endpoint responded, but /api/terminal/status is not wired or unavailable.");
          return;
        }
      } catch {
        // ignore secondary failure
      }

      setBackendAvailable(false);
      setBackendReason(String(e?.message || e || "Terminal backend unavailable"));
      append("meta", "Terminal backend unavailable. Execution may fail until the API is reachable.");
    }
  }, [append, desktopAllowed, enabled]);

  useEffect(() => {
    void checkTerminalBackend();
  }, [checkTerminalBackend]);

  useEffect(() => {
    if (!desktopAllowed) return;
    void checkAgentRadar(false);
    const timer = window.setInterval(() => {
      void checkAgentRadar(false);
    }, 30000);
    return () => window.clearInterval(timer);
  }, [checkAgentRadar, desktopAllowed]);

  const run = useCallback(
    async (overrideInput?: string) => {
      if (!desktopAllowed) {
        toast.error("Terminal is desktop-only.");
        return;
      }

      const original = String(overrideInput ?? input).trim();
      if (!original || busy) return;

      const handledLocally = await handleLocalDirective(original);
      if (handledLocally) return;

      const explicitShell = /^\/run\s+/i.test(original) || /^!\s*/.test(original);
      const explicitSmugcc = /^\/smugcc(\s+|$)/i.test(original);
      const explicitAgent = /^\/agent\s+/i.test(original) || explicitSmugcc;
      const explicitAi = /^(\/ai|\/task|\/agent|\/smugcc)(\s+|$)/i.test(original);

      const normalizedCommand = explicitShell
        ? original.replace(/^\/run\s+/i, "").replace(/^!\s*/, "").trim()
        : explicitSmugcc
          ? original
        : explicitAi
          ? original.replace(/^(\/ai|\/task|\/agent)\s+/i, "").trim()
          : original;

      const route: TerminalRoute = explicitShell
        ? "shell"
        : explicitAi
          ? "ai"
          : looksLikeShellCommand(original)
            ? "shell"
            : "ai";

      if (!normalizedCommand) {
        toast.error("Nothing to run.");
        return;
      }

      append("input", `${prompt} ${original}`);
      pushHistory(original);
      setInput("");
      setBusy(true);

      emitDlTrace({
        id: uid(),
        ts: nowIso(),
        title: route === "shell" ? "Terminal shell request" : "Terminal AI task request",
        content: normalizedCommand,
        source: route === "shell" ? "terminal.shell" : "terminal.ai",
        level: route === "shell" ? "thinking" : "info",
        tags: route === "shell" ? ["terminal", "shell"] : ["terminal", "ai-task", "agent"],
      });

      try {
        if (route === "shell") {
          const execResp = await requestJSON<TerminalExecuteResponse>("/api/terminal/execute", {
            method: "POST",
            body: JSON.stringify({
              command: normalizedCommand,
              mode,
              session_id: sessionId,
              timeout_s: 25,
              max_output_chars: 50000,
              caller: "WebUI.TerminalPanel",
              // Explicit local-user confirmation for the backend governance preflight.
              // The Run button is a direct user action; this does not grant autonomous authority.
              confirmed: true,
              user_confirmed: true,
              confirm_phrase: "APPROVE GOVERNED ACTION",
            }),
          });

          const resp = execResp.data;

          if (resp?.session_id) setSessionId(resp.session_id);
          if (resp?.cwd) setCwd(resp.cwd);
          setBackendAvailable(execResp.status < 500);

          if (resp?.blocked) {
            const reason = resp.reason || "Blocked by policy.";
            setBackendReason(reason);
            append("stderr", reason);
            emitDlTrace({
              id: uid(),
              ts: nowIso(),
              title: "Terminal shell request blocked",
              content: `${normalizedCommand}\n\n${reason}`,
              source: "terminal.shell",
              level: "error",
              tags: ["terminal", "shell", "blocked"],
            });
            toast.error(reason);
            return;
          }

          if (resp?.stdout) append("stdout", resp.stdout);
          if (resp?.stderr) append("stderr", resp.stderr);

          const metaLine = `exit_code=${resp?.exit_code ?? -1} duration_ms=${resp?.duration_ms ?? 0}${resp?.engine ? ` engine=${resp.engine}` : ""}`;
          append("meta", metaLine);
          setBackendReason(resp?.reason || "");

          const summary = [resp?.stdout || "", resp?.stderr || "", metaLine]
            .filter(Boolean)
            .join("\n\n")
            .trim();

          emitDlTrace({
            id: uid(),
            ts: nowIso(),
            title: `Terminal command ${resp?.exit_code === 0 ? "completed" : "returned non-zero exit"}`,
            content: summary || normalizedCommand,
            source: "terminal.shell",
            level: resp?.exit_code === 0 ? "success" : "warning",
            tags: ["terminal", "shell", resp?.engine || mode],
          });

          emitDlSubject({
            id: `terminal-shell-${Date.now()}`,
            title: shorten(normalizedCommand, 72) || "Terminal command",
            summary: shorten(summary || normalizedCommand, 220) || "Terminal command executed.",
            source: "terminal.shell",
            stage: resp?.exit_code === 0 ? "observed" : "hold",
            confidence: resp?.exit_code === 0 ? 82 : 46,
            risk: resp?.exit_code === 0 ? 24 : 58,
            sandboxRecommended: resp?.exit_code !== 0,
            notes: summary || normalizedCommand,
            tags: ["terminal", "shell", resp?.engine || mode],
            updatedAt: nowIso(),
          });

          return;
        }

        if (explicitAgent) {
          append("meta", "Routing request through Sarah AI-agent governance lane...");

          const agentResp = await requestJSON<TerminalAIResponse>("/api/terminal/agent", {
            method: "POST",
            body: JSON.stringify({
              task: normalizedCommand,
              text: normalizedCommand,
              source: "terminal_panel",
              mode: "governed_agent_task",
              session_id: sessionId,
              workdir: cwd,
              caller: "WebUI.TerminalPanel",
              smoke_test: false,
              agent_request: true,
              agent_authority: "inspect_or_propose_only",
            }),
          });

          const resp = agentResp.data || {};
          const reply = buildAiReply(resp) || normalizeText(resp?.stdout) || "Sarah accepted the terminal agent task, but no response text was returned.";
          const actions = Array.isArray(resp?.actions)
            ? resp.actions
            : Array.isArray(resp?.meta?.actions)
              ? resp.meta?.actions
              : [];

          if (typeof resp?.session_id === "string" && resp.session_id) setSessionId(resp.session_id);
          if (typeof resp?.cwd === "string" && resp.cwd) setCwd(resp.cwd);
          setBackendAvailable(agentResp.status < 500);
          setBackendReason(String(resp?.reason || ""));

          if (resp?.blocked) {
            append("stderr", String(resp.reason || resp.stderr || "Agent request blocked by policy."));
          }
          append("stdout", reply);

          if (resp?.agent_status) {
            const status = resp.agent_status as any;
            const smoke = status?.smoke_tests;
            if (smoke) append("meta", `AgentFirewall smoke tests: ${smoke.passed ?? 0}/${smoke.total ?? 0} passed.`);
            // SARAHMEMORY_PATCH_NOTE 2026-08-04:
            // Display-only governance evidence for Terminal Bay adapter/launch results.
            // The UI does not authorize execution; it only surfaces backend proof.
            const taskId = resp?.task_id || status?.task_id;
            if (taskId) append("meta", `Task ID: ${taskId}`);
            if (status?.launch_gate) append("meta", `Launch gate: blocked=${Boolean(status.launch_gate.blocked)} reason=${status.launch_gate.reason || "none"}`);
            if (status?.verified_answer_state) append("meta", `Verified answer state: ${status.verified_answer_state}`);
            const receiptIds = Array.isArray(resp?.receipt_ids) ? resp.receipt_ids : Array.isArray(status?.receipt_ids) ? status.receipt_ids : [];
            if (receiptIds.length > 0) append("meta", `Receipt IDs: ${receiptIds.slice(0, 6).join(", ")}`);
          }

          if (actions.length > 0) {
            dispatchUiActions(actions);
            append("meta", `Applied ${actions.length} UI action(s) returned by Sarah.`);
          }

          emitDlTrace({
            id: uid(),
            ts: nowIso(),
            title: "Terminal AI-agent task response",
            content: `${normalizedCommand}

${reply}`,
            source: "terminal.agent",
            level: resp?.blocked ? "error" : "success",
            tags: ["terminal", "ai-agent", "governed", "inspect-propose"],
          });

          emitDlSubject({
            id: `terminal-agent-${Date.now()}`,
            title: shorten(normalizedCommand, 72) || "Terminal AI-agent task",
            summary: shorten(reply, 220) || "AI-agent task processed from terminal.",
            source: "terminal.agent",
            stage: resp?.blocked ? "hold" : "observed",
            confidence: resp?.blocked ? 52 : 84,
            risk: resp?.blocked ? 72 : 28,
            sandboxRecommended: true,
            notes: `${normalizedCommand}

${reply}`,
            tags: ["terminal", "ai-agent", "governed"],
            updatedAt: nowIso(),
          });

          return;
        }

        append("meta", "Routing request through Sarah AI operator...");

        const aiPayload = {
          text: normalizedCommand,
          source: "terminal_panel",
          intent: "developer_terminal",
          tone: "direct",
          complexity: "detailed",
          terminal: true,
          terminal_mode: mode,
          session_id: sessionId,
          workdir: cwd,
          route_mode: (() => {
            try {
              return window.localStorage.getItem("route_mode") || "Any";
            } catch {
              return "Any";
            }
          })(),
          agent_request: explicitAgent,
          agent_runtime_preference: explicitAgent ? "OLLAMA_API" : undefined,
          agent_authority: explicitAgent ? "inspect_or_propose_only" : undefined,
        };

        // `/api/terminal/ai` exists as a governed agent alias. This non-shell
        // prompt still routes through chat so Neuron/CognitiveServices/SMGET
        // remain the authority layer for ordinary terminal-language tasks.
        const aiResp = await requestJSON<TerminalAIResponse>("/api/chat", {
          method: "POST",
          body: JSON.stringify(aiPayload),
        });

        const resp = aiResp.data || {};
        const reply = buildAiReply(resp) || "Sarah accepted the terminal task, but no response text was returned.";
        const actions = Array.isArray(resp?.actions)
          ? resp.actions
          : Array.isArray(resp?.meta?.actions)
            ? resp.meta?.actions
            : [];

        setBackendAvailable(aiResp.status < 500);
        setBackendReason(String(resp?.reason || ""));

        append("stdout", reply);

        if (Array.isArray(resp?.links) && resp.links.length > 0) {
          append("meta", `links=${(resp.links as unknown[]).map((item) => String(item)).join(" | ")}`);
        }

        if (actions.length > 0) {
          dispatchUiActions(actions);
          append("meta", `Applied ${actions.length} UI action(s) returned by Sarah.`);
        }

        emitDlTrace({
          id: uid(),
          ts: nowIso(),
          title: "Terminal AI task response",
          content: `${normalizedCommand}\n\n${reply}`,
          source: "terminal.ai",
          level: resp?.blocked ? "error" : "success",
          tags: ["terminal", "ai-task", "agent"],
        });

        emitDlSubject({
          id: `terminal-ai-${Date.now()}`,
          title: shorten(normalizedCommand, 72) || "Terminal AI task",
          summary: shorten(reply, 220) || "AI task processed from terminal.",
          source: "terminal.ai",
          stage: actions.length > 0 ? "testing" : "sandbox",
          confidence: actions.length > 0 ? 76 : 68,
          risk: actions.length > 0 ? 38 : 46,
          sandboxRecommended: true,
          notes: `${normalizedCommand}\n\n${reply}`,
          tags: ["terminal", "ai-task", "agent"],
          updatedAt: nowIso(),
        });
      } catch (e: any) {
        const msg = String(e?.message || e || "Terminal execution failed");
        setBackendAvailable(false);
        setBackendReason(msg);
        append("stderr", msg);
        emitDlTrace({
          id: uid(),
          ts: nowIso(),
          title: "Terminal execution failure",
          content: `${normalizedCommand}\n\n${msg}`,
          source: route === "shell" ? "terminal.shell" : "terminal.ai",
          level: "error",
          tags: route === "shell" ? ["terminal", "shell", "error"] : ["terminal", "ai-task", "error"],
        });
        toast.error(msg);
      } finally {
        setBusy(false);
        focusInput();
      }
    },
    [
      append,
      busy,
      cwd,
      desktopAllowed,
      emitDlSubject,
      emitDlTrace,
      enabled,
      focusInput,
      handleLocalDirective,
      input,
      mode,
      prompt,
      pushHistory,
      sessionId,
    ]
  );

  const onKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    e.stopPropagation();

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy) {
        void run();
      }
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (history.length === 0) return;

      const nextIndex = historyIndex < 0 ? 0 : Math.min(historyIndex + 1, history.length - 1);
      setHistoryIndex(nextIndex);
      setInput(history[nextIndex] || "");
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (history.length === 0) return;

      const nextIndex = historyIndex - 1;
      if (nextIndex < 0) {
        setHistoryIndex(-1);
        setInput("");
        return;
      }

      setHistoryIndex(nextIndex);
      setInput(history[nextIndex] || "");
    }
  };

  const agentRadarLine = formatAgentRadarLine(agentRadar);
  const agentVisibility = agentRadar?.agent_visibility;
  const agentSummary = agentVisibility?.summary || {};
  const agentRisk = agentRadarRiskLabel(agentVisibility?.risk);
  const externalProcesses = Number(agentVisibility?.summary?.active_external_ai_processes || 0);
  const externalConnections = Number(agentVisibility?.summary?.active_external_ai_connections || 0);
  const observedExternalProcesses = Number(agentVisibility?.summary?.observed_external_ai_processes || 0);
  const agentButtonCounters: Record<AgentScoreboardSection, number> = {
    overview: 0,
    observed: observedExternalProcesses,
    outbound: Number(agentSummary.outbound || 0),
    returned: Number(agentRadar?.summary?.tasks_completed || 0),
    quarantined: Number(agentSummary.quarantined || 0),
    blocked: Number(agentSummary.blocked || 0),
    review: Number(agentSummary.review_required || 0),
    passports: Number(agentSummary.active_passports || agentRadar?.summary?.passports_active || 0),
  };

  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions) || actions.length === 0) return;

      for (const a of actions) {
        if (!a?.type) continue;

        if (a.type === "terminal.set_input" || a.type === "terminal_set_input") {
          const v = String(a.payload?.value ?? a.payload?.command ?? "");
          setInput(v);
          focusInput();
        }

        if (a.type === "terminal.run" || a.type === "terminal_run") {
          const v = a.payload?.command ?? a.payload?.value;
          if (v != null) {
            void run(String(v));
          } else {
            void run();
          }
        }

        if (a.type === "terminal.clear" || a.type === "terminal_clear") {
          resetConsole();
          setInput("");
          focusInput();
        }
      }
    };

    window.addEventListener("sarah:ui", handler);
    return () => window.removeEventListener("sarah:ui", handler);
  }, [focusInput, resetConsole, run]);

  if (!desktopAllowed) {
    return (
      <div className="w-full h-full flex flex-col rounded-2xl border bg-background">
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="max-w-md text-center">
            <Terminal className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
            <p className="text-sm font-medium">Terminal unavailable on mobile</p>
            <p className="text-xs text-muted-foreground mt-2">
              The developer terminal is desktop-only.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="w-full h-full flex flex-col rounded-2xl border bg-background"
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between px-3 py-2 border-b gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="h-4 w-4 text-primary shrink-0" />
          <div className="text-sm font-semibold">Sarah Terminal</div>
          <div className="text-xs text-muted-foreground truncate">
            Session: <span className="font-mono">{sessionId}</span>
          </div>
          {!frontendDevMode ? (
            <div className="text-[11px] text-yellow-500 truncate">
              Backend-gated
            </div>
          ) : null}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            {backendAvailable === true ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                <span>Backend Ready</span>
              </>
            ) : backendAvailable === false ? (
              <>
                <AlertTriangle className="h-3.5 w-3.5 text-yellow-500" />
                <span>Backend Issue</span>
              </>
            ) : (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Checking</span>
              </>
            )}
          </div>

          <select
            className="h-9 rounded-md border bg-background px-2 text-sm"
            value={mode}
            onChange={(e) => setMode(e.target.value as TerminalEngineMode)}
            aria-label="Terminal mode"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <option value="auto">AUTO</option>
            <option value="windows">WINDOWS (cmd)</option>
            <option value="powershell">POWERSHELL</option>
            <option value="bash">BASH</option>
          </select>

          <Button
            variant="secondary"
            onClick={() => {
              append("meta", "New session started.");
              setSessionId(`webui_${Date.now()}`);
              setCwd(null);
              focusInput();
            }}
            disabled={busy}
          >
            New Session
          </Button>

          <Button
            variant="secondary"
            onClick={() => {
              resetConsole();
              focusInput();
            }}
            disabled={busy}
          >
            <Eraser className="h-4 w-4 mr-1" />
            Clear
          </Button>

          <Button
            variant="secondary"
            onClick={() => void checkTerminalBackend()}
            disabled={busy}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Check
          </Button>
        </div>
      </div>

      <div className="px-3 py-2 border-b bg-muted/20 text-xs text-muted-foreground flex items-center justify-between gap-3">
        <div className="truncate">
          <span className="font-medium text-foreground/80">Prompt:</span>{" "}
          <span className="font-mono text-foreground">{prompt}</span>
        </div>
        <div className="truncate">
          <span className="font-medium text-foreground/80">CWD:</span>{" "}
          <span className="font-mono">{cwd || "BASE_DIR"}</span>
        </div>
        <div className="truncate">
          <span className="font-medium text-foreground/80">Mode:</span> {mode.toUpperCase()}
        </div>
      </div>

      <div className="px-3 py-2 border-b bg-muted/10 text-xs text-muted-foreground flex items-center gap-1.5 overflow-x-auto">
        <Shield className={cn("h-3.5 w-3.5 shrink-0", agentRisk === "CRITICAL" || agentRisk === "HIGH" ? "text-yellow-500" : "text-green-500")} />
        <span className="shrink-0 font-medium text-foreground/80">Agents</span>
        {[
          ["observed", "OBS"],
          ["outbound", "OUT"],
          ["returned", "RETURN"],
          ["quarantined", "Q"],
          ["blocked", "BLOCK"],
          ["review", "REVIEW"],
          ["passports", "PASS"],
        ].map(([section, label]) => (
          <Button
            key={section}
            type="button"
            size="sm"
            variant={agentScoreboardSection === section ? "secondary" : "ghost"}
            className="h-6 px-2 text-[11px]"
            disabled={agentScoreboardBusy}
            onClick={() => void inspectAgentScoreboard(section as AgentScoreboardSection, true)}
            title={`Inspect ${label} agent scoreboard rows`}
          >
            {label} {agentButtonCounters[section as AgentScoreboardSection] ?? 0}
          </Button>
        ))}
        <span className="shrink-0 pl-2">Risk: {agentRadarUnavailable ? "UNKNOWN" : agentRisk}</span>
        {externalProcesses > 0 ? <span className="shrink-0">SM Proc: {externalProcesses}</span> : null}
        {externalConnections > 0 ? <span className="shrink-0">SM Conn: {externalConnections}</span> : null}
        {agentRadarUnavailable ? <span className="shrink-0">{agentRadarLine}</span> : null}
        {agentRadarBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" /> : null}
        {agentScoreboardBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" /> : null}
      </div>

      {backendReason ? (
        <div className="px-3 py-2 border-b bg-muted/10 text-xs text-muted-foreground flex items-center gap-2">
          <Bot className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{backendReason}</span>
        </div>
      ) : null}

      <div
        ref={scrollerRef}
        className="flex-1 overflow-auto p-3 font-mono text-sm leading-5 bg-black text-green-300"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {lines.map((l) => (
          <div
            key={l.id}
            className={cn(
              "whitespace-pre-wrap break-words",
              l.kind === "input" && "text-cyan-300",
              l.kind === "stdout" && "text-green-300",
              l.kind === "stderr" && "text-red-300",
              l.kind === "meta" && "text-zinc-400"
            )}
            title={prettyTime(l.ts)}
          >
            {l.text}
          </div>
        ))}
      </div>

      <div className="border-t p-3 flex items-center gap-2 bg-background">
        <div className="text-xs text-muted-foreground font-mono shrink-0 max-w-[35%] truncate">
          {prompt}
        </div>

        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          spellCheck={false}
          placeholder="Enter shell command or AI task..."
          disabled={busy}
          className={cn(
            "flex-1 h-10 rounded-md border bg-background px-3 text-sm font-mono outline-none",
            "border-input focus-visible:ring-2 focus-visible:ring-ring"
          )}
        />

        <Button
          onClick={() => void run()}
          disabled={busy || !input.trim()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {busy ? (
            <>
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-1" />
              Run
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Terminal,
  Bot,
  Send,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Loader2,
  History,
  Wand2,
  Shield,
  ChevronRight,
} from "lucide-react";

import { TerminalPanel } from "@/components/panels/TerminalPanel";
import { useIsMobile } from "@/hooks/use-mobile";
import { useSarahStore } from "@/stores/useSarahStore";
import { useNavigationStore } from "@/stores/useNavigationStore";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

type TerminalEntryLevel = "info" | "success" | "warning" | "error" | "command";

interface TerminalEntry {
  id: string;
  ts: string;
  level: TerminalEntryLevel;
  source: string;
  text: string;
}

const TERMINAL_HISTORY_KEY = "sarahmemory:terminal:history";
const TERMINAL_LOG_KEY = "sarahmemory:terminal:log";
const TERMINAL_MAX_HISTORY = 80;
const TERMINAL_MAX_LOG = 250;

function nowIso(): string {
  return new Date().toISOString();
}

function uid(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
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

function prettyTime(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString();
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

function dispatchSarahUi(actions: any[]) {
  try {
    window.dispatchEvent(
      new CustomEvent("sarah:ui", {
        detail: {
          source: "TerminalScreen",
          ts: Date.now(),
          actions,
        },
      })
    );
  } catch {
    // non-fatal
  }
}

export default function TerminalScreen() {
  const { setCurrentScreen } = useNavigationStore();
  const isMobile = useIsMobile();
  const store = useSarahStore();

  const devMode = Boolean((store as any)?.developersMode ?? (store as any)?.DEVELOPERSMODE ?? false);
  const enabled = !isMobile && devMode;

  const [command, setCommand] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [statusText, setStatusText] = useState("Terminal bridge ready.");
  const [devBridgeGate, setDevBridgeGate] = useState<any>(null);
  const [historyIndex, setHistoryIndex] = useState<number>(-1);

  const [commandHistory, setCommandHistory] = useState<string[]>(() =>
    loadJson<string[]>(TERMINAL_HISTORY_KEY, [])
  );
  const [logEntries, setLogEntries] = useState<TerminalEntry[]>(() =>
    loadJson<TerminalEntry[]>(TERMINAL_LOG_KEY, [
      {
        id: uid("boot"),
        ts: nowIso(),
        level: "info",
        source: "terminal",
        text: "SarahMemory terminal initialized. Developer Mode required for interaction.",
      },
    ])
  );

  const pushLog = useCallback((entry: Omit<TerminalEntry, "id" | "ts">) => {
    setLogEntries((prev) => {
      const next = [
        {
          id: uid("log"),
          ts: nowIso(),
          ...entry,
        },
        ...prev,
      ].slice(0, TERMINAL_MAX_LOG);

      saveJson(TERMINAL_LOG_KEY, next);
      return next;
    });
  }, []);

  const pushHistory = useCallback((value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    setCommandHistory((prev) => {
      const next = [trimmed, ...prev.filter((item) => item !== trimmed)].slice(0, TERMINAL_MAX_HISTORY);
      saveJson(TERMINAL_HISTORY_KEY, next);
      return next;
    });
    setHistoryIndex(-1);
  }, []);

  const quickActions = useMemo(
    () => [
      {
        label: "Open Chat",
        command: "/screen chat",
      },
      {
        label: "Open History",
        command: "/screen history",
      },
      {
        label: "Refresh DL Engine",
        command: "/screen dlengine",
      },
      {
        label: "Open Media",
        command: "/screen media",
      },
      {
        label: "Focus Research",
        command: "/screen research",
      },
      {
        label: "Show Settings",
        command: "/screen settings",
      },
    ],
    []
  );

  const tryBackendCall = useCallback(async (path: string, payload?: any) => {
    try {
      return await api.proxy.call(path, payload);
    } catch {
      return null;
    }
  }, []);

  const runStructuredUiCommand = useCallback(
    async (raw: string): Promise<boolean> => {
      const text = raw.trim();
      if (!text.startsWith("/")) return false;

      const [verb, ...rest] = text.slice(1).split(/\s+/);
      const arg = rest.join(" ").trim();

      if (verb === "screen" && arg) {
        dispatchSarahUi([{ type: "set_screen", payload: { screen: arg } }]);
        setCurrentScreen(arg as any);
        pushLog({
          level: "success",
          source: "ui-bus",
          text: `Switched screen to "${arg}".`,
        });
        return true;
      }

      if (verb === "refresh" && arg) {
        dispatchSarahUi([{ type: `${arg}_refresh`, payload: {} }]);
        pushLog({
          level: "success",
          source: "ui-bus",
          text: `Requested refresh for "${arg}".`,
        });
        return true;
      }

      if (verb === "help") {
        pushLog({
          level: "info",
          source: "terminal",
          text:
            'Built-in commands: /screen <name>, /refresh <name>, /clear, /help. Any other input is sent to the AI/backend bridge.',
        });
        return true;
      }

      if (verb === "clear") {
        const next = [
          {
            id: uid("cleared"),
            ts: nowIso(),
            level: "info" as TerminalEntryLevel,
            source: "terminal",
            text: "Terminal log cleared.",
          },
        ];
        setLogEntries(next);
        saveJson(TERMINAL_LOG_KEY, next);
        return true;
      }

      return false;
    },
    [pushLog, setCurrentScreen]
  );

  const handleCommandSubmit = useCallback(
    async (rawCommand?: string) => {
      const text = (rawCommand ?? command).trim();
      if (!text || isRunning || !enabled) return;

      setIsRunning(true);
      setStatusText("Processing terminal request...");
      pushHistory(text);
      pushLog({
        level: "command",
        source: "user",
        text,
      });
      setCommand("");
      setHistoryIndex(-1);

      try {
        const handledLocally = await runStructuredUiCommand(text);
        if (handledLocally) {
          setBackendAvailable(true);
          setStatusText("Command handled locally.");
          return;
        }

        const payload = {
          command: text,
          query: text,
          source: "terminal_screen",
          mode: "developer_terminal",
        };

        const response =
          (await tryBackendCall("/api/terminal/execute", payload)) ||
          (await tryBackendCall("/api/terminal/ai", payload)) ||
          (await tryBackendCall("/api/chat", {
            message: text,
            input: text,
            source: "terminal_screen",
          }));

        if (response) {
          setBackendAvailable(true);

          const reply =
            normalizeText((response as any)?.reply) ||
            normalizeText((response as any)?.response) ||
            normalizeText((response as any)?.message) ||
            normalizeText((response as any)?.output) ||
            normalizeText((response as any)?.result) ||
            "Command processed.";

          pushLog({
            level: "success",
            source: "ai",
            text: reply,
          });

          const actions = (response as any)?.actions;
          if (Array.isArray(actions) && actions.length > 0) {
            dispatchSarahUi(actions);
            pushLog({
              level: "info",
              source: "ui-bus",
              text: `Applied ${actions.length} UI action(s) from AI response.`,
            });
          }

          setStatusText("AI/backend command completed.");
        } else {
          setBackendAvailable(false);
          pushLog({
            level: "warning",
            source: "fallback",
            text:
              "No backend terminal endpoint responded. The command was not executed remotely.",
          });
          setStatusText("Backend terminal endpoints unavailable.");
        }
      } catch (error) {
        console.error("[TerminalScreen] Command failed:", error);
        setBackendAvailable(false);
        pushLog({
          level: "error",
          source: "terminal",
          text: `Command failed: ${normalizeText(error) || "Unknown error"}`,
        });
        setStatusText("Command failed.");
      } finally {
        setIsRunning(false);
      }
    },
    [
      command,
      enabled,
      isRunning,
      pushHistory,
      pushLog,
      runStructuredUiCommand,
      tryBackendCall,
    ]
  );

  const checkBackend = useCallback(async () => {
    setStatusText("Checking terminal bridge...");
    const devBridge = await tryBackendCall("/api/devbridge/status");
    if (devBridge) setDevBridgeGate((devBridge as any)?.apply_gate || null);

    const response =
      (await tryBackendCall("/api/terminal/status")) ||
      devBridge ||
      (await tryBackendCall("/api/health")) ||
      (await tryBackendCall("/api/ui/bootstrap"));

    if (response) {
      setBackendAvailable(true);
      setStatusText("Terminal bridge connected.");
      pushLog({
        level: "success",
        source: "health",
        text: "Backend terminal/AI bridge is reachable.",
      });
    } else {
      setBackendAvailable(false);
      setStatusText("Terminal bridge unavailable.");
      pushLog({
        level: "warning",
        source: "health",
        text: "Backend terminal/AI bridge could not be reached.",
      });
    }
  }, [pushLog, tryBackendCall]);

  useEffect(() => {
    if (enabled) {
      void checkBackend();
    }
  }, [checkBackend, enabled]);

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

          if (a.type === "terminal_append_log") {
            pushLog({
              level: (a.payload?.level as TerminalEntryLevel) || "info",
              source: a.payload?.source || "ui-bus",
              text: normalizeText(a.payload?.text || a.payload?.message || ""),
            });
          }

          if (a.type === "terminal_run" && a.payload?.command) {
            void handleCommandSubmit(String(a.payload.command));
          }

          if (a.type === "terminal_set_input") {
            setCommand(normalizeText(a.payload?.value || ""));
          }

          if (a.type === "terminal_clear") {
            const next = [
              {
                id: uid("cleared"),
                ts: nowIso(),
                level: "info" as TerminalEntryLevel,
                source: "ui-bus",
                text: "Terminal log cleared by UI action.",
              },
            ];
            setLogEntries(next);
            saveJson(TERMINAL_LOG_KEY, next);
          }
        } catch (e) {
          console.warn("[TerminalScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [handleCommandSubmit, pushLog, setCurrentScreen]);

  if (!enabled) return null;

  return (
    <div className="w-full h-full p-3">
      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-3 h-full">
        {/* Left Rail */}
        <div className="min-h-0 rounded-xl border border-border bg-card overflow-hidden flex flex-col">
          <div className="p-4 border-b border-border bg-card/80">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-primary" />
              <h1 className="text-base font-semibold">AI Terminal</h1>
              <Button
                variant="ghost"
                size="icon"
                className="ml-auto h-8 w-8"
                onClick={() => void checkBackend()}
                disabled={isRunning}
              >
                <RefreshCw className={cn("h-4 w-4", isRunning && "animate-spin")} />
              </Button>
            </div>

            <div className="mt-3 space-y-2 text-xs">
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/50 px-3 py-2">
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Shield className="h-3.5 w-3.5" />
                  Developer Mode
                </span>
                <span className="text-foreground font-medium">{devBridgeGate?.developer_mode ? "Backend gated" : "UI gated"}</span>
              </div>

              <div className="flex items-center justify-between rounded-lg border border-border bg-background/50 px-3 py-2">
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Bot className="h-3.5 w-3.5" />
                  AI Bridge
                </span>
                <span
                  className={cn(
                    "font-medium",
                    backendAvailable === true
                      ? "text-green-500"
                      : backendAvailable === false
                        ? "text-yellow-500"
                        : "text-muted-foreground"
                  )}
                >
                  {backendAvailable === true
                    ? "Connected"
                    : backendAvailable === false
                      ? "Fallback"
                      : "Checking"}
                </span>
              </div>
            </div>

            <p className="text-[11px] text-muted-foreground mt-3">{statusText}</p>
          </div>

          {/* Command Input */}
          <div className="p-4 border-b border-border">
            <div className="flex items-center gap-2">
              <Input
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                placeholder="Type terminal or AI command..."
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void handleCommandSubmit();
                    return;
                  }

                  if (e.key === "ArrowUp") {
                    e.preventDefault();
                    if (commandHistory.length === 0) return;
                    const nextIndex =
                      historyIndex < 0
                        ? 0
                        : Math.min(historyIndex + 1, commandHistory.length - 1);
                    setHistoryIndex(nextIndex);
                    setCommand(commandHistory[nextIndex] || "");
                  }

                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    if (commandHistory.length === 0) return;
                    const nextIndex = historyIndex - 1;
                    if (nextIndex < 0) {
                      setHistoryIndex(-1);
                      setCommand("");
                      return;
                    }
                    setHistoryIndex(nextIndex);
                    setCommand(commandHistory[nextIndex] || "");
                  }
                }}
              />
              <Button onClick={() => void handleCommandSubmit()} disabled={isRunning || !command.trim()}>
                {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>

            <p className="text-[11px] text-muted-foreground mt-2">
              Built-in: <span className="font-mono">/screen</span>, <span className="font-mono">/refresh</span>, <span className="font-mono">/help</span>, <span className="font-mono">/clear</span>. Raw OS execution remains backend-governed.
            </p>
          </div>

          {/* Quick Actions */}
          <div className="p-4 border-b border-border">
            <div className="flex items-center gap-2 mb-2">
              <Wand2 className="h-4 w-4 text-primary" />
              <p className="text-sm font-medium">Quick Actions</p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {quickActions.map((item) => (
                <Button
                  key={item.label}
                  variant="outline"
                  size="sm"
                  className="justify-start"
                  onClick={() => {
                    setCommand(item.command);
                    void handleCommandSubmit(item.command);
                  }}
                >
                  <ChevronRight className="h-3.5 w-3.5 mr-2" />
                  {item.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Session Log */}
          <div className="flex-1 min-h-0">
            <div className="px-4 pt-4 pb-2 flex items-center gap-2">
              <History className="h-4 w-4 text-primary" />
              <p className="text-sm font-medium">Session Log</p>
            </div>

            <ScrollArea className="h-[calc(100%-40px)] px-4 pb-4">
              <div className="space-y-2">
                {logEntries.map((entry) => (
                  <div
                    key={entry.id}
                    className={cn(
                      "rounded-lg border p-3 text-xs",
                      entry.level === "success" && "border-green-500/30 bg-green-500/5",
                      entry.level === "warning" && "border-yellow-500/30 bg-yellow-500/5",
                      entry.level === "error" && "border-red-500/30 bg-red-500/5",
                      entry.level === "command" && "border-blue-500/30 bg-blue-500/5",
                      entry.level === "info" && "border-border bg-background/50"
                    )}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="font-medium">{entry.source}</span>
                      <span className="text-muted-foreground">{prettyTime(entry.ts)}</span>
                    </div>
                    <pre className="whitespace-pre-wrap break-words font-sans text-xs">
                      {entry.text}
                    </pre>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Terminal Core */}
        <div className="min-h-0 rounded-xl border border-border bg-card overflow-hidden">
          <div className="p-3 border-b border-border bg-card/80 flex items-center gap-2">
            <Terminal className="h-4 w-4 text-primary" />
            <p className="text-sm font-medium">Interactive Terminal Surface</p>
            <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
              {backendAvailable === true ? (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                  <span>AI linked</span>
                </>
              ) : backendAvailable === false ? (
                <>
                  <AlertCircle className="h-3.5 w-3.5 text-yellow-500" />
                  <span>Local fallback</span>
                </>
              ) : (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Checking</span>
                </>
              )}
            </div>
          </div>

          <div className="h-[calc(100%-49px)]">
            <TerminalPanel />
          </div>
        </div>
      </div>
    </div>
  );
}
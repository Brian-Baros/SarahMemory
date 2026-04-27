import { useState, useEffect, useMemo, useCallback } from "react";
import {
  Cpu,
  Activity,
  BarChart3,
  Loader2,
  AlertCircle,
  TrendingUp,
  Brain,
  Eye,
  FlaskConical,
  ShieldCheck,
  RefreshCw,
  Search,
  CheckCircle2,
  XCircle,
  PauseCircle,
  PlayCircle,
  Settings2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useNavigationStore } from "@/stores/useNavigationStore";

interface TrainingJob {
  id: string;
  name: string;
  status: "pending" | "running" | "complete" | "error" | "queued" | "paused" | "sandboxed";
  progress: number;
  startedAt?: Date;
  details?: string;
}

interface EngineStats {
  modelsLoaded: number;
  activeJobs: number;
  memoryUsage: number;
  gpuUsage: number;
  cpuUsage?: number;
  thinkingLoad?: number;
  subjectsOpen?: number;
  ticketsPending?: number;
}

type ThoughtLevel = "info" | "warning" | "success" | "error" | "thinking";
type SubjectStage =
  | "new"
  | "observed"
  | "sandbox"
  | "testing"
  | "evaluation"
  | "approved"
  | "hold"
  | "rejected";

interface ThoughtTrace {
  id: string;
  ts: string;
  title: string;
  content: string;
  source: string;
  level: ThoughtLevel;
  tags: string[];
}

interface SubjectBox {
  id: string;
  title: string;
  summary: string;
  source: string;
  stage: SubjectStage;
  confidence: number;
  risk: number;
  sandboxRecommended: boolean;
  notes: string;
  tags: string[];
  updatedAt: string;
}

interface FineTuneControlState {
  autonomyEnabled: boolean;
  sandboxFirst: boolean;
  requireEvaluation: boolean;
  requireApproval: boolean;
  showOnlyHighSignal: boolean;
  pollIntervalSec: number;
}

const DL_ENGINE_SUBJECTS_KEY = "sarahmemory:dlengine:subjects";
const DL_ENGINE_THOUGHTS_KEY = "sarahmemory:dlengine:thoughts";
const DL_ENGINE_CONTROLS_KEY = "sarahmemory:dlengine:controls";
const DL_ENGINE_TRACE_LIMIT = 120;

function safeDate(value: unknown): Date {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
  const parsed = new Date(value as any);
  if (!Number.isNaN(parsed.getTime())) return parsed;
  return new Date();
}

function nowIso(): string {
  return new Date().toISOString();
}

function clampPercent(value: unknown): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function uid(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeStage(value: unknown): SubjectStage {
  const s = String(value || "").toLowerCase().trim();
  if (
    s === "new" ||
    s === "observed" ||
    s === "sandbox" ||
    s === "testing" ||
    s === "evaluation" ||
    s === "approved" ||
    s === "hold" ||
    s === "rejected"
  ) {
    return s;
  }
  return "new";
}

function normalizeThoughtLevel(value: unknown): ThoughtLevel {
  const s = String(value || "").toLowerCase().trim();
  if (s === "warning" || s === "success" || s === "error" || s === "thinking") return s;
  return "info";
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return (parsed as T) ?? fallback;
  } catch {
    return fallback;
  }
}

function saveJson<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // non-fatal
  }
}

function buildFallbackThoughts(stats: EngineStats | null, jobs: TrainingJob[]): ThoughtTrace[] {
  const out: ThoughtTrace[] = [];

  if (stats) {
    out.push({
      id: uid("trace"),
      ts: nowIso(),
      title: "Engine telemetry observed",
      content: `DL engine reports ${stats.modelsLoaded} models loaded, ${stats.activeJobs} active jobs, memory ${stats.memoryUsage}% and GPU ${stats.gpuUsage}%.`,
      source: "dlengine.status",
      level: "thinking",
      tags: ["telemetry", "runtime", "engine"],
    });
  }

  for (const job of jobs.slice(0, 4)) {
    out.push({
      id: uid("trace"),
      ts: job.startedAt ? safeDate(job.startedAt).toISOString() : nowIso(),
      title: `Training activity: ${job.name}`,
      content:
        job.status === "running"
          ? `${job.name} is running at ${job.progress}% complete.`
          : `${job.name} is currently ${job.status} with reported progress ${job.progress}%.`,
      source: "dlengine.jobs",
      level: job.status === "error" ? "error" : job.status === "complete" ? "success" : "info",
      tags: ["training", "job", job.status],
    });
  }

  if (out.length === 0) {
    out.push({
      id: uid("trace"),
      ts: nowIso(),
      title: "No live thought stream available",
      content:
        "The backend did not provide a cognitive trace feed, so the screen is operating in fallback introspection mode.",
      source: "ui.fallback",
      level: "warning",
      tags: ["fallback", "trace"],
    });
  }

  return out;
}

function deriveSubjectsFromThoughts(thoughts: ThoughtTrace[]): SubjectBox[] {
  const groups = new Map<string, SubjectBox>();

  for (const trace of thoughts) {
    const key = trace.title.trim().toLowerCase() || trace.id;
    if (!groups.has(key)) {
      groups.set(key, {
        id: `subject-${key.replace(/[^a-z0-9]+/gi, "-").slice(0, 48) || trace.id}`,
        title: trace.title || "Unlabeled Concept",
        summary: trace.content || "No summary available.",
        source: trace.source || "unknown",
        stage: "new",
        confidence: trace.level === "success" ? 82 : trace.level === "warning" ? 48 : 63,
        risk: trace.level === "error" ? 82 : trace.level === "warning" ? 61 : 34,
        sandboxRecommended: trace.level !== "success",
        notes: "",
        tags: Array.isArray(trace.tags) ? trace.tags : [],
        updatedAt: trace.ts || nowIso(),
      });
    } else {
      const existing = groups.get(key)!;
      existing.summary = existing.summary.length >= (trace.content || "").length ? existing.summary : trace.content;
      existing.tags = Array.from(new Set([...(existing.tags || []), ...(trace.tags || [])]));
      existing.updatedAt = trace.ts || existing.updatedAt;
      if (trace.level === "error") existing.risk = Math.max(existing.risk, 85);
      if (trace.level === "warning") existing.sandboxRecommended = true;
    }
  }

  return Array.from(groups.values()).sort(
    (a, b) => safeDate(b.updatedAt).getTime() - safeDate(a.updatedAt).getTime()
  );
}

export function DLEngineScreen() {
  const { setCurrentScreen } = useNavigationStore();

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<string>("");

  const [thoughts, setThoughts] = useState<ThoughtTrace[]>(() => loadJson<ThoughtTrace[]>(DL_ENGINE_THOUGHTS_KEY, []));
  const [subjects, setSubjects] = useState<SubjectBox[]>(() => loadJson<SubjectBox[]>(DL_ENGINE_SUBJECTS_KEY, []));
  const [selectedSubjectId, setSelectedSubjectId] = useState<string>("");

  const [controls, setControls] = useState<FineTuneControlState>(() =>
    loadJson<FineTuneControlState>(DL_ENGINE_CONTROLS_KEY, {
      autonomyEnabled: false,
      sandboxFirst: true,
      requireEvaluation: true,
      requireApproval: true,
      showOnlyHighSignal: false,
      pollIntervalSec: 8,
    })
  );

  const selectedSubject = useMemo(
    () => subjects.find((s) => s.id === selectedSubjectId) || null,
    [selectedSubjectId, subjects]
  );

  const setControl = useCallback(
    (patch: Partial<FineTuneControlState>) => {
      setControls((prev) => {
        const next = { ...prev, ...patch };
        saveJson(DL_ENGINE_CONTROLS_KEY, next);
        return next;
      });
    },
    []
  );

  const tryCall = useCallback(async (path: string) => {
    try {
      return await api.proxy.call(path);
    } catch {
      return null;
    }
  }, []);

  const normalizeJobs = useCallback((rawJobs: any[]): TrainingJob[] => {
    return (rawJobs || []).map((job: any, idx: number) => ({
      id: String(job?.id || job?.job_id || `job-${idx}`),
      name: String(job?.name || job?.title || job?.job_name || "Training Job"),
      status: String(job?.status || "pending") as TrainingJob["status"],
      progress: clampPercent(job?.progress ?? job?.pct ?? 0),
      startedAt: job?.startedAt || job?.started_at ? safeDate(job.startedAt || job.started_at) : undefined,
      details: String(job?.details || job?.notes || ""),
    }));
  }, []);

  const normalizeStats = useCallback((raw: any): EngineStats => {
    return {
      modelsLoaded: Number(raw?.modelsLoaded ?? raw?.models_loaded ?? 0) || 0,
      activeJobs: Number(raw?.activeJobs ?? raw?.active_jobs ?? 0) || 0,
      memoryUsage: clampPercent(raw?.memoryUsage ?? raw?.memory_usage ?? 0),
      gpuUsage: clampPercent(raw?.gpuUsage ?? raw?.gpu_usage ?? 0),
      cpuUsage: clampPercent(raw?.cpuUsage ?? raw?.cpu_usage ?? 0),
      thinkingLoad: clampPercent(raw?.thinkingLoad ?? raw?.thinking_load ?? raw?.reasoning_load ?? 0),
      subjectsOpen: Number(raw?.subjectsOpen ?? raw?.subjects_open ?? 0) || 0,
      ticketsPending: Number(raw?.ticketsPending ?? raw?.tickets_pending ?? 0) || 0,
    };
  }, []);

  const normalizeThoughts = useCallback((raw: any[]): ThoughtTrace[] => {
    return (raw || []).map((item: any, idx: number) => ({
      id: String(item?.id || item?.trace_id || uid(`trace-${idx}`)),
      ts: String(item?.ts || item?.timestamp || nowIso()),
      title: String(item?.title || item?.subject || item?.event || "Cognitive trace"),
      content: String(item?.content || item?.details || item?.summary || ""),
      source: String(item?.source || item?.module || "unknown"),
      level: normalizeThoughtLevel(item?.level || item?.severity || item?.status),
      tags: Array.isArray(item?.tags)
        ? item.tags.map((t: any) => String(t))
        : String(item?.category || item?.kind || "")
            .split(",")
            .map((x) => x.trim())
            .filter(Boolean),
    }));
  }, []);

  const normalizeSubjects = useCallback((raw: any[]): SubjectBox[] => {
    return (raw || []).map((item: any, idx: number) => ({
      id: String(item?.id || item?.ticket_id || item?.subject_id || `subject-${idx}`),
      title: String(item?.title || item?.name || item?.subject || "Concept"),
      summary: String(item?.summary || item?.rationale || item?.details || ""),
      source: String(item?.source || item?.module || item?.category || "unknown"),
      stage: normalizeStage(item?.stage || item?.state || item?.status),
      confidence: clampPercent(item?.confidence ?? item?.score?.confidence ?? 50),
      risk: clampPercent(item?.risk ?? item?.risk_score ?? 35),
      sandboxRecommended: Boolean(
        item?.sandboxRecommended ??
          item?.sandbox_recommended ??
          item?.proposed_action?.dry_run ??
          item?.state === "sandbox"
      ),
      notes: String(item?.notes || ""),
      tags: Array.isArray(item?.tags)
        ? item.tags.map((t: any) => String(t))
        : Array.isArray(item?.score?.missed)
          ? item.score.missed.map((t: any) => String(t))
          : [],
      updatedAt: String(item?.updatedAt || item?.updated_at || item?.ts || nowIso()),
    }));
  }, []);

  const mergeThoughts = useCallback((incoming: ThoughtTrace[]) => {
    setThoughts((prev) => {
      const map = new Map<string, ThoughtTrace>();

      const put = (item: ThoughtTrace) => {
        const key = String(item?.id || `${item?.title || ""}|${item?.source || ""}|${item?.content || ""}`)
          .trim()
          .toLowerCase();
        if (!key) return;

        const existing = map.get(key);
        if (!existing) {
          map.set(key, item);
          return;
        }

        map.set(key, {
          ...existing,
          ...item,
          tags: Array.from(new Set([...(existing.tags || []), ...(item.tags || [])])),
        });
      };

      for (const item of prev || []) put(item);
      for (const item of incoming || []) put(item);

      const merged = Array.from(map.values())
        .sort((a, b) => safeDate(b.ts).getTime() - safeDate(a.ts).getTime())
        .slice(0, DL_ENGINE_TRACE_LIMIT);

      saveJson(DL_ENGINE_THOUGHTS_KEY, merged);
      return merged;
    });
  }, []);

  const mergeSubjects = useCallback((incoming: SubjectBox[]) => {
    setSubjects((prev) => {
      const map = new Map<string, SubjectBox>();

      for (const s of prev) map.set(s.id, s);
      for (const s of incoming) {
        const existing = map.get(s.id);
        if (!existing) {
          map.set(s.id, s);
          continue;
        }
        map.set(s.id, {
          ...existing,
          ...s,
          notes: existing.notes || s.notes || "",
          tags: Array.from(new Set([...(existing.tags || []), ...(s.tags || [])])),
        });
      }

      const merged = Array.from(map.values()).sort(
        (a, b) => safeDate(b.updatedAt).getTime() - safeDate(a.updatedAt).getTime()
      );
      saveJson(DL_ENGINE_SUBJECTS_KEY, merged);
      return merged;
    });
  }, []);

  const checkStatus = useCallback(async () => {
    setIsLoading(true);
    setStatusMessage("");

    try {
      const statusResponse = await tryCall("/api/dlengine/status");
      const thoughtResponse =
        (await tryCall("/api/dlengine/thoughts")) ||
        (await tryCall("/api/cognitive/thoughts")) ||
        (await tryCall("/api/cognitive/trace"));
      const subjectResponse =
        (await tryCall("/api/dlengine/subjects")) ||
        (await tryCall("/api/dlengine/tickets")) ||
        (await tryCall("/api/cognitive/tickets"));

      if (statusResponse) {
        setIsAvailable(true);

        const normalizedStats = normalizeStats((statusResponse as any)?.stats || statusResponse);
        setStats(normalizedStats);

        const normalizedJobs = normalizeJobs((statusResponse as any)?.jobs || []);
        setJobs(normalizedJobs);

        const normalizedThoughts =
          thoughtResponse && Array.isArray((thoughtResponse as any)?.thoughts)
            ? normalizeThoughts((thoughtResponse as any).thoughts)
            : thoughtResponse && Array.isArray((thoughtResponse as any)?.events)
              ? normalizeThoughts((thoughtResponse as any).events)
              : buildFallbackThoughts(normalizedStats, normalizedJobs);

        const trimmedThoughts = normalizedThoughts
          .sort((a, b) => safeDate(b.ts).getTime() - safeDate(a.ts).getTime())
          .slice(0, DL_ENGINE_TRACE_LIMIT);

        mergeThoughts(trimmedThoughts);

        const backendSubjects =
          subjectResponse && Array.isArray((subjectResponse as any)?.subjects)
            ? normalizeSubjects((subjectResponse as any).subjects)
            : subjectResponse && Array.isArray((subjectResponse as any)?.tickets)
              ? normalizeSubjects((subjectResponse as any).tickets)
              : deriveSubjectsFromThoughts(trimmedThoughts);

        mergeSubjects(backendSubjects);
      } else {
        setIsAvailable(false);
        setStats(null);
        setJobs([]);
        const fallbackThoughts = buildFallbackThoughts(null, []);
        mergeThoughts(fallbackThoughts);
        mergeSubjects(deriveSubjectsFromThoughts(fallbackThoughts));
      }

      setLastUpdated(nowIso());
    } catch (error) {
      console.warn("[DLEngine] Not available:", error);
      setIsAvailable(false);
      setStatusMessage("DL Engine status fetch failed. Operating in local review mode.");
    } finally {
      setIsLoading(false);
    }
  }, [mergeSubjects, mergeThoughts, normalizeJobs, normalizeStats, normalizeSubjects, normalizeThoughts, tryCall]);

  useEffect(() => {
    void checkStatus();
  }, [checkStatus]);

  useEffect(() => {
    const every = Math.max(3, Number(controls.pollIntervalSec || 8));
    const id = window.setInterval(() => {
      void checkStatus();
    }, every * 1000);
    return () => window.clearInterval(id);
  }, [checkStatus, controls.pollIntervalSec]);

  const updateSubject = useCallback((subjectId: string, patch: Partial<SubjectBox>) => {
    setSubjects((prev) => {
      const next = prev.map((item) =>
        item.id === subjectId ? { ...item, ...patch, updatedAt: nowIso() } : item
      );
      saveJson(DL_ENGINE_SUBJECTS_KEY, next);
      return next;
    });
  }, []);

  const pushThought = useCallback((entry: ThoughtTrace) => {
    setThoughts((prev) => {
      const next = [entry, ...prev].slice(0, DL_ENGINE_TRACE_LIMIT);
      return next;
    });
  }, []);

  const submitSubjectAction = useCallback(
    async (subject: SubjectBox, nextStage: SubjectStage) => {
      updateSubject(subject.id, { stage: nextStage });

      pushThought({
        id: uid("trace"),
        ts: nowIso(),
        title: `Subject moved to ${nextStage}`,
        content: `${subject.title} was marked for ${nextStage} by the user review workflow.`,
        source: "ui.review",
        level:
          nextStage === "approved"
            ? "success"
            : nextStage === "rejected"
              ? "error"
              : nextStage === "sandbox" || nextStage === "testing" || nextStage === "evaluation"
                ? "thinking"
                : "info",
        tags: ["subject", "review", nextStage],
      });

      const payload = {
        id: subject.id,
        title: subject.title,
        stage: nextStage,
        notes: subject.notes,
        confidence: subject.confidence,
        risk: subject.risk,
        sandboxRecommended: subject.sandboxRecommended,
      };

      const endpoints = [
        "/api/dlengine/subject_action",
        "/api/dlengine/ticket_action",
        "/api/cognitive/ticket_action",
      ];

      for (const path of endpoints) {
        try {
          await api.proxy.call(path, payload as any);
          setStatusMessage(`Subject action sent to backend: ${subject.title} -> ${nextStage}`);
          return;
        } catch {
          // continue
        }
      }

      setStatusMessage(`Subject staged locally: ${subject.title} -> ${nextStage}`);
    },
    [pushThought, updateSubject]
  );

  const submitFineTuneControls = useCallback(async () => {
    const endpoints = [
      "/api/dlengine/finetune/config",
      "/api/dlengine/controls",
      "/api/cognitive/controls",
    ];

    for (const path of endpoints) {
      try {
        await api.proxy.call(path, controls as any);
        setStatusMessage("Fine-tune / governance control state sent to backend.");
        pushThought({
          id: uid("trace"),
          ts: nowIso(),
          title: "Fine-tune controls updated",
          content:
            "The operator updated the DL Engine governance and tuning controls from the DLEngine screen.",
          source: "ui.controls",
          level: "success",
          tags: ["finetune", "controls", "governance"],
        });
        return;
      } catch {
        // continue
      }
    }

    setStatusMessage("Fine-tune controls saved locally. Backend endpoint not available.");
  }, [controls, pushThought]);

  const filteredSubjects = useMemo(() => {
    const items = [...subjects].sort((a, b) => safeDate(b.updatedAt).getTime() - safeDate(a.updatedAt).getTime());
    if (!controls.showOnlyHighSignal) return items;
    return items.filter((s) => s.confidence >= 60 || s.risk >= 55 || s.sandboxRecommended);
  }, [controls.showOnlyHighSignal, subjects]);

  const getStatusColor = (status: TrainingJob["status"]) => {
    switch (status) {
      case "running":
        return "text-blue-500";
      case "complete":
        return "text-green-500";
      case "error":
        return "text-red-500";
      case "paused":
        return "text-yellow-500";
      case "sandboxed":
        return "text-purple-500";
      default:
        return "text-muted-foreground";
    }
  };

  const thoughtLevelClass = (level: ThoughtLevel) => {
    switch (level) {
      case "success":
        return "border-green-500/30 bg-green-500/5";
      case "warning":
        return "border-yellow-500/30 bg-yellow-500/5";
      case "error":
        return "border-red-500/30 bg-red-500/5";
      case "thinking":
        return "border-blue-500/30 bg-blue-500/5";
      default:
        return "border-border bg-card";
    }
  };

  const stageBadgeClass = (stage: SubjectStage) => {
    switch (stage) {
      case "approved":
        return "text-green-500";
      case "rejected":
        return "text-red-500";
      case "sandbox":
      case "testing":
      case "evaluation":
        return "text-blue-500";
      case "hold":
        return "text-yellow-500";
      default:
        return "text-muted-foreground";
    }
  };

  useEffect(() => {
    if (!selectedSubjectId && filteredSubjects.length > 0) {
      setSelectedSubjectId(filteredSubjects[0].id);
    } else if (selectedSubjectId && !filteredSubjects.some((s) => s.id === selectedSubjectId)) {
      setSelectedSubjectId(filteredSubjects[0]?.id || "");
    }
  }, [filteredSubjects, selectedSubjectId]);

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

          if (a.type === "dlengine_refresh") {
            void checkStatus();
          }

          if (a.type === "dlengine_add_trace" && a.payload) {
            const incomingThoughts = normalizeThoughts([a.payload]);
            if (incomingThoughts.length > 0) {
              mergeThoughts(incomingThoughts);
            }
          }

          if (a.type === "dlengine_add_subject" && a.payload) {
            const incomingSubjects = normalizeSubjects([a.payload]);
            if (incomingSubjects.length > 0) {
              mergeSubjects(incomingSubjects);
              if (!selectedSubjectId) {
                setSelectedSubjectId(incomingSubjects[0].id);
              }
            }
          }

          if (a.type === "dlengine_select_subject" && a.payload?.id) {
            setSelectedSubjectId(String(a.payload.id));
          }

          if (a.type === "dlengine_set_stage" && a.payload?.id && a.payload?.stage) {
            const subject = subjects.find((s) => s.id === a.payload.id);
            if (subject) {
              void submitSubjectAction(subject, normalizeStage(a.payload.stage));
            }
          }

          if (a.type === "dlengine_toggle_high_signal") {
            setControl({ showOnlyHighSignal: !controls.showOnlyHighSignal });
          }
        } catch (e) {
          console.warn("[DLEngineScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [checkStatus, controls.showOnlyHighSignal, mergeSubjects, mergeThoughts, normalizeSubjects, normalizeThoughts, selectedSubjectId, setControl, setCurrentScreen, subjects, submitSubjectAction]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">DL Engine</h1>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto h-8 w-8"
            onClick={() => void checkStatus()}
            disabled={isLoading}
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Governed thought tracing, subject review, sandbox routing, and fine-tune control
        </p>
        {lastUpdated && (
          <p className="text-[11px] text-muted-foreground/80 mt-1">
            Last updated: {safeDate(lastUpdated).toLocaleString()}
          </p>
        )}
        {statusMessage && <p className="text-[11px] text-primary mt-1">{statusMessage}</p>}
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Availability Notice */}
          {isAvailable === false && (
            <div className="p-4 rounded-xl bg-muted/50 border border-border">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Backend DL Engine unavailable</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    The screen remains active in local review mode so the operator can still inspect subjects,
                    notes, and fine-tune controls.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Stats */}
          {stats && (
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground">Models Loaded</p>
                <p className="text-2xl font-bold mt-1">{stats.modelsLoaded}</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground">Active Jobs</p>
                <p className="text-2xl font-bold mt-1">{stats.activeJobs}</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground mb-2">Memory</p>
                <Progress value={stats.memoryUsage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{stats.memoryUsage}%</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground mb-2">GPU</p>
                <Progress value={stats.gpuUsage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{stats.gpuUsage}%</p>
              </div>

              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground mb-2">Thinking Load</p>
                <Progress value={stats.thinkingLoad || 0} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{stats.thinkingLoad || 0}%</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground">Subjects / Tickets</p>
                <p className="text-2xl font-bold mt-1">
                  {(stats.subjectsOpen ?? filteredSubjects.length) + (stats.ticketsPending ? 0 : 0)}
                </p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  open {stats.subjectsOpen ?? filteredSubjects.length} • pending {stats.ticketsPending ?? 0}
                </p>
              </div>
            </div>
          )}

          {/* Fine-Tune / Governance Controls */}
          <div className="p-4 rounded-xl bg-card border border-border">
            <p className="text-sm font-medium mb-3 flex items-center gap-2">
              <Settings2 className="h-4 w-4" />
              Fine-Tune / Governance Controls
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Autonomy Visible</span>
                <input
                  type="checkbox"
                  checked={controls.autonomyEnabled}
                  onChange={(e) => setControl({ autonomyEnabled: e.target.checked })}
                />
              </label>

              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Sandbox First</span>
                <input
                  type="checkbox"
                  checked={controls.sandboxFirst}
                  onChange={(e) => setControl({ sandboxFirst: e.target.checked })}
                />
              </label>

              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Require Evaluation</span>
                <input
                  type="checkbox"
                  checked={controls.requireEvaluation}
                  onChange={(e) => setControl({ requireEvaluation: e.target.checked })}
                />
              </label>

              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Require Approval</span>
                <input
                  type="checkbox"
                  checked={controls.requireApproval}
                  onChange={(e) => setControl({ requireApproval: e.target.checked })}
                />
              </label>

              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Show High Signal Only</span>
                <input
                  type="checkbox"
                  checked={controls.showOnlyHighSignal}
                  onChange={(e) => setControl({ showOnlyHighSignal: e.target.checked })}
                />
              </label>

              <label className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-background/50">
                <span className="text-sm">Poll Interval (sec)</span>
                <input
                  type="number"
                  min={3}
                  max={60}
                  value={controls.pollIntervalSec}
                  onChange={(e) => setControl({ pollIntervalSec: Math.max(3, Number(e.target.value || 8)) })}
                  className="w-20 rounded border border-border bg-background px-2 py-1 text-sm"
                />
              </label>
            </div>

            <div className="flex items-center gap-2 mt-3">
              <Button variant="default" size="sm" onClick={() => void submitFineTuneControls()}>
                Save Controls
              </Button>
              <Button variant="outline" size="sm" onClick={() => void checkStatus()}>
                Sync Now
              </Button>
            </div>
          </div>

          {/* Thought Stream */}
          <div>
            <p className="text-sm font-medium mb-3 flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Chat Thinking Process / Cognitive Trace
            </p>

            {isLoading && thoughts.length === 0 ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : thoughts.length === 0 ? (
              <div className="text-center py-8 rounded-xl bg-card border border-border">
                <Eye className="h-10 w-10 mx-auto text-muted-foreground/50 mb-2" />
                <p className="text-sm text-muted-foreground">No trace available</p>
              </div>
            ) : (
              <div className="space-y-2">
                {thoughts.map((trace) => (
                  <div
                    key={trace.id}
                    className={cn("p-3 rounded-xl border", thoughtLevelClass(trace.level))}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="font-medium text-sm truncate">{trace.title}</p>
                        <p className="text-[11px] text-muted-foreground mt-1">
                          {trace.source} • {safeDate(trace.ts).toLocaleString()}
                        </p>
                      </div>
                      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {trace.level}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 whitespace-pre-wrap">
                      {trace.content}
                    </p>
                    {trace.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {trace.tags.map((tag) => (
                          <span
                            key={`${trace.id}-${tag}`}
                            className="px-2 py-0.5 rounded-full text-[10px] bg-muted text-muted-foreground"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Subjects / Concepts */}
          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-4">
            <div>
              <p className="text-sm font-medium mb-3 flex items-center gap-2">
                <Search className="h-4 w-4" />
                Subject Boxes / New Concepts
              </p>

              {filteredSubjects.length === 0 ? (
                <div className="text-center py-8 rounded-xl bg-card border border-border">
                  <Search className="h-10 w-10 mx-auto text-muted-foreground/50 mb-2" />
                  <p className="text-sm text-muted-foreground">No concepts queued</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {filteredSubjects.map((subject) => {
                    const isActive = selectedSubjectId === subject.id;
                    return (
                      <button
                        key={subject.id}
                        onClick={() => setSelectedSubjectId(subject.id)}
                        className={cn(
                          "w-full text-left p-3 rounded-xl border transition-all",
                          "bg-card hover:bg-card/80 border-border",
                          isActive && "border-primary/50 bg-primary/5"
                        )}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="font-medium text-sm truncate">{subject.title}</p>
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {subject.summary}
                            </p>
                          </div>
                          <span className={cn("text-[11px] font-medium capitalize", stageBadgeClass(subject.stage))}>
                            {subject.stage}
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 mt-3">
                          <div>
                            <p className="text-[10px] text-muted-foreground mb-1">Confidence</p>
                            <Progress value={subject.confidence} className="h-2" />
                          </div>
                          <div>
                            <p className="text-[10px] text-muted-foreground mb-1">Risk</p>
                            <Progress value={subject.risk} className="h-2" />
                          </div>
                        </div>

                        <div className="flex items-center justify-between mt-3 text-[11px] text-muted-foreground">
                          <span>{subject.source}</span>
                          <span>{safeDate(subject.updatedAt).toLocaleString()}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <div>
              <p className="text-sm font-medium mb-3 flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                Subject Review / Operator Decision
              </p>

              {!selectedSubject ? (
                <div className="text-center py-8 rounded-xl bg-card border border-border">
                  <ShieldCheck className="h-10 w-10 mx-auto text-muted-foreground/50 mb-2" />
                  <p className="text-sm text-muted-foreground">Select a subject to review</p>
                </div>
              ) : (
                <div className="p-4 rounded-xl bg-card border border-border space-y-3">
                  <div>
                    <p className="font-medium text-sm">{selectedSubject.title}</p>
                    <p className="text-xs text-muted-foreground mt-1">{selectedSubject.source}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <p className="text-[11px] text-muted-foreground">Confidence</p>
                      <p className="text-lg font-semibold mt-1">{selectedSubject.confidence}%</p>
                    </div>
                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <p className="text-[11px] text-muted-foreground">Risk</p>
                      <p className="text-lg font-semibold mt-1">{selectedSubject.risk}%</p>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-background/50">
                    <p className="text-[11px] text-muted-foreground mb-1">Summary</p>
                    <p className="text-sm text-foreground whitespace-pre-wrap">{selectedSubject.summary}</p>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-background/50">
                    <p className="text-[11px] text-muted-foreground mb-2">Review Notes</p>
                    <textarea
                      value={selectedSubject.notes}
                      onChange={(e) => updateSubject(selectedSubject.id, { notes: e.target.value })}
                      rows={5}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm resize-y"
                      placeholder="Operator notes, tuning feedback, guardrail notes, or sandbox instructions..."
                    />
                  </div>

                  {selectedSubject.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {selectedSubject.tags.map((tag) => (
                        <span
                          key={`${selectedSubject.id}-${tag}`}
                          className="px-2 py-0.5 rounded-full text-[10px] bg-muted text-muted-foreground"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "sandbox")}
                    >
                      <FlaskConical className="h-4 w-4 mr-2" />
                      Sandbox
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "testing")}
                    >
                      <PlayCircle className="h-4 w-4 mr-2" />
                      Test
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "evaluation")}
                    >
                      <Activity className="h-4 w-4 mr-2" />
                      Evaluate
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "hold")}
                    >
                      <PauseCircle className="h-4 w-4 mr-2" />
                      Hold
                    </Button>
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "approved")}
                    >
                      <CheckCircle2 className="h-4 w-4 mr-2" />
                      Approve
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => void submitSubjectAction(selectedSubject, "rejected")}
                    >
                      <XCircle className="h-4 w-4 mr-2" />
                      Reject
                    </Button>
                  </div>

                  <div className="text-[11px] text-muted-foreground">
                    Current stage:{" "}
                    <span className={cn("font-medium capitalize", stageBadgeClass(selectedSubject.stage))}>
                      {selectedSubject.stage}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Training Jobs */}
          <div>
            <p className="text-sm font-medium mb-3 flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Training Jobs
            </p>

            {isLoading && jobs.length === 0 ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : jobs.length === 0 ? (
              <div className="text-center py-8 rounded-xl bg-card border border-border">
                <BarChart3 className="h-10 w-10 mx-auto text-muted-foreground/50 mb-2" />
                <p className="text-sm text-muted-foreground">
                  {isAvailable ? "No active jobs" : "Engine not available"}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {jobs.map((job) => (
                  <div key={job.id} className="p-3 rounded-xl bg-card border border-border">
                    <div className="flex items-center justify-between mb-2 gap-3">
                      <div className="min-w-0">
                        <p className="font-medium text-sm truncate">{job.name}</p>
                        {job.details && (
                          <p className="text-xs text-muted-foreground truncate mt-1">{job.details}</p>
                        )}
                      </div>
                      <span className={cn("text-xs font-medium capitalize", getStatusColor(job.status))}>
                        {job.status}
                      </span>
                    </div>

                    <Progress value={job.progress} className="h-2 mb-1" />
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{job.progress}% complete</span>
                      <span>{job.startedAt ? job.startedAt.toLocaleString() : "Queued"}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
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
  Moon,
  Sun,
  SlidersHorizontal,
  Gauge,
  Power,
  Zap,
  Save,
  RotateCcw,
  Database,
  Layers,
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

type DLRuntimeMode = "auto" | "manual" | "paused";

interface ModelWeightControlState {
  reasoning: number;
  coding: number;
  memory: number;
  research: number;
  creativity: number;
  safety: number;
  autonomy: number;
  precision: number;
  speed: number;
}

interface WeightSliderSpec {
  key: keyof ModelWeightControlState;
  label: string;
  description: string;
}

interface RemStatusState {
  enabled?: boolean;
  running?: boolean;
  phase?: string;
  reason?: string;
  cycle_id?: string;
  cycles_completed?: number;
  auto_applied?: number;
  staged?: number;
  rejected?: number;
  idle_ready?: boolean;
  neosky_enabled?: boolean;
  manual_force_allowed?: boolean;
  observation_only?: boolean;
  self_evolution_allowed?: boolean;
  thread_alive?: boolean;
  protected_files?: string[];
  last_report?: any;
  reports?: any[];
  updated_at?: string | null;
  started_at?: string | null;
}

interface RemDerivedState {
  phase: string;
  running: boolean;
  enabled: boolean;
  neoskyEnabled: boolean;
  idleReady: boolean;
  cycleId: string;
  cyclesCompleted: number;
  dreams: number;
  results: number;
  autoApplied: number;
  staged: number;
  rejected: number;
  sandboxPassed: number;
  sandboxFailed: number;
  lanesOk: number;
  lanesDegraded: number;
  lanesFailed: number;
  lanesTotal: number;
  activeProvider: string;
  activeModel: string;
  activeModelCount: number;
  laneEntries: Array<[string, any]>;
  latestCycles: any[];
}

interface ActiveCapabilityState {
  activeProvider: string;
  activeModel: string;
  activeModelCount: number;
  source?: string;
}

interface DevBridgeSummaryState {
  available: boolean;
  developerMode: boolean;
  envAllowApply: boolean;
  loopback: boolean;
  bridgeRoot: string;
  counts: {
    packets: number;
    responses: number;
    stages: number;
  };
  latestResponseId: string;
  latestStageId: string;
  latestStageStatus: string;
  latestStageFiles: number;
  latestStageValidated: boolean;
  latestStageApplied: boolean;
  latestSummary: string;
  rollbackAvailable: boolean;
  rollbackStatus: string;
  backupRoot: string;
  repairTickets: number;
  repairBatches: number;
  sandboxes: number;
  cmdTicketsPending: number;
  cmdTicketsProcessed: number;
  cmdTicketsFailed: number;
  cmdTicketsArchivedFailed: number;
  cmdTicketsInvalidPending: number;
  cmdTicketsGeneratedAt: string;
  cmdTicketsInventoryNote: string;
  updatedAt: string;
}

const DL_ENGINE_SUBJECTS_KEY = "sarahmemory:dlengine:subjects";
const DL_ENGINE_THOUGHTS_KEY = "sarahmemory:dlengine:thoughts";
const DL_ENGINE_CONTROLS_KEY = "sarahmemory:dlengine:controls";
const DL_ENGINE_WEIGHTS_KEY = "sarahmemory:dlengine:model_weights";
const DL_ENGINE_WEIGHT_CATEGORY_KEY = "sarahmemory:dlengine:weight_category";
const DL_ENGINE_WEIGHT_MODEL_KEY = "sarahmemory:dlengine:weight_model";
const DL_ENGINE_MODE_KEY = "sarahmemory:dlengine:runtime_mode";
const DL_ENGINE_TRACE_LIMIT = 120;
const DL_ENGINE_REM_REPORT_LIMIT = 3;

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

function weightProfileStorageKey(category: string, modelId = ""): string {
  const safeCategory = String(category || "reasoning").trim() || "reasoning";
  const safeModelId = String(modelId || "__category_default__").trim() || "__category_default__";
  return `${DL_ENGINE_WEIGHTS_KEY}:${safeCategory}:${safeModelId}`;
}

function defaultModelWeightsForCategory(category: string): ModelWeightControlState {
  const base = defaultModelWeights();
  const cat = String(category || "reasoning").trim().toLowerCase();
  const presets: Record<string, Partial<ModelWeightControlState>> = {
    reasoning: { reasoning: 70, coding: 35, memory: 60, research: 60, creativity: 40, safety: 90, autonomy: 35, precision: 75, speed: 50 },
    coder: { reasoning: 55, coding: 82, memory: 55, research: 55, creativity: 35, safety: 88, autonomy: 30, precision: 82, speed: 55 },
    embeddings: { reasoning: 45, coding: 25, memory: 88, research: 60, creativity: 25, safety: 90, autonomy: 25, precision: 78, speed: 65 },
    vision: { reasoning: 55, coding: 20, memory: 65, research: 45, creativity: 35, safety: 92, autonomy: 25, precision: 82, speed: 60 },
    image_generation: { reasoning: 45, coding: 20, memory: 50, research: 35, creativity: 86, safety: 92, autonomy: 25, precision: 70, speed: 55 },
    tts: { reasoning: 40, coding: 15, memory: 45, research: 25, creativity: 55, safety: 90, autonomy: 20, precision: 68, speed: 72 },
    stt: { reasoning: 45, coding: 15, memory: 55, research: 30, creativity: 30, safety: 90, autonomy: 20, precision: 82, speed: 72 },
    unknown: { reasoning: 50, coding: 35, memory: 50, research: 45, creativity: 35, safety: 95, autonomy: 15, precision: 70, speed: 45 },
  };
  return normalizeModelWeights({ ...base, ...(presets[cat] || {}) });
}

function defaultModelWeights(): ModelWeightControlState {
  return {
    reasoning: 65,
    coding: 55,
    memory: 60,
    research: 55,
    creativity: 45,
    safety: 90,
    autonomy: 35,
    precision: 70,
    speed: 50,
  };
}

function normalizeModelWeights(raw: Partial<ModelWeightControlState> | null | undefined): ModelWeightControlState {
  const base = defaultModelWeights();
  const incoming = raw && typeof raw === "object" ? raw : {};
  return {
    reasoning: clampPercent(incoming.reasoning ?? base.reasoning),
    coding: clampPercent(incoming.coding ?? base.coding),
    memory: clampPercent(incoming.memory ?? base.memory),
    research: clampPercent(incoming.research ?? base.research),
    creativity: clampPercent(incoming.creativity ?? base.creativity),
    safety: clampPercent(incoming.safety ?? base.safety),
    autonomy: clampPercent(incoming.autonomy ?? base.autonomy),
    precision: clampPercent(incoming.precision ?? base.precision),
    speed: clampPercent(incoming.speed ?? base.speed),
  };
}

function normalizeDlMode(value: unknown): DLRuntimeMode {
  const mode = String(value || "auto").toLowerCase().trim();
  if (mode === "manual" || mode === "run" || mode === "start" || mode === "active") return "manual";
  if (mode === "paused" || mode === "pause" || mode === "stop" || mode === "stopped") return "paused";
  return "auto";
}

function normalizeCapability(raw: any, fallback?: ActiveCapabilityState): ActiveCapabilityState {
  const model = raw && typeof raw === "object" ? raw : {};
  return {
    activeProvider: String(
      model.active_provider ||
        model.selected_provider ||
        model.provider ||
        fallback?.activeProvider ||
        "core"
    ),
    activeModel: String(
      model.active_model ||
        model.selected_model ||
        model.model ||
        fallback?.activeModel ||
        "SarahMemory Core Runtime"
    ),
    activeModelCount: Math.max(
      1,
      Number(model.active_model_count ?? model.models_loaded ?? model.model_count ?? fallback?.activeModelCount ?? 1) || 1
    ),
    source: String(model.source || fallback?.source || "backend"),
  };
}

const MODEL_WEIGHT_SLIDERS: WeightSliderSpec[] = [
  { key: "reasoning", label: "Reasoning", description: "Logical planning and answer quality" },
  { key: "coding", label: "Coding", description: "Code generation / repair priority" },
  { key: "memory", label: "Memory", description: "Recall and context consolidation" },
  { key: "research", label: "Research", description: "Web/API/local research depth" },
  { key: "creativity", label: "Creative", description: "Idea generation and avatar/personality variation" },
  { key: "safety", label: "Safety", description: "Governance strictness and restraint" },
  { key: "autonomy", label: "Autonomy", description: "How aggressively low-risk work is attempted" },
  { key: "precision", label: "Precision", description: "Bias toward exactness over broad exploration" },
  { key: "speed", label: "Speed", description: "Bias toward quicker, lighter passes" },
];

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


function safeNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === "object" ? (value as Record<string, any>) : {};
}

function safeArray<T = any>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function summarizeDevBridgeState(rawLatest: any, rawStatus: any, rawCmdTickets?: any): DevBridgeSummaryState | null {
  const latestRoot = asRecord(rawLatest);
  const statusRoot = asRecord(rawStatus);
  const cmdTicketsRoot = asRecord(rawCmdTickets);
  if (!Object.keys(latestRoot).length && !Object.keys(statusRoot).length && !Object.keys(cmdTicketsRoot).length) return null;

  const countsRoot = asRecord(latestRoot.counts);
  const repairCountsRoot = asRecord(latestRoot.repair_counts || statusRoot.repair_counts);
  const cmdTicketCountsRoot = asRecord(
    cmdTicketsRoot.counts || latestRoot.cmd_tickets || statusRoot.cmd_tickets || repairCountsRoot.cmd_tickets || asRecord(statusRoot.repair_counts).cmd_tickets
  );
  const cmdTicketExtendedRoot = asRecord(cmdTicketsRoot.extended_counts || latestRoot.cmd_ticket_inventory || statusRoot.cmd_ticket_inventory);
  const applyGate = asRecord(statusRoot.apply_gate);
  const latestResponse = asRecord(latestRoot.latest_response || latestRoot.latest);
  const latestStage = asRecord(latestRoot.latest_stage);
  const parsedJson = asRecord(latestResponse.parsed_json);
  const stageValidation = asRecord(latestStage.validation);
  const stageIntegrity = asRecord(stageValidation.integrity);
  const stageApply = asRecord(latestStage.apply_result);
  const stageRollback = asRecord(latestStage.rollback_result);
  const latestStageStatus = String(latestStage.status || "").trim();

  return {
    available: Boolean(latestRoot.ok || statusRoot.ok),
    developerMode: Boolean(applyGate.developer_mode),
    envAllowApply: Boolean(applyGate.env_allow_apply),
    loopback: Boolean(applyGate.loopback),
    bridgeRoot: String(latestRoot.bridge_root || statusRoot.bridge_root || ""),
    counts: {
      packets: safeNumber(countsRoot.packets, 0),
      responses: safeNumber(countsRoot.responses, 0),
      stages: safeNumber(countsRoot.stages, 0),
    },
    latestResponseId: String(latestResponse.response_id || ""),
    latestStageId: String(latestStage.stage_id || latestResponse.stage_id || ""),
    latestStageStatus: latestStageStatus || (latestResponse.staged ? "staged" : "idle"),
    latestStageFiles: safeArray(latestStage.files).length,
    latestStageValidated: latestStageStatus === "validated" || Boolean(stageValidation.ok || stageIntegrity.ok),
    latestStageApplied: latestStageStatus === "applied" || Boolean(stageApply.ok),
    latestSummary: String(parsedJson.summary || latestResponse.summary || "").slice(0, 320),
    rollbackAvailable: Boolean(stageApply.ok && safeArray(stageApply.applied).length > 0 && latestStageStatus !== "rolled_back"),
    rollbackStatus: String(stageRollback.ok ? "rolled_back" : latestStageStatus || "idle"),
    backupRoot: String(stageApply.backup_root || ""),
    repairTickets: safeNumber(repairCountsRoot.tickets, 0),
    repairBatches: safeNumber(repairCountsRoot.batches, 0),
    sandboxes: safeNumber(repairCountsRoot.sandboxes, 0),
    cmdTicketsPending: safeNumber(cmdTicketsRoot.pending_count ?? cmdTicketCountsRoot.pending, 0),
    cmdTicketsProcessed: safeNumber(cmdTicketCountsRoot.processed, 0),
    cmdTicketsFailed: safeNumber(cmdTicketsRoot.failed_count ?? cmdTicketCountsRoot.failed, 0),
    cmdTicketsArchivedFailed: safeNumber(cmdTicketsRoot.archived_failed_count ?? cmdTicketExtendedRoot.archived_failed, 0),
    cmdTicketsInvalidPending: safeNumber(cmdTicketsRoot.invalid_count, 0),
    cmdTicketsGeneratedAt: String(cmdTicketsRoot.generated_at || cmdTicketsRoot.updated_at || ""),
    cmdTicketsInventoryNote: String(cmdTicketsRoot.inventory_note || "Cmd ticket counts are filesystem inventory counters."),
    updatedAt: nowIso(),
  };
}

function buildDevBridgeThoughts(summary: DevBridgeSummaryState | null): ThoughtTrace[] {
  if (!summary) return [];
  const stageLabel = summary.latestStageId
    ? `${summary.latestStageId} (${summary.latestStageStatus || "staged"})`
    : "no active stage";

  return [
    {
      id: `devbridge-trace-${summary.latestStageId || summary.latestResponseId || "status"}`,
      ts: summary.updatedAt,
      title: "DevBridge repair lane observed",
      content:
        `DevBridge reports ${summary.counts.packets} packets, ${summary.counts.responses} responses, and ${summary.counts.stages} staged patch sets. ` +
        `Repair tickets: ${summary.repairTickets}; batches: ${summary.repairBatches}; sandboxes: ${summary.sandboxes}. ` +
        `Cmd ticket inventory: pending ${summary.cmdTicketsPending}; processed archive ${summary.cmdTicketsProcessed}; active failed ${summary.cmdTicketsFailed}; archived failed ${summary.cmdTicketsArchivedFailed}. ` +
        `Latest stage: ${stageLabel}. Rollback: ${summary.rollbackAvailable ? "available" : summary.rollbackStatus || "idle"}. Developer gate: ${summary.developerMode || summary.envAllowApply ? "open" : "closed"}.`,
      source: "devbridge.status",
      level: summary.latestStageApplied ? "success" : summary.latestStageValidated ? "thinking" : "info",
      tags: ["devbridge", "repair", "governance", summary.latestStageStatus || "status"].filter(Boolean),
    },
  ];
}

function buildDevBridgeSubjects(summary: DevBridgeSummaryState | null): SubjectBox[] {
  if (!summary || (!summary.latestStageId && !summary.latestResponseId)) return [];
  const stage: SubjectStage = summary.latestStageApplied
    ? "approved"
    : summary.latestStageValidated
      ? "evaluation"
      : summary.latestStageId
        ? "sandbox"
        : "observed";

  return [
    {
      id: `devbridge-subject-${summary.latestStageId || summary.latestResponseId}`,
      title: summary.latestStageId ? `DevBridge staged repair: ${summary.latestStageId}` : `DevBridge response: ${summary.latestResponseId}`,
      summary:
        summary.latestSummary ||
        `DevBridge repair lane has ${summary.counts.responses} imported responses and ${summary.counts.stages} staged patch sets.`,
      source: "devbridge.repair_lane",
      stage,
      confidence: summary.latestStageValidated || summary.latestStageApplied ? 82 : 62,
      risk: summary.latestStageFiles > 0 ? 42 : 24,
      sandboxRecommended: !summary.latestStageValidated && !summary.latestStageApplied,
      notes: `Generated from /api/devbridge/latest, /api/devbridge/status, and /api/devbridge/cmd-tickets. Repair tickets ${summary.repairTickets}, batches ${summary.repairBatches}, sandboxes ${summary.sandboxes}. Cmd ticket inventory pending ${summary.cmdTicketsPending}, processed archive ${summary.cmdTicketsProcessed}, active failed ${summary.cmdTicketsFailed}, archived failed ${summary.cmdTicketsArchivedFailed}. Rollback ${summary.rollbackAvailable ? "available from DevBridge backup" : summary.rollbackStatus || "idle"}.`,
      tags: ["devbridge", "repair", "stage", summary.latestStageStatus || "status"].filter(Boolean),
      updatedAt: summary.updatedAt,
    },
  ];
}

function extractRemStatus(raw: any): RemStatusState | null {
  if (!raw) return null;
  const root = asRecord(raw);
  const rem = asRecord(root.rem || root.status || root);
  return Object.keys(rem).length > 0 ? (rem as RemStatusState) : null;
}

function isRemCycleLike(value: unknown): boolean {
  const rec = asRecord(value);
  return (
    safeArray(rec.dreams).length > 0 ||
    safeArray(rec.results).length > 0 ||
    Object.keys(asRecord(rec.subprocesses)).length > 0 ||
    typeof rec.cycle_number !== "undefined"
  );
}

function reportFromCycle(cycle: any): any {
  const rec = asRecord(cycle);
  if (!Object.keys(rec).length) return {};
  return {
    ...rec,
    cycles: [rec],
    summary: rec.summary || {},
  };
}

function extractReportRoot(rawReport: any, status: RemStatusState | null): any {
  const root = asRecord(rawReport);
  if (Array.isArray(root.cycles)) return root;
  if (isRemCycleLike(root)) return reportFromCycle(root);

  const rootLast = asRecord(root.last_report);
  if (Array.isArray(rootLast.cycles)) return rootLast;
  if (isRemCycleLike(rootLast)) return reportFromCycle(rootLast);

  const reports = safeArray<any>(root.reports);
  const lastFromReports = reports.length ? reports[reports.length - 1] : null;
  const lastReportRec = asRecord(lastFromReports);
  if (Array.isArray(lastReportRec.cycles)) return lastReportRec;
  if (isRemCycleLike(lastReportRec)) return reportFromCycle(lastReportRec);

  const statusLast = asRecord(status?.last_report);
  if (Array.isArray(statusLast.cycles)) return statusLast;
  if (isRemCycleLike(statusLast)) return reportFromCycle(statusLast);

  return {};
}

function getRemCycles(rawReport: any, status: RemStatusState | null): any[] {
  const report = extractReportRoot(rawReport, status);
  const cycles = safeArray<any>(report.cycles);
  if (cycles.length > 0) return cycles;
  return isRemCycleLike(report) ? [report] : [];
}

function flattenRemResults(cycles: any[]): any[] {
  return cycles.flatMap((cycle) => safeArray<any>(asRecord(cycle).results));
}

function getRemLaneEntries(cycles: any[]): Array<[string, any]> {
  const entries: Array<[string, any]> = [];
  for (const cycle of cycles) {
    const subprocesses = asRecord(asRecord(cycle).subprocesses);
    for (const [name, value] of Object.entries(subprocesses)) {
      entries.push([name, value]);
    }
  }
  return entries;
}

function remDecisionStage(decisionValue: unknown): SubjectStage {
  const decision = String(decisionValue || "").toLowerCase();
  if (decision.includes("auto") || decision.includes("allow")) return "approved";
  if (decision.includes("stage") || decision.includes("review")) return "evaluation";
  if (decision.includes("hold") || decision.includes("defer")) return "hold";
  if (decision.includes("deny") || decision.includes("reject") || decision.includes("fail")) return "rejected";
  return "observed";
}

function remRiskPercent(value: unknown): number {
  const s = String(value || "").toLowerCase();
  if (s === "low") return 18;
  if (s === "medium") return 55;
  if (s === "high") return 85;
  return clampPercent(value ?? 35);
}

function deriveRemDashboard(status: RemStatusState | null, rawReport: any): RemDerivedState {
  const cycles = getRemCycles(rawReport, status);
  const results = flattenRemResults(cycles);
  const lanes = getRemLaneEntries(cycles);
  const summary = asRecord(extractReportRoot(rawReport, status).summary);

  let sandboxPassed = safeNumber(summary.sandbox_passed, 0);
  let sandboxFailed = safeNumber(summary.sandbox_failed, 0);
  let lanesOk = safeNumber(summary.lanes_ok, 0);
  let lanesDegraded = safeNumber(summary.lanes_degraded, 0);
  let lanesFailed = safeNumber(summary.lanes_failed, 0);
  let computedAutoApplied = 0;
  let computedStaged = 0;
  let computedRejected = 0;
  let activeProvider = "";
  let activeModel = "";
  let activeModelCount = 0;

  for (const item of results) {
    const decision = String(asRecord(item).promotion?.decision || asRecord(item).decision || "").toLowerCase();
    if (decision.includes("auto") || decision.includes("applied")) computedAutoApplied += 1;
    else if (decision.includes("stage") || decision.includes("review") || decision.includes("defer")) computedStaged += 1;
    else if (decision.includes("reject") || decision.includes("deny") || decision.includes("fail")) computedRejected += 1;
  }

  if (!sandboxPassed && !sandboxFailed) {
    for (const item of results) {
      const sandbox = asRecord(asRecord(item).sandbox);
      if (Object.keys(sandbox).length === 0) continue;
      if (sandbox.passed) sandboxPassed += 1;
      else sandboxFailed += 1;
    }
  }

  if (!lanesOk && !lanesDegraded && !lanesFailed) {
    for (const [, lane] of lanes) {
      const l = asRecord(lane);
      const laneStatus = String(l.status || "ok").toLowerCase();
      if (l.ok && (laneStatus === "ok" || laneStatus === "")) lanesOk += 1;
      else if (l.ok && (laneStatus === "degraded" || laneStatus === "disabled")) lanesDegraded += 1;
      else if (l.skipped) lanesDegraded += 1;
      else lanesFailed += 1;
    }
  }

  for (const [name, lane] of lanes) {
    if (name !== "api") continue;
    const apiLane = asRecord(lane);
    const providers = asRecord(apiLane.providers);
    activeProvider = String(apiLane.selected_provider || "");
    activeModel = String(apiLane.selected_model || "");
    const availableProviders = Object.entries(providers).filter(([, info]) => Boolean(asRecord(info).available));
    activeModelCount = Math.max(activeModelCount, Number(apiLane.models_loaded || 0), availableProviders.length > 0 ? 1 : 0);
    if (!activeProvider && availableProviders.length > 0) {
      activeProvider = availableProviders[0][0];
    }
    if (!activeModel && activeProvider && providers[activeProvider]) {
      activeModel = String(asRecord(providers[activeProvider]).default_model || "");
    }
  }

  const cycleId = String(status?.cycle_id || extractReportRoot(rawReport, status).cycle_id || "");
  const phase = String(status?.phase || (status?.running ? "rem_dreaming" : "awake") || "awake");

  return {
    phase,
    running: Boolean(status?.running),
    enabled: Boolean(status?.enabled ?? status?.manual_force_allowed ?? status),
    neoskyEnabled: Boolean(status?.neosky_enabled),
    idleReady: Boolean(status?.idle_ready),
    cycleId,
    cyclesCompleted: Math.max(safeNumber(status?.cycles_completed, 0), safeNumber(summary.cycles, 0), cycles.length),
    dreams: Math.max(safeNumber(summary.dreams, 0), cycles.reduce((acc, cycle) => acc + safeArray(asRecord(cycle).dreams).length, 0)),
    results: Math.max(safeNumber(summary.results, 0), results.length),
    autoApplied: Math.max(safeNumber(status?.auto_applied, 0), safeNumber(summary.auto_applied, 0), computedAutoApplied),
    staged: Math.max(safeNumber(status?.staged, 0), safeNumber(summary.staged, 0), computedStaged),
    rejected: Math.max(safeNumber(status?.rejected, 0), safeNumber(summary.rejected, 0), computedRejected),
    sandboxPassed,
    sandboxFailed,
    lanesOk,
    lanesDegraded,
    lanesFailed,
    lanesTotal: Math.max(safeNumber(summary.lanes_total, 0), new Set(lanes.map(([name]) => name)).size),
    activeProvider,
    activeModel,
    activeModelCount,
    laneEntries: lanes.slice(-12).reverse(),
    latestCycles: cycles.slice(-3).reverse(),
  };
}

function buildRemThoughts(status: RemStatusState | null, rawReport: any): ThoughtTrace[] {
  const derived = deriveRemDashboard(status, rawReport);
  const cycles = getRemCycles(rawReport, status);
  const results = flattenRemResults(cycles).slice(-12).reverse();
  const laneEntries = getRemLaneEntries(cycles).slice(-12).reverse();
  const cycleId = derived.cycleId || "current";
  const out: ThoughtTrace[] = [];

  if (status) {
    out.push({
      id: `rem-phase-${cycleId}-${derived.phase}`,
      ts: String(status.updated_at || status.started_at || nowIso()),
      title: `REM Sleep phase: ${derived.phase}`,
      content: `REM ${derived.enabled ? "enabled" : "disabled"}; idle-ready ${derived.idleReady}; cycles ${derived.cyclesCompleted}; auto-applied ${derived.autoApplied}; staged ${derived.staged}; rejected ${derived.rejected}. Protected files: ${(status.protected_files || ["SarahMemoryGlobals.py"]).join(", ")}.`,
      source: "avatar.rem.status",
      level: derived.running ? "thinking" : derived.rejected > 0 || derived.lanesFailed > 0 ? "warning" : "success",
      tags: ["rem", "sleep", derived.phase],
    });
  }

  for (const [name, laneValue] of laneEntries) {
    const lane = asRecord(laneValue);
    const laneStatus = String(lane.status || (lane.ok ? "ok" : "error"));
    const timedOut = Boolean(lane.timed_out);
    out.push({
      id: `rem-lane-${cycleId}-${name}-${safeNumber(lane.duration_ms, 0)}-${laneStatus}`,
      ts: nowIso(),
      title: `REM lane ${name}: ${laneStatus}`,
      content: timedOut
        ? `${name} exceeded its REM timeout (${lane.timeout_seconds || "configured"} seconds).`
        : `${name} reported ok=${Boolean(lane.ok)} in ${safeNumber(lane.duration_ms, 0)}ms. ${lane.reason || lane.error || "No blocking issue reported."}`,
      source: `rem.${name}`,
      level: timedOut || !lane.ok ? "warning" : laneStatus === "degraded" || laneStatus === "disabled" ? "warning" : "success",
      tags: ["rem", "lane", name, laneStatus],
    });
  }

  for (const item of results) {
    const dream = asRecord(asRecord(item).dream);
    const promotion = asRecord(asRecord(item).promotion);
    const assurance = asRecord(asRecord(item).assurance);
    const governance = asRecord(asRecord(item).governance);
    const decision = String(promotion.decision || item.decision || "reviewed");
    out.push({
      id: `rem-decision-${String(dream.dream_id || dream.ticket_id || dream.title || uid("dream"))}`,
      ts: String(promotion.ts || assurance.ts || governance.ts || dream.ts || nowIso()),
      title: `REM reviewed: ${String(dream.title || "Dream candidate")}`,
      content: `Decision: ${decision}. Category: ${dream.category || "unknown"}. Risk: ${dream.risk_tier || governance.risk_tier || "unknown"}. ${safeArray<string>(promotion.reasons).join(" ")}`.trim(),
      source: "rem.dream.review",
      level: decision.toLowerCase().includes("auto") ? "success" : decision.toLowerCase().includes("deny") || decision.toLowerCase().includes("reject") ? "warning" : "thinking",
      tags: ["rem", "dream", String(dream.category || "candidate"), decision],
    });
  }

  return out;
}

function buildRemSubjects(rawReport: any, status: RemStatusState | null): SubjectBox[] {
  const cycles = getRemCycles(rawReport, status);
  const results = flattenRemResults(cycles);

  return results.map((item: any, idx: number) => {
    const result = asRecord(item);
    const dream = asRecord(result.dream);
    const promotion = asRecord(result.promotion);
    const assurance = asRecord(result.assurance);
    const governance = asRecord(result.governance);
    const decision = String(promotion.decision || result.decision || "reviewed");
    const id = String(dream.dream_id || dream.ticket_id || `rem-subject-${idx}`);
    const confidence = clampPercent((safeNumber(assurance.assurance_score, safeNumber(assurance.confidence, 0.75)) || 0.75) * 100);
    const risk = governance.risk_score !== undefined ? clampPercent(governance.risk_score) : remRiskPercent(dream.risk_tier || governance.risk_tier);

    return {
      id: `rem-${id}`,
      title: String(dream.title || "REM candidate"),
      summary: String(dream.rationale || safeArray<string>(promotion.reasons).join(" ") || decision),
      source: String(dream.category || "rem"),
      stage: remDecisionStage(decision),
      confidence,
      risk,
      sandboxRecommended: !decision.toLowerCase().includes("auto"),
      notes: "",
      tags: Array.from(new Set(["rem", String(dream.category || "candidate"), String(dream.risk_tier || governance.risk_tier || "risk")])),
      updatedAt: String(promotion.ts || assurance.ts || governance.ts || dream.ts || nowIso()),
    };
  });
}

export function DLEngineScreen() {
  const { setCurrentScreen } = useNavigationStore();

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [remStatus, setRemStatus] = useState<RemStatusState | null>(null);
  const [remReport, setRemReport] = useState<any>(null);
  const [modelCapability, setModelCapability] = useState<ActiveCapabilityState>(() =>
    normalizeCapability(null)
  );
  const [modelManagerStatus, setModelManagerStatus] = useState<any>(null);
  const [selectedWeightCategory, setSelectedWeightCategory] = useState<string>(() =>
    loadJson<string>(DL_ENGINE_WEIGHT_CATEGORY_KEY, "reasoning")
  );
  const [selectedWeightModelId, setSelectedWeightModelId] = useState<string>(() =>
    loadJson<string>(DL_ENGINE_WEIGHT_MODEL_KEY, "")
  );
  const [isLoadingWeightProfile, setIsLoadingWeightProfile] = useState(false);

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

  const [modelWeights, setModelWeights] = useState<ModelWeightControlState>(() =>
    normalizeModelWeights(loadJson<Partial<ModelWeightControlState>>(DL_ENGINE_WEIGHTS_KEY, defaultModelWeights()))
  );
  const [dlMode, setDlMode] = useState<DLRuntimeMode>(() =>
    loadJson<DLRuntimeMode>(DL_ENGINE_MODE_KEY, "auto")
  );
  const [isRemCommanding, setIsRemCommanding] = useState(false);
  const [isDlCommanding, setIsDlCommanding] = useState(false);
  const [isSavingWeights, setIsSavingWeights] = useState(false);
  const [devBridgeSummary, setDevBridgeSummary] = useState<DevBridgeSummaryState | null>(null);

  const selectedSubject = useMemo(
    () => subjects.find((s) => s.id === selectedSubjectId) || null,
    [selectedSubjectId, subjects]
  );

  const remSummary = useMemo(() => deriveRemDashboard(remStatus, remReport), [remReport, remStatus]);

  const effectiveModelsLoaded = Math.max(
    stats?.modelsLoaded || 0,
    remSummary.activeModelCount || 0,
    modelCapability.activeModelCount || 0,
    Number(modelManagerStatus?.model_count || 0)
  );

  const weightCategories = useMemo(() => {
    const backendCategories = Array.isArray(modelManagerStatus?.categories) ? modelManagerStatus.categories : [];
    const defaults = [
      { id: "reasoning", label: "General Thinking" },
      { id: "coder", label: "Coding Help" },
      { id: "embeddings", label: "Memory Search" },
      { id: "vision", label: "Vision / Camera" },
      { id: "image_generation", label: "Image Creation" },
      { id: "tts", label: "Voice / Speech" },
      { id: "unknown", label: "Unclassified" },
    ];
    const merged = [...backendCategories, ...defaults];
    const seen = new Set<string>();
    return merged.filter((cat: any) => {
      const id = String(cat?.id || "").trim();
      if (!id || seen.has(id)) return false;
      seen.add(id);
      return true;
    });
  }, [modelManagerStatus]);

  const allManagedModels = useMemo(() => {
    return Array.isArray(modelManagerStatus?.models) ? modelManagerStatus.models : [];
  }, [modelManagerStatus]);

  const activeWeightModelId = String(modelManagerStatus?.active_models?.[selectedWeightCategory] || "");

  const selectedWeightCategoryLabel = useMemo(() => {
    const found = weightCategories.find((cat: any) => String(cat?.id || "") === selectedWeightCategory);
    return String(found?.label || selectedWeightCategory || "General Thinking");
  }, [selectedWeightCategory, weightCategories]);

  const weightModelOptions = useMemo(() => {
    const activeId = String(modelManagerStatus?.active_models?.[selectedWeightCategory] || "");
    return allManagedModels.filter((model: any) => {
      const category = String(model?.category || model?.detected_category || "unknown");
      return category === selectedWeightCategory || String(model?.id || "") === activeId;
    });
  }, [allManagedModels, modelManagerStatus, selectedWeightCategory]);

  // Empty selectedWeightModelId is intentional: it means category-default profile.
  // Do not silently substitute the active model or first model here, or the sliders
  // will not switch to category defaults when the Model job dropdown changes.
  const effectiveWeightModelId = selectedWeightModelId;

  const selectedWeightModel = useMemo(() => {
    return weightModelOptions.find((model: any) => String(model?.id || "") === String(selectedWeightModelId)) || null;
  }, [selectedWeightModelId, weightModelOptions]);

  const selectedWeightContextLabel = selectedWeightModelId
    ? String(selectedWeightModel?.simple_label || selectedWeightModel?.display_name || selectedWeightModel?.repo || selectedWeightModelId)
    : `${selectedWeightCategoryLabel} / Category default`;

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

  const setWeight = useCallback((key: keyof ModelWeightControlState, value: number) => {
    setModelWeights((prev) => {
      const next = normalizeModelWeights({ ...prev, [key]: value });
      saveJson(weightProfileStorageKey(selectedWeightCategory, selectedWeightModelId), next);
      saveJson(DL_ENGINE_WEIGHTS_KEY, next);
      return next;
    });
  }, [selectedWeightCategory, selectedWeightModelId]);

  const resetWeights = useCallback(async () => {
    setIsLoadingWeightProfile(true);
    try {
      let result: any = null;
      try {
        const res = await fetch("/api/dlengine/weights/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            category: selectedWeightCategory,
            model_id: selectedWeightModelId,
            source: "dlengine_screen_reset",
          }),
        });
        result = await res.json().catch(() => null);
      } catch {
        result = null;
      }
      const next = normalizeModelWeights(result?.weights || defaultModelWeightsForCategory(selectedWeightCategory));
      setModelWeights(next);
      saveJson(weightProfileStorageKey(selectedWeightCategory, selectedWeightModelId), next);
      saveJson(DL_ENGINE_WEIGHTS_KEY, next);
      setStatusMessage(`Model weight controller reset for ${selectedWeightContextLabel}.`);
    } finally {
      setIsLoadingWeightProfile(false);
    }
  }, [selectedWeightCategory, selectedWeightContextLabel, selectedWeightModelId]);

  const tryCall = useCallback(async (path: string, payload?: any) => {
    try {
      const result = await api.proxy.call(path, payload);
      if (result) return result;
    } catch {
      // Fall through to direct fetch. Some API proxy builds only support a subset
      // of methods, but this panel needs the controls to work directly.
    }

    try {
      const isPost = typeof payload !== "undefined";
      const res = await fetch(path, {
        method: isPost ? "POST" : "GET",
        headers: isPost ? { "Content-Type": "application/json" } : undefined,
        credentials: "include",
        body: isPost ? JSON.stringify(payload) : undefined,
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) return body || { ok: false, error: `HTTP ${res.status}` };
      return body;
    } catch {
      return null;
    }
  }, []);

  const loadWeightProfile = useCallback(
    async (category: string, modelId: string) => {
      const cat = String(category || "reasoning");
      const mid = String(modelId || "");
      setIsLoadingWeightProfile(true);
      try {
        const qs = new URLSearchParams();
        qs.set("category", cat);
        if (mid) qs.set("model_id", mid);
        const result = await tryCall(`/api/dlengine/weights?${qs.toString()}`);
        const storageKey = weightProfileStorageKey(cat, mid);
        if (result && (result as any).ok !== false && (result as any).weights) {
          const nextWeights = normalizeModelWeights((result as any).weights);
          setModelWeights(nextWeights);
          saveJson(storageKey, nextWeights);
          saveJson(DL_ENGINE_WEIGHTS_KEY, nextWeights);
          setStatusMessage(`Loaded governed weights for ${String((result as any)?.context?.model_label || mid || cat)}.`);
        } else {
          const fallback = normalizeModelWeights(loadJson<Partial<ModelWeightControlState>>(storageKey, defaultModelWeightsForCategory(cat)));
          setModelWeights(fallback);
          saveJson(DL_ENGINE_WEIGHTS_KEY, fallback);
          setStatusMessage(`Loaded local governed weights for ${mid || cat}.`);
        }
      } finally {
        setIsLoadingWeightProfile(false);
      }
    },
    [tryCall]
  );

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
      const modelStatusResponse = await tryCall("/api/models/status?refresh=1");
      const remStatusResponse = await tryCall("/api/avatar/rem/status");
      const remReportResponse = await tryCall(`/api/avatar/rem/report?limit=${DL_ENGINE_REM_REPORT_LIMIT}`);
      const thoughtResponse =
        (await tryCall("/api/dlengine/thoughts")) ||
        (await tryCall("/api/cognitive/thoughts")) ||
        (await tryCall("/api/cognitive/trace"));
      const subjectResponse =
        (await tryCall("/api/dlengine/subjects")) ||
        (await tryCall("/api/dlengine/tickets")) ||
        (await tryCall("/api/cognitive/tickets"));
      const devBridgeStatusResponse = await tryCall("/api/devbridge/status");
      const devBridgeLatestResponse = await tryCall("/api/devbridge/latest");
      const devBridgeCmdTicketsResponse = await tryCall("/api/devbridge/cmd-tickets?limit=25&detail_limit=10");
      const nextDevBridgeSummary = summarizeDevBridgeState(devBridgeLatestResponse, devBridgeStatusResponse, devBridgeCmdTicketsResponse);
      setDevBridgeSummary(nextDevBridgeSummary);

      if (modelStatusResponse && (modelStatusResponse as any).ok !== false) {
        setModelManagerStatus(modelStatusResponse);
      }

      const nextRemStatus =
        extractRemStatus(remStatusResponse) ||
        extractRemStatus((statusResponse as any)?.rem ? { rem: (statusResponse as any).rem } : null);
      if (nextRemStatus) setRemStatus(nextRemStatus);
      if (remReportResponse) setRemReport(remReportResponse);

      if ((statusResponse as any)?.model) {
        setModelCapability((prev) => normalizeCapability((statusResponse as any).model, prev));
      } else if ((statusResponse as any)?.rem_summary) {
        setModelCapability((prev) =>
          normalizeCapability(
            {
              active_provider: (statusResponse as any).rem_summary.active_provider,
              active_model: (statusResponse as any).rem_summary.active_model,
              active_model_count: (statusResponse as any).rem_summary.active_model_count,
            },
            prev
          )
        );
      }

      if ((statusResponse as any)?.controls && typeof (statusResponse as any).controls === "object") {
        setControls((prev) => {
          const next = { ...prev, ...(statusResponse as any).controls };
          saveJson(DL_ENGINE_CONTROLS_KEY, next);
          return next;
        });
      }

      // Weight sliders are profile-scoped. Do not overwrite them from the
      // generic DL status payload during polling; the selected category/model
      // profile is loaded by loadWeightProfile().

      const backendMode =
        (statusResponse as any)?.runtime?.runtime_mode ||
        (statusResponse as any)?.runtime?.mode ||
        (statusResponse as any)?.state?.mode;
      if (backendMode) {
        const normalizedMode = normalizeDlMode(backendMode);
        setDlMode(normalizedMode);
        saveJson(DL_ENGINE_MODE_KEY, normalizedMode);
      }

      if (statusResponse || remStatusResponse || remReportResponse) {
        setIsAvailable(true);

        const normalizedStats = statusResponse
          ? normalizeStats((statusResponse as any)?.stats || statusResponse)
          : normalizeStats({
              modelsLoaded: nextRemStatus ? 1 : 0,
              activeJobs: nextRemStatus?.running ? 1 : 0,
              memoryUsage: 0,
              gpuUsage: 0,
              thinkingLoad: nextRemStatus?.running ? 66 : 15,
              subjectsOpen: 0,
              ticketsPending: 0,
            });
        setStats(normalizedStats);

        const normalizedJobs = normalizeJobs((statusResponse as any)?.jobs || []);
        setJobs(normalizedJobs);

        const remThoughts = buildRemThoughts(nextRemStatus || remStatus, remReportResponse || remReport);
        const normalizedThoughts =
          thoughtResponse && Array.isArray((thoughtResponse as any)?.thoughts)
            ? normalizeThoughts((thoughtResponse as any).thoughts)
            : thoughtResponse && Array.isArray((thoughtResponse as any)?.events)
              ? normalizeThoughts((thoughtResponse as any).events)
              : buildFallbackThoughts(normalizedStats, normalizedJobs);

        const trimmedThoughts = [...remThoughts, ...normalizedThoughts]
          .sort((a, b) => safeDate(b.ts).getTime() - safeDate(a.ts).getTime())
          .slice(0, DL_ENGINE_TRACE_LIMIT);

        const devBridgeThoughts = buildDevBridgeThoughts(nextDevBridgeSummary);
        mergeThoughts([...devBridgeThoughts, ...trimmedThoughts]);

        const remSubjects = buildRemSubjects(remReportResponse || remReport, nextRemStatus || remStatus);
        const backendSubjects =
          subjectResponse && Array.isArray((subjectResponse as any)?.subjects)
            ? normalizeSubjects((subjectResponse as any).subjects)
            : subjectResponse && Array.isArray((subjectResponse as any)?.tickets)
              ? normalizeSubjects((subjectResponse as any).tickets)
              : deriveSubjectsFromThoughts(trimmedThoughts);

        const devBridgeSubjects = buildDevBridgeSubjects(nextDevBridgeSummary);
        mergeSubjects([...devBridgeSubjects, ...remSubjects, ...backendSubjects]);
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
  }, [isLoadingWeightProfile, isSavingWeights, mergeSubjects, mergeThoughts, normalizeJobs, normalizeStats, normalizeSubjects, normalizeThoughts, remReport, remStatus, tryCall]);

  useEffect(() => {
    void checkStatus();
  }, [checkStatus]);

  useEffect(() => {
    // If a model profile disappears from the selected category, fall back to
    // the category-default profile instead of silently using another model.
    if (selectedWeightModelId && !weightModelOptions.some((m: any) => String(m?.id || "") === selectedWeightModelId)) {
      setSelectedWeightModelId("");
      saveJson(DL_ENGINE_WEIGHT_MODEL_KEY, "");
      saveJson(`${DL_ENGINE_WEIGHT_MODEL_KEY}:${selectedWeightCategory}`, "");
    }
  }, [selectedWeightCategory, selectedWeightModelId, weightModelOptions]);

  useEffect(() => {
    saveJson(DL_ENGINE_WEIGHT_CATEGORY_KEY, selectedWeightCategory);
    saveJson(DL_ENGINE_WEIGHT_MODEL_KEY, selectedWeightModelId);
    saveJson(`${DL_ENGINE_WEIGHT_MODEL_KEY}:${selectedWeightCategory}`, selectedWeightModelId);
    void loadWeightProfile(selectedWeightCategory, selectedWeightModelId);
  }, [loadWeightProfile, selectedWeightCategory, selectedWeightModelId]);

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

  const toggleRemSleep = useCallback(async () => {
    setIsRemCommanding(true);
    const goingToWake = Boolean(remSummary.running || remStatus?.running);
    const path = goingToWake ? "/api/avatar/rem/stop" : "/api/avatar/rem/start";
    const payload = {
      reason: goingToWake ? "manual_dlengine_wake" : "manual_dlengine_sleep",
      force: !goingToWake,
      source: "DLEngineScreen",
    };

    try {
      const result = await tryCall(path, payload);
      if (result && (result as any).ok !== false) {
        setStatusMessage(goingToWake ? "Manual wake request sent to REM controller." : "Manual REM Sleep request sent to controller.");
        pushThought({
          id: uid("trace"),
          ts: nowIso(),
          title: goingToWake ? "Manual REM wake requested" : "Manual REM sleep requested",
          content: goingToWake
            ? "The operator requested SarahMemory to wake from REM mode from the DL Engine control deck."
            : "The operator manually placed SarahMemory into REM Sleep from the DL Engine control deck.",
          source: "ui.rem.control",
          level: goingToWake ? "info" : "thinking",
          tags: ["rem", "manual", goingToWake ? "wake" : "sleep"],
        });
      } else {
        const blockedReason = String(
          (result as any)?.checks?.blocked_reason ||
            (result as any)?.state?.reason ||
            (result as any)?.error ||
            "backend endpoint unavailable"
        );
        setStatusMessage(`REM controller did not accept ${goingToWake ? "wake" : "sleep"} request: ${blockedReason}`);
      }
      await checkStatus();
    } finally {
      setIsRemCommanding(false);
    }
  }, [checkStatus, pushThought, remStatus?.running, remSummary.running, tryCall]);

  const submitDlRuntimeMode = useCallback(async (mode: DLRuntimeMode) => {
    setIsDlCommanding(true);
    setDlMode(mode);
    saveJson(DL_ENGINE_MODE_KEY, mode);

    const payload = {
      mode,
      source: "dlengine_screen",
      category: selectedWeightCategory,
      model_id: effectiveWeightModelId,
      context: {
        category: selectedWeightCategory,
        model_id: effectiveWeightModelId,
        model_label: selectedWeightContextLabel,
      },
      controls,
      weights: modelWeights,
      ts: nowIso(),
    };

    const endpoints = [
      "/api/dlengine/mode",
      "/api/dlengine/control",
      "/api/dlengine/controls",
      mode === "manual" ? "/api/dlengine/start" : mode === "paused" ? "/api/dlengine/stop" : "/api/dlengine/auto",
    ];

    let sent = false;
    for (const path of endpoints) {
      const result = await tryCall(path, payload);
      if (result && (result as any).ok !== false) {
        sent = true;
        break;
      }
    }

    setStatusMessage(
      sent
        ? `Deep Learning runtime mode set to ${mode.toUpperCase()}.`
        : `Deep Learning runtime mode saved locally as ${mode.toUpperCase()}; backend endpoint not available yet.`
    );

    pushThought({
      id: uid("trace"),
      ts: nowIso(),
      title: `DL runtime mode: ${mode.toUpperCase()}`,
      content: sent
        ? `The operator switched the Deep Learning Engine to ${mode} mode and synced it with the backend.`
        : `The operator switched the Deep Learning Engine to ${mode} mode locally. Backend runtime endpoint did not respond.`,
      source: "ui.dl.control",
      level: mode === "paused" ? "warning" : mode === "manual" ? "thinking" : "success",
      tags: ["deep-learning", "mode", mode],
    });

    setIsDlCommanding(false);
  }, [controls, effectiveWeightModelId, modelWeights, pushThought, selectedWeightCategory, selectedWeightContextLabel, tryCall]);

  const submitWeightControls = useCallback(async () => {
    setIsSavingWeights(true);

    const payload = {
      weights: modelWeights,
      mode: dlMode,
      category: selectedWeightCategory,
      model_id: effectiveWeightModelId,
      context: {
        category: selectedWeightCategory,
        model_id: effectiveWeightModelId,
        model_label: selectedWeightContextLabel,
      },
      source: "dlengine_model_weight_controller",
      note: "Governed AI tuner weights. These are policy/routing weights, not raw tensor edits.",
      ts: nowIso(),
    };

    const endpoints = [
      "/api/dlengine/weights",
      "/api/dlengine/tuning_weights",
      "/api/dlengine/controls",
      "/api/cognitive/weights",
    ];

    let sent = false;
    for (const path of endpoints) {
      const result = await tryCall(path, payload);
      if (result && (result as any).ok !== false) {
        sent = true;
        break;
      }
    }

    saveJson(weightProfileStorageKey(selectedWeightCategory, effectiveWeightModelId), modelWeights);
    saveJson(DL_ENGINE_WEIGHTS_KEY, modelWeights);
    setStatusMessage(sent ? `Model weight profile synced for ${selectedWeightContextLabel}.` : `Model weight profile saved locally for ${selectedWeightContextLabel}; backend endpoint not available yet.`);
    pushThought({
      id: uid("trace"),
      ts: nowIso(),
      title: "Model Weight Controller updated",
      content: `Weights saved for ${selectedWeightContextLabel}: reasoning ${modelWeights.reasoning}, coding ${modelWeights.coding}, memory ${modelWeights.memory}, research ${modelWeights.research}, safety ${modelWeights.safety}, autonomy ${modelWeights.autonomy}.`,
      source: "ui.weight_controller",
      level: sent ? "success" : "warning",
      tags: ["weights", "tuning", "governance"],
    });

    setIsSavingWeights(false);
  }, [dlMode, effectiveWeightModelId, modelWeights, pushThought, selectedWeightCategory, selectedWeightContextLabel, tryCall]);

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

          if (a.type === "dlengine_rem_toggle") {
            void toggleRemSleep();
          }

          if (a.type === "dlengine_rem_sleep") {
            if (!remSummary.running) void toggleRemSleep();
          }

          if (a.type === "dlengine_rem_wake") {
            if (remSummary.running) void toggleRemSleep();
          }

          if (a.type === "dlengine_set_mode" && a.payload?.mode) {
            const mode = String(a.payload.mode).toLowerCase();
            if (mode === "auto" || mode === "manual" || mode === "paused") {
              void submitDlRuntimeMode(mode as DLRuntimeMode);
            }
          }
        } catch (e) {
          console.warn("[DLEngineScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [checkStatus, controls.showOnlyHighSignal, mergeSubjects, mergeThoughts, normalizeSubjects, normalizeThoughts, remSummary.running, selectedSubjectId, setControl, setCurrentScreen, subjects, submitDlRuntimeMode, submitSubjectAction, toggleRemSleep]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="shrink-0 border-b border-border bg-card/50 p-4">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">DL Engine</h1>
          <div className="ml-auto flex items-center gap-2">
            <span className={cn("rounded-full px-2 py-1 text-[11px]", remSummary.running ? "bg-blue-500/10 text-blue-500" : "bg-muted text-muted-foreground")}>
              {remSummary.running ? "REM ACTIVE" : "AWAKE"}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => void checkStatus()}
              disabled={isLoading}
              title="Refresh DL Engine / REM status"
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
          </div>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          Master control surface for Deep Learning, REM Sleep, subject review, sandbox routing, governance, and AI tuning.
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground/80">
          {lastUpdated && <span>Last updated: {safeDate(lastUpdated).toLocaleString()}</span>}
          {remSummary.cycleId && <span>• REM cycle: {remSummary.cycleId}</span>}
          {statusMessage && <span className="text-primary">• {statusMessage}</span>}
        </div>
      </div>

      {/* Fixed control deck: separated from live trace scroll so polling cannot jump controls around. */}
      <div className="shrink-0 border-b border-border bg-background/95 p-3 shadow-sm">
        <div className="max-h-[46vh] overflow-y-auto pr-1">
          {isAvailable === false && (
            <div className="mb-3 rounded-xl border border-border bg-muted/50 p-3">
              <div className="flex items-start gap-3">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Backend DL Engine unavailable</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    The screen remains active in local review mode. Controls are preserved and will sync when backend endpoints respond.
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_1.05fr_1.15fr]">
            {/* Runtime command center */}
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <p className="flex items-center gap-2 text-sm font-medium">
                    <Power className="h-4 w-4 text-primary" />
                    Runtime Command Center
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Manual REM and DL controls for fast validation without PowerShell.
                  </p>
                </div>
                <span className={cn("rounded-full px-2 py-1 text-[11px]", remSummary.neoskyEnabled ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500")}>
                  NeoSky {remSummary.neoskyEnabled ? "ON" : "LOCKED"}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                <Button
                  variant={remSummary.running ? "destructive" : "default"}
                  size="sm"
                  onClick={() => void toggleRemSleep()}
                  disabled={isRemCommanding}
                  className="justify-start"
                  title="Toggle SarahMemory REM Sleep manually. When NeoSky is locked, REM runs in observation-only mode."
                >
                  {remSummary.running ? <Sun className="mr-2 h-4 w-4" /> : <Moon className="mr-2 h-4 w-4" />}
                  {isRemCommanding ? "Sending…" : remSummary.running ? "Wake Sarah" : "Force REM Sleep"}
                </Button>

                <Button
                  variant={dlMode === "manual" ? "default" : "outline"}
                  size="sm"
                  onClick={() => void submitDlRuntimeMode("manual")}
                  disabled={isDlCommanding}
                  className="justify-start"
                >
                  <Zap className="mr-2 h-4 w-4" />
                  Manual DL Run
                </Button>

                <Button
                  variant={dlMode === "auto" ? "default" : "outline"}
                  size="sm"
                  onClick={() => void submitDlRuntimeMode("auto")}
                  disabled={isDlCommanding}
                  className="justify-start"
                >
                  <Activity className="mr-2 h-4 w-4" />
                  DL Auto Mode
                </Button>

                <Button
                  variant={dlMode === "paused" ? "destructive" : "outline"}
                  size="sm"
                  onClick={() => void submitDlRuntimeMode("paused")}
                  disabled={isDlCommanding}
                  className="justify-start"
                >
                  <PauseCircle className="mr-2 h-4 w-4" />
                  Pause DL
                </Button>
              </div>

              <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
                <div className="rounded-lg border border-border bg-background/50 p-2">
                  <p className="text-muted-foreground">REM Phase</p>
                  <p className="truncate font-semibold">{remSummary.phase}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-2">
                  <p className="text-muted-foreground">Idle Ready</p>
                  <p className="font-semibold">{remSummary.idleReady ? "Yes" : "No"}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-2">
                  <p className="text-muted-foreground">DL Mode</p>
                  <p className="font-semibold uppercase">{dlMode}</p>
                </div>
              </div>
            </div>

            {/* Metrics deck */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="mb-3 flex items-center gap-2 text-sm font-medium">
                <Gauge className="h-4 w-4 text-primary" />
                Live Counters / Runtime Health
              </p>

              <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-2 2xl:grid-cols-4">
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Models / Providers</p>
                  <p className="mt-1 text-xl font-bold">{effectiveModelsLoaded}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Active Jobs / REM</p>
                  <p className="mt-1 text-xl font-bold">{Math.max(stats?.activeJobs || 0, remSummary.running ? 1 : 0)}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">REM Cycles</p>
                  <p className="mt-1 text-xl font-bold">{remSummary.cyclesCompleted}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Dreams</p>
                  <p className="mt-1 text-xl font-bold">{remSummary.dreams}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Auto-Applied</p>
                  <p className="mt-1 text-xl font-bold text-green-500">{remSummary.autoApplied}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Staged</p>
                  <p className="mt-1 text-xl font-bold text-blue-500">{remSummary.staged}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Rejected</p>
                  <p className="mt-1 text-xl font-bold text-yellow-500">{remSummary.rejected}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Lane Issues</p>
                  <p className="mt-1 text-xl font-bold">{remSummary.lanesDegraded + remSummary.lanesFailed}</p>
                </div>
              </div>

              <div className="mt-3 rounded-lg border border-border bg-background/50 p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="flex items-center gap-2 text-xs font-medium">
                      <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                      DevBridge Repair Lane
                    </p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      {devBridgeSummary?.available
                        ? `Packets ${devBridgeSummary.counts.packets} • Responses ${devBridgeSummary.counts.responses} • Stages ${devBridgeSummary.counts.stages} • Cmd inventory pending ${devBridgeSummary.cmdTicketsPending}`
                        : "No DevBridge telemetry loaded yet."}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={cn(
                        "rounded-full px-2 py-1 text-[10px] font-medium",
                        devBridgeSummary?.developerMode || devBridgeSummary?.envAllowApply
                          ? "bg-green-500/10 text-green-500"
                          : "bg-muted text-muted-foreground"
                      )}
                    >
                      {devBridgeSummary?.developerMode || devBridgeSummary?.envAllowApply ? "DEV GATE OPEN" : "DEV GATE CLOSED"}
                    </span>
                    <Button variant="outline" size="sm" onClick={() => setCurrentScreen("research" as any)}>
                      Open Bridge
                    </Button>
                  </div>
                </div>

                <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-3">
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Latest Stage</p>
                    <p className="mt-1 truncate text-xs font-semibold">{devBridgeSummary?.latestStageId || "—"}</p>
                    <p className="mt-1 text-[10px] capitalize text-muted-foreground">{devBridgeSummary?.latestStageStatus || "idle"}</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Validation</p>
                    <p className="mt-1 text-xs font-semibold">
                      {devBridgeSummary?.latestStageApplied
                        ? "Applied"
                        : devBridgeSummary?.latestStageValidated
                          ? "Validated"
                          : devBridgeSummary?.latestStageId
                            ? "Pending"
                            : "Idle"}
                    </p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Files: {devBridgeSummary?.latestStageFiles || 0}</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Latest Response</p>
                    <p className="mt-1 truncate text-xs font-semibold">{devBridgeSummary?.latestResponseId || "—"}</p>
                    <p className="mt-1 truncate text-[10px] text-muted-foreground">{devBridgeSummary?.latestSummary || "No imported repair summary."}</p>
                  </div>
                </div>

                <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-4 2xl:grid-cols-5">
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Repair Tickets</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.repairTickets || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Buffered issues</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Repair Batches</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.repairBatches || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Grouped by file</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Sandboxes</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.sandboxes || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Apps / panels / IoT drafts</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Pending Queue</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.cmdTicketsPending || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Processable now</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Processed Archive</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.cmdTicketsProcessed || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Historical success files</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Active Failed</p>
                    <p className={cn("mt-1 text-xs font-semibold", devBridgeSummary?.cmdTicketsFailed ? "text-red-500" : "")}>{devBridgeSummary?.cmdTicketsFailed || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Failed folder inventory</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Failed Archive</p>
                    <p className="mt-1 text-xs font-semibold">{devBridgeSummary?.cmdTicketsArchivedFailed || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Preserved audit evidence</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Invalid Pending</p>
                    <p className={cn("mt-1 text-xs font-semibold", devBridgeSummary?.cmdTicketsInvalidPending ? "text-red-500" : "")}>{devBridgeSummary?.cmdTicketsInvalidPending || 0}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Malformed queued JSON</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Cmd Snapshot</p>
                    <p className="mt-1 truncate text-xs font-semibold">{devBridgeSummary?.cmdTicketsGeneratedAt ? safeDate(devBridgeSummary.cmdTicketsGeneratedAt).toLocaleTimeString() : "—"}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">Static inventory refresh</p>
                  </div>
                  <div className="rounded-md border border-border bg-card/50 p-2">
                    <p className="text-[10px] text-muted-foreground">Rollback</p>
                    <p className={cn("mt-1 text-xs font-semibold", devBridgeSummary?.rollbackAvailable ? "text-yellow-500" : "")}>{devBridgeSummary?.rollbackAvailable ? "Available" : devBridgeSummary?.rollbackStatus || "Idle"}</p>
                    <p className="mt-1 truncate text-[10px] text-muted-foreground">{devBridgeSummary?.backupRoot || "No active backup root."}</p>
                  </div>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground">Active AI Capability</p>
                  <p className="mt-1 truncate text-sm font-semibold">{modelCapability.activeProvider || remSummary.activeProvider || "core"}</p>
                  <p className="mt-1 truncate text-xs text-muted-foreground">{modelCapability.activeModel || remSummary.activeModel || "SarahMemory Core Runtime"}</p>
                </div>
                <div className="rounded-lg border border-border bg-background/50 p-3">
                  <p className="text-[11px] text-muted-foreground mb-2">Memory / GPU / Thinking</p>
                  <div className="space-y-1.5">
                    <div className="grid grid-cols-[58px_1fr_34px] items-center gap-2 text-[11px]"><span>Memory</span><Progress value={stats?.memoryUsage || 0} className="h-2" /><span>{stats?.memoryUsage || 0}%</span></div>
                    <div className="grid grid-cols-[58px_1fr_34px] items-center gap-2 text-[11px]"><span>GPU</span><Progress value={stats?.gpuUsage || 0} className="h-2" /><span>{stats?.gpuUsage || 0}%</span></div>
                    <div className="grid grid-cols-[58px_1fr_34px] items-center gap-2 text-[11px]"><span>Think</span><Progress value={stats?.thinkingLoad || 0} className="h-2" /><span>{stats?.thinkingLoad || 0}%</span></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Governance + model tuner */}
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <p className="flex items-center gap-2 text-sm font-medium">
                    <SlidersHorizontal className="h-4 w-4 text-primary" />
                    Governance + Model Weight Controller
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Operator-facing AI tuner. These sliders are governed routing/policy weights, not raw tensor edits.
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => void resetWeights()} title="Reset current model profile to defaults">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  <Button variant="default" size="sm" onClick={() => void submitWeightControls()} disabled={isSavingWeights}>
                    <Save className="mr-2 h-4 w-4" />
                    Save Weights
                  </Button>
                </div>
              </div>

              <div className="mb-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                <label className="rounded-lg border border-border bg-background/50 p-2">
                  <span className="text-[11px] text-muted-foreground">Model job</span>
                  <select
                    value={selectedWeightCategory}
                    onChange={(e) => {
                      const nextCategory = e.target.value;
                      const rememberedModelId = loadJson<string>(`${DL_ENGINE_WEIGHT_MODEL_KEY}:${nextCategory}`, "");
                      setSelectedWeightCategory(nextCategory);
                      setSelectedWeightModelId(rememberedModelId);
                      saveJson(DL_ENGINE_WEIGHT_CATEGORY_KEY, nextCategory);
                      saveJson(DL_ENGINE_WEIGHT_MODEL_KEY, rememberedModelId);
                    }}
                    className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs outline-none"
                  >
                    {weightCategories.map((cat: any) => (
                      <option key={String(cat.id)} value={String(cat.id)}>
                        {String(cat.label || cat.id)}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="rounded-lg border border-border bg-background/50 p-2">
                  <span className="text-[11px] text-muted-foreground">Model profile</span>
                  <select
                    value={selectedWeightModelId || "__category_default__"}
                    onChange={(e) => {
                      const nextModelId = e.target.value === "__category_default__" ? "" : e.target.value;
                      setSelectedWeightModelId(nextModelId);
                      saveJson(DL_ENGINE_WEIGHT_MODEL_KEY, nextModelId);
                      saveJson(`${DL_ENGINE_WEIGHT_MODEL_KEY}:${selectedWeightCategory}`, nextModelId);
                    }}
                    className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs outline-none"
                  >
                    <option value="__category_default__">Category default</option>
                    {weightModelOptions.map((model: any) => (
                      <option key={String(model.id)} value={String(model.id)}>
                        {String(model.simple_label || model.display_name || model.repo || model.id)}
                        {model.status_label ? ` · ${String(model.status_label)}` : ""}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="mb-3 rounded-lg border border-border bg-background/50 p-2 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Active weight profile:</span>{" "}
                {selectedWeightContextLabel}
                {isLoadingWeightProfile ? " · loading profile…" : ""}
                <span className="block pt-1 text-[10px]">
                  Profiles are stored by category/model context. These sliders change governed routing weights only.
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2 md:grid-cols-2 2xl:grid-cols-3">
                {MODEL_WEIGHT_SLIDERS.map((item) => (
                  <label key={item.key} className="rounded-lg border border-border bg-background/50 p-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-medium">{item.label}</span>
                      <span className="text-[11px] text-muted-foreground">{modelWeights[item.key]}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={modelWeights[item.key]}
                      onChange={(e) => setWeight(item.key, Number(e.target.value))}
                      className="mt-2 w-full accent-primary"
                    />
                    <p className="mt-1 truncate text-[10px] text-muted-foreground">{item.description}</p>
                  </label>
                ))}
              </div>

              <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2 2xl:grid-cols-3">
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>Autonomy Visible</span>
                  <input type="checkbox" checked={controls.autonomyEnabled} onChange={(e) => setControl({ autonomyEnabled: e.target.checked })} />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>Sandbox First</span>
                  <input type="checkbox" checked={controls.sandboxFirst} onChange={(e) => setControl({ sandboxFirst: e.target.checked })} />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>Require Evaluation</span>
                  <input type="checkbox" checked={controls.requireEvaluation} onChange={(e) => setControl({ requireEvaluation: e.target.checked })} />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>Require Approval</span>
                  <input type="checkbox" checked={controls.requireApproval} onChange={(e) => setControl({ requireApproval: e.target.checked })} />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>High Signal Only</span>
                  <input type="checkbox" checked={controls.showOnlyHighSignal} onChange={(e) => setControl({ showOnlyHighSignal: e.target.checked })} />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/50 p-2 text-xs">
                  <span>Poll sec</span>
                  <input
                    type="number"
                    min={3}
                    max={60}
                    value={controls.pollIntervalSec}
                    onChange={(e) => setControl({ pollIntervalSec: Math.max(3, Number(e.target.value || 8)) })}
                    className="w-16 rounded border border-border bg-background px-2 py-1 text-xs"
                  />
                </label>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button variant="default" size="sm" onClick={() => void submitFineTuneControls()}>
                  Save Governance
                </Button>
                <Button variant="outline" size="sm" onClick={() => void checkStatus()}>
                  Sync Now
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom split-screen: live trace and subject review are isolated in scroll panes. */}
      <div className="min-h-0 flex-1 p-4">
        <div className="grid h-full min-h-0 grid-cols-1 gap-4 xl:grid-cols-[1.15fr_0.85fr]">
          <div className="flex min-h-0 flex-col gap-4">
            {/* Thought Stream */}
            <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-border bg-card p-4">
              <div className="flex shrink-0 items-center justify-between gap-3">
                <p className="flex items-center gap-2 text-sm font-medium">
                  <Brain className="h-4 w-4" />
                  Chat Thinking Process / Cognitive Trace
                </p>
                <span className="rounded-full bg-muted px-2 py-1 text-[11px] text-muted-foreground">
                  {thoughts.length} records
                </span>
              </div>

              <ScrollArea className="mt-3 min-h-0 flex-1 pr-3">
                {isLoading && thoughts.length === 0 ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : thoughts.length === 0 ? (
                  <div className="rounded-xl border border-border bg-background/50 py-8 text-center">
                    <Eye className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">No trace available</p>
                  </div>
                ) : (
                  <div className="space-y-2 pb-2">
                    {thoughts.map((trace) => (
                      <div key={trace.id} className={cn("rounded-xl border p-3", thoughtLevelClass(trace.level))}>
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">{trace.title}</p>
                            <p className="mt-1 text-[11px] text-muted-foreground">
                              {trace.source} • {safeDate(trace.ts).toLocaleString()}
                            </p>
                          </div>
                          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{trace.level}</span>
                        </div>
                        <p className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">{trace.content}</p>
                        {trace.tags.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {trace.tags.map((tag) => (
                              <span key={`${trace.id}-${tag}`} className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </div>

            {/* Training Jobs */}
            <div className="flex max-h-[28vh] min-h-[160px] flex-col rounded-xl border border-border bg-card p-4">
              <div className="flex shrink-0 items-center justify-between gap-3">
                <p className="flex items-center gap-2 text-sm font-medium">
                  <TrendingUp className="h-4 w-4" />
                  Training Jobs
                </p>
                <span className="rounded-full bg-muted px-2 py-1 text-[11px] text-muted-foreground">
                  {jobs.length} active
                </span>
              </div>

              <ScrollArea className="mt-3 min-h-0 flex-1 pr-3">
                {isLoading && jobs.length === 0 ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : jobs.length === 0 ? (
                  <div className="rounded-xl border border-border bg-background/50 py-8 text-center">
                    <BarChart3 className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">{isAvailable ? "No active jobs" : "Engine not available"}</p>
                  </div>
                ) : (
                  <div className="space-y-2 pb-2">
                    {jobs.map((job) => (
                      <div key={job.id} className="rounded-xl border border-border bg-background/50 p-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">{job.name}</p>
                            {job.details && <p className="mt-1 truncate text-xs text-muted-foreground">{job.details}</p>}
                          </div>
                          <span className={cn("text-xs font-medium capitalize", getStatusColor(job.status))}>{job.status}</span>
                        </div>
                        <Progress value={job.progress} className="mb-1 h-2" />
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>{job.progress}% complete</span>
                          <span>{job.startedAt ? job.startedAt.toLocaleString() : "Queued"}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </div>
          </div>

          <div className="grid min-h-0 grid-rows-[0.95fr_1.05fr] gap-4">
            {/* Subjects / Concepts */}
            <div className="flex min-h-0 flex-col rounded-xl border border-border bg-card p-4">
              <div className="flex shrink-0 items-center justify-between gap-3">
                <p className="flex items-center gap-2 text-sm font-medium">
                  <Search className="h-4 w-4" />
                  Subject Boxes / New Concepts
                </p>
                <span className="rounded-full bg-muted px-2 py-1 text-[11px] text-muted-foreground">
                  {filteredSubjects.length} queued
                </span>
              </div>

              <ScrollArea className="mt-3 min-h-0 flex-1 pr-3">
                {filteredSubjects.length === 0 ? (
                  <div className="rounded-xl border border-border bg-background/50 py-8 text-center">
                    <Search className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">No concepts queued</p>
                  </div>
                ) : (
                  <div className="space-y-2 pb-2">
                    {filteredSubjects.map((subject) => {
                      const isActive = selectedSubjectId === subject.id;
                      return (
                        <button
                          key={subject.id}
                          onClick={() => setSelectedSubjectId(subject.id)}
                          className={cn(
                            "w-full rounded-xl border p-3 text-left transition-all",
                            "border-border bg-background/50 hover:bg-card/80",
                            isActive && "border-primary/50 bg-primary/5"
                          )}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">{subject.title}</p>
                              <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{subject.summary}</p>
                            </div>
                            <span className={cn("text-[11px] font-medium capitalize", stageBadgeClass(subject.stage))}>{subject.stage}</span>
                          </div>

                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <div>
                              <p className="mb-1 text-[10px] text-muted-foreground">Confidence</p>
                              <Progress value={subject.confidence} className="h-2" />
                            </div>
                            <div>
                              <p className="mb-1 text-[10px] text-muted-foreground">Risk</p>
                              <Progress value={subject.risk} className="h-2" />
                            </div>
                          </div>

                          <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
                            <span>{subject.source}</span>
                            <span>{safeDate(subject.updatedAt).toLocaleString()}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </ScrollArea>
            </div>

            {/* Subject review */}
            <div className="flex min-h-0 flex-col rounded-xl border border-border bg-card p-4">
              <div className="shrink-0">
                <p className="flex items-center gap-2 text-sm font-medium">
                  <ShieldCheck className="h-4 w-4" />
                  Subject Review / Operator Decision
                </p>
              </div>

              <ScrollArea className="mt-3 min-h-0 flex-1 pr-3">
                {!selectedSubject ? (
                  <div className="rounded-xl border border-border bg-background/50 py-8 text-center">
                    <ShieldCheck className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">Select a subject to review</p>
                  </div>
                ) : (
                  <div className="space-y-3 pb-2">
                    <div>
                      <p className="text-sm font-medium">{selectedSubject.title}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{selectedSubject.source}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-lg border border-border bg-background/50 p-3">
                        <p className="text-[11px] text-muted-foreground">Confidence</p>
                        <p className="mt-1 text-lg font-semibold">{selectedSubject.confidence}%</p>
                      </div>
                      <div className="rounded-lg border border-border bg-background/50 p-3">
                        <p className="text-[11px] text-muted-foreground">Risk</p>
                        <p className="mt-1 text-lg font-semibold">{selectedSubject.risk}%</p>
                      </div>
                    </div>

                    <div className="rounded-lg border border-border bg-background/50 p-3">
                      <p className="mb-1 text-[11px] text-muted-foreground">Summary</p>
                      <p className="whitespace-pre-wrap text-sm text-foreground">{selectedSubject.summary}</p>
                    </div>

                    <div className="rounded-lg border border-border bg-background/50 p-3">
                      <p className="mb-2 text-[11px] text-muted-foreground">Review Notes</p>
                      <textarea
                        value={selectedSubject.notes}
                        onChange={(e) => updateSubject(selectedSubject.id, { notes: e.target.value })}
                        rows={5}
                        className="w-full resize-y rounded-md border border-border bg-background px-3 py-2 text-sm"
                        placeholder="Operator notes, tuning feedback, guardrail notes, or sandbox instructions..."
                      />
                    </div>

                    {selectedSubject.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {selectedSubject.tags.map((tag) => (
                          <span key={`${selectedSubject.id}-${tag}`} className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="outline" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "sandbox")}>
                        <FlaskConical className="mr-2 h-4 w-4" />
                        Sandbox
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "testing")}>
                        <PlayCircle className="mr-2 h-4 w-4" />
                        Test
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "evaluation")}>
                        <Activity className="mr-2 h-4 w-4" />
                        Evaluate
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "hold")}>
                        <PauseCircle className="mr-2 h-4 w-4" />
                        Hold
                      </Button>
                      <Button variant="default" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "approved")}>
                        <CheckCircle2 className="mr-2 h-4 w-4" />
                        Approve
                      </Button>
                      <Button variant="destructive" size="sm" onClick={() => void submitSubjectAction(selectedSubject, "rejected")}>
                        <XCircle className="mr-2 h-4 w-4" />
                        Reject
                      </Button>
                    </div>

                    <div className="text-[11px] text-muted-foreground">
                      Current stage:{" "}
                      <span className={cn("font-medium capitalize", stageBadgeClass(selectedSubject.stage))}>{selectedSubject.stage}</span>
                    </div>
                  </div>
                )}
              </ScrollArea>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
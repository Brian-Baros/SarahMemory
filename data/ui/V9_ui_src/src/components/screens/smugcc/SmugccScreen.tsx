import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Boxes,
  CheckCircle2,
  FileCode2,
  GitBranch,
  Loader2,
  RefreshCw,
  Send,
  ShieldCheck,
  Terminal,
  Wand2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { apiFetch } from "@/lib/config";

type SmugccPacket = {
  status?: any;
  schema?: any;
  compatibility?: any;
  receipts?: any;
  passports?: any;
  quarantine?: any;
  tasks?: any;
  pending?: any;
};

type BuilderState = {
  objective: string;
  intent: string;
  provider: string;
  sourceProtocol: string;
  adapterId: string;
  requestedCapabilities: string;
  allowedSources: string;
  allowedMethods: string;
  expectedOutputType: string;
  riskLevel: string;
};

const DEFAULT_BUILDER: BuilderState = {
  objective: "Validate an external adapter contract without execution.",
  intent: "contract_validation",
  provider: "generic",
  sourceProtocol: "generic_rest_tool",
  adapterId: "generic_rest_tool",
  requestedCapabilities: "return_data",
  allowedSources: "",
  allowedMethods: "GET",
  expectedOutputType: "contract",
  riskLevel: "medium",
};

async function safeGet(path: string): Promise<any> {
  try {
    return await apiFetch(path, { method: "GET" });
  } catch (error) {
    return { ok: false, error: String(error), path };
  }
}

async function safePost(path: string, body: any): Promise<any> {
  try {
    return await apiFetch(path, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body || {}) });
  } catch (error) {
    return { ok: false, error: String(error), path };
  }
}

function asArray(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}

function csv(value: string): string[] {
  return value.split(",").map((x) => x.trim()).filter(Boolean);
}

function groupedErrors(validation: any): Record<string, any[]> {
  const direct = validation?.owner_trace?.errors_by_owner || validation?.errors_by_owner;
  if (direct && typeof direct === "object") return direct;
  const grouped: Record<string, any[]> = {};
  for (const err of asArray(validation?.errors)) {
    const owner = String(err?.owner || "SMUGCC");
    grouped[owner] = grouped[owner] || [];
    grouped[owner].push(err);
  }
  return grouped;
}

function dispatchUi(actions: any[]) {
  try {
    window.dispatchEvent(new CustomEvent("sarah:ui", { detail: { source: "SmugccScreen", ts: Date.now(), actions } }));
  } catch {
    // non-fatal UI bridge
  }
}

function openSurface(screen: string, extraActions: any[] = []) {
  dispatchUi([
    { type: "navigate", payload: { screen } },
    { type: "window.open", payload: { id: screen } },
    { type: "window.focus", payload: { id: screen } },
    ...extraActions,
  ]);
}

function StatusTile({ label, value, good = true }: { label: string; value: string | number; good?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-card/70 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={good ? "mt-1 text-sm font-semibold" : "mt-1 text-sm font-semibold text-destructive"}>{value}</div>
    </div>
  );
}

function JsonBlock({ value, height = "max-h-[360px]" }: { value: any; height?: string }) {
  return (
    <pre className={`${height} overflow-auto rounded-md bg-background p-3 text-xs text-muted-foreground`}>
      {JSON.stringify(value || {}, null, 2)}
    </pre>
  );
}

export function SmugccScreen() {
  const [packet, setPacket] = useState<SmugccPacket>({});
  const [builder, setBuilder] = useState<BuilderState>(DEFAULT_BUILDER);
  const [builtEnvelope, setBuiltEnvelope] = useState<any>(null);
  const [validatorText, setValidatorText] = useState("{}");
  const [lastValidation, setLastValidation] = useState<any>(null);
  const [lastTrace, setLastTrace] = useState<any>(null);
  const [lastStage, setLastStage] = useState<any>(null);
  const [naildeResult, setNaildeResult] = useState<any>(null);
  const [adapterDraft, setAdapterDraft] = useState("{}");
  const [adapterValidation, setAdapterValidation] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const refresh = async () => {
    setIsLoading(true);
    setError("");
    try {
      const [status, schema, compatibility, receipts, passports, quarantine, tasks, pending] = await Promise.all([
        safeGet("/api/smugcc/status"),
        safeGet("/api/smugcc/schema"),
        safeGet("/api/smugcc/compatibility"),
        safeGet("/api/smugcc/receipts"),
        safeGet("/api/smugcc/passports"),
        safeGet("/api/smugcc/quarantine"),
        safeGet("/api/tasks"),
        safeGet("/api/actions/pending"),
      ]);
      setPacket({ status, schema, compatibility, receipts, passports, quarantine, tasks, pending });
      if (!status?.ok || !schema?.ok || !compatibility?.ok) setError("One or more SMUGCC bridge endpoints returned a non-OK packet.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const status = packet.status || {};
  const schema = packet.schema || {};
  const compatibility = packet.compatibility || {};
  const doctrine = status.doctrine || {};
  const adapters = asArray(compatibility.known_adapters);
  const lifecycle = asArray(status.lifecycle || compatibility.lifecycle);
  const gatewayNames = ["SarahMemorySMLProtocol", "SarahMemoryAgentFirewall", "SarahMemoryTrustRegistry", "SarahMemorySafetyPolicies", "SarahMemorySecurityGovernor", "SarahMemoryAssuranceGate", "SarahMemoryOperatorCore", "SarahMemoryLedger"];

  const allAdaptersValid = useMemo(() => adapters.every((adapter: any) => adapter?.valid !== false && adapter?.execution_authority === false), [adapters]);

  const buildEnvelope = async () => {
    setBusy("build");
    const result = await safePost("/api/smugcc/build", {
      objective: builder.objective,
      intent: builder.intent,
      provider: builder.provider,
      source_protocol: builder.sourceProtocol,
      adapter_id: builder.adapterId,
      requested_capabilities: csv(builder.requestedCapabilities),
      allowed_sources: csv(builder.allowedSources),
      allowed_methods: csv(builder.allowedMethods),
      expected_output_type: builder.expectedOutputType,
      risk_level: builder.riskLevel,
    });
    setBusy("");
    setBuiltEnvelope(result?.envelope || result);
    setValidatorText(JSON.stringify(result?.envelope || result, null, 2));
    setLastValidation(result?.validation || null);
  };

  const validateEnvelope = async (envelope: any = builtEnvelope) => {
    setBusy("validate");
    let parsed = envelope;
    if (!parsed) {
      try { parsed = JSON.parse(validatorText || "{}"); } catch (err) { parsed = { parse_error: String(err) }; }
    }
    const result = await safePost("/api/smugcc/validate", { envelope: parsed });
    setBusy("");
    setLastValidation(result);
    return result;
  };

  const traceEnvelope = async () => {
    setBusy("trace");
    let parsed = builtEnvelope;
    try { parsed = JSON.parse(validatorText || JSON.stringify(builtEnvelope || {})); } catch {}
    const result = await safePost("/api/smugcc/trace", { envelope: parsed });
    setBusy("");
    setLastTrace(result);
  };

  const stageMission = async () => {
    setBusy("stage");
    let parsed = builtEnvelope;
    try { parsed = JSON.parse(validatorText || JSON.stringify(builtEnvelope || {})); } catch {}
    const result = await safePost("/api/smugcc/mission/stage", { envelope: parsed, mode: "draft" });
    setBusy("");
    setLastStage(result);
    void refresh();
  };

  const sendToChat = () => {
    const draft = `Validate this SMUGCC envelope and show the governed trace:\n\n${validatorText || JSON.stringify(builtEnvelope || {}, null, 2)}`;
    try { window.sessionStorage.setItem("sarah:chat-pending-draft", draft); } catch {}
    openSurface("chat");
    try { window.dispatchEvent(new CustomEvent("sarah:chat-draft", { detail: { draft } })); } catch {}
  };

  const sendToTerminal = () => {
    const command = "/smugcc validate " + (validatorText || JSON.stringify(builtEnvelope || {}));
    openSurface("terminal", [{ type: "terminal_set_input", payload: { value: command } }]);
    window.setTimeout(() => dispatchUi([{ type: "terminal_set_input", payload: { value: command } }]), 150);
  };

  const sendToNailde = async (kind = "Create adapter", createSandbox = false) => {
    setBusy("nailde");
    const goal = `${kind} tool for SMUGCC: ${builder.objective || "SMUGCC adapter mission"}`;
    const result = await safePost("/api/nailde/mission/create", { text: goal, goal, plan_only: !createSandbox, confirmed: createSandbox, source: "smugcc_panel", envelope: builtEnvelope });
    setBusy("");
    setNaildeResult(result);
    openSurface("nailde");
  };

  const validateAdapter = async () => {
    setBusy("adapter");
    let declaration: any = {};
    try { declaration = JSON.parse(adapterDraft || "{}"); } catch (err) { declaration = { parse_error: String(err) }; }
    const result = await safePost("/api/smugcc/adapter/validate", { declaration });
    setBusy("");
    setAdapterValidation(result);
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="shrink-0 border-b border-border bg-card/60 p-4 backdrop-blur">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">SMUGCC Contract Cockpit</h1>
          <Badge variant={status.ok ? "default" : "destructive"} className="ml-auto">{status.ok ? "Contract online" : "Unavailable"}</Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void refresh()} title="Refresh SMUGCC cockpit">
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          </Button>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">Governed contract cockpit. Build, validate, stage, trace, and route SMUGCC envelopes without granting execution authority.</p>
      </div>

      <div className="grid grid-cols-2 gap-2 border-b border-border bg-background/60 p-3 md:grid-cols-4">
        <StatusTile label="Schema" value={String(status.contract_schema || "unknown")} good={status.ok} />
        <StatusTile label="Version" value={String(status.contract_version || "unknown")} good={status.ok} />
        <StatusTile label="Adapters" value={Number(status.adapter_count || adapters.length || 0)} good={allAdaptersValid} />
        <StatusTile label="Execution Authority" value={String(Boolean(doctrine.execution_authority))} good={!doctrine.execution_authority} />
      </div>

      {error ? <div className="flex items-center gap-2 border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive"><AlertCircle className="h-4 w-4" />{error}</div> : null}

      <Tabs defaultValue="overview" className="flex min-h-0 flex-1 flex-col">
        <TabsList className="mx-3 mt-3 grid grid-cols-4 lg:grid-cols-8">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="builder">Builder</TabsTrigger>
          <TabsTrigger value="validator">Validator</TabsTrigger>
          <TabsTrigger value="adapters">Adapters</TabsTrigger>
          <TabsTrigger value="trace">Trace</TabsTrigger>
          <TabsTrigger value="tasks">Tasks</TabsTrigger>
          <TabsTrigger value="quarantine">Security</TabsTrigger>
          <TabsTrigger value="nailde">NAILDE</TabsTrigger>
        </TabsList>

        <ScrollArea className="min-h-0 flex-1">
          <TabsContent value="overview" className="m-0 space-y-3 p-3">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-border bg-card/70 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold"><ShieldCheck className="h-4 w-4 text-primary" />Doctrine</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(doctrine).map(([key, value]) => <div key={key} className="rounded-md border border-border/70 bg-background/60 p-2"><div className="text-muted-foreground">{key.replace(/_/g, " ")}</div><div className="font-medium">{String(value)}</div></div>)}
                </div>
              </div>
              <div className="rounded-lg border border-border bg-card/70 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold"><GitBranch className="h-4 w-4 text-primary" />Gateway Status</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {gatewayNames.map((name) => <div key={name} className="flex items-center gap-2 rounded-md border border-border/70 bg-background/60 p-2"><CheckCircle2 className="h-3.5 w-3.5 text-primary" /><span className="truncate">{name}</span></div>)}
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 text-sm font-semibold">Lifecycle</div><div className="flex flex-wrap gap-2">{lifecycle.map((stage) => <Badge key={String(stage)} variant="secondary">{String(stage)}</Badge>)}</div></div>
          </TabsContent>

          <TabsContent value="builder" className="m-0 space-y-3 p-3">
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(builder).map(([key, value]) => <label key={key} className="text-xs text-muted-foreground">{key.replace(/[A-Z]/g, " $&")}<Input className="mt-1" value={String(value)} onChange={(e) => setBuilder((prev) => ({ ...prev, [key]: e.target.value }))} /></label>)}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void buildEnvelope()} disabled={Boolean(busy)}><Wand2 className="mr-2 h-4 w-4" />Build Envelope</Button>
              <Button variant="outline" onClick={() => void validateEnvelope()} disabled={Boolean(busy)}>Validate Envelope</Button>
              <Button variant="outline" onClick={() => void stageMission()} disabled={Boolean(busy)}>Stage Mission</Button>
              <Button variant="outline" onClick={sendToChat}><Send className="mr-2 h-4 w-4" />Send to Chat</Button>
              <Button variant="outline" onClick={sendToTerminal}><Terminal className="mr-2 h-4 w-4" />Send to Terminal</Button>
              <Button variant="secondary" onClick={() => void sendToNailde("Create adapter")}>Send to NAILDE</Button>
            </div>
            <JsonBlock value={builtEnvelope || { message: "No envelope built yet." }} />
          </TabsContent>

          <TabsContent value="validator" className="m-0 space-y-3 p-3">
            <Textarea value={validatorText} onChange={(e) => setValidatorText(e.target.value)} className="min-h-[240px] font-mono text-xs" />
            <div className="flex flex-wrap gap-2"><Button onClick={() => void validateEnvelope(null)} disabled={Boolean(busy)}>Validate</Button><Button variant="outline" onClick={traceEnvelope} disabled={Boolean(busy)}>Trace</Button><Button variant="outline" onClick={() => setValidatorText(JSON.stringify(builtEnvelope || {}, null, 2))}>Repair Draft</Button></div>
            <div className="grid gap-2 md:grid-cols-2">
              {Object.entries(groupedErrors(lastValidation)).map(([owner, errors]) => <div key={owner} className="rounded-lg border border-border bg-card/70 p-3"><div className="text-sm font-semibold">{owner}</div>{asArray(errors).map((err, idx) => <div key={idx} className="mt-2 rounded border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">{String(err?.code || "error")}: {String(err?.message || err)}</div>)}</div>)}
            </div>
            <JsonBlock value={lastValidation || { message: "Validation not run." }} />
          </TabsContent>

          <TabsContent value="adapters" className="m-0 space-y-3 p-3">
            <div className="grid gap-2 md:grid-cols-2">{adapters.map((adapter: any) => <div key={String(adapter.adapter_id)} className="rounded-lg border border-border bg-card/70 p-3"><div className="flex items-center gap-2"><Boxes className="h-4 w-4 text-primary" /><div className="min-w-0 flex-1"><div className="truncate text-sm font-semibold">{String(adapter.adapter_id)}</div><div className="truncate text-xs text-muted-foreground">{String(adapter.source_protocol || "")}</div></div><Badge variant={adapter.valid === false ? "destructive" : "secondary"}>{adapter.valid === false ? "Invalid" : "Declared"}</Badge></div></div>)}</div>
            <Textarea value={adapterDraft} onChange={(e) => setAdapterDraft(e.target.value)} className="min-h-[140px] font-mono text-xs" />
            <div className="flex flex-wrap gap-2"><Button onClick={validateAdapter} disabled={Boolean(busy)}>Validate Adapter Declaration</Button><Button variant="outline" onClick={() => void sendToNailde("Stage adapter builder")}>Stage Adapter Builder in NAILDE</Button><Button variant="outline" onClick={() => void sendToNailde("Generate adapter skeleton", true)}>Generate Adapter Skeleton</Button></div>
            <JsonBlock value={adapterValidation || { credential_values: "never displayed", evidence_support: "declare hashes/traces only" }} />
          </TabsContent>

          <TabsContent value="trace" className="m-0 space-y-3 p-3"><div className="flex gap-2"><Button onClick={traceEnvelope} disabled={Boolean(busy)}>Refresh Trace</Button><Button variant="outline" onClick={stageMission} disabled={Boolean(busy)}>Stage Mission</Button></div><JsonBlock value={lastTrace || { pipeline: gatewayNames, message: "Trace not run." }} height="max-h-[560px]" /></TabsContent>

          <TabsContent value="tasks" className="m-0 grid gap-3 p-3 md:grid-cols-2"><div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 text-sm font-semibold">Tasks / Pending</div><JsonBlock value={{ tasks: packet.tasks?.tasks || [], pending: packet.pending?.pending_actions || [], lastStage }} /></div><div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 text-sm font-semibold">Ledger Receipts</div><JsonBlock value={packet.receipts || { receipts: [] }} /></div></TabsContent>

          <TabsContent value="quarantine" className="m-0 grid gap-3 p-3 md:grid-cols-2"><div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 text-sm font-semibold">Quarantine</div><JsonBlock value={packet.quarantine || { quarantine: {} }} /></div><div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 text-sm font-semibold">Passports</div><JsonBlock value={packet.passports || { passports: [] }} /></div></TabsContent>

          <TabsContent value="nailde" className="m-0 space-y-3 p-3"><div className="flex flex-wrap gap-2"><Button onClick={() => void sendToNailde("Create adapter")}>Stage Create Adapter</Button><Button variant="outline" onClick={() => void sendToNailde("Create test harness")}>Stage Test Harness</Button><Button variant="outline" onClick={() => void sendToNailde("Create app/tool", true)}>Stage App/Tool</Button><Button variant="secondary" onClick={() => openSurface("nailde")}>Open NAILDE</Button></div><JsonBlock value={naildeResult || { workspace_status: "No NAILDE mission staged from this panel yet.", sandbox_first: true }} /></TabsContent>

          <TabsContent value="schema" className="m-0 p-3"><div className="rounded-lg border border-border bg-card/70 p-3"><div className="mb-2 flex items-center gap-2 text-sm font-semibold"><FileCode2 className="h-4 w-4 text-primary" />Canonical Envelope</div><JsonBlock value={schema.envelope || {}} height="max-h-[560px]" /></div></TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}




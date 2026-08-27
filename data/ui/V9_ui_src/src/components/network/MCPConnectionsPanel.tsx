import { useCallback, useEffect, useMemo, useState } from "react";
import { Cable, RefreshCw, Play, ShieldCheck, AlertTriangle, Wrench, Database, MessageSquareCode } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiFetch } from "@/lib/config";

type RpcItem = { name?: string; uri?: string; description?: string; inputSchema?: unknown };
type Receipt = { ts: string; method: string; ok: boolean; detail: unknown };

const PROFILE_KEY = "sarahmemory:mcp:gateway-profile";

function readProfile() {
  try {
    return JSON.parse(localStorage.getItem(PROFILE_KEY) || "{}") as { name?: string; endpoint?: string };
  } catch {
    return {};
  }
}

function pretty(value: unknown) {
  try { return JSON.stringify(value, null, 2); } catch { return String(value ?? ""); }
}

export function MCPConnectionsPanel() {
  const saved = useMemo(readProfile, []);
  const [name, setName] = useState(saved.name || "Local MCP Gateway");
  const [endpoint, setEndpoint] = useState(saved.endpoint || "/api/mcp/rpc");
  const [interop, setInterop] = useState<any>(null);
  const [policy, setPolicy] = useState<any>(null);
  const [session, setSession] = useState<any>(null);
  const [tools, setTools] = useState<RpcItem[]>([]);
  const [resources, setResources] = useState<RpcItem[]>([]);
  const [prompts, setPrompts] = useState<RpcItem[]>([]);
  const [selectedTool, setSelectedTool] = useState("");
  const [argumentsText, setArgumentsText] = useState("{}");
  const [resourceUri, setResourceUri] = useState("");
  const [promptName, setPromptName] = useState("");
  const [result, setResult] = useState<any>(null);
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const record = useCallback((method: string, ok: boolean, detail: unknown) => {
    setReceipts((items) => [{ ts: new Date().toISOString(), method, ok, detail }, ...items].slice(0, 50));
  }, []);

  const rpc = useCallback(async (method: string, params: Record<string, unknown> = {}) => {
    const packet = await apiFetch<any>(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: crypto.randomUUID(), method, params }),
    });
    if (packet?.error) throw new Error(packet.error?.message || pretty(packet.error));
    return packet?.result ?? packet;
  }, [endpoint]);

  const refreshPolicy = useCallback(async () => {
    const [statusPacket, policyPacket] = await Promise.all([
      apiFetch<any>("/api/net/interop/status").catch((error) => ({ ok: false, error: String(error) })),
      apiFetch<any>("/api/net/interop/policy").catch((error) => ({ ok: false, error: String(error) })),
    ]);
    setInterop(statusPacket);
    setPolicy(policyPacket);
  }, []);

  useEffect(() => { void refreshPolicy(); }, [refreshPolicy]);

  const initialize = async () => {
    setBusy(true); setError("");
    localStorage.setItem(PROFILE_KEY, JSON.stringify({ name, endpoint }));
    try {
      const initialized = await rpc("initialize", {
        protocolVersion: "2025-06-18",
        capabilities: { roots: { listChanged: true }, sampling: {}, elicitation: {} },
        clientInfo: { name: "SarahMemory-AiOS-UI", version: "9.0.0" },
      });
      setSession(initialized); setResult(initialized); record("initialize", true, initialized);
      await rpc("notifications/initialized").catch(() => null);
      const [toolPacket, resourcePacket, promptPacket] = await Promise.all([
        rpc("tools/list").catch((error) => ({ tools: [], error: String(error) })),
        rpc("resources/list").catch((error) => ({ resources: [], error: String(error) })),
        rpc("prompts/list").catch((error) => ({ prompts: [], error: String(error) })),
      ]);
      setTools(Array.isArray(toolPacket?.tools) ? toolPacket.tools : []);
      setResources(Array.isArray(resourcePacket?.resources) ? resourcePacket.resources : []);
      setPrompts(Array.isArray(promptPacket?.prompts) ? promptPacket.prompts : []);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message); setSession(null); record("initialize", false, message);
    } finally { setBusy(false); }
  };

  const invoke = async (method: string, params: Record<string, unknown>) => {
    setBusy(true); setError("");
    try {
      const packet = await rpc(method, params);
      setResult(packet); record(method, true, packet);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message); record(method, false, message);
    } finally { setBusy(false); }
  };

  const passiveIngest = async () => {
    setBusy(true); setError("");
    try {
      const packet = await apiFetch<any>("/api/net/interop/ingest", {
        method: "POST", body: JSON.stringify({ protocol: "mcp", source: name, payload: result }),
      });
      setResult(packet); record("/api/net/interop/ingest", true, packet);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message); record("/api/net/interop/ingest", false, message);
    } finally { setBusy(false); }
  };

  const adapterOnly = String(interop?.mcp || "unknown") === "adapter_only";

  return (
    <div className="space-y-3 p-1">
      <div className="grid gap-2 md:grid-cols-[1fr_1.4fr_auto_auto]">
        <Input value={name} onChange={(event) => setName(event.target.value)} aria-label="MCP profile name" placeholder="Profile name" />
        <Input value={endpoint} onChange={(event) => setEndpoint(event.target.value)} aria-label="MCP gateway endpoint" placeholder="/api/mcp/rpc" />
        <Button variant="outline" onClick={() => void refreshPolicy()}><RefreshCw className="mr-1 h-4 w-4" />Policy</Button>
        <Button onClick={() => void initialize()} disabled={busy}><Cable className="mr-1 h-4 w-4" />{busy ? "Working" : "Connect"}</Button>
      </div>

      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card/70 p-3 text-xs">
        <Badge variant={session ? "default" : "secondary"}>{session ? "Gateway initialized" : "Gateway not initialized"}</Badge>
        <Badge variant={adapterOnly ? "secondary" : "outline"}>Backend MCP: {String(interop?.mcp || "unreported")}</Badge>
        <span>Tools {tools.length}</span><span>Resources {resources.length}</span><span>Prompts {prompts.length}</span>
      </div>

      {adapterOnly && (
        <div className="flex gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500" />
          <span>The current API Bridge advertises passive MCP adapter mode. Direct tool execution remains blocked until a governed local MCP gateway serves the configured JSON-RPC endpoint.</span>
        </div>
      )}
      {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">{error}</div>}

      <Tabs defaultValue="tools">
        <TabsList className="grid h-auto w-full grid-cols-4">
          <TabsTrigger value="tools"><Wrench className="mr-1 h-3.5 w-3.5" />Tools</TabsTrigger>
          <TabsTrigger value="resources"><Database className="mr-1 h-3.5 w-3.5" />Resources</TabsTrigger>
          <TabsTrigger value="prompts"><MessageSquareCode className="mr-1 h-3.5 w-3.5" />Prompts</TabsTrigger>
          <TabsTrigger value="receipts"><ShieldCheck className="mr-1 h-3.5 w-3.5" />Receipts</TabsTrigger>
        </TabsList>
        <TabsContent value="tools" className="space-y-2">
          <select className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm" value={selectedTool} onChange={(event) => setSelectedTool(event.target.value)}>
            <option value="">Select a discovered tool</option>{tools.map((tool) => <option key={tool.name} value={tool.name}>{tool.name}</option>)}
          </select>
          <Textarea value={argumentsText} onChange={(event) => setArgumentsText(event.target.value)} className="min-h-28 font-mono text-xs" aria-label="Tool arguments JSON" />
          <Button disabled={!session || !selectedTool || busy} onClick={() => { try { void invoke("tools/call", { name: selectedTool, arguments: JSON.parse(argumentsText || "{}") }); } catch { setError("Tool arguments must be valid JSON."); } }}><Play className="mr-1 h-4 w-4" />Call governed tool</Button>
        </TabsContent>
        <TabsContent value="resources" className="space-y-2">
          <select className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm" value={resourceUri} onChange={(event) => setResourceUri(event.target.value)}><option value="">Select a resource</option>{resources.map((item) => <option key={item.uri} value={item.uri}>{item.name || item.uri}</option>)}</select>
          <Button disabled={!session || !resourceUri || busy} onClick={() => void invoke("resources/read", { uri: resourceUri })}>Read resource</Button>
        </TabsContent>
        <TabsContent value="prompts" className="space-y-2">
          <select className="h-9 w-full rounded-md border border-border bg-background px-2 text-sm" value={promptName} onChange={(event) => setPromptName(event.target.value)}><option value="">Select a prompt</option>{prompts.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select>
          <Button disabled={!session || !promptName || busy} onClick={() => void invoke("prompts/get", { name: promptName, arguments: {} })}>Get prompt</Button>
        </TabsContent>
        <TabsContent value="receipts"><Textarea readOnly value={pretty(receipts)} className="min-h-52 font-mono text-xs" /></TabsContent>
      </Tabs>

      <div className="space-y-2">
        <div className="flex items-center justify-between"><span className="text-xs font-medium">Last result</span><Button size="sm" variant="outline" disabled={!result} onClick={() => void passiveIngest()}>Submit as passive evidence</Button></div>
        <Textarea readOnly value={pretty(result || { interop, policy })} className="min-h-40 font-mono text-xs" />
      </div>
    </div>
  );
}

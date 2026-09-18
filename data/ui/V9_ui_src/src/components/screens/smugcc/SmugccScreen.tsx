import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  Boxes,
  FileCode2,
  GitBranch,
  Loader2,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiFetch } from "@/lib/config";

type SmugccPacket = {
  status?: any;
  schema?: any;
  compatibility?: any;
};

async function safeGet(path: string): Promise<any> {
  try {
    return await apiFetch(path, { method: "GET" });
  } catch (error) {
    return { ok: false, error: String(error), path };
  }
}

function asArray(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}

function boolLabel(value: unknown) {
  return value ? "Yes" : "No";
}

function StatusTile({ label, value, good = true }: { label: string; value: string | number; good?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-card/70 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={good ? "mt-1 text-sm font-semibold" : "mt-1 text-sm font-semibold text-destructive"}>
        {value}
      </div>
    </div>
  );
}

export function SmugccScreen() {
  const [packet, setPacket] = useState<SmugccPacket>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    setIsLoading(true);
    setError("");
    try {
      const [status, schema, compatibility] = await Promise.all([
        safeGet("/api/smugcc/status"),
        safeGet("/api/smugcc/schema"),
        safeGet("/api/smugcc/compatibility"),
      ]);
      setPacket({ status, schema, compatibility });
      if (!status?.ok || !schema?.ok || !compatibility?.ok) {
        setError("One or more SMUGCC read-only endpoints returned a non-OK packet.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const status = packet.status || {};
  const schema = packet.schema || {};
  const compatibility = packet.compatibility || {};
  const doctrine = status.doctrine || {};
  const adapters = asArray(compatibility.known_adapters);
  const lifecycle = asArray(status.lifecycle || compatibility.lifecycle);
  const owners = Object.entries(status.owners || compatibility.owners || {});

  const allAdaptersValid = useMemo(
    () => adapters.every((adapter: any) => adapter?.valid !== false && adapter?.execution_authority === false),
    [adapters],
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="shrink-0 border-b border-border bg-card/60 p-4 backdrop-blur">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">SMUGCC / Cognitive Contract</h1>
          <Badge variant={status.ok ? "default" : "destructive"} className="ml-auto">
            {status.ok ? "Contract online" : "Unavailable"}
          </Badge>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => void refresh()}
            title="Refresh SMUGCC status"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          </Button>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          Read-only external cognitive ABI visibility. SMUGCC adapts providers to SarahMemory without creating authority.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2 border-b border-border bg-background/60 p-3 md:grid-cols-4">
        <StatusTile label="Schema" value={String(status.contract_schema || "unknown")} good={status.ok} />
        <StatusTile label="Version" value={String(status.contract_version || "unknown")} good={status.ok} />
        <StatusTile label="Adapters" value={Number(status.adapter_count || adapters.length || 0)} good={allAdaptersValid} />
        <StatusTile label="Execution Authority" value={boolLabel(doctrine.execution_authority)} good={!doctrine.execution_authority} />
      </div>

      {error && (
        <div className="flex items-center gap-2 border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      <Tabs defaultValue="overview" className="flex min-h-0 flex-1 flex-col">
        <TabsList className="mx-3 mt-3 grid grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="adapters">Adapters</TabsTrigger>
          <TabsTrigger value="owners">Owners</TabsTrigger>
          <TabsTrigger value="schema">Schema</TabsTrigger>
        </TabsList>

        <ScrollArea className="min-h-0 flex-1">
          <TabsContent value="overview" className="m-0 space-y-3 p-3">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-border bg-card/70 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <ShieldCheck className="h-4 w-4 text-primary" />
                  Doctrine
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(doctrine).map(([key, value]) => (
                    <div key={key} className="rounded-md border border-border/70 bg-background/60 p-2">
                      <div className="text-muted-foreground">{key.replace(/_/g, " ")}</div>
                      <div className={value ? "font-medium" : "font-medium text-primary"}>{String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-border bg-card/70 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <GitBranch className="h-4 w-4 text-primary" />
                  Lifecycle
                </div>
                <div className="flex flex-wrap gap-2">
                  {lifecycle.map((stage) => (
                    <Badge key={String(stage)} variant="secondary">
                      {String(stage)}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="adapters" className="m-0 grid gap-2 p-3 md:grid-cols-2">
            {adapters.map((adapter: any) => (
              <div key={String(adapter.adapter_id)} className="rounded-lg border border-border bg-card/70 p-3">
                <div className="flex items-center gap-2">
                  <Boxes className="h-4 w-4 text-primary" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{String(adapter.adapter_id)}</div>
                    <div className="truncate text-xs text-muted-foreground">{String(adapter.source_protocol || "")}</div>
                  </div>
                  {adapter.valid === false ? (
                    <Badge variant="destructive">Invalid</Badge>
                  ) : (
                    <Badge variant="secondary">Declared</Badge>
                  )}
                </div>
              </div>
            ))}
          </TabsContent>

          <TabsContent value="owners" className="m-0 space-y-2 p-3">
            {owners.map(([key, value]: [string, any]) => (
              <div key={key} className="rounded-lg border border-border bg-card/70 p-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-primary" />
                  <div className="text-sm font-semibold">{String(value?.owner || key)}</div>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {asArray(value?.owns).map((item) => (
                    <Badge key={String(item)} variant="outline">
                      {String(item)}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </TabsContent>

          <TabsContent value="schema" className="m-0 p-3">
            <div className="rounded-lg border border-border bg-card/70 p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                <FileCode2 className="h-4 w-4 text-primary" />
                Canonical Envelope
              </div>
              <pre className="max-h-[420px] overflow-auto rounded-md bg-background p-3 text-xs text-muted-foreground">
                {JSON.stringify(schema.envelope || {}, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

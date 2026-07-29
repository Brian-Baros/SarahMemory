import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  LayoutGrid,
  RefreshCw,
  Shield,
  Package,
  FileJson,
  MonitorCog,
  AlertTriangle,
  CheckCircle2,
  Lock,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { apiFetch } from '@/lib/config';
import { cn } from '@/lib/utils';

interface AddonRegistryItem {
  id: string;
  name: string;
  zone?: string;
  path?: string;
  has_manifest?: boolean;
  has_ui?: boolean;
  version?: string;
  author?: string;
  description?: string;
  permissions?: string[];
  risk_tier?: string;
  trust_status?: string;
  activation_status?: string;
  governance?: Record<string, unknown>;
}

interface AddonRegistryPacket {
  ok?: boolean;
  count?: number;
  addons?: AddonRegistryItem[];
  candidates?: AddonRegistryItem[];
  governance?: Record<string, unknown>;
  fallback?: boolean;
  error?: string;
}

function normalizeItems(packet: AddonRegistryPacket | null): AddonRegistryItem[] {
  if (!packet) return [];
  const raw = Array.isArray(packet.addons) ? packet.addons : Array.isArray(packet.candidates) ? packet.candidates : [];
  return raw.map((item: any) => ({
    id: String(item.id || item.name || 'unknown'),
    name: String(item.name || item.id || 'Unnamed Addon'),
    zone: item.zone,
    path: item.path,
    has_manifest: Boolean(item.has_manifest),
    has_ui: Boolean(item.has_ui),
    version: String(item.version || item.manifest?.version || 'unknown'),
    author: String(item.author || item.manifest?.author || 'unknown'),
    description: String(item.description || item.manifest?.description || 'No description provided.'),
    permissions: Array.isArray(item.permissions) ? item.permissions.map(String) : Array.isArray(item.manifest?.permissions) ? item.manifest.permissions.map(String) : [],
    risk_tier: String(item.risk_tier || item.manifest?.risk_tier || 'UNDECLARED'),
    trust_status: String(item.trust_status || (item.has_manifest ? 'manifest_present' : 'manifest_missing')),
    activation_status: String(item.activation_status || 'review_required'),
    governance: item.governance || {},
  }));
}

async function safeLocalGet(path: string): Promise<any> {
  try {
    return await apiFetch(path, { method: 'GET' });
  } catch (error) {
    return { ok: false, fallback: true, error: String(error), path };
  }
}

export function AddonsScreen() {
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<any>(null);
  const [governance, setGovernance] = useState<any>(null);
  const [registry, setRegistry] = useState<AddonRegistryPacket | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [healthResp, govResp, registryResp] = await Promise.all([
        safeLocalGet('/api/store/health'),
        safeLocalGet('/api/store/governance'),
        safeLocalGet('/api/store/addons/registry'),
      ]);
      setHealth(healthResp || null);
      setGovernance(govResp || null);
      const reg = registryResp && !(registryResp as any).fallback ? registryResp : await safeLocalGet('/api/store/addons/candidates');
      setRegistry((reg || null) as AddonRegistryPacket | null);
    } catch (err) {
      console.warn('[AddonsScreen] refresh failed:', err);
      setHealth({ ok: false, error: String(err) });
      setRegistry(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const addons = useMemo(() => normalizeItems(registry), [registry]);
  const selected = addons.find((a) => a.id === selectedId) || addons[0] || null;
  const approvedCount = addons.filter((a) => a.has_manifest && a.trust_status !== 'manifest_missing').length;
  const quarantineCount = addons.filter((a) => String(a.activation_status || '').includes('quarantine') || !a.has_manifest).length;

  return (
    <div className="flex flex-col h-full bg-background min-h-0">
      <div className="shrink-0 p-4 border-b border-border bg-card/60 backdrop-blur">
        <div className="flex items-center gap-2">
          <LayoutGrid className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Addons & Capability Registry</h1>
          <Badge variant="secondary" className="ml-auto">Read-only review</Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void refresh()} disabled={loading}>
            <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Local addon discovery, manifest visibility, and capability risk review. Launch/activation remains governed.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-3 border-b border-border bg-background/60">
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Package className="h-4 w-4" /> Candidates</div>
          <p className="mt-1 text-sm font-medium">{addons.length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><FileJson className="h-4 w-4" /> Manifests</div>
          <p className="mt-1 text-sm font-medium">{approvedCount}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Shield className="h-4 w-4" /> Quarantine</div>
          <p className="mt-1 text-sm font-medium">{quarantineCount}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Lock className="h-4 w-4" /> Auto-run</div>
          <p className="mt-1 text-sm font-medium">Blocked</p>
        </div>
      </div>

      <Tabs defaultValue="registry" className="flex-1 flex flex-col min-h-0">
        <div className="shrink-0 border-b border-border px-2">
          <TabsList className="w-full h-12 bg-transparent justify-start gap-1">
            <TabsTrigger value="registry" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Package className="h-4 w-4" />
              <span className="text-xs">Registry</span>
            </TabsTrigger>
            <TabsTrigger value="detail" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10" disabled={!selected}>
              <MonitorCog className="h-4 w-4" />
              <span className="text-xs">Manifest</span>
            </TabsTrigger>
            <TabsTrigger value="governance" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Shield className="h-4 w-4" />
              <span className="text-xs">Governance</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          <TabsContent value="registry" className="m-0 p-4">
            {loading ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                <RefreshCw className="h-8 w-8 mx-auto mb-3 animate-spin" />
                Scanning local addon candidates...
              </div>
            ) : addons.length === 0 ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                <LayoutGrid className="h-10 w-10 mx-auto mb-3 opacity-50" />
                No local addon candidates found.
                <p className="text-xs mt-1">Place addons under the configured addons directories and include a manifest.json for review.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {addons.map((addon) => {
                  const active = selected?.id === addon.id;
                  return (
                    <button
                      key={addon.id}
                      className={cn(
                        'text-left rounded-xl border p-3 bg-card hover:bg-accent/50 transition-colors',
                        active ? 'border-primary/60' : 'border-border'
                      )}
                      onClick={() => setSelectedId(addon.id)}
                    >
                      <div className="flex items-start gap-2">
                        <LayoutGrid className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                        <div className="min-w-0 flex-1">
                          <p className="font-medium text-sm truncate">{addon.name}</p>
                          <p className="text-xs text-muted-foreground truncate">{addon.id}</p>
                        </div>
                        <Badge variant={addon.has_manifest ? 'default' : 'secondary'}>
                          {addon.has_manifest ? 'Manifest' : 'No manifest'}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2 line-clamp-2">{addon.description}</p>
                      <div className="flex flex-wrap gap-1 mt-3">
                        <Badge variant="outline">{addon.zone || 'unknown-zone'}</Badge>
                        <Badge variant="outline">risk: {addon.risk_tier}</Badge>
                        <Badge variant="outline">{addon.has_ui ? 'UI' : 'No UI'}</Badge>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </TabsContent>

          <TabsContent value="detail" className="m-0 p-4">
            {!selected ? (
              <div className="text-sm text-muted-foreground">Select an addon to review its manifest.</div>
            ) : (
              <div className="space-y-4">
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="flex items-start gap-3">
                    <FileJson className="h-5 w-5 text-primary mt-0.5" />
                    <div className="min-w-0 flex-1">
                      <h2 className="font-semibold truncate">{selected.name}</h2>
                      <p className="text-xs text-muted-foreground truncate">{selected.path}</p>
                    </div>
                    <Badge variant={selected.has_manifest ? 'default' : 'secondary'}>{selected.trust_status}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mt-3">{selected.description}</p>
                  <div className="grid grid-cols-2 gap-2 mt-4 text-xs">
                    <div><span className="text-muted-foreground">Version:</span> {selected.version}</div>
                    <div><span className="text-muted-foreground">Author:</span> {selected.author}</div>
                    <div><span className="text-muted-foreground">Risk:</span> {selected.risk_tier}</div>
                    <div><span className="text-muted-foreground">Activation:</span> {selected.activation_status}</div>
                  </div>
                </div>

                <div className="rounded-xl border border-border bg-card p-4">
                  <h3 className="text-sm font-medium mb-2">Declared Permissions</h3>
                  {selected.permissions && selected.permissions.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {selected.permissions.map((p) => <Badge key={p} variant="outline">{p}</Badge>)}
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">No permissions declared. Treat as untrusted until manifest is completed.</p>
                  )}
                </div>

                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
                  <div className="flex gap-2">
                    <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0" />
                    <div>
                      <p className="text-sm font-medium">Activation is intentionally disabled from this screen.</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Addons must pass manifest review, TrustRegistry validation, explicit approval, and the governed launcher path before execution.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="governance" className="m-0 p-4">
            <div className="space-y-3">
              <div className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <h3 className="text-sm font-medium">Addon Governance Contract</h3>
                </div>
                <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-5">
                  <li>Candidate scans are read-only.</li>
                  <li>Generated or discovered addons do not auto-run.</li>
                  <li>Sandbox promotion requires explicit approval.</li>
                  <li>Runtime activation must pass registration and trust review.</li>
                  <li>This UI is a registry/review surface, not an execution surface.</li>
                </ul>
              </div>
              <pre className="text-xs bg-muted/60 border border-border rounded-xl p-3 overflow-auto max-h-80">
{JSON.stringify({ health, governance, registry_governance: registry?.governance }, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

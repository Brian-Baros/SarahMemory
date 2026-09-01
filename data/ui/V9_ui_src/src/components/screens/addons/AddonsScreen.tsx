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
  Play,
  Copy,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { apiFetch } from '@/lib/config';
import { cn } from '@/lib/utils';

interface AddonRegistryItem {
  id: string;
  addon_id?: string;
  name: string;
  zone?: string;
  path?: string;
  has_manifest?: boolean;
  has_ui?: boolean;
  has_icon?: boolean;
  icon?: string;
  icon_data_url?: string;
  runtime?: string;
  buttons?: string[];
  version?: string;
  author?: string;
  description?: string;
  permissions?: string[];
  risk_tier?: string;
  trust_status?: string;
  activation_status?: string;
  manifest?: Record<string, unknown>;
  ui?: Record<string, unknown>;
  install_state?: Record<string, unknown>;
  governance?: Record<string, unknown>;
}

interface AddonRegistryPacket {
  ok?: boolean;
  count?: number;
  addons?: AddonRegistryItem[];
  candidates?: AddonRegistryItem[];
  data?: {
    count?: number;
    addons?: AddonRegistryItem[];
    candidates?: AddonRegistryItem[];
    governance?: Record<string, unknown>;
    scan_roots?: Array<Record<string, unknown>>;
  };
  scan_roots?: Array<Record<string, unknown>>;
  governance?: Record<string, unknown>;
  fallback?: boolean;
  error?: string;
}

function dataSource(packet: AddonRegistryPacket | null): AddonRegistryPacket | NonNullable<AddonRegistryPacket['data']> | null {
  if (!packet) return null;
  return packet.data && typeof packet.data === 'object' ? packet.data : packet;
}

function normalizeItems(packet: AddonRegistryPacket | null): AddonRegistryItem[] {
  const source = dataSource(packet);
  if (!source) return [];
  const raw = Array.isArray(source.addons) ? source.addons : Array.isArray(source.candidates) ? source.candidates : [];
  return raw.map((item: any) => {
    const manifest = item.manifest || {};
    const ui = item.ui || {};
    const addonId = String(item.addon_id || item.id || manifest.addon_id || manifest.id || item.name || 'unknown');
    return {
      id: addonId,
      addon_id: addonId,
      name: String(item.name || ui.title || manifest.name || addonId || 'Unnamed Addon'),
      zone: item.zone,
      path: item.path,
      has_manifest: Boolean(item.has_manifest),
      has_ui: Boolean(item.has_ui),
      has_icon: Boolean(item.has_icon || item.icon || item.icon_data_url || ui.icon),
      icon: String(item.icon || ui.icon || ''),
      icon_data_url: String(item.icon_data_url || ''),
      runtime: String(item.runtime || ui.runtime || manifest?.execution?.mode || 'manifest'),
      buttons: Array.isArray(item.buttons) ? item.buttons.map(String) : Array.isArray(ui.buttons) ? ui.buttons.map(String) : ['RUN', 'COPY', 'REMOVE', 'UPDATE'],
      version: String(item.version || manifest.version || 'unknown'),
      author: String(item.author || manifest.author || 'unknown'),
      description: String(item.description || ui.description || manifest.description || 'No description provided.'),
      permissions: Array.isArray(item.permissions) ? item.permissions.map(String) : Array.isArray(manifest.permissions) ? manifest.permissions.map(String) : [],
      risk_tier: String(item.risk_tier || manifest.risk_tier || 'UNDECLARED'),
      trust_status: String(item.trust_status || (item.has_manifest ? 'manifest_present' : 'manifest_missing')),
      activation_status: String(item.activation_status || item.install_state?.activation_status || 'installed_not_running'),
      manifest,
      ui,
      install_state: item.install_state || {},
      governance: item.governance || {},
    };
  });
}

async function safeLocalGet(path: string): Promise<any> {
  try {
    return await apiFetch(path, { method: 'GET' });
  } catch (error) {
    return { ok: false, fallback: true, error: String(error), path };
  }
}

async function safeLocalPost(path: string, body: Record<string, unknown>): Promise<any> {
  try {
    return await apiFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (error) {
    return { ok: false, fallback: true, error: String(error), path };
  }
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value ?? '');
  }
}

function AddonIcon({ addon }: { addon: AddonRegistryItem }) {
  if (addon.icon_data_url) {
    return (
      <div className="grid h-12 w-12 shrink-0 place-items-center overflow-hidden rounded-xl border border-primary/40 bg-background">
        <img src={addon.icon_data_url} alt={`${addon.name} icon`} className="h-10 w-10 object-contain" />
      </div>
    );
  }
  return (
    <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl border border-primary/40 bg-primary/10 text-primary">
      <LayoutGrid className="h-7 w-7" />
    </div>
  );
}

export function AddonsScreen() {
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<any>(null);
  const [powerStoreStatus, setPowerStoreStatus] = useState<any>(null);
  const [governance, setGovernance] = useState<any>(null);
  const [registry, setRegistry] = useState<AddonRegistryPacket | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [actionResult, setActionResult] = useState<any>(null);
  const [runningId, setRunningId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [healthResp, powerStatusResp, govResp, registryResp] = await Promise.all([
        safeLocalGet('/api/store/health'),
        safeLocalGet('/api/store/powerstore/status'),
        safeLocalGet('/api/store/governance'),
        safeLocalGet('/api/store/addons/registry'),
      ]);
      setHealth(healthResp || null);
      setPowerStoreStatus(powerStatusResp || null);
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
  const visibleApps = addons.filter((a) => a.has_manifest);
  const approvedCount = addons.filter((a) => a.has_manifest && a.trust_status !== 'manifest_missing').length;
  const reviewCount = addons.filter((a) => !a.has_manifest || String(a.activation_status || '').includes('review')).length;
  const scanRoots = useMemo(() => {
    const source = dataSource(registry);
    return Array.isArray((source as any)?.scan_roots) ? (source as any).scan_roots : [];
  }, [registry]);

  const runAddon = useCallback(async (addon: AddonRegistryItem) => {
    if (!addon?.id) return;
    const confirmed = window.confirm(`Run ${addon.name}?\n\nThis starts the addon through the governed local Addons launcher path. No auto-run is performed.`);
    if (!confirmed) return;
    setRunningId(addon.id);
    try {
      const result = await safeLocalPost('/api/store/addons/run', {
        addon_id: addon.id,
        confirm: true,
        confirmed: true,
        user_confirmed: true,
        source: 'AddonsScreen.RUN',
      });
      setActionResult(result);
      await refresh();
    } finally {
      setRunningId(null);
    }
  }, [refresh]);

  const lifecycleAction = useCallback(async (addon: AddonRegistryItem, action: 'copy' | 'remove' | 'update') => {
    if (!addon?.id) return;
    const confirmed = window.confirm(`${action.toUpperCase()} ${addon.name}?\n\nThis action is governed, local-only, and requires backend approval.`);
    if (!confirmed) return;
    const body: Record<string, unknown> = {
      addon_id: addon.id,
      confirm: true,
      confirmed: true,
      user_confirmed: true,
      source: `AddonsScreen.${action.toUpperCase()}`,
    };
    if (action === 'update') {
      setActionResult({ ok: false, blocked: true, action, addon_id: addon.id, reason: 'UPDATE requires a validated NAILDE source package path. Use NAILDE Add to Addons or PowerStore install authorize.', execution_authority: false });
      return;
    }
    const result = await safeLocalPost(`/api/store/addons/${action}`, body);
    setActionResult(result);
    await refresh();
  }, [refresh]);

  const exportForPowerStore = useCallback(async (addon: AddonRegistryItem) => {
    if (!addon?.id) return;
    const confirmed = window.confirm(`Prepare ${addon.name} for SarahMemory PowerStore?\n\nThis creates a local signed package and does not upload it yet.`);
    if (!confirmed) return;
    const result = await safeLocalPost('/api/store/powerstore/publish/prepare', {
      addon_id: addon.id,
      distribution: 'private',
      license: 'creator_defined',
      confirm: true,
      confirmed: true,
      user_confirmed: true,
      source: 'AddonsScreen.PowerStorePrepare',
    });
    setActionResult(result);
  }, []);

  const handshakePowerStore = useCallback(async () => {
    const result = await safeLocalGet('/api/store/powerstore/handshake');
    setPowerStoreStatus(result?.data?.store_status || result?.store_status || result);
    setActionResult(result);
  }, []);

  return (
    <div className="flex flex-col h-full bg-background min-h-0">
      <div className="shrink-0 p-4 border-b border-border bg-card/60 backdrop-blur">
        <div className="flex items-center gap-2">
          <LayoutGrid className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Addons & Applications</h1>
          <Badge variant="secondary" className="ml-auto">Runtime icons</Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void refresh()} disabled={loading} title="Refresh Addons">
            <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Installed NAILDE applications are discovered from the Addons folder at runtime. Refresh should show new manifest/ui.json apps without rebuilding the UI.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-3 border-b border-border bg-background/60">
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Package className="h-4 w-4" /> Applications</div>
          <p className="mt-1 text-sm font-medium">{visibleApps.length}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><FileJson className="h-4 w-4" /> Manifests</div>
          <p className="mt-1 text-sm font-medium">{approvedCount}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Shield className="h-4 w-4" /> Review</div>
          <p className="mt-1 text-sm font-medium">{reviewCount}</p>
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
              <span className="text-xs">Applications</span>
            </TabsTrigger>
            <TabsTrigger value="detail" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10" disabled={!selected}>
              <MonitorCog className="h-4 w-4" />
              <span className="text-xs">Manifest</span>
            </TabsTrigger>
            <TabsTrigger value="governance" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Shield className="h-4 w-4" />
              <span className="text-xs">Governance</span>
            </TabsTrigger>
            <TabsTrigger value="powerstore" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <UploadCloud className="h-4 w-4" />
              <span className="text-xs">PowerStore</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          <TabsContent value="registry" className="m-0 p-4">
            {loading ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                <RefreshCw className="h-8 w-8 mx-auto mb-3 animate-spin" />
                Scanning local addon applications...
              </div>
            ) : visibleApps.length === 0 ? (
              <div className="py-10 text-center text-sm text-muted-foreground">
                <LayoutGrid className="h-10 w-10 mx-auto mb-3 opacity-50" />
                No installed addon applications found.
                <p className="text-xs mt-1">Expected generated apps under data/addons with manifest.json and ui.json.</p>
                {scanRoots.length ? <pre className="mx-auto mt-3 max-w-xl rounded border border-border bg-muted/60 p-2 text-left text-[10px]">{pretty(scanRoots)}</pre> : null}
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
                {visibleApps.map((addon) => {
                  const active = selected?.id === addon.id;
                  const canRun = addon.has_manifest && addon.buttons?.some((b) => b.toUpperCase() === 'RUN');
                  return (
                    <div
                      key={addon.id}
                      className={cn('rounded-xl border p-3 bg-card transition-colors', active ? 'border-primary/60' : 'border-border')}
                      onClick={() => setSelectedId(addon.id)}
                    >
                      <div className="flex items-start gap-3">
                        <AddonIcon addon={addon} />
                        <div className="min-w-0 flex-1">
                          <p className="font-semibold text-sm truncate">{addon.name}</p>
                          <p className="text-xs text-muted-foreground truncate">{addon.id}</p>
                          <div className="mt-1 flex flex-wrap gap-1">
                            <Badge variant={addon.has_manifest ? 'default' : 'secondary'}>{addon.has_manifest ? 'Manifest' : 'No manifest'}</Badge>
                            <Badge variant="outline">{addon.runtime || 'runtime'}</Badge>
                          </div>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3 line-clamp-2">{addon.description}</p>
                      <div className="flex flex-wrap gap-1 mt-3">
                        <Badge variant="outline">{addon.zone || 'unknown-zone'}</Badge>
                        <Badge variant="outline">risk: {addon.risk_tier}</Badge>
                        <Badge variant="outline">{addon.has_ui ? 'UI card' : 'No ui.json'}</Badge>
                      </div>
                      <div className="mt-3 grid grid-cols-5 gap-1">
                        <Button size="sm" className="h-8 gap-1 text-xs" onClick={(e) => { e.stopPropagation(); void runAddon(addon); }} disabled={!canRun || runningId === addon.id}>
                          {runningId === addon.id ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} RUN
                        </Button>
                        <Button size="sm" variant="outline" className="h-8 gap-1 text-xs" onClick={(e) => { e.stopPropagation(); void lifecycleAction(addon, 'copy'); }}><Copy className="h-3.5 w-3.5" /> COPY</Button>
                        <Button size="sm" variant="outline" className="h-8 gap-1 text-xs" onClick={(e) => { e.stopPropagation(); void lifecycleAction(addon, 'remove'); }}><Trash2 className="h-3.5 w-3.5" /> REMOVE</Button>
                        <Button size="sm" variant="outline" className="h-8 gap-1 text-xs" onClick={(e) => { e.stopPropagation(); void lifecycleAction(addon, 'update'); }}><RefreshCw className="h-3.5 w-3.5" /> UPDATE</Button>
                        <Button size="sm" variant="outline" className="h-8 gap-1 text-xs" onClick={(e) => { e.stopPropagation(); void exportForPowerStore(addon); }}><UploadCloud className="h-3.5 w-3.5" /> STORE</Button>
                      </div>
                    </div>
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
                    <AddonIcon addon={selected} />
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
                    <div><span className="text-muted-foreground">Runtime:</span> {selected.runtime}</div>
                    <div><span className="text-muted-foreground">Icon:</span> {selected.has_icon ? 'present' : 'generic fallback'}</div>
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

                <div className="rounded-xl border border-border bg-card p-4">
                  <h3 className="text-sm font-medium mb-2">Runtime Action Result</h3>
                  <pre className="max-h-60 overflow-auto rounded border border-border bg-muted/60 p-2 text-xs">{pretty(actionResult || { status: 'No addon action run yet.' })}</pre>
                </div>
              </div>
            )}
          </TabsContent>


          <TabsContent value="powerstore" className="m-0 p-4">
            <div className="space-y-4">
              <div className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-center gap-2">
                  <UploadCloud className="h-5 w-5 text-primary" />
                  <h2 className="font-semibold">SarahMemory PowerStore Gateway</h2>
                  <Button size="sm" variant="outline" className="ml-auto h-8 text-xs" onClick={() => void handshakePowerStore()}>Handshake</Button>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Local Addons and NAILDE work without the PowerStore connection. This panel can detect whether store.sarahmemory.com is UP or DOWN, but local creation/install/run remains available either way. Upload is not automatic. Downloads must be staged, verified, scanned, validated, and approved before install.
                </p>
                <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
                  <div className="rounded border border-border p-2"><b>Local-only</b><br />Install privately into data/addons.</div>
                  <div className="rounded border border-border p-2"><b>Publish-ready</b><br />Create package, hash manifest, and creator signature.</div>
                  <div className="rounded border border-border p-2"><b>Marketplace import</b><br />Stage, verify, malware scan, NAILDE validate, approve.</div>
                </div>
              </div>
              {selected ? (
                <div className="rounded-xl border border-border bg-card p-4">
                  <h3 className="font-semibold text-sm">Selected application</h3>
                  <p className="mt-1 text-xs text-muted-foreground">{selected.name} · {selected.id}</p>
                  <Button size="sm" className="mt-3 gap-1 text-xs" onClick={() => void exportForPowerStore(selected)}>
                    <UploadCloud className="h-3.5 w-3.5" /> Prepare Signed PowerStore Package
                  </Button>
                </div>
              ) : null}
              <pre className="max-h-[420px] overflow-auto rounded-xl border border-border bg-muted/60 p-3 text-[11px]">{pretty(actionResult || { powerstore_status: powerStoreStatus || { status: 'No PowerStore action run yet.' } })}</pre>
            </div>
          </TabsContent>

          <TabsContent value="governance" className="m-0 p-4">
            <div className="space-y-3">
              <div className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <h3 className="text-sm font-medium">Addon Runtime Governance Contract</h3>
                </div>
                <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-5">
                  <li>Registry scan is read-only and safe for refresh.</li>
                  <li>Installed addon icons come from manifest.json / ui.json metadata.</li>
                  <li>Small local SVG/PNG icons are returned as safe data URLs.</li>
                  <li>No generated addon auto-runs because it exists.</li>
                  <li>RUN requires an explicit click and backend local-only launcher route.</li>
                  <li>UI source rebuild is not required for new addon cards.</li>
                </ul>
              </div>
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
                <div className="flex gap-2">
                  <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0" />
                  <div>
                    <p className="text-sm font-medium">Auto-run remains blocked.</p>
                    <p className="text-xs text-muted-foreground mt-1">This screen displays and launches installed addons only after user interaction. Shell execution and live core mutation remain outside this UI.</p>
                  </div>
                </div>
              </div>
              <pre className="text-xs bg-muted/60 border border-border rounded-xl p-3 overflow-auto max-h-80">
{pretty({ health, governance, registry_governance: dataSource(registry)?.governance, scan_roots: scanRoots })}
              </pre>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

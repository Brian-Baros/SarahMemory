import { useEffect, useMemo, useState } from 'react';
import {
  Network,
  Users,
  MessageSquare,
  Phone,
  FileText,
  Loader2,
  AlertCircle,
  WifiOff,
  RefreshCw,
  Shield,
  Server,
  Activity,
  Cable,
  Boxes,
  Cpu,
  Gauge,
  Orbit,
  Database,
  ShieldCheck,
  Box,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { apiFetch } from '@/lib/config';
import { useSarahStore } from '@/stores/useSarahStore';
import { useNavigationStore } from '@/stores/useNavigationStore';
import { cn } from '@/lib/utils';

import { DialerPanel } from '@/components/panels/dialer/DialerPanel';
import { ContactsPanel } from '@/components/panels/contacts/ContactsPanel';
import { MCPConnectionsPanel } from '@/components/network/MCPConnectionsPanel';

interface NodeInfo {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'busy';
  lastSeen?: Date;
  meta?: Record<string, unknown>;
}

interface SarahNetStatus {
  ui?: any;
  netHealth?: any;
  net2Health?: any;
  governance?: any;
  interop?: any;
  diagnostics?: any;
  nodeList?: any;
  fabric?: any;
  realtime?: any;
  gcaios?: any;
}

function readBool(value: unknown): boolean {
  return value === true || value === 'true' || value === 1;
}

function isFallback(value: any): boolean {
  return Boolean(value?.fallback || value?.error || value?.ok === false);
}

function asArray(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}

function nodeFromBackend(row: any): NodeInfo | null {
  const id = String(row?.node_id || row?.id || row?.name || '').trim();
  if (!id) return null;
  const lastRaw = row?.last_ts || row?.lastSeen || row?.updated_ts || row?.ts;
  let lastSeen: Date | undefined;
  if (typeof lastRaw === 'number') lastSeen = new Date(lastRaw * 1000);
  if (typeof lastRaw === 'string') {
    const d = new Date(lastRaw);
    if (!Number.isNaN(d.getTime())) lastSeen = d;
  }
  return {
    id,
    name: String(row?.meta?.name || row?.display_name || row?.name || id),
    status: readBool(row?.online) ? 'online' : 'offline',
    lastSeen,
    meta: row?.meta || {},
  };
}

async function safeLocalGet(path: string): Promise<any> {
  try {
    return await apiFetch(path, { method: 'GET' });
  } catch (error) {
    return { ok: false, fallback: true, error: String(error), path };
  }
}

/**
 * SarahNet Screen - Local-first one-way broker status and communications surface.
 *
 * This screen never calls the public SarahMemory cloud endpoint directly. All
 * network status is resolved through the local backend, which preserves the
 * one-way broker doctrine and lets SarahMemory decide whether cloud, LAN, or
 * offline behavior is permitted.
 */
export function SarahNetScreen() {
  const { setCurrentScreen } = useNavigationStore();
  const { contacts, settings } = useSarahStore();
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [statusPacket, setStatusPacket] = useState<SarahNetStatus>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const localOnlyMode = Boolean((settings as any)?.localOnlyMode);

  const checkAvailability = async () => {
    setIsLoading(true);
    try {
      const [ui, netHealth, net2Health, governance, interop, diagnostics, nodeList, fabric, realtime, gcaios] = await Promise.all([
        safeLocalGet('/api/net/ui/status'),
        safeLocalGet('/api/net/health'),
        safeLocalGet('/api/net2/health'),
        safeLocalGet('/api/net/governance'),
        safeLocalGet('/api/net/interop/status'),
        safeLocalGet('/api/net/diagnostics'),
        safeLocalGet('/api/net2/node/list'),
        safeLocalGet('/api/net2/fabric/status'),
        safeLocalGet('/api/net/rt/status'),
        safeLocalGet('/api/self/gcaios/status?workload=sarahnet_xr'),
      ]);

      const packet: SarahNetStatus = { ui, netHealth, net2Health, governance, interop, diagnostics, nodeList, fabric, realtime, gcaios };
      setStatusPacket(packet);

      const localOk = !isFallback(ui) || !isFallback(netHealth) || !isFallback(net2Health);
      const storageReady = readBool((ui as any)?.storage_ready) || readBool((netHealth as any)?.enabled) || readBool((net2Health as any)?.enabled);
      setIsAvailable(Boolean(localOk && (storageReady || !localOnlyMode)));

      const backendNodes = asArray((nodeList as any)?.nodes || (nodeList as any)?.data?.nodes)
        .map(nodeFromBackend)
        .filter(Boolean) as NodeInfo[];
      setNodes(backendNodes);
    } catch (error) {
      console.warn('[SarahNet] Local broker status check failed:', error);
      setStatusPacket({});
      setIsAvailable(false);
      setNodes([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void checkAvailability();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions) || actions.length === 0) return;
      for (const a of actions) {
        if (!a || !a.type) continue;
        try {
          if (a.type === 'navigate' || a.type === 'set_screen') {
            const screen = a.payload?.screen || a.payload?.route;
            if (typeof screen === 'string' && screen) {
              const s = screen.replace(/^\//, '');
              if (s) setCurrentScreen(s as any);
            }
          }
          if (a.type === 'sarahnet_refresh') void checkAvailability();
        } catch (e) {
          console.warn('[SarahNetScreen] UI action failed:', a, e);
        }
      }
    };
    window.addEventListener('sarah:ui', handler as any);
    return () => window.removeEventListener('sarah:ui', handler as any);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const displayNodes: NodeInfo[] = useMemo(() => {
    if (nodes.length > 0) return nodes;
    return contacts.map((c) => ({
      id: c.id,
      name: c.name,
      status: c.status === 'online' ? 'online' : 'offline',
    }));
  }, [nodes, contacts]);

  const brokerMode = String((statusPacket.ui as any)?.broker?.mode || (statusPacket.interop as any)?.broker_mode || 'ONE_WAY_STORE_AND_FORWARD');
  const storageReady = readBool((statusPacket.ui as any)?.storage_ready) || readBool((statusPacket.netHealth as any)?.enabled);
  const directExec = readBool((statusPacket.ui as any)?.broker?.executes_remote_commands) || readBool((statusPacket.governance as any)?.governance?.executes_remote_commands);
  const tableOk = readBool((statusPacket.ui as any)?.table_status?.ok);
  const fabric = ((statusPacket.fabric as any)?.data || statusPacket.fabric || {}) as any;
  const fabricCounts = (fabric?.counts || {}) as any;
  const worlds = asArray(fabric?.worlds);
  const regions = asArray(fabric?.regions);
  const entities = asArray(fabric?.entities);
  const realtime = ((statusPacket.realtime as any)?.data || statusPacket.realtime || {}) as any;
  const gcaios = ((statusPacket.gcaios as any)?.data || statusPacket.gcaios || {}) as any;
  const computePassport = (gcaios?.compute?.passport || {}) as any;
  const computePlan = (gcaios?.compute?.plan || {}) as any;
  const readiness = (gcaios?.readiness || {}) as any;
  const energetics = (gcaios?.energetics || {}) as any;
  const gcop = (gcaios?.gcop || {}) as any;
  const doctrine = (fabric?.doctrine || {}) as any;
  const readinessCount = Object.values(readiness).filter(Boolean).length;
  const readinessTotal = Object.keys(readiness).length;

  if (isLoading) {
    return (
      <div className="flex flex-col h-full bg-background items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground mt-2">Checking local SarahNet broker...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-background min-h-0">
      <div className="shrink-0 p-4 border-b border-border bg-card/60 backdrop-blur">
        <div className="flex items-center gap-2">
          <Orbit className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">SarahNet Reality Fabric</h1>
          <Badge variant={isAvailable ? 'default' : 'secondary'} className="ml-auto">
            {isAvailable ? 'Local broker reachable' : 'Local/offline mode'}
          </Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void checkAvailability()} title="Refresh SarahNet status">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Full SML semantic control, bounded SML-RT presence, and subjective client rendering. Cloud is never contacted directly from this UI.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-2 p-3 border-b border-border bg-background/60">
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Orbit className="h-4 w-4 text-primary" /> Worlds</div>
          <p className="mt-1 text-sm font-semibold">{Number(fabricCounts.worlds || worlds.length || 0)}</p>
        </div>
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Box className="h-4 w-4 text-primary" /> Entities</div>
          <p className="mt-1 text-sm font-semibold">{Number(fabricCounts.entities || entities.length || 0)}</p>
        </div>
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Gauge className="h-4 w-4 text-primary" /> SML-RT</div>
          <p className="mt-1 text-sm font-semibold">{Number(realtime?.records || 0)} live</p>
        </div>
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Cpu className="h-4 w-4 text-primary" /> Compute</div>
          <p className="mt-1 text-sm font-semibold truncate">{String(computePlan?.mode || 'SAFE_DISCOVERY')}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Shield className="h-4 w-4" /> Broker</div>
          <p className="mt-1 text-sm font-medium truncate">{brokerMode}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Server className="h-4 w-4" /> Storage</div>
          <p className="mt-1 text-sm font-medium">{storageReady ? 'Ready' : 'Not mounted'}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Activity className="h-4 w-4" /> Tables</div>
          <p className="mt-1 text-sm font-medium">{tableOk ? 'Schema OK' : 'Check diagnostics'}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/70 p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground"><WifiOff className="h-4 w-4" /> Remote Exec</div>
          <p className="mt-1 text-sm font-medium">{directExec ? 'Unsafe' : 'Blocked'}</p>
        </div>
      </div>

      {isAvailable === false && (
        <div className="m-4 p-4 rounded-xl bg-muted/50 border border-border">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium">SarahNet is local-first and currently offline/degraded.</p>
              <p className="text-xs text-muted-foreground mt-1">
                Communications panels remain available, but brokered network actions require local backend support. No cloud status endpoint was called.
              </p>
            </div>
          </div>
        </div>
      )}

      <Tabs defaultValue="fabric" className="flex-1 flex flex-col min-h-0">
        <div className="shrink-0 border-b border-border px-2">
          <TabsList className="w-full h-12 bg-transparent justify-start gap-1 overflow-x-auto">
            <TabsTrigger value="fabric" className="min-w-[92px] flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Network className="h-4 w-4" />
              <span className="text-xs">Fabric</span>
            </TabsTrigger>
            <TabsTrigger value="worlds" className="min-w-[92px] flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Boxes className="h-4 w-4" />
              <span className="text-xs">Worlds</span>
            </TabsTrigger>
            <TabsTrigger value="nodes" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Users className="h-4 w-4" />
              <span className="text-xs">Nodes</span>
            </TabsTrigger>
            <TabsTrigger value="messages" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <MessageSquare className="h-4 w-4" />
              <span className="text-xs">Messages</span>
            </TabsTrigger>
            <TabsTrigger value="calls" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Phone className="h-4 w-4" />
              <span className="text-xs">Calls</span>
            </TabsTrigger>
            <TabsTrigger value="files" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <FileText className="h-4 w-4" />
              <span className="text-xs">Files</span>
            </TabsTrigger>
            <TabsTrigger value="mcp" className="flex-1 gap-1.5 data-[state=active]:bg-primary/10">
              <Cable className="h-4 w-4" />
              <span className="text-xs">MCP</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          <TabsContent value="fabric" className="m-0 p-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div className="rounded-2xl border border-primary/25 bg-gradient-to-br from-primary/10 via-card to-card p-4">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-primary" />
                  <h2 className="text-sm font-semibold">Governed Cognitive Operating Plane</h2>
                  <Badge variant={readinessCount === readinessTotal && readinessTotal > 0 ? 'default' : 'secondary'} className="ml-auto">
                    {readinessCount}/{readinessTotal || 8} ready
                  </Badge>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Intent is semantic until OperatorCore. Models advise; SML, policy, assurance, security, Energetics, and human authority govern transitions.
                </p>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(readiness).map(([name, ready]) => (
                    <div key={name} className="flex items-center justify-between rounded-lg border border-border bg-background/50 px-2 py-1.5">
                      <span className="truncate mr-2">{name.replaceAll('_', ' ')}</span>
                      <span className={ready ? 'text-emerald-500' : 'text-amber-500'}>{ready ? 'ready' : 'defer'}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-card p-4">
                <div className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-primary" />
                  <h2 className="text-sm font-semibold">Adaptive Compute Passport</h2>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                  <div><p className="text-muted-foreground">Machine class</p><p className="font-medium mt-1">{String(computePassport?.machine_class || 'unknown')}</p></div>
                  <div><p className="text-muted-foreground">Confidence</p><p className="font-medium mt-1">{Math.round(Number(computePassport?.confidence || 0) * 100)}%</p></div>
                  <div><p className="text-muted-foreground">Workload plan</p><p className="font-medium mt-1">{String(computePlan?.mode || 'SAFE_DISCOVERY')}</p></div>
                  <div><p className="text-muted-foreground">SML-RT target</p><p className="font-medium mt-1">{Number(computePlan?.workloads?.sarahnet_rt?.target_state_hz || 0)} Hz</p></div>
                </div>
                {readBool(computePassport?.unknown_machine) && (
                  <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-300">
                    Unknown-machine safe discovery is active. Heavy and embodied workloads remain reduced.
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-border bg-card p-4">
                <div className="flex items-center gap-2"><Gauge className="h-5 w-5 text-primary" /><h2 className="text-sm font-semibold">Continuity & Energetics</h2></div>
                <div className="mt-3 space-y-2 text-xs">
                  <div className="flex items-center justify-between rounded-lg bg-muted/40 px-3 py-2"><span>GCOP continuity</span><Badge variant={gcop?.available ? 'default' : 'secondary'}>{gcop?.available ? 'available' : 'deferred'}</Badge></div>
                  <div className="flex items-center justify-between rounded-lg bg-muted/40 px-3 py-2"><span>Energetics governor</span><Badge variant={energetics?.available ? 'default' : 'secondary'}>{energetics?.available ? 'online' : 'lockout'}</Badge></div>
                  <div className="flex items-center justify-between rounded-lg bg-muted/40 px-3 py-2"><span>Operator execution from this screen</span><Badge variant="secondary">none</Badge></div>
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-card p-4">
                <div className="flex items-center gap-2"><Database className="h-5 w-5 text-primary" /><h2 className="text-sm font-semibold">Reality Fabric Doctrine</h2></div>
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg bg-muted/40 p-2">Full SML for consequential state: <strong>{doctrine.full_sml_for_consequential_state ? 'yes' : 'unknown'}</strong></div>
                  <div className="rounded-lg bg-muted/40 p-2">SML-RT latest-wins state: <strong>{doctrine.sml_rt_for_bounded_ephemeral_state ? 'bounded' : 'unknown'}</strong></div>
                  <div className="rounded-lg bg-muted/40 p-2">Rendering is subjective: <strong>{doctrine.rendering_is_subjective ? 'yes' : 'unknown'}</strong></div>
                  <div className="rounded-lg bg-muted/40 p-2">Ledger per frame: <strong>{doctrine.ledger_per_frame === false ? 'no' : 'unknown'}</strong></div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="worlds" className="m-0 p-4">
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2"><h2 className="text-sm font-semibold">Semantic worlds</h2><Badge variant="outline">{worlds.length} visible</Badge></div>
                {worlds.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-border p-8 text-center"><Orbit className="h-10 w-10 mx-auto text-muted-foreground/40" /><p className="mt-2 text-sm text-muted-foreground">No governed worlds registered yet.</p></div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
                    {worlds.map((world: any) => (
                      <div key={String(world.world_id)} className="rounded-xl border border-border bg-card p-3">
                        <div className="flex items-center gap-2"><Orbit className="h-4 w-4 text-primary" /><p className="font-medium text-sm truncate">{String(world.name || world.world_id)}</p><Badge variant="outline" className="ml-auto text-[10px]">{String(world.status || 'unknown')}</Badge></div>
                        <p className="mt-2 text-xs text-muted-foreground truncate">{String(world.world_id)}</p>
                        <p className="mt-1 text-xs">{String(world.reality_class || 'UNKNOWN').replaceAll('_', ' ')}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div>
                  <div className="flex items-center justify-between mb-2"><h2 className="text-sm font-semibold">Authority regions</h2><Badge variant="outline">{regions.length}</Badge></div>
                  <div className="space-y-2">
                    {regions.slice(0, 30).map((region: any) => (
                      <div key={`${region.world_id}:${region.region_id}`} className="rounded-xl border border-border bg-card p-3 text-xs">
                        <div className="flex items-center justify-between"><span className="font-medium">{String(region.region_id)}</span><Badge variant="secondary">v{Number(region.version || 1)}</Badge></div>
                        <p className="mt-1 text-muted-foreground truncate">Authority: {String(region.authority_node || 'unassigned')}</p>
                      </div>
                    ))}
                    {regions.length === 0 && <p className="text-xs text-muted-foreground">No region authorities registered.</p>}
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2"><h2 className="text-sm font-semibold">Persistent entities</h2><Badge variant="outline">{entities.length}</Badge></div>
                  <div className="space-y-2">
                    {entities.slice(0, 30).map((entity: any) => (
                      <div key={String(entity.entity_id)} className="rounded-xl border border-border bg-card p-3 text-xs">
                        <div className="flex items-center justify-between"><span className="font-medium truncate">{String(entity.semantic_type || entity.entity_type)}</span><Badge variant="secondary">v{Number(entity.state_version || 1)}</Badge></div>
                        <p className="mt-1 text-muted-foreground truncate">{String(entity.entity_id)} · {String(entity.persistence_class || 'PERSISTENT')}</p>
                      </div>
                    ))}
                    {entities.length === 0 && <p className="text-xs text-muted-foreground">No persistent entities registered.</p>}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="nodes" className="m-0 p-4">
            <div className="space-y-2">
              {displayNodes.length === 0 ? (
                <div className="text-center py-8">
                  <Users className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
                  <p className="text-sm text-muted-foreground">No local nodes found</p>
                  <p className="text-xs text-muted-foreground mt-1">Register a trusted node through /api/net2/node/register.</p>
                </div>
              ) : (
                displayNodes.map((node) => (
                  <div key={node.id} className="p-3 rounded-xl bg-card border border-border flex items-center gap-3">
                    <div
                      className={cn(
                        'w-2.5 h-2.5 rounded-full',
                        node.status === 'online' && 'bg-green-500',
                        node.status === 'busy' && 'bg-yellow-500',
                        node.status === 'offline' && 'bg-muted-foreground'
                      )}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm truncate">{node.name}</p>
                      <p className="text-xs text-muted-foreground capitalize">
                        {node.status}{node.lastSeen ? ` • ${node.lastSeen.toLocaleTimeString()}` : ''}
                      </p>
                    </div>
                    <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!isAvailable} title="Brokered message only">
                      <MessageSquare className="h-4 w-4" />
                    </Button>
                  </div>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="messages" className="m-0 p-4">
            <ContactsPanel />
          </TabsContent>

          <TabsContent value="calls" className="m-0 p-4">
            <DialerPanel />
          </TabsContent>

          <TabsContent value="files" className="m-0 p-4">
            <div className="text-center py-8">
              <FileText className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm font-medium">Brokered file transfer</p>
              <p className="text-xs text-muted-foreground mt-1 max-w-md mx-auto">
                File transfer routes are available through /api/net/file/* and remain store-and-forward only. Receiving nodes must explicitly accept, verify CRC/SHA, and decide locally.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="mcp" className="m-0 p-4">
            <MCPConnectionsPanel />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

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
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { apiFetch } from '@/lib/config';
import { useSarahStore } from '@/stores/useSarahStore';
import { useNavigationStore } from '@/stores/useNavigationStore';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';

import { DialerPanel } from '@/components/panels/DialerPanel';
import { ContactsPanel } from '@/components/panels/ContactsPanel';

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
      const [ui, netHealth, net2Health, governance, interop, diagnostics, nodeList] = await Promise.all([
        safeLocalGet('/api/net/ui/status'),
        safeLocalGet('/api/net/health'),
        safeLocalGet('/api/net2/health'),
        safeLocalGet('/api/net/governance'),
        safeLocalGet('/api/net/interop/status'),
        safeLocalGet('/api/net/diagnostics'),
        safeLocalGet('/api/net2/node/list'),
      ]);

      const packet: SarahNetStatus = { ui, netHealth, net2Health, governance, interop, diagnostics, nodeList };
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
          <Network className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">SarahNet</h1>
          <Badge variant={isAvailable ? 'default' : 'secondary'} className="ml-auto">
            {isAvailable ? 'Local broker reachable' : 'Local/offline mode'}
          </Badge>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => void checkAvailability()} title="Refresh SarahNet status">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Local-first store-and-forward broker. Cloud is not contacted directly from the UI.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-3 border-b border-border bg-background/60">
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

      <Tabs defaultValue="nodes" className="flex-1 flex flex-col min-h-0">
        <div className="shrink-0 border-b border-border px-2">
          <TabsList className="w-full h-12 bg-transparent justify-start gap-1">
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
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
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
        </ScrollArea>
      </Tabs>
    </div>
  );
}

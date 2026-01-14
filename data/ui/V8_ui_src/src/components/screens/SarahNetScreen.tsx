import { useState, useEffect } from 'react';
import {
  Network,
  Users,
  MessageSquare,
  Phone,
  FileText,
  Loader2,
  AlertCircle,
  WifiOff,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useSarahStore } from '@/stores/useSarahStore';
import { cn } from '@/lib/utils';

// ✅ Add these imports
import { DialerPanel } from '@/components/panels/DialerPanel';
import { ContactsPanel } from '@/components/panels/ContactsPanel';

interface NodeInfo {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'busy';
  lastSeen?: Date;
}

/**
 * SarahNet Screen - Network presence and communication
 * Shows node status, messaging, calls, and file transfer
 */
export function SarahNetScreen() {
  const { contacts } = useSarahStore();
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    checkAvailability();
  }, []);

  const checkAvailability = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('https://api.sarahmemory.com/api/sarahnet/status', {
        method: 'GET',
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        setIsAvailable(true);
        if (data.nodes) setNodes(data.nodes);
      } else {
        setIsAvailable(false);
      }
    } catch (error) {
      console.warn('[SarahNet] Not available:', error);
      setIsAvailable(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Convert contacts to nodes for display
  const displayNodes: NodeInfo[] =
    nodes.length > 0
      ? nodes
      : contacts.map((c) => ({
          id: c.id,
          name: c.name,
          status: c.status === 'online' ? 'online' : 'offline',
        }));

  if (isLoading) {
    return (
      <div className="flex flex-col h-full bg-background items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground mt-2">Connecting to SarahNet...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">SarahNet</h1>
          {isAvailable === false && (
            <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
              <WifiOff className="h-3.5 w-3.5" />
              Offline
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1">Network presence and communication</p>
      </div>

      {/* Not available notice */}
      {isAvailable === false && (
        <div className="m-4 p-4 rounded-xl bg-muted/50 border border-border">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium">Not available (server configuration)</p>
              <p className="text-xs text-muted-foreground mt-1">
                SarahNet features require server-side support. Local tools still work below.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
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
                  <p className="text-sm text-muted-foreground">No nodes found</p>
                </div>
              ) : (
                displayNodes.map((node) => (
                  <div
                    key={node.id}
                    className="p-3 rounded-xl bg-card border border-border flex items-center gap-3"
                  >
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
                      <p className="text-xs text-muted-foreground capitalize">{node.status}</p>
                    </div>
                    <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!isAvailable}>
                      <MessageSquare className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!isAvailable}>
                      <Phone className="h-4 w-4" />
                    </Button>
                  </div>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="messages" className="m-0 p-4">
            <div className="text-center py-8">
              <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                {isAvailable ? 'No messages' : 'Not available (server configuration)'}
              </p>
            </div>
          </TabsContent>

          {/* ✅ Calls tab now contains the dialer + contacts (local tools) */}
          <TabsContent value="calls" className="m-0 p-0">
            <div className="border-b border-border">
              <DialerPanel />
            </div>
            <div className="p-0">
              <ContactsPanel />
            </div>

            {/* Optional hint when server is offline */}
            {isAvailable === false && (
              <div className="px-4 pb-4">
                <p className="text-xs text-muted-foreground">
                  SarahNet calling (P2P/VoIP) requires backend support. The dialer UI is available for testing.
                </p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="files" className="m-0 p-4">
            <div className="text-center py-8">
              <FileText className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                {isAvailable ? 'No shared files' : 'Not available (server configuration)'}
              </p>
            </div>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

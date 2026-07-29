import { useState } from 'react';
import { MessageSquare, Users, Phone, PhoneOff } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ContactsPanel } from '@/components/panels/ContactsPanel';
import { DialerPanel } from '@/components/panels/DialerPanel';
import { useSarahStore } from '@/stores/useSarahStore';
import { usePreviewStore } from '@/stores/usePreviewStore';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { toast } from 'sonner';

type CommTab = 'messaging' | 'contacts' | 'dialer';

export function CommunicationModule() {
  const [activeTab, setActiveTab] = useState<CommTab>('messaging');
  const [messageText, setMessageText] = useState('');
  
  const addMessage = useSarahStore((s) => s.addMessage);
  const { current, showCall, endCall } = usePreviewStore();
  const isInCall = current.type === 'call';

  const handleSendMessage = () => {
    if (!messageText.trim()) {
      toast.error('Please enter a message');
      return;
    }

    // Log to chat as a note
    addMessage({
      role: 'user',
      content: `[Communication Note] ${messageText}`,
    });

    toast.success('Note logged to chat');
    setMessageText('');
  };

  const handleStartCall = (callId: string) => {
    showCall(callId);
    addMessage({
      role: 'assistant',
      content: '[Call Started] Video call connected',
    });
    toast.success('Call started');
  };

  const handleEndCall = () => {
    endCall();
    toast.info('Call ended');
    addMessage({
      role: 'assistant',
      content: '[Call Ended] Video call disconnected',
    });
  };

  return (
    <div className="space-y-0">
      {/* Active Call View */}
      {isInCall && (
        <div className="p-3 border-b border-sidebar-border bg-destructive/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-destructive rounded-full animate-pulse" />
              <span className="text-sm font-medium">Call Active</span>
            </div>
            <Button
              variant="destructive"
              size="sm"
              className="h-7 text-xs"
              onClick={handleEndCall}
            >
              <PhoneOff className="h-3 w-3 mr-1" />
              End Call
            </Button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as CommTab)}>
        <TabsList className="w-full grid grid-cols-3 h-9 rounded-none border-b border-sidebar-border bg-transparent">
          <TabsTrigger value="messaging" className="text-xs data-[state=active]:bg-sidebar-accent">
            <MessageSquare className="h-3 w-3 mr-1" />
            Notes
          </TabsTrigger>
          <TabsTrigger value="contacts" className="text-xs data-[state=active]:bg-sidebar-accent">
            <Users className="h-3 w-3 mr-1" />
            Contacts
          </TabsTrigger>
          <TabsTrigger value="dialer" className="text-xs data-[state=active]:bg-sidebar-accent">
            <Phone className="h-3 w-3 mr-1" />
            Dial
          </TabsTrigger>
        </TabsList>

        <TabsContent value="messaging" className="mt-0 p-3 space-y-2">
          <Textarea
            value={messageText}
            onChange={(e) => setMessageText(e.target.value)}
            placeholder="Add a note to chat history..."
            className="min-h-[60px] text-sm bg-sidebar-accent border-sidebar-border resize-none"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!messageText.trim()}
            className="w-full h-8 text-sm"
          >
            <MessageSquare className="h-3 w-3 mr-1.5" />
            Log Note
          </Button>
        </TabsContent>

        <TabsContent value="contacts" className="mt-0">
          <ContactsPanel />
        </TabsContent>

        <TabsContent value="dialer" className="mt-0">
          <DialerPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}

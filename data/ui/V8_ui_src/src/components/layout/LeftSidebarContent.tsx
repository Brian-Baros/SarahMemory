import { useState, useEffect } from 'react';
import { Calendar, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSarahStore } from '@/stores/useSarahStore';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messageCount: number;
}

/**
 * Left sidebar content - Chat history list
 * Used by both desktop sidebar and mobile drawer
 */
export function LeftSidebarContent() {
  const { threads, activeThreadId, setActiveThread, clearMessages, addMessage, setLeftDrawerOpen } = useSarahStore();
  const [searchDate, setSearchDate] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

  // Fetch conversations from backend
  const fetchConversations = async () => {
    setIsLoading(true);
    try {
      const response = await api.proxy.getConversations();
      if (response.conversations && Array.isArray(response.conversations)) {
        const mapped = response.conversations.map((conv: any) => ({
          id: conv.id || conv.conversation_id || String(Math.random()),
          title: conv.title || conv.summary || 'Conversation',
          preview: conv.preview || conv.last_message || '',
          timestamp: new Date(conv.timestamp || conv.created_at || Date.now()),
          messageCount: conv.message_count || conv.messageCount || 0,
        }));
        setConversations(mapped);
      }
      setHasLoaded(true);
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
      setConversations(threads.map(t => ({
        id: t.id,
        title: t.title,
        preview: t.preview,
        timestamp: t.timestamp,
        messageCount: t.messageCount,
      })));
      setHasLoaded(true);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!hasLoaded) {
      fetchConversations();
    }
  }, [hasLoaded]);

  const displayConversations = conversations.length > 0 ? conversations : threads.map(t => ({
    id: t.id,
    title: t.title,
    preview: t.preview,
    timestamp: t.timestamp,
    messageCount: t.messageCount,
  }));

  const filteredConversations = searchDate 
    ? displayConversations.filter(conv => {
        const convDate = new Date(conv.timestamp).toISOString().split('T')[0];
        return convDate === searchDate;
      })
    : displayConversations;

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  const handleLoadConversation = async (conv: Conversation) => {
    setActiveThread(conv.id);
    setLeftDrawerOpen(false); // Close drawer on mobile
    
    try {
      const response = await api.proxy.call(`/api/conversations/${conv.id}`);
      if (response && (response as any).messages) {
        clearMessages();
        (response as any).messages.forEach((msg: any) => {
          addMessage({
            role: msg.role || (msg.is_user ? 'user' : 'assistant'),
            content: msg.content || msg.text || '',
          });
        });
        toast.success('Conversation loaded');
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Date Search & Refresh */}
      <div className="p-3 border-b border-sidebar-border space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs text-muted-foreground">Search by date</label>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={fetchConversations}
            disabled={isLoading}
            className="text-sidebar-foreground hover:text-foreground hover:bg-sidebar-accent h-7 w-7"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
          </Button>
        </div>
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input 
            type="date" 
            value={searchDate}
            onChange={(e) => setSearchDate(e.target.value)}
            className="pl-9 bg-sidebar-accent border-sidebar-border text-sidebar-foreground"
          />
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoading && !hasLoaded ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : filteredConversations.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            No conversations found
          </div>
        ) : (
          filteredConversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => handleLoadConversation(conv)}
              className={cn(
                "w-full text-left p-3 rounded-lg transition-all duration-200",
                "hover:bg-sidebar-accent",
                activeThreadId === conv.id 
                  ? "bg-sidebar-accent border border-sidebar-primary/30" 
                  : "bg-transparent"
              )}
            >
              <div className="font-medium text-sm text-sidebar-foreground truncate">
                {conv.title}
              </div>
              <div className="text-xs text-muted-foreground truncate mt-1">
                {conv.preview}
              </div>
              <div className="text-xs text-muted-foreground mt-1.5 flex items-center justify-between">
                <span>{formatDate(conv.timestamp)}</span>
                <span>{conv.messageCount} msgs</span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

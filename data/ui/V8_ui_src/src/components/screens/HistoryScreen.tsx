import { useState, useEffect } from 'react';
import { Clock, Calendar, Loader2, RefreshCw, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useSarahStore } from '@/stores/useSarahStore';
import { useNavigationStore } from '@/stores/useNavigationStore';
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
 * History Screen - Chat history with date search
 * Mobile-first panel that shows all past conversations
 */
export function HistoryScreen() {
  const { threads, activeThreadId, setActiveThread, clearMessages, addMessage } = useSarahStore();
  const { setCurrentScreen } = useNavigationStore();
  const [searchDate, setSearchDate] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

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
      // Use local threads as fallback
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
    setCurrentScreen('chat'); // Navigate to chat
    
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
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Chat History</h1>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={fetchConversations}
            disabled={isLoading}
            className="ml-auto h-8 w-8"
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
        </div>
        
        {/* Date Search */}
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input 
            type="date" 
            value={searchDate}
            onChange={(e) => setSearchDate(e.target.value)}
            className="pl-9"
            placeholder="Search by date"
          />
        </div>
      </div>

      {/* Conversation List */}
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {isLoading && !hasLoaded ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="text-center py-12">
              <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-muted-foreground">No conversations found</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Start chatting to build your history
              </p>
            </div>
          ) : (
            filteredConversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => handleLoadConversation(conv)}
                className={cn(
                  "w-full text-left p-4 rounded-xl transition-all duration-200",
                  "bg-card hover:bg-card/80 border border-border/50",
                  activeThreadId === conv.id && "border-primary/50 bg-primary/5"
                )}
              >
                <div className="font-medium text-sm truncate">
                  {conv.title}
                </div>
                <div className="text-xs text-muted-foreground truncate mt-1">
                  {conv.preview}
                </div>
                <div className="text-xs text-muted-foreground/70 mt-2 flex items-center justify-between">
                  <span>{formatDate(conv.timestamp)}</span>
                  <span>{conv.messageCount} messages</span>
                </div>
              </button>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

import { useState, useEffect, useMemo, useCallback } from "react";
import { Clock, Calendar, Loader2, RefreshCw, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useSarahStore } from "@/stores/useSarahStore";
import { useNavigationStore } from "@/stores/useNavigationStore";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messageCount: number;
}

interface CachedConversationMessages {
  id: string;
  messages: Array<{
    role: string;
    content: string;
  }>;
  updatedAt: string;
}

const HISTORY_CACHE_KEY = "sarahmemory:conversation-history:index";
const HISTORY_MESSAGES_CACHE_KEY = "sarahmemory:conversation-history:messages";

/**
 * History Screen - Chat history with date search
 * Mobile-first panel that shows all past conversations
 */
export function HistoryScreen() {
  const { threads, activeThreadId, setActiveThread, clearMessages, addMessage } = useSarahStore();
  const { setCurrentScreen } = useNavigationStore();

  const [searchDate, setSearchDate] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);

  const safeDate = (value: unknown): Date => {
    if (value instanceof Date && !Number.isNaN(value.getTime())) return value;

    if (typeof value === "string" || typeof value === "number") {
      const parsed = new Date(value);
      if (!Number.isNaN(parsed.getTime())) return parsed;
    }

    return new Date();
  };

  const normalizeConversation = useCallback((conv: any): Conversation | null => {
    const id = String(conv?.id || conv?.conversation_id || conv?.thread_id || "").trim();
    if (!id) return null;

    const timestamp = safeDate(conv?.timestamp || conv?.created_at || conv?.updated_at || Date.now());
    const title =
      String(conv?.title || conv?.summary || conv?.name || "Conversation").trim() || "Conversation";
    const preview = String(conv?.preview || conv?.last_message || conv?.snippet || "").trim();

    const rawCount =
      conv?.message_count ??
      conv?.messageCount ??
      (Array.isArray(conv?.messages) ? conv.messages.length : 0);

    const messageCount = Number.isFinite(Number(rawCount)) ? Number(rawCount) : 0;

    return {
      id,
      title,
      preview,
      timestamp,
      messageCount,
    };
  }, []);

  const loadCachedConversationIndex = useCallback((): Conversation[] => {
    try {
      const raw = window.localStorage.getItem(HISTORY_CACHE_KEY);
      if (!raw) return [];

      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      return parsed
        .map((item) => normalizeConversation(item))
        .filter(Boolean) as Conversation[];
    } catch {
      return [];
    }
  }, [normalizeConversation]);

  const saveCachedConversationIndex = useCallback((items: Conversation[]) => {
    try {
      const payload = items.map((item) => ({
        ...item,
        timestamp: item.timestamp.toISOString(),
      }));
      window.localStorage.setItem(HISTORY_CACHE_KEY, JSON.stringify(payload));
    } catch {
      // non-fatal
    }
  }, []);

  const loadCachedMessageMap = useCallback((): Record<string, CachedConversationMessages> => {
    try {
      const raw = window.localStorage.getItem(HISTORY_MESSAGES_CACHE_KEY);
      if (!raw) return {};

      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }, []);

  const saveCachedConversationMessages = useCallback(
    (conversationId: string, messages: Array<{ role: string; content: string }>) => {
      if (!conversationId || !Array.isArray(messages) || messages.length === 0) return;

      try {
        const existing = loadCachedMessageMap();
        existing[conversationId] = {
          id: conversationId,
          messages: messages.map((msg) => ({
            role: String(msg?.role || "assistant"),
            content: String(msg?.content || ""),
          })),
          updatedAt: new Date().toISOString(),
        };
        window.localStorage.setItem(HISTORY_MESSAGES_CACHE_KEY, JSON.stringify(existing));
      } catch {
        // non-fatal
      }
    },
    [loadCachedMessageMap]
  );

  const mergeConversationSources = useCallback(
    (sources: Conversation[][]): Conversation[] => {
      const map = new Map<string, Conversation>();

      for (const source of sources) {
        for (const item of source) {
          if (!item?.id) continue;

          const existing = map.get(item.id);
          if (!existing) {
            map.set(item.id, item);
            continue;
          }

          const existingTime = safeDate(existing.timestamp).getTime();
          const incomingTime = safeDate(item.timestamp).getTime();

          map.set(item.id, {
            id: item.id,
            title:
              item.title && item.title !== "Conversation"
                ? item.title
                : existing.title || "Conversation",
            preview: item.preview || existing.preview || "",
            timestamp: incomingTime >= existingTime ? item.timestamp : existing.timestamp,
            messageCount: Math.max(existing.messageCount || 0, item.messageCount || 0),
          });
        }
      }

      return Array.from(map.values()).sort(
        (a, b) => safeDate(b.timestamp).getTime() - safeDate(a.timestamp).getTime()
      );
    },
    []
  );

  const localThreadConversations = useMemo<Conversation[]>(() => {
    return (threads || [])
      .map((thread: any) =>
        normalizeConversation({
          id: thread?.id,
          title: thread?.title,
          preview: thread?.preview,
          timestamp: thread?.timestamp,
          messageCount: thread?.messageCount,
          messages: thread?.messages,
        })
      )
      .filter(Boolean) as Conversation[];
  }, [threads, normalizeConversation]);

  const hydrateConversationIndex = useCallback(
    (remoteItems: Conversation[] = []) => {
      const cached = loadCachedConversationIndex();
      const merged = mergeConversationSources([remoteItems, localThreadConversations, cached]);
      setConversations(merged);
      saveCachedConversationIndex(merged);
      return merged;
    },
    [
      loadCachedConversationIndex,
      localThreadConversations,
      mergeConversationSources,
      saveCachedConversationIndex,
    ]
  );

  const fetchConversations = useCallback(async () => {
    setIsLoading(true);

    try {
      const response = await api.proxy.getConversations();
      const remoteItems = Array.isArray(response?.conversations)
        ? (response.conversations
            .map((conv: any) => normalizeConversation(conv))
            .filter(Boolean) as Conversation[])
        : [];

      hydrateConversationIndex(remoteItems);
      setHasLoadedOnce(true);
    } catch (error) {
      console.error("Failed to fetch conversations:", error);
      hydrateConversationIndex([]);
      setHasLoadedOnce(true);
    } finally {
      setIsLoading(false);
    }
  }, [hydrateConversationIndex, normalizeConversation]);

  // Initial load
  useEffect(() => {
    if (!hasLoadedOnce) {
      void fetchConversations();
    }
  }, [hasLoadedOnce, fetchConversations]);

  // Keep history updated when local threads change
  useEffect(() => {
    hydrateConversationIndex([]);
  }, [hydrateConversationIndex, localThreadConversations]);

  const displayConversations = useMemo<Conversation[]>(() => {
    if (conversations.length > 0) return conversations;
    return mergeConversationSources([localThreadConversations, loadCachedConversationIndex()]);
  }, [
    conversations,
    localThreadConversations,
    mergeConversationSources,
    loadCachedConversationIndex,
  ]);

  const filteredConversations = useMemo(() => {
    if (!searchDate) return displayConversations;

    return displayConversations.filter((conv) => {
      const convDate = safeDate(conv.timestamp).toISOString().split("T")[0];
      return convDate === searchDate;
    });
  }, [displayConversations, searchDate]);

  const formatDate = (date: Date) => {
    const normalized = safeDate(date);
    const now = new Date();
    const diff = now.getTime() - normalized.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days <= 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;
    return normalized.toLocaleDateString();
  };

  const handleLoadConversation = useCallback(
    async (conv: Conversation) => {
      setActiveThread(conv.id);
      setCurrentScreen("chat");

      try {
        const response = await api.proxy.call(`/api/conversations/${conv.id}`);
        const remoteMessages = Array.isArray((response as any)?.messages)
          ? (response as any).messages
              .map((msg: any) => ({
                role: String(msg?.role || (msg?.is_user ? "user" : "assistant")),
                content: String(msg?.content || msg?.text || ""),
              }))
              .filter((msg: { role: string; content: string }) => msg.content.trim().length > 0)
          : [];

        if (remoteMessages.length > 0) {
          clearMessages();
          remoteMessages.forEach((msg: { role: string; content: string }) => {
            addMessage({
              role: msg.role as any,
              content: msg.content,
            });
          });

          saveCachedConversationMessages(conv.id, remoteMessages);

          // Keep preview/count fresh in the index cache
          const updatedIndex = mergeConversationSources([
            displayConversations,
            [
              {
                ...conv,
                preview: remoteMessages[remoteMessages.length - 1]?.content || conv.preview,
                messageCount: remoteMessages.length,
                timestamp: new Date(),
              },
            ],
          ]);
          setConversations(updatedIndex);
          saveCachedConversationIndex(updatedIndex);

          toast.success("Conversation loaded");
          return;
        }

        throw new Error("No messages returned from backend");
      } catch (error) {
        console.error("Failed to load conversation from backend:", error);

        const cachedMessageMap = loadCachedMessageMap();
        const cached = cachedMessageMap[conv.id];

        if (cached && Array.isArray(cached.messages) && cached.messages.length > 0) {
          clearMessages();
          cached.messages.forEach((msg) => {
            addMessage({
              role: msg.role as any,
              content: msg.content,
            });
          });
          toast.success("Loaded cached conversation");
          return;
        }

        const localThread = (threads || []).find((t: any) => String(t?.id) === conv.id);
        const localThreadMessages = Array.isArray((localThread as any)?.messages)
          ? (localThread as any).messages
              .map((msg: any) => ({
                role: String(msg?.role || (msg?.is_user ? "user" : "assistant")),
                content: String(msg?.content || msg?.text || ""),
              }))
              .filter((msg: { role: string; content: string }) => msg.content.trim().length > 0)
          : [];

        if (localThreadMessages.length > 0) {
          clearMessages();
          localThreadMessages.forEach((msg: { role: string; content: string }) => {
            addMessage({
              role: msg.role as any,
              content: msg.content,
            });
          });

          saveCachedConversationMessages(conv.id, localThreadMessages);
          toast.success("Loaded local conversation");
          return;
        }

        toast.error("Conversation metadata found, but no saved messages were available");
      }
    },
    [
      addMessage,
      clearMessages,
      displayConversations,
      loadCachedMessageMap,
      mergeConversationSources,
      saveCachedConversationIndex,
      saveCachedConversationMessages,
      setActiveThread,
      setCurrentScreen,
      threads,
    ]
  );

  // ---------------------------------------------------------------------------
  // SarahMemory UI Control Bus listener (Chat-driven automation)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const handler = (ev: any) => {
      const actions = ev?.detail?.actions || [];
      if (!Array.isArray(actions) || actions.length === 0) return;

      for (const a of actions) {
        if (!a || !a.type) continue;

        try {
          if (a.type === "navigate" || a.type === "set_screen") {
            const screen = a.payload?.screen || a.payload?.route;
            if (typeof screen === "string" && screen) {
              const s = screen.replace(/^\//, "");
              if (s) setCurrentScreen(s as any);
            }
          }

          if (a.type === "history_refresh") {
            void fetchConversations();
          }

          if (a.type === "history_open" && a.payload?.id) {
            const target = displayConversations.find((c) => c.id === a.payload.id);
            if (target) void handleLoadConversation(target);
          }
        } catch (e) {
          console.warn("[HistoryScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [displayConversations, fetchConversations, handleLoadConversation, setCurrentScreen]);

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
            onClick={() => void fetchConversations()}
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
          {isLoading && !hasLoadedOnce ? (
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
                onClick={() => void handleLoadConversation(conv)}
                className={cn(
                  "w-full text-left p-4 rounded-xl transition-all duration-200",
                  "bg-card hover:bg-card/80 border border-border/50",
                  activeThreadId === conv.id && "border-primary/50 bg-primary/5"
                )}
              >
                <div className="font-medium text-sm truncate">{conv.title}</div>

                <div className="text-xs text-muted-foreground truncate mt-1">
                  {conv.preview || "No preview available"}
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
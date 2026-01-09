import { useState } from "react";
import { Search, BookOpen, Link, Loader2, AlertCircle, Plus, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useSarahStore } from "@/stores/useSarahStore";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface SearchResult {
  id: string;
  title: string;
  snippet: string;
  url?: string;
  source?: string;
}

/**
 * Research Screen - Search and knowledge gathering
 * Supports web search, summarization, and add-to-memory
 */
export function ResearchScreen() {
  const { addMessage } = useSarahStore();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    try {
      // Try research endpoint
      const response = await api.proxy.call("/api/research/search", {
        method: "POST",
        body: { query: query.trim() },
      });

      if (response && (response as any).results) {
        setResults((response as any).results);
        setIsAvailable(true);
      } else {
        // Fallback: use chat with research mode
        const chatResponse = await api.chat.sendMessage([{ role: "user", content: query.trim() }], {
          researchMode: true,
        });

        if (chatResponse.sources && chatResponse.sources.length > 0) {
          setResults(
            chatResponse.sources.map((src, idx) => ({
              id: String(idx),
              title: src,
              snippet: chatResponse.content.substring(0, 200),
              url: src,
            })),
          );
          setIsAvailable(true);
        } else {
          setResults([
            {
              id: "0",
              title: "AI Response",
              snippet: chatResponse.content,
            },
          ]);
          setIsAvailable(true);
        }
      }

      // Log to chat
      addMessage({
        role: "user",
        content: `[Research] ${query.trim()}`,
      });
    } catch (error) {
      console.warn("[Research] Search failed:", error);
      setIsAvailable(false);
      toast.error("Research search not available");
    } finally {
      setIsSearching(false);
    }
  };

  const handleAddToMemory = async (result: SearchResult) => {
    try {
      await api.proxy.call("/api/memory/add", {
        method: "POST",
        body: {
          content: result.snippet,
          title: result.title,
          source: result.url,
        },
      });
      toast.success("Added to memory");

      addMessage({
        role: "assistant",
        content: `[Memory] Added: ${result.title}`,
      });
    } catch (error) {
      console.warn("[Research] Add to memory failed:", error);
      toast.error("Could not add to memory");
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Research Hub</h1>
        </div>
        <p className="text-xs text-muted-foreground mt-1">Search, summarize, and save knowledge</p>
      </div>

      {/* Search Input */}
      <div className="p-4 border-b border-border">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search for information..."
              className="pl-9"
            />
          </div>
          <Button onClick={handleSearch} disabled={isSearching || !query.trim()}>
            {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* Results */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {isAvailable === false && (
            <div className="p-4 rounded-xl bg-muted/50 border border-border">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Not available (server configuration)</p>
                  <p className="text-xs text-muted-foreground mt-1">Research features require server-side support.</p>
                </div>
              </div>
            </div>
          )}

          {results.length === 0 && !isSearching && (
            <div className="text-center py-12">
              <Search className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">Enter a query to search for information</p>
            </div>
          )}

          {isSearching && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          )}

          {results.map((result) => (
            <div key={result.id} className="p-4 rounded-xl bg-card border border-border">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-medium text-sm">{result.title}</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={() => handleAddToMemory(result)}
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground mt-2 line-clamp-3">{result.snippet}</p>
              {result.url && (
                <a
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-2"
                >
                  <Link className="h-3 w-3" />
                  View source
                  <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

import { useMemo, useState } from 'react';
import { User, Download, X } from 'lucide-react';
import sarahIcon from '@/assets/sarah-icon.ico';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import type { Message } from '@/types/sarah';

interface ChatMessageProps {
  message: Message;
  onSendFollowUp?: (text: string) => void;
}

function parseFollowUpSuggestions(content: string): { text: string; suggestions: string[] } {
  // Only match bracket tokens that are standalone (preceded by start/whitespace and followed by whitespace/end)
  const suggestionRegex = /(^|\s)\[([^\]\n]+)\](?=\s|$)/g;

  const suggestions: string[] = [];
  let match: RegExpExecArray | null;

  while ((match = suggestionRegex.exec(content)) !== null) {
    suggestions.push(match[2].trim());
  }

  // Remove only those standalone tokens (preserves markdown links like [docs](url))
  const text = content.replace(suggestionRegex, '$1').trim();
  return { text, suggestions };
}

// Detect media in message (images/videos)
function parseMedia(content: string): { text: string; images: string[]; videos: string[] } {
  const images: string[] = [];
  const videos: string[] = [];

  // Markdown images: ![alt](url)
  const mdImageRegex = /!\[[^\]]*?\]\(([^)]+)\)/gi;
  let m: RegExpExecArray | null;

  while ((m = mdImageRegex.exec(content)) !== null) {
    const url = (m[1] || '').trim();
    if (url && !images.includes(url)) images.push(url);
  }

  // Raw image URLs
  const rawImageRegex = /https?:\/\/[^\s)]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s)]*)?/gi;
  let im: RegExpExecArray | null;

  while ((im = rawImageRegex.exec(content)) !== null) {
    const url = (im[0] || '').trim();
    if (url && !images.includes(url)) images.push(url);
  }

  // Raw video URLs
  const rawVideoRegex = /https?:\/\/[^\s)]+\.(?:mp4|webm|mov|avi)(?:\?[^\s)]*)?/gi;
  let vm: RegExpExecArray | null;

  while ((vm = rawVideoRegex.exec(content)) !== null) {
    const url = (vm[0] || '').trim();
    if (url && !videos.includes(url)) videos.push(url);
  }

  // Clean up text: remove markdown image blocks + extracted URLs
  let text = content;

  // Remove markdown image tokens entirely
  text = text.replace(mdImageRegex, '').trim();

  // Remove raw URLs we extracted (images/videos)
  [...images, ...videos].forEach((u) => {
    // escape for regex
    const escaped = u.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    text = text.replace(new RegExp(escaped, 'g'), '').trim();
  });

  // Normalize extra whitespace left behind
  text = text.replace(/\n{3,}/g, '\n\n').replace(/[ \t]{2,}/g, ' ').trim();

  return { text, images, videos };
}

export function ChatMessage({ message, onSendFollowUp }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const formatTime = (ts: any) => {
    const date = ts instanceof Date ? ts : new Date(ts ?? Date.now());
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const { finalText, suggestions, images, videos } = useMemo(() => {
    const { text: withoutSuggestions, suggestions } = parseFollowUpSuggestions(message.content || '');
    const { text: finalText, images, videos } = parseMedia(withoutSuggestions);
    return { finalText, suggestions, images, videos };
  }, [message.content]);

  const handleFollowUp = (suggestion: string) => {
    if (!onSendFollowUp) return;

    const s = suggestion.trim();
    const lower = s.toLowerCase();

    const followUpMessage =
      lower.includes('yes') || lower.includes('more') || lower.includes('deeper')
        ? `Yes, please ${lower}`
        : s;

    onSendFollowUp(followUpMessage);
  };

  const handleDownload = async (url: string, filename?: string) => {
    try {
      const response = await fetch(url, { mode: 'cors' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename || 'download';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className={cn('flex gap-3 animate-fade-in', isUser ? 'justify-end' : 'justify-start')}>
      {/* Avatar for assistant */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 mt-1 ring-2 ring-primary/40 shadow-[0_0_10px_rgba(var(--primary-rgb),0.3)]">
          <img src={sarahIcon} alt="Sarah AI" className="w-full h-full object-cover" />
        </div>
      )}

      <div className={cn('max-w-[85%] sm:max-w-[75%] space-y-2', isUser && 'items-end')}>
        {/* Message Bubble */}
        <div className={cn('px-4 py-2.5 whitespace-pre-wrap', isUser ? 'bubble-user' : 'bubble-assistant')}>
          <p className="text-sm leading-relaxed">{finalText || message.content}</p>
        </div>

        {/* Media Display - Images */}
        {images.length > 0 && (
          <div
            className={cn(
              'grid gap-2',
              images.length === 1 ? 'grid-cols-1' : images.length === 2 ? 'grid-cols-2' : images.length <= 4 ? 'grid-cols-2' : 'grid-cols-3',
            )}
          >
            {images.slice(0, 4).map((img, idx) => (
              <div
                key={`${img}-${idx}`}
                className="relative group rounded-lg overflow-hidden cursor-pointer bg-secondary"
                onClick={() => setSelectedImage(img)}
              >
                <img
                  src={img}
                  alt={`Generated image ${idx + 1}`}
                  className="w-full h-auto max-h-48 object-cover"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 text-white hover:bg-white/20"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(img, `image-${idx + 1}.png`);
                    }}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Media Display - Videos */}
        {videos.length > 0 && (
          <div className="space-y-2">
            {videos.map((vid, idx) => (
              <div key={`${vid}-${idx}`} className="relative rounded-lg overflow-hidden bg-secondary">
                <video src={vid} className="w-full max-h-64" controls playsInline />
                <div className="absolute top-2 right-2 flex gap-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 bg-background/80 hover:bg-background"
                    onClick={() => handleDownload(vid, `video-${idx + 1}.mp4`)}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Follow-up Suggestions */}
        {!isUser && suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {suggestions.map((suggestion, idx) => (
              <Button
                key={`${suggestion}-${idx}`}
                variant="outline"
                size="sm"
                className="h-7 text-xs bg-secondary/50 hover:bg-primary/20 border-border hover:border-primary/50"
                onClick={() => handleFollowUp(suggestion)}
              >
                {suggestion}
              </Button>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <div className={cn('text-xs text-muted-foreground px-1', isUser ? 'text-right' : 'text-left')}>
          {formatTime(message.timestamp)}
        </div>
      </div>

      {/* Avatar for user */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-1">
          <User className="w-4 h-4 text-secondary-foreground" />
        </div>
      )}

      {/* Image Lightbox */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setSelectedImage(null)}>
          <div className="relative max-w-4xl max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img src={selectedImage} alt="Full size" className="max-w-full max-h-[90vh] object-contain rounded-lg" />
            <div className="absolute top-2 right-2 flex gap-2">
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 bg-background/80 hover:bg-background text-foreground"
                onClick={() => handleDownload(selectedImage, 'image.png')}
              >
                <Download className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 bg-background/80 hover:bg-background text-foreground"
                onClick={() => setSelectedImage(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState } from 'react';
import { User, Download, Play, Pause, Square, X, Image as ImageIcon, Video as VideoIcon } from 'lucide-react';
import sarahIcon from '@/assets/sarah-icon.ico';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import type { Message } from '@/types/sarah';

interface ChatMessageProps {
  message: Message;
  onSendFollowUp?: (text: string) => void;
}

// Parse follow-up suggestions like "Should I dig deeper? [Yes] [No]"
function parseFollowUpSuggestions(content: string): { text: string; suggestions: string[] } {
  const suggestionRegex = /\[([^\]]+)\]/g;
  const suggestions: string[] = [];
  let match;
  
  while ((match = suggestionRegex.exec(content)) !== null) {
    suggestions.push(match[1]);
  }
  
  // Remove the suggestion brackets from the text
  const text = content.replace(suggestionRegex, '').trim();
  
  return { text, suggestions };
}

// Detect media in message (images/videos)
function parseMedia(content: string): { text: string; images: string[]; videos: string[] } {
  const images: string[] = [];
  const videos: string[] = [];
  
  // Match image URLs
  const imageRegex = /(?:!\[.*?\]\(([^)]+)\)|(?:https?:\/\/[^\s]+\.(?:jpg|jpeg|png|gif|webp)))/gi;
  let imageMatch;
  while ((imageMatch = imageRegex.exec(content)) !== null) {
    const url = imageMatch[1] || imageMatch[0];
    if (url && !images.includes(url)) {
      images.push(url);
    }
  }
  
  // Match video URLs
  const videoRegex = /(?:https?:\/\/[^\s]+\.(?:mp4|webm|mov|avi))/gi;
  let videoMatch;
  while ((videoMatch = videoRegex.exec(content)) !== null) {
    if (!videos.includes(videoMatch[0])) {
      videos.push(videoMatch[0]);
    }
  }
  
  // Clean up text by removing raw URLs that we've extracted
  let text = content;
  images.forEach(img => {
    text = text.replace(img, '').replace(/!\[.*?\]\(\)/g, '');
  });
  videos.forEach(vid => {
    text = text.replace(vid, '');
  });
  
  return { text: text.trim(), images, videos };
}

export function ChatMessage({ message, onSendFollowUp }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Parse content for follow-ups and media
  const { text: textWithoutSuggestions, suggestions } = parseFollowUpSuggestions(message.content);
  const { text: finalText, images, videos } = parseMedia(textWithoutSuggestions);

  const handleFollowUp = (suggestion: string) => {
    if (onSendFollowUp) {
      // Convert suggestion to a follow-up message
      const followUpMessage = suggestion.toLowerCase().includes('yes') || 
                             suggestion.toLowerCase().includes('more') ||
                             suggestion.toLowerCase().includes('deeper')
        ? `Yes, please ${suggestion.toLowerCase()}`
        : suggestion;
      onSendFollowUp(followUpMessage);
    }
  };

  const handleDownload = async (url: string, filename?: string) => {
    try {
      const response = await fetch(url);
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
    <div className={cn(
      "flex gap-3 animate-fade-in",
      isUser ? "justify-end" : "justify-start"
    )}>
      {/* Avatar for assistant */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 mt-1 ring-2 ring-primary/40 shadow-[0_0_10px_rgba(var(--primary-rgb),0.3)]">
          <img src={sarahIcon} alt="Sarah AI" className="w-full h-full object-cover" />
        </div>
      )}

      <div className={cn(
        "max-w-[85%] sm:max-w-[75%] space-y-2",
        isUser && "items-end"
      )}>
        {/* Message Bubble */}
        <div className={cn(
          "px-4 py-2.5 whitespace-pre-wrap",
          isUser ? "bubble-user" : "bubble-assistant"
        )}>
          <p className="text-sm leading-relaxed">{finalText || message.content}</p>
        </div>

        {/* Media Display - Images */}
        {images.length > 0 && (
          <div className={cn(
            "grid gap-2",
            images.length === 1 ? "grid-cols-1" : 
            images.length === 2 ? "grid-cols-2" :
            images.length <= 4 ? "grid-cols-2" : "grid-cols-3"
          )}>
            {images.slice(0, 4).map((img, idx) => (
              <div 
                key={idx} 
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
              <div key={idx} className="relative rounded-lg overflow-hidden bg-secondary">
                <video 
                  src={vid}
                  className="w-full max-h-64"
                  controls
                  playsInline
                />
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
                key={idx}
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
        <div className={cn(
          "text-xs text-muted-foreground px-1",
          isUser ? "text-right" : "text-left"
        )}>
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
        <div 
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh]">
            <img 
              src={selectedImage} 
              alt="Full size"
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
            />
            <div className="absolute top-2 right-2 flex gap-2">
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8 bg-background/80 hover:bg-background text-foreground"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDownload(selectedImage, 'image.png');
                }}
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
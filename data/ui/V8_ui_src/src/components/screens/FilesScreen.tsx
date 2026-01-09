import { useState, useEffect } from 'react';
import { 
  FolderOpen, 
  FileText, 
  Image, 
  Music, 
  Video, 
  Download,
  Loader2,
  AlertCircle,
  Folder,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';

interface FileItem {
  id: string;
  name: string;
  type: 'folder' | 'document' | 'image' | 'audio' | 'video' | 'other';
  size?: number;
  modifiedAt?: Date;
  path?: string;
}

/**
 * Files Screen - File explorer
 * Browses local/remote files with download support
 */
export function FilesScreen() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);
  const [currentPath, setCurrentPath] = useState('/');

  useEffect(() => {
    loadFiles();
  }, [currentPath]);

  const loadFiles = async () => {
    setIsLoading(true);
    try {
      const response = await api.proxy.call('/api/files/list', {
        method: 'POST',
        body: { path: currentPath },
      });
      
      if (response && (response as any).files) {
        setFiles((response as any).files);
        setIsAvailable(true);
      } else {
        setIsAvailable(false);
      }
    } catch (error) {
      console.warn('[Files] Not available:', error);
      setIsAvailable(false);
    } finally {
      setIsLoading(false);
    }
  };

  const getFileIcon = (type: FileItem['type']) => {
    switch (type) {
      case 'folder': return Folder;
      case 'document': return FileText;
      case 'image': return Image;
      case 'audio': return Music;
      case 'video': return Video;
      default: return FileText;
    }
  };

  const formatSize = (bytes?: number) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleItemClick = (file: FileItem) => {
    if (file.type === 'folder') {
      setCurrentPath(file.path || `${currentPath}${file.name}/`);
    }
  };

  const handleDownload = async (file: FileItem) => {
    try {
      const response = await api.proxy.call('/api/files/download', {
        method: 'POST',
        body: { path: file.path || `${currentPath}${file.name}` },
      });
      
      if (response && (response as any).url) {
        window.open((response as any).url, '_blank');
      }
    } catch (error) {
      console.warn('[Files] Download failed:', error);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <FolderOpen className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Files</h1>
        </div>
        <p className="text-xs text-muted-foreground mt-1 truncate">
          {currentPath}
        </p>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          {isAvailable === false && (
            <div className="p-4 rounded-xl bg-muted/50 border border-border mb-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Not available (server configuration)</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    File browsing requires server-side support.
                  </p>
                </div>
              </div>
            </div>
          )}

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : files.length === 0 ? (
            <div className="text-center py-12">
              <FolderOpen className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                {isAvailable ? "No files found" : "File browser not available"}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {currentPath !== '/' && (
                <button
                  onClick={() => {
                    const parts = currentPath.split('/').filter(Boolean);
                    parts.pop();
                    setCurrentPath(parts.length ? `/${parts.join('/')}/` : '/');
                  }}
                  className="w-full text-left p-3 rounded-xl bg-card border border-border flex items-center gap-3 hover:bg-card/80"
                >
                  <Folder className="h-5 w-5 text-muted-foreground" />
                  <span className="text-sm">..</span>
                </button>
              )}
              
              {files.map((file) => {
                const Icon = getFileIcon(file.type);
                return (
                  <div
                    key={file.id}
                    className="p-3 rounded-xl bg-card border border-border flex items-center gap-3"
                  >
                    <button
                      onClick={() => handleItemClick(file)}
                      className={cn(
                        "flex items-center gap-3 flex-1 min-w-0 text-left",
                        file.type === 'folder' && "cursor-pointer hover:text-primary"
                      )}
                    >
                      <Icon className={cn(
                        "h-5 w-5 shrink-0",
                        file.type === 'folder' ? "text-primary" : "text-muted-foreground"
                      )} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm truncate">{file.name}</p>
                        {file.size && (
                          <p className="text-xs text-muted-foreground">{formatSize(file.size)}</p>
                        )}
                      </div>
                    </button>
                    {file.type !== 'folder' && (
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="h-8 w-8 shrink-0"
                        onClick={() => handleDownload(file)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

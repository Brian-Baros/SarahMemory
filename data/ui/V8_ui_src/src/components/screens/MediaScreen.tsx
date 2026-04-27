import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Loader2,
  Music,
  Video,
  Image,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { useCreativeCacheStore, type CachedItem } from "@/stores/useCreativeCacheStore";
import { cn } from "@/lib/utils";
import { useNavigationStore } from "@/stores/useNavigationStore";

type ResolvedMediaKind = "audio" | "video" | "image" | "unknown";

const SEEK_STEP_SECONDS = 10;

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function safeDate(value: unknown): Date {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
  const parsed = new Date(value as any);
  if (!Number.isNaN(parsed.getTime())) return parsed;
  return new Date();
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function getMediaKind(item: CachedItem | null): ResolvedMediaKind {
  if (!item) return "unknown";

  const type = String((item as any)?.type || "").toLowerCase();
  if (type === "music" || type === "voice" || type === "audio") return "audio";
  if (type === "video") return "video";
  if (type === "image") return "image";

  const mime = String(
    (item as any)?.mimeType || (item as any)?.mime || (item as any)?.contentType || ""
  ).toLowerCase();

  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("image/")) return "image";

  const candidates = [
    (item as any)?.url,
    (item as any)?.preview,
    (item as any)?.thumbnail,
    (item as any)?.previewUrl,
    (item as any)?.downloadUrl,
    (item as any)?.src,
    (item as any)?.path,
    (item as any)?.filePath,
    (item as any)?.file_url,
  ];

  const joined = candidates.filter(Boolean).join(" ").toLowerCase();

  if (/\.(mp3|wav|ogg|m4a|aac|flac)(\?|$)/i.test(joined)) return "audio";
  if (/\.(mp4|webm|mov|mkv|avi)(\?|$)/i.test(joined)) return "video";
  if (/\.(png|jpg|jpeg|gif|webp|bmp|svg)(\?|$)/i.test(joined)) return "image";

  return "unknown";
}

function getBestPreview(item: CachedItem | null): string | null {
  if (!item) return null;

  const candidates = [
    (item as any)?.preview,
    (item as any)?.thumbnail,
    (item as any)?.poster,
    (item as any)?.cover,
    (item as any)?.image,
    (item as any)?.imageUrl,
    (item as any)?.previewUrl,
    (item as any)?.url,
  ];

  for (const value of candidates) {
    if (isNonEmptyString(value)) return value;
  }

  return null;
}

function getCandidateMediaSources(item: CachedItem | null): string[] {
  if (!item) return [];

  const candidates = [
    (item as any)?.url,
    (item as any)?.src,
    (item as any)?.mediaUrl,
    (item as any)?.media_url,
    (item as any)?.downloadUrl,
    (item as any)?.download_url,
    (item as any)?.previewUrl,
    (item as any)?.preview,
    (item as any)?.thumbnail,
    (item as any)?.path,
    (item as any)?.filePath,
    (item as any)?.file_path,
    (item as any)?.localPath,
    (item as any)?.blobUrl,
    (item as any)?.blob_url,
  ];

  const out: string[] = [];
  for (const value of candidates) {
    if (isNonEmptyString(value) && !out.includes(value)) {
      out.push(value);
    }
  }
  return out;
}

/**
 * Media Screen - Media player for creative outputs
 * Plays audio/video from the creative cache
 */
export function MediaScreen() {
  const { setCurrentScreen } = useNavigationStore();
  const { items, downloadItem } = useCreativeCacheStore();

  const [currentItem, setCurrentItem] = useState<CachedItem | null>(null);
  const [currentSrc, setCurrentSrc] = useState<string | null>(null);
  const [resolvedKind, setResolvedKind] = useState<ResolvedMediaKind>("unknown");

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState([75]);
  const [progress, setProgress] = useState([0]);

  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const [isResolving, setIsResolving] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const userSeekingRef = useRef(false);

  const cleanupObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      try {
        URL.revokeObjectURL(objectUrlRef.current);
      } catch {
        // ignore
      }
      objectUrlRef.current = null;
    }
  }, []);

  const stopActiveMedia = useCallback(() => {
    try {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if (videoRef.current) {
        videoRef.current.pause();
      }
    } catch {
      // ignore
    }
  }, []);

  const applyElementVolume = useCallback(
    (muted: boolean, nextVolume: number) => {
      const normalized = Math.max(0, Math.min(1, nextVolume / 100));
      const elements = [audioRef.current, videoRef.current];

      for (const el of elements) {
        if (!el) continue;
        el.muted = muted;
        el.volume = normalized;
      }
    },
    []
  );

  const allMedia = useMemo(() => {
    const list = [...items.music, ...items.voice, ...items.video, ...items.image];
    return list
      .slice()
      .sort((a, b) => safeDate((b as any)?.createdAt).getTime() - safeDate((a as any)?.createdAt).getTime());
  }, [items]);

  const currentIndex = useMemo(() => {
    if (!currentItem) return -1;
    return allMedia.findIndex((item) => item.id === currentItem.id);
  }, [allMedia, currentItem]);

  const getMediaIcon = (type: CachedItem["type"]) => {
    switch (type) {
      case "music":
      case "voice":
        return Music;
      case "video":
        return Video;
      case "image":
        return Image;
      default:
        return Music;
    }
  };

  const syncPlaybackStateFromElement = useCallback(() => {
    const mediaEl = resolvedKind === "video" ? videoRef.current : audioRef.current;
    if (!mediaEl) return;

    if (!userSeekingRef.current) {
      const d = Number.isFinite(mediaEl.duration) ? mediaEl.duration : 0;
      const t = Number.isFinite(mediaEl.currentTime) ? mediaEl.currentTime : 0;

      setDuration(d);
      setCurrentTime(t);

      if (d > 0) {
        setProgress([Math.min(100, Math.max(0, (t / d) * 100))]);
      } else {
        setProgress([0]);
      }
    }
  }, [resolvedKind]);

  const playCurrent = useCallback(async () => {
    const mediaEl = resolvedKind === "video" ? videoRef.current : audioRef.current;
    if (!mediaEl) {
      if (resolvedKind === "image") {
        setIsPlaying(false);
      }
      return;
    }

    try {
      await mediaEl.play();
      setIsPlaying(true);
      setMediaError(null);
    } catch (error) {
      console.error("[MediaScreen] play failed:", error);
      setIsPlaying(false);
      setMediaError("Playback could not start.");
    }
  }, [resolvedKind]);

  const pauseCurrent = useCallback(() => {
    try {
      audioRef.current?.pause();
      videoRef.current?.pause();
    } catch {
      // ignore
    }
    setIsPlaying(false);
  }, []);

  const togglePlayback = useCallback(async () => {
    if (!currentItem) return;

    if (resolvedKind === "image") {
      setIsPlaying(false);
      return;
    }

    if (isPlaying) {
      pauseCurrent();
    } else {
      await playCurrent();
    }
  }, [currentItem, isPlaying, pauseCurrent, playCurrent, resolvedKind]);

  const seekToPercent = useCallback(
    (nextPercent: number) => {
      const mediaEl = resolvedKind === "video" ? videoRef.current : audioRef.current;
      if (!mediaEl || duration <= 0) return;

      const clampedPercent = Math.max(0, Math.min(100, nextPercent));
      const targetTime = (clampedPercent / 100) * duration;

      mediaEl.currentTime = targetTime;
      setCurrentTime(targetTime);
      setProgress([clampedPercent]);
    },
    [duration, resolvedKind]
  );

  const skipRelative = useCallback(
    (deltaSeconds: number) => {
      const mediaEl = resolvedKind === "video" ? videoRef.current : audioRef.current;
      if (!mediaEl) return;

      const d = Number.isFinite(mediaEl.duration) ? mediaEl.duration : 0;
      const next = Math.max(0, Math.min(d || Number.MAX_SAFE_INTEGER, mediaEl.currentTime + deltaSeconds));
      mediaEl.currentTime = next;
      setCurrentTime(next);

      if (d > 0) {
        setProgress([Math.min(100, Math.max(0, (next / d) * 100))]);
      }
    },
    [resolvedKind]
  );

  const selectItemByIndex = useCallback(
    async (index: number, autoPlay = true) => {
      if (index < 0 || index >= allMedia.length) return;
      const item = allMedia[index];
      if (!item) return;

      stopActiveMedia();
      cleanupObjectUrl();

      setCurrentItem(item);
      setResolvedKind(getMediaKind(item));
      setCurrentSrc(null);
      setCurrentTime(0);
      setDuration(0);
      setProgress([0]);
      setMediaError(null);
      setIsPlaying(false);
      setIsResolving(true);

      try {
        const directSources = getCandidateMediaSources(item);
        let resolvedSrc: string | null = directSources[0] || null;

        if (!resolvedSrc && typeof downloadItem === "function") {
          let downloaded: any = null;

          try {
            downloaded = await (downloadItem as any)(item);
          } catch {
            try {
              downloaded = await (downloadItem as any)(item.id);
            } catch {
              try {
                downloaded = await (downloadItem as any)(item.id, item.type);
              } catch (error) {
                console.warn("[MediaScreen] downloadItem failed:", error);
              }
            }
          }

          if (typeof downloaded === "string" && downloaded.trim()) {
            resolvedSrc = downloaded;
          } else if (downloaded instanceof Blob) {
            const objectUrl = URL.createObjectURL(downloaded);
            objectUrlRef.current = objectUrl;
            resolvedSrc = objectUrl;
          } else if (downloaded && typeof downloaded === "object") {
            const blobCandidate =
              downloaded.blob instanceof Blob
                ? downloaded.blob
                : downloaded.data instanceof Blob
                  ? downloaded.data
                  : null;

            if (blobCandidate) {
              const objectUrl = URL.createObjectURL(blobCandidate);
              objectUrlRef.current = objectUrl;
              resolvedSrc = objectUrl;
            } else {
              const objCandidates = [
                downloaded.url,
                downloaded.src,
                downloaded.path,
                downloaded.localPath,
                downloaded.preview,
                downloaded.downloadUrl,
              ];
              for (const candidate of objCandidates) {
                if (isNonEmptyString(candidate)) {
                  resolvedSrc = candidate;
                  break;
                }
              }
            }
          }
        }

        if (!resolvedSrc && getMediaKind(item) === "image") {
          resolvedSrc = getBestPreview(item);
        }

        if (!resolvedSrc) {
          setMediaError("No playable source was found for this media item.");
          setCurrentSrc(null);
          return;
        }

        setCurrentSrc(resolvedSrc);
        setResolvedKind(getMediaKind(item));

        if (autoPlay && getMediaKind(item) !== "image") {
          requestAnimationFrame(() => {
            void playCurrent();
          });
        }
      } finally {
        setIsResolving(false);
      }
    },
    [allMedia, cleanupObjectUrl, downloadItem, playCurrent, stopActiveMedia]
  );

  const playNext = useCallback(async () => {
    if (allMedia.length === 0) return;
    const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % allMedia.length : 0;
    await selectItemByIndex(nextIndex, true);
  }, [allMedia.length, currentIndex, selectItemByIndex]);

  const playPrevious = useCallback(async () => {
    if (allMedia.length === 0) return;
    const prevIndex = currentIndex >= 0 ? (currentIndex - 1 + allMedia.length) % allMedia.length : 0;
    await selectItemByIndex(prevIndex, true);
  }, [allMedia.length, currentIndex, selectItemByIndex]);

  const handlePlayItem = useCallback(
    async (item: CachedItem) => {
      const index = allMedia.findIndex((x) => x.id === item.id);
      if (index >= 0) {
        await selectItemByIndex(index, true);
      }
    },
    [allMedia, selectItemByIndex]
  );

  useEffect(() => {
    applyElementVolume(isMuted, volume[0] ?? 75);
  }, [applyElementVolume, isMuted, volume]);

  useEffect(() => {
    if (allMedia.length > 0 && !currentItem) {
      void selectItemByIndex(0, false);
    }
  }, [allMedia, currentItem, selectItemByIndex]);

  useEffect(() => {
    const audio = audioRef.current;
    const video = videoRef.current;
    const mediaEl = resolvedKind === "video" ? video : audio;

    if (!mediaEl) return;

    const onLoadedMetadata = () => {
      const d = Number.isFinite(mediaEl.duration) ? mediaEl.duration : 0;
      setDuration(d);
      setCurrentTime(Number.isFinite(mediaEl.currentTime) ? mediaEl.currentTime : 0);
      setMediaError(null);
    };

    const onTimeUpdate = () => {
      syncPlaybackStateFromElement();
    };

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => {
      setIsPlaying(false);
      void playNext();
    };

    const onError = () => {
      setIsPlaying(false);
      setMediaError("Playback failed for this media item.");
    };

    mediaEl.addEventListener("loadedmetadata", onLoadedMetadata);
    mediaEl.addEventListener("timeupdate", onTimeUpdate);
    mediaEl.addEventListener("play", onPlay);
    mediaEl.addEventListener("pause", onPause);
    mediaEl.addEventListener("ended", onEnded);
    mediaEl.addEventListener("error", onError);

    return () => {
      mediaEl.removeEventListener("loadedmetadata", onLoadedMetadata);
      mediaEl.removeEventListener("timeupdate", onTimeUpdate);
      mediaEl.removeEventListener("play", onPlay);
      mediaEl.removeEventListener("pause", onPause);
      mediaEl.removeEventListener("ended", onEnded);
      mediaEl.removeEventListener("error", onError);
    };
  }, [playNext, resolvedKind, syncPlaybackStateFromElement, currentSrc]);

  useEffect(() => {
    return () => {
      stopActiveMedia();
      cleanupObjectUrl();
    };
  }, [cleanupObjectUrl, stopActiveMedia]);

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

          if (a.type === "media_play") {
            if (a.payload?.id) {
              const target = allMedia.find((m) => m.id === a.payload.id);
              if (target) {
                void handlePlayItem(target);
              } else {
                void playCurrent();
              }
            } else {
              void playCurrent();
            }
          }

          if (a.type === "media_pause") {
            pauseCurrent();
          }

          if (a.type === "media_toggle") {
            void togglePlayback();
          }

          if (a.type === "media_next") {
            void playNext();
          }

          if (a.type === "media_previous") {
            void playPrevious();
          }

          if (a.type === "media_seek") {
            const percent = Number(a.payload?.percent);
            const seconds = Number(a.payload?.seconds);

            if (Number.isFinite(percent)) {
              seekToPercent(percent);
            } else if (Number.isFinite(seconds)) {
              const mediaEl = resolvedKind === "video" ? videoRef.current : audioRef.current;
              if (mediaEl) {
                mediaEl.currentTime = Math.max(0, seconds);
              }
            }
          }

          if (a.type === "media_set_volume") {
            const nextVolume = Number(a.payload?.value);
            if (Number.isFinite(nextVolume)) {
              const clamped = Math.max(0, Math.min(100, nextVolume));
              setVolume([clamped]);
              if (clamped > 0 && isMuted) setIsMuted(false);
            }
          }

          if (a.type === "media_mute") {
            setIsMuted(Boolean(a.payload?.value ?? true));
          }

          if (a.type === "media_open" && a.payload?.id) {
            const target = allMedia.find((m) => m.id === a.payload.id);
            if (target) {
              void handlePlayItem(target);
            }
          }
        } catch (e) {
          console.warn("[MediaScreen] UI action failed:", a, e);
        }
      }
    };

    window.addEventListener("sarah:ui", handler as any);
    return () => window.removeEventListener("sarah:ui", handler as any);
  }, [
    allMedia,
    handlePlayItem,
    isMuted,
    pauseCurrent,
    playCurrent,
    playNext,
    playPrevious,
    resolvedKind,
    seekToPercent,
    setCurrentScreen,
    togglePlayback,
  ]);

  const renderNowPlayingVisual = () => {
    if (!currentItem) return null;

    const preview = getBestPreview(currentItem);

    if (isResolving) {
      return (
        <div className="aspect-video max-h-56 rounded-lg bg-muted flex items-center justify-center mb-4">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      );
    }

    if (resolvedKind === "video" && currentSrc) {
      return (
        <div className="aspect-video max-h-56 rounded-lg bg-black overflow-hidden mb-4">
          <video
            ref={videoRef}
            src={currentSrc}
            className="w-full h-full object-contain bg-black"
            playsInline
            preload="metadata"
            poster={preview || undefined}
          />
        </div>
      );
    }

    if (resolvedKind === "image" && (currentSrc || preview)) {
      return (
        <div className="aspect-video max-h-56 rounded-lg bg-muted overflow-hidden mb-4">
          <img
            src={(currentSrc || preview) as string}
            alt={(currentItem as any)?.prompt || "Media preview"}
            className="w-full h-full object-contain"
          />
        </div>
      );
    }

    if ((resolvedKind === "audio" || resolvedKind === "unknown") && currentSrc) {
      return (
        <div className="aspect-video max-h-56 rounded-lg bg-muted flex items-center justify-center mb-4 overflow-hidden">
          {preview ? (
            <img
              src={preview}
              alt={(currentItem as any)?.prompt || "Media artwork"}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="text-center">
              <Music className="h-12 w-12 text-muted-foreground/50 mx-auto" />
              <p className="text-xs text-muted-foreground mt-2 px-4 truncate max-w-full">
                {(currentItem as any)?.prompt || "Audio"}
              </p>
            </div>
          )}
          <audio ref={audioRef} src={currentSrc} preload="metadata" hidden />
        </div>
      );
    }

    return (
      <div className="aspect-video max-h-56 rounded-lg bg-muted flex items-center justify-center mb-4">
        <div className="text-center">
          {(() => {
            const Icon = getMediaIcon(currentItem.type);
            return <Icon className="h-12 w-12 text-muted-foreground/50 mx-auto" />;
          })()}
          <p className="text-xs text-muted-foreground mt-2 px-4 truncate max-w-full">
            {(currentItem as any)?.prompt || "Media item"}
          </p>
        </div>
      </div>
    );
  };

  const currentPrompt =
    (currentItem as any)?.prompt ||
    (currentItem as any)?.title ||
    (currentItem as any)?.name ||
    "Untitled media";

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Play className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Media Player</h1>
        </div>
      </div>

      {/* Now Playing */}
      {currentItem && (
        <div className="p-4 border-b border-border bg-card/30">
          {renderNowPlayingVisual()}

          <div className="mb-3">
            <div className="text-sm font-medium truncate">{currentPrompt}</div>
            <div className="text-xs text-muted-foreground capitalize mt-1">
              {currentItem.type} • {safeDate((currentItem as any)?.createdAt).toLocaleString()}
            </div>
            {mediaError && <div className="text-xs text-destructive mt-2">{mediaError}</div>}
          </div>

          {/* Progress */}
          <div className="mb-2">
            <Slider
              value={progress}
              onValueChange={(value) => {
                userSeekingRef.current = true;
                setProgress(value);
              }}
              onValueCommit={(value) => {
                userSeekingRef.current = false;
                seekToPercent(value[0] ?? 0);
              }}
              max={100}
              step={0.1}
              className="mb-2"
              disabled={resolvedKind === "image" || duration <= 0}
            />
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center gap-2 mt-4">
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10"
              onClick={() => {
                if (resolvedKind === "image") {
                  void playPrevious();
                } else {
                  skipRelative(-SEEK_STEP_SECONDS);
                }
              }}
              disabled={allMedia.length === 0}
            >
              <SkipBack className="h-5 w-5" />
            </Button>

            <Button
              variant="default"
              size="icon"
              className="h-12 w-12"
              onClick={() => void togglePlayback()}
              disabled={resolvedKind === "image" || !currentSrc || isResolving}
            >
              {isResolving ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : isPlaying ? (
                <Pause className="h-6 w-6" />
              ) : (
                <Play className="h-6 w-6" />
              )}
            </Button>

            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10"
              onClick={() => {
                if (resolvedKind === "image") {
                  void playNext();
                } else {
                  skipRelative(SEEK_STEP_SECONDS);
                }
              }}
              disabled={allMedia.length === 0}
            >
              <SkipForward className="h-5 w-5" />
            </Button>
          </div>

          {/* Playlist Nav */}
          <div className="flex items-center justify-center gap-2 mt-3">
            <Button variant="outline" size="sm" onClick={() => void playPrevious()} disabled={allMedia.length === 0}>
              Previous
            </Button>
            <Button variant="outline" size="sm" onClick={() => void playNext()} disabled={allMedia.length === 0}>
              Next
            </Button>
          </div>

          {/* Volume */}
          <div className="flex items-center gap-2 mt-4">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setIsMuted((prev) => !prev)}
              disabled={resolvedKind === "image"}
            >
              {isMuted || (volume[0] ?? 0) === 0 ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            <Slider
              value={isMuted ? [0] : volume}
              onValueChange={(value) => {
                setVolume(value);
                if ((value[0] ?? 0) > 0 && isMuted) setIsMuted(false);
                if ((value[0] ?? 0) === 0 && !isMuted) setIsMuted(true);
              }}
              max={100}
              step={1}
              className="flex-1"
              disabled={resolvedKind === "image"}
            />
          </div>
        </div>
      )}

      {/* Media Library */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          <p className="text-xs text-muted-foreground mb-3">Library ({allMedia.length} items)</p>

          {allMedia.length === 0 ? (
            <div className="text-center py-12">
              <Music className="h-12 w-12 mx-auto text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">No media in session</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Generate content in Studios to see it here
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {allMedia.map((item) => {
                const Icon = getMediaIcon(item.type);
                const isActive = currentItem?.id === item.id;

                return (
                  <button
                    key={item.id}
                    onClick={() => void handlePlayItem(item)}
                    className={cn(
                      "w-full text-left p-3 rounded-xl flex items-center gap-3 transition-all",
                      "bg-card border border-border hover:bg-card/80",
                      isActive && "border-primary/50 bg-primary/5"
                    )}
                  >
                    <div
                      className={cn(
                        "w-10 h-10 rounded-lg flex items-center justify-center shrink-0 overflow-hidden",
                        isActive ? "bg-primary/20" : "bg-muted"
                      )}
                    >
                      {getBestPreview(item) && getMediaKind(item) === "image" ? (
                        <img
                          src={getBestPreview(item) as string}
                          alt={(item as any)?.prompt || "Preview"}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <Icon
                          className={cn(
                            "h-5 w-5",
                            isActive ? "text-primary" : "text-muted-foreground"
                          )}
                        />
                      )}
                    </div>

                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">
                        {(item as any)?.prompt || (item as any)?.title || "Untitled media"}
                      </p>
                      <p className="text-xs text-muted-foreground capitalize">
                        {item.type} • {safeDate((item as any)?.createdAt).toLocaleTimeString()}
                      </p>
                    </div>

                    {isActive && isPlaying && resolvedKind !== "image" && (
                      <div className="flex gap-0.5">
                        <div className="w-0.5 h-3 bg-primary animate-pulse" />
                        <div className="w-0.5 h-4 bg-primary animate-pulse delay-75" />
                        <div className="w-0.5 h-2 bg-primary animate-pulse delay-150" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2, ExternalLink, ArrowLeft, ArrowRight, RefreshCw, Globe, Search } from "lucide-react";
import { toast } from "sonner";
import { useNavigationStore } from "@/stores/useNavigationStore";

type FetchBundle = {
  ok: boolean;
  url?: string;
  title?: string;
  clean_html?: string;
  text?: string;
  links?: Array<{ text: string; url: string }>;
  content_type?: string;
  error?: string;
  detail?: string;
};

function normalizeUrl(raw: string): string {
  const u = (raw || "").trim();
  if (!u) return "";
  if (u.startsWith("http://") || u.startsWith("https://")) return u;
  // basic host/path heuristic
  if (/^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$/i.test(u)) return `https://${u}`;
  return u; // treat as query
}

function isProbablyUrl(raw: string): boolean {
  const u = (raw || "").trim();
  if (!u) return false;
  if (u.startsWith("http://") || u.startsWith("https://")) return true;
  return /^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$/i.test(u);
}

async function postJson<T>(url: string, body: any): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const data = (await res.json().catch(() => ({}))) as any;
  if (!res.ok || data?.ok === false) {
    const msg = data?.error || `Request failed (${res.status})`;
    const detail = data?.detail ? ` — ${data.detail}` : "";
    throw new Error(`${msg}${detail}`);
  }
  return data as T;
}

export function ResearchScreen() {
  const { setCurrentScreen } = useNavigationStore();
  const [address, setAddress] = useState("https://duckduckgo.com/");
  const [loading, setLoading] = useState(false);


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
if (a.type === "research_open") {
  const url = a.payload?.url || a.payload?.href || a.payload?.address;
  if (typeof url === "string" && url.trim()) void navigateReader(url.trim());
}
if (a.type === "research_back") {
  if (canBack) {
    const u = hist[histIdx - 1];
    if (u) {
      setHistIdx((x) => x - 1);
      void navigateReader(u, true);
    }
  }
}
if (a.type === "research_forward") {
  if (canFwd) {
    const u = hist[histIdx + 1];
    if (u) {
      setHistIdx((x) => x + 1);
      void navigateReader(u, true);
    }
  }
}

            } catch (e) {
              console.warn("[ResearchScreen] UI action failed:", a, e);
            }
          }
        };
        window.addEventListener("sarah:ui", handler as any);
        return () => window.removeEventListener("sarah:ui", handler as any);
      }, []);

  // Reader Mode bundle
  const [bundle, setBundle] = useState<FetchBundle | null>(null);

  // very lightweight history (Reader Mode)
  const [hist, setHist] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState<number>(-1);

  const canBack = histIdx > 0;
  const canFwd = histIdx >= 0 && histIdx < hist.length - 1;

  const currentUrl = useMemo(() => {
    if (histIdx >= 0 && histIdx < hist.length) return hist[histIdx];
    return bundle?.url || "";
  }, [bundle?.url, hist, histIdx]);

  const htmlRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // init first entry into history so back/forward behaves immediately
    const u = normalizeUrl(address);
    if (u && isProbablyUrl(u)) {
      setHist([u]);
      setHistIdx(0);
      // auto-fetch first page
      void navigateReader(u, true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function navigateReader(url: string, replaceHistory = false) {
    const u = normalizeUrl(url);
    if (!u) return;

    setLoading(true);
    try {
      const data = await postJson<FetchBundle>("/api/browser/fetch", { url: u });
      setBundle(data);

      if (replaceHistory) {
        setHist([data.url || u]);
        setHistIdx(0);
      } else {
        const nextUrl = data.url || u;
        setHist((prev) => {
          const base = prev.slice(0, histIdx + 1);
          base.push(nextUrl);
          return base;
        });
        setHistIdx((prev) => prev + 1);
      }

      setAddress(data.url || u);
    } catch (e: any) {
      toast.error(e?.message || "Fetch failed");
      setBundle({
        ok: false,
        error: e?.message || "Fetch failed",
        url: u,
        title: u,
        clean_html: `<pre>${String(e?.message || "Fetch failed")}</pre>`,
        text: String(e?.message || "Fetch failed"),
        links: [],
      });
    } finally {
      setLoading(false);
    }
  }

  function openInLiveBrowser(url: string) {
    const u = normalizeUrl(url);
    if (!u) return;

    // DesktopHost bridge (WebView2)
    const anyWin = window as any;
    if (anyWin?.chrome?.webview?.postMessage) {
      try {
        anyWin.chrome.webview.postMessage(JSON.stringify({ type: "NAVIGATE_BROWSER", url: u }));
        toast.success("Routed to Live Browser");
        return;
      } catch {
        // fall through
      }
    }

    // Local Flask can open a pywebview window (when local-only)
    postJson("/api/browser/open_native", { url: u })
      .then(() => toast.success("Opened native browser"))
      .catch(() => {
        // final fallback
        window.open(u, "_blank", "noopener,noreferrer");
      });
  }

  async function handleGo() {
    const raw = address.trim();
    if (!raw) return;

    // If user typed a URL → Reader navigate
    if (isProbablyUrl(raw)) {
      await navigateReader(raw);
      return;
    }

    // Otherwise treat as search query → DuckDuckGo
    const q = encodeURIComponent(raw);
    const ddg = `https://duckduckgo.com/?q=${q}`;
    await navigateReader(ddg);
  }

  async function handleBack() {
    if (!canBack) return;
    const nextIdx = histIdx - 1;
    const url = hist[nextIdx];
    setHistIdx(nextIdx);
    await navigateReader(url, true);
  }

  async function handleForward() {
    if (!canFwd) return;
    const nextIdx = histIdx + 1;
    const url = hist[nextIdx];
    setHistIdx(nextIdx);
    await navigateReader(url, true);
  }

  async function handleReload() {
    const u = currentUrl || address;
    if (!u) return;
    await navigateReader(u, true);
  }

  function onReaderClickCapture(e: React.MouseEvent) {
    // Intercept anchor clicks inside Reader HTML and keep navigation in-app
    const target = e.target as HTMLElement | null;
    if (!target) return;

    const a = target.closest("a") as HTMLAnchorElement | null;
    if (!a) return;

    const href = (a.getAttribute("href") || "").trim();
    if (!href) return;

    e.preventDefault();
    e.stopPropagation();

    // Open in Reader by default; user can route to Live Browser with button
    void navigateReader(href);
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 p-3">
      {/* Toolbar */}
      <div className="flex items-center gap-2">
        <Button variant="outline" size="icon" disabled={!canBack || loading} onClick={handleBack}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={!canFwd || loading} onClick={handleForward}>
          <ArrowRight className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" disabled={loading} onClick={handleReload}>
          <RefreshCw className="h-4 w-4" />
        </Button>

        <div className="flex-1 flex items-center gap-2">
          <Input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleGo();
            }}
            placeholder="Enter URL or search query"
          />
          <Button onClick={() => void handleGo()} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            <span className="ml-2">Go</span>
          </Button>
        </div>

        <Button variant="secondary" onClick={() => openInLiveBrowser(currentUrl || address)} disabled={loading}>
          <Globe className="h-4 w-4" />
          <span className="ml-2">Live Browser</span>
        </Button>

        <Button variant="outline" onClick={() => window.open(normalizeUrl(currentUrl || address) || "about:blank", "_blank")}>
          <ExternalLink className="h-4 w-4" />
        </Button>
      </div>

      {/* Title / status */}
      <div className="flex items-center justify-between gap-2 text-sm opacity-80">
        <div className="truncate">
          <span className="font-medium">Reader:</span>{" "}
          <span className="truncate">{bundle?.title || currentUrl || "—"}</span>
        </div>
        <div className="shrink-0">
          {loading ? "Loading…" : bundle?.content_type ? bundle.content_type : ""}
        </div>
      </div>

      {/* Reader surface */}
      <div className="flex-1 rounded-lg border bg-background overflow-auto">
        <div
          ref={htmlRef}
          className="prose prose-invert max-w-none p-4"
          onClickCapture={onReaderClickCapture}
          // backend already sanitizes via bleach; this is still a trust boundary so keep it restricted to that endpoint
          dangerouslySetInnerHTML={{ __html: bundle?.clean_html || "<p>Enter a URL or query, then press Go.</p>" }}
        />
      </div>

      {/* Quick links (optional metadata) */}
      {!!bundle?.links?.length && (
        <div className="rounded-lg border p-3">
          <div className="text-sm font-medium mb-2">Links</div>
          <div className="grid gap-2">
            {bundle.links.slice(0, 12).map((l, idx) => (
              <div key={`${l.url}-${idx}`} className="flex items-center justify-between gap-2">
                <button
                  className="text-left text-sm underline truncate"
                  onClick={() => void navigateReader(l.url)}
                  title={l.url}
                >
                  {l.text || l.url}
                </button>
                <Button variant="outline" size="sm" onClick={() => openInLiveBrowser(l.url)}>
                  Live
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

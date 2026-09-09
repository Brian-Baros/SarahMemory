/**
 * SarahMemory Configuration
 * 
 * Configuration for connecting to the SarahMemory Flask backend.
 * Uses runtime backend contract first, then environment variables.
 * SARAHMEMORY_PATCH_NOTE 2026-06-24:
 * V9 local UI must bind to the backend that served it. It must not silently
 * drift to api.sarahmemory.com when running inside pywebview/local Flask.
 */

// Environment detection
export const isProduction = import.meta.env.PROD;
export const isDevelopment = import.meta.env.DEV;

/**
 * Get API base URL - supports window.SARAH_API_BASE override
 */
export const getApiBase = (): string => {
  const isProduction = import.meta.env.PROD;

  // Priority 1: Explicit runtime override injected by Flask /api/ui/runtime-config.js.
  if (typeof window !== "undefined" && (window as any).SARAH_API_BASE) {
    return String((window as any).SARAH_API_BASE).replace(/\/$/, "");
  }

  // Priority 1B: Structured V9 runtime boot object injected into index.html.
  if (typeof window !== "undefined" && (window as any).SARAH_UI_BOOT?.apiBase) {
    return String((window as any).SARAH_UI_BOOT.apiBase).replace(/\/$/, "");
  }

  if (typeof window !== "undefined") {
    const { hostname, origin } = window.location;

    // Priority 2: Hosted WebUI → API subdomain
    if (hostname === "ai.sarahmemory.com") {
      return "https://api.sarahmemory.com";
    }

    // Priority 3: Local same-origin (Flask serving UI + API)
    if (hostname === "127.0.0.1" || hostname === "localhost") {
      return origin;
    }
  }

  // Priority 4: Environment variable
  if (import.meta.env.VITE_SARAH_API_URL) {
    return String(import.meta.env.VITE_SARAH_API_URL);
  }

  // Priority 5: Public API base
  if (import.meta.env.VITE_PUBLIC_API_BASE) {
    return String(import.meta.env.VITE_PUBLIC_API_BASE);
  }

  // Priority 6: Production local-first default. A Vite production build served
  // by local Flask is still a LOCAL runtime; do not choose the public API unless
  // the operator explicitly configured VITE_PUBLIC_API_BASE above.
  if (isProduction) {
    return "http://127.0.0.1:8000";
  }

  // Priority 7: Absolute local fallback
  return "http://127.0.0.1:8000";
};



export type SarahRequestInit = RequestInit & {
  timeoutMs?: number;
};

function timeoutForPath(path: string): number {
  if (path.includes("/api/health")) return config.timeouts.health;
  if (path.includes("/api/voice")) return config.timeouts.voice;
  return config.timeouts.api;
}

/**
 * Robust fetch helper with proper error handling and a governed client-side
 * watchdog so UI controls cannot spin forever when the local backend stalls.
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: SarahRequestInit = {}
): Promise<T> {
  const baseUrl = getApiBase();
  const url = `${baseUrl}${path}`;

  const { timeoutMs, signal, ...fetchOptions } = options;
  const requestTimeoutMs = timeoutMs ?? timeoutForPath(path);
  const controller = new AbortController();
  const abortFromCaller = () => {
    try {
      controller.abort(signal?.reason);
    } catch {
      controller.abort();
    }
  };
  const timeout = setTimeout(() => controller.abort(new Error(`Request timeout after ${requestTimeoutMs}ms`)), requestTimeoutMs);

  if (signal?.aborted) {
    abortFromCaller();
  } else {
    signal?.addEventListener("abort", abortFromCaller, { once: true });
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        ...fetchOptions.headers,
      },
      credentials: 'include',
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) {
      const reason = controller.signal.reason;
      if (reason instanceof Error && reason.message) {
        throw reason;
      }
      throw new Error(`Request aborted: ${path}`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abortFromCaller);
  }
  
  // Try to parse JSON
  let data: unknown;
  const contentType = response.headers.get('content-type');
  
  if (contentType?.includes('application/json')) {
    try {
      data = await response.json();
    } catch {
      data = { error: 'Invalid JSON response', status: response.status };
    }
  } else {
    const text = await response.text();
    data = { error: text || `HTTP ${response.status}`, status: response.status };
  }
  
  if (!response.ok) {
    const errorMessage = (data as any)?.error || `Backend returned ${response.status}`;
    throw new Error(errorMessage);
  }
  
  return data as T;
}

export const config = {
  // API Configuration - use getter for dynamic resolution
  get apiBaseUrl() {
    return getApiBase();
  },
  
  // Mode detection
  get isCloudMode() {
    // SARAHMEMORY_PATCH_NOTE 2026-06-24:
    // A production Vite bundle can still be served by the LOCAL Flask backend.
    // Cloud mode is true only when the resolved API base is a SarahMemory cloud
    // host, not merely because import.meta.env.PROD is true.
    return this.apiBaseUrl.includes('sarahmemory.com');
  },
  
  get isLocalMode() {
    return !this.isCloudMode;
  },
  
  // Project Info
  version: '9.0.0',
  projectName: 'SarahMemory AiOS',
  
  // External Links
  githubUrl: 'https://github.com/Brian-Baros/SarahMemory',
  donateUrl: 'https://www.paypal.com/donate/?hosted_button_id=ZV43V3NYR6FDY',
  websiteUrl: 'https://www.sarahmemory.com',
  
  // Supabase Edge Function URLs (for proxying to backend)
  supabase: {
    projectId: import.meta.env.VITE_SUPABASE_PROJECT_ID || 'mflfjcipyzcdvsuprclt',
    url: import.meta.env.VITE_SUPABASE_URL || 'https://mflfjcipyzcdvsuprclt.supabase.co',
  },
  
  // Timeouts
  timeouts: {
    api: 30000,
    health: 5000,
    voice: 15000,
  },
  
  // Default settings
  defaults: {
    voice: 'sarah',
    theme: 'default',
    autoSpeak: true,
  },
  
  // Feature flags (will be updated by bootstrap)
  features: {
    voip: true,
    videoConference: true,
    creativeTools: true,
    avatar3d: true,
    desktopMirror: true,
    fileTransfer: true,
  },
} as const;

export default config;

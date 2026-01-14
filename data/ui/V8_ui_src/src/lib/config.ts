/**
 * SarahMemory Configuration
 * 
 * Configuration for connecting to the SarahMemory Flask backend.
 * Uses environment variables when available, with sensible defaults.
 */

// Environment detection
export const isProduction = import.meta.env.PROD;
export const isDevelopment = import.meta.env.DEV;

/**
 * Get API base URL - supports window.SARAH_API_BASE override
 */
export const getApiBase = (): string => {
  const isProduction = import.meta.env.PROD;

  // Priority 1: Explicit runtime override
  if (typeof window !== "undefined" && (window as any).SARAH_API_BASE) {
    return String((window as any).SARAH_API_BASE);
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

  // Priority 6: Production default
  if (isProduction) {
    return "https://api.sarahmemory.com";
  }

  // Priority 7: Absolute fallback
  return "http://127.0.0.1:8000";
};



/**
 * Robust fetch helper with proper error handling
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = getApiBase();
  const url = `${baseUrl}${path}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    credentials: 'include',
  });
  
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
    return isProduction || this.apiBaseUrl.includes('sarahmemory.com');
  },
  
  get isLocalMode() {
    return !this.isCloudMode;
  },
  
  // Project Info
  version: '8.0.0',
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

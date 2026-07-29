import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

/**
 * SarahMemory API Proxy Edge Function
 * 
 * This function proxies requests from the WebUI to the SarahMemory Flask backend.
 * It handles all API endpoints: chat, contacts, reminders, voice, avatar, dialer, etc.
 */

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
};

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const SARAH_API_URL = Deno.env.get("SARAH_MEMORY_API_URL") || "https://api.sarahmemory.com";
    
    // Parse the request body for POST requests
    let requestBody: { endpoint?: string; method?: string; payload?: unknown } = {};
    if (req.method === "POST") {
      try {
        requestBody = await req.json();
      } catch {
        // If body is empty or not JSON, continue with defaults
      }
    }
    
    // Extract the endpoint from body or query params
    const url = new URL(req.url);
    const endpoint = requestBody.endpoint || url.searchParams.get("endpoint") || "/";
    const method = requestBody.method || req.method;
    const apiUrl = `${SARAH_API_URL}${endpoint}`;
    
    console.log(`[sarah-api] Proxying ${method} request to: ${apiUrl}`);
    
    // Forward the request to the Flask backend
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    
    // Forward authorization if present
    const authHeader = req.headers.get("authorization");
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }
    
    // Prepare body for the forwarded request
    let body: string | undefined;
    if (requestBody.payload) {
      body = JSON.stringify(requestBody.payload);
    }
    
    const response = await fetch(apiUrl, {
      method: method,
      headers,
      body: method !== "GET" && method !== "HEAD" ? body : undefined,
    });
    
    // Handle different response types
    const contentType = response.headers.get("content-type") || "";
    
    // If response is not OK, return a JSON fallback instead of raw HTML error
    if (!response.ok) {
      console.error(`[sarah-api] Backend returned ${response.status} for ${apiUrl}`);
      return new Response(
        JSON.stringify({ 
          error: `Backend returned ${response.status}`,
          fallback: true,
          status: response.status,
          endpoint: endpoint
        }),
        {
          status: 200, // Return 200 so frontend can handle gracefully
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }
    
    if (contentType.includes("application/json")) {
      const data = await response.json();
      return new Response(JSON.stringify(data), {
        status: response.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    } else if (contentType.includes("audio")) {
      // Handle audio responses (TTS)
      const audioData = await response.arrayBuffer();
      return new Response(audioData, {
        status: response.status,
        headers: { 
          ...corsHeaders, 
          "Content-Type": contentType,
        },
      });
    } else {
      // For text/html or other content, wrap in JSON
      const text = await response.text();
      return new Response(JSON.stringify({ data: text, contentType }), {
        status: response.status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  } catch (error) {
    console.error("[sarah-api] Error:", error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : "Unknown error",
        fallback: true,
        message: "Backend unavailable - using fallback mode" 
      }),
      {
        status: 503,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});

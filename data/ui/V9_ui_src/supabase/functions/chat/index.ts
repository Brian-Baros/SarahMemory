import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

/**
 * Chat Edge Function
 * 
 * This function handles chat messages, routing them through the SarahMemory
 * Flask backend at https://api.sarahmemory.com
 * 
 * Flask endpoint: POST /api/chat
 * Request: { "text": "...", "intent"?, "tone"?, "complexity"?, "files"? }
 * Response: { "ok": true, "reply": "...", "meta": {...} }
 */

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages, useAI, mode, research_mode, conversation_id } = await req.json();
    const SARAH_API_URL = Deno.env.get("SARAH_MEMORY_API_URL") || "https://api.sarahmemory.com";
    
    // Get the latest user message
    const latestMessage = messages[messages.length - 1];
    
    console.log(`[chat] Mode: ${mode || 'any'}, Research: ${research_mode || false}`);
    
    // Try SarahMemory backend first (unless explicitly using AI-only mode)
    if (mode !== "api" && !useAI) {
      try {
        console.log("[chat] Attempting SarahMemory backend at /api/chat...");
        
        // Build request per app.py: POST /api/chat with { text, intent?, tone?, complexity?, files? }
        const backendResponse = await fetch(`${SARAH_API_URL}/api/chat`, {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "Accept": "application/json",
          },
          body: JSON.stringify({ 
            text: latestMessage.content,
            intent: research_mode ? "research" : undefined,
            mode: mode || "any",
            context: messages.slice(0, -1).map((m: any) => ({
              role: m.role,
              content: m.content
            })),
            conversation_id,
          }),
        });
        
        if (backendResponse.ok) {
          const data = await backendResponse.json();
          console.log("[chat] Backend response received:", JSON.stringify(data).slice(0, 200));
          
          return new Response(JSON.stringify({ 
            content: data.reply || data.response || data.message || data.content,
            source: "sarah_backend",
            audio_url: data.audio_url || null,
            web_augmented: data.web_augmented || false,
            sources: data.sources || [],
            meta: data.meta || {},
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
        console.log(`[chat] Backend returned status ${backendResponse.status}; local-first fallback will report unavailable`);
      } catch (e) {
        console.log("[chat] Backend unavailable:", e);
      }
    }

    return new Response(JSON.stringify({ 
      content: "SarahMemory backend is unavailable. Cloud fallback is disabled by local-first policy.",
      source: "sarah_backend_unavailable",
      audio_url: null,
      ok: false,
      fallback_policy: "local_first_no_third_party_gateway",
    }), {
      status: 503,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
    
  } catch (error) {
    console.error("[chat] Error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

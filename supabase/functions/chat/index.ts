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
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
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
        console.log(`[chat] Backend returned status ${backendResponse.status}, falling back to AI`);
      } catch (e) {
        console.log("[chat] Backend unavailable, using AI fallback:", e);
      }
    }
    
    // Fallback to Lovable AI Gateway
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY not configured");
    }
    
    console.log("[chat] Using Lovable AI Gateway");
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { 
            role: "system", 
            content: `You are Sarah, an intelligent AI assistant from the SarahMemory AiOS system. 
You are helpful, friendly, and knowledgeable. You assist users with tasks, answer questions, 
provide information, and help manage their digital life including contacts, reminders, and research.
Keep responses clear, concise, and helpful. When appropriate, ask clarifying questions.
You have access to the user's contacts, reminders, and conversation history.`
          },
          ...messages,
        ],
        stream: false,
      }),
    });
    
    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again later." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits depleted. Please add funds." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      throw new Error(`AI Gateway error: ${response.status}`);
    }
    
    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || "I'm sorry, I couldn't process that request.";
    
    return new Response(JSON.stringify({ 
      content,
      source: "lovable_ai",
      audio_url: null,
    }), {
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

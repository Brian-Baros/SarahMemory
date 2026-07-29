import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

/**
 * Avatar Edge Function
 * 
 * Handles avatar state management and control operations by proxying to
 * the SarahMemory Flask backend at https://api.sarahmemory.com
 * 
 * Flask endpoints from app.py:
 * - GET /api/avatar/state
 * - POST /api/avatar/mode
 * - POST /api/avatar/emotion
 * - GET /api/avatar/frame?width=...&height=...
 * - POST /api/avatar/lipsync
 * - POST /api/avatar/conference/start|answer|end|toggle
 * - GET /api/avatar/conference/info
 * - POST /api/avatar/media/image|video|stop
 * - GET /api/avatar/media/info
 * - POST /api/avatar/desktop/mirror
 * - POST /api/avatar/panel/size|maximize|popout
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
    const SARAH_API_URL = Deno.env.get("SARAH_MEMORY_API_URL") || "https://api.sarahmemory.com";
    const body = await req.json();
    const { action, mode, expression, emotion, animation, speaking, listening, description, width, height } = body;
    
    console.log(`[avatar] Action: ${action}, Mode: ${mode || 'n/a'}`);
    
    // =========================================================================
    // GET AVATAR STATE - GET /api/avatar/state
    // =========================================================================
    if (action === "get_state") {
      try {
        const response = await fetch(`${SARAH_API_URL}/api/avatar/state`, {
          method: "GET",
          headers: { "Accept": "application/json" },
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            success: true,
            state: data,
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[avatar] Get state error:", e);
      }
      
      // Return default state on error
      return new Response(JSON.stringify({ 
        success: true,
        state: {
          mode: "avatar_2d",
          expression: "neutral",
          speaking: false,
          listening: false,
        },
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // SET AVATAR MODE - POST /api/avatar/mode
    // =========================================================================
    if (action === "set_mode") {
      try {
        console.log(`[avatar] Setting mode to ${mode} via /api/avatar/mode`);
        const response = await fetch(`${SARAH_API_URL}/api/avatar/mode`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            success: true, 
            mode,
            ...data,
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
        console.log(`[avatar] Set mode returned ${response.status}`);
      } catch (e) {
        console.error("[avatar] Set mode error:", e);
      }
      
      return new Response(JSON.stringify({ 
        success: false,
        error: "Avatar backend unavailable",
        fallback: true,
        mode,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // SET EMOTION - POST /api/avatar/emotion
    // =========================================================================
    if (action === "set_expression" || action === "set_emotion") {
      const emotionValue = emotion || expression;
      try {
        console.log(`[avatar] Setting emotion to ${emotionValue} via /api/avatar/emotion`);
        const response = await fetch(`${SARAH_API_URL}/api/avatar/emotion`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ emotion: emotionValue }),
        });
        
        return new Response(JSON.stringify({ 
          success: response.ok, 
          expression: emotionValue,
          emotion: emotionValue,
          fallback: !response.ok,
        }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      } catch (e) {
        console.error("[avatar] Set emotion error:", e);
      }
      
      return new Response(JSON.stringify({ 
        success: false, 
        expression: emotionValue,
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // GET AVATAR FRAME - GET /api/avatar/frame
    // =========================================================================
    if (action === "get_frame") {
      try {
        const params = new URLSearchParams();
        if (width) params.set("width", String(width));
        if (height) params.set("height", String(height));
        
        const response = await fetch(`${SARAH_API_URL}/api/avatar/frame?${params}`, {
          method: "GET",
        });
        
        if (response.ok) {
          const contentType = response.headers.get("content-type") || "";
          
          if (contentType.includes("image")) {
            const arrayBuffer = await response.arrayBuffer();
            const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
            return new Response(JSON.stringify({ 
              success: true,
              frame_base64: base64,
              content_type: contentType,
            }), {
              headers: { ...corsHeaders, "Content-Type": "application/json" },
            });
          }
          
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[avatar] Get frame error:", e);
      }
      
      return new Response(JSON.stringify({ 
        success: false,
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // LIPSYNC - POST /api/avatar/lipsync
    // =========================================================================
    if (action === "lipsync") {
      try {
        const response = await fetch(`${SARAH_API_URL}/api/avatar/lipsync`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[avatar] Lipsync error:", e);
      }
      
      return new Response(JSON.stringify({ success: false, fallback: true }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // SPEAKING STATE
    // =========================================================================
    if (action === "speaking") {
      const speakingState = speaking ?? false;
      // Just acknowledge - backend tracks this internally
      return new Response(JSON.stringify({ success: true, speaking: speakingState }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // LISTENING STATE
    // =========================================================================
    if (action === "listening") {
      const listeningState = listening ?? false;
      return new Response(JSON.stringify({ success: true, listening: listeningState }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // CONFERENCE CONTROLS - POST /api/avatar/conference/*
    // =========================================================================
    if (action?.startsWith("conference_")) {
      const conferenceAction = action.replace("conference_", "");
      const endpoint = `/api/avatar/conference/${conferenceAction}`;
      
      try {
        const method = conferenceAction === "info" ? "GET" : "POST";
        const response = await fetch(`${SARAH_API_URL}${endpoint}`, {
          method,
          headers: { "Content-Type": "application/json" },
          body: method === "POST" ? JSON.stringify(body) : undefined,
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error(`[avatar] Conference ${conferenceAction} error:`, e);
      }
      
      return new Response(JSON.stringify({ 
        success: false, 
        error: "Conference feature unavailable",
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // MEDIA DISPLAY - POST /api/avatar/media/*
    // =========================================================================
    if (action?.startsWith("media_")) {
      const mediaAction = action.replace("media_", "");
      const endpoint = `/api/avatar/media/${mediaAction}`;
      
      try {
        const method = mediaAction === "info" ? "GET" : "POST";
        const response = await fetch(`${SARAH_API_URL}${endpoint}`, {
          method,
          headers: { "Content-Type": "application/json" },
          body: method === "POST" ? JSON.stringify(body) : undefined,
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error(`[avatar] Media ${mediaAction} error:`, e);
      }
      
      return new Response(JSON.stringify({ success: false, fallback: true }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // DESKTOP MIRROR - POST /api/avatar/desktop/mirror
    // =========================================================================
    if (action === "desktop_mirror") {
      try {
        const response = await fetch(`${SARAH_API_URL}/api/avatar/desktop/mirror`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[avatar] Desktop mirror error:", e);
      }
      
      return new Response(JSON.stringify({ success: false, fallback: true }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // PANEL CONTROLS - POST /api/avatar/panel/*
    // =========================================================================
    if (action?.startsWith("panel_")) {
      const panelAction = action.replace("panel_", "");
      const endpoint = `/api/avatar/panel/${panelAction}`;
      
      try {
        const response = await fetch(`${SARAH_API_URL}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ success: true, ...data }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error(`[avatar] Panel ${panelAction} error:`, e);
      }
      
      return new Response(JSON.stringify({ success: false, fallback: true }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // TRIGGER ANIMATION - maps to emotion/expression changes
    // =========================================================================
    if (action === "trigger_animation") {
      try {
        console.log(`[avatar] Triggering animation: ${animation}`);
        const response = await fetch(`${SARAH_API_URL}/api/avatar/emotion`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ emotion: animation, animation }),
        });
        
        return new Response(JSON.stringify({ 
          success: response.ok, 
          animation,
          fallback: !response.ok,
        }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      } catch (e) {
        console.error("[avatar] Trigger animation error:", e);
      }
      
      return new Response(JSON.stringify({ 
        success: true, 
        animation,
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // UNKNOWN ACTION
    // =========================================================================
    return new Response(JSON.stringify({ 
      error: `Unknown action: ${action}` 
    }), {
      status: 400,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
    
  } catch (error) {
    console.error("[avatar] Error:", error);
    return new Response(
      JSON.stringify({ 
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        fallback: true,
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

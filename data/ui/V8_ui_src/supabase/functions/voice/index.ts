import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

/**
 * Voice Edge Function
 * 
 * Handles TTS (text-to-speech) and voice management by proxying to
 * the SarahMemory Flask backend at https://api.sarahmemory.com
 * 
 * Flask endpoints:
 * - GET /get_available_voices - List available voices
 * - POST /set_user_setting - Set voice_profile setting
 */

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Default voices when backend is unavailable
const DEFAULT_VOICES = [
  { id: "sarah", name: "Sarah (Default)", language: "en-US", gender: "female" },
  { id: "emma", name: "Emma", language: "en-GB", gender: "female" },
  { id: "alex", name: "Alex", language: "en-US", gender: "male" },
  { id: "zoe", name: "Zoe", language: "en-US", gender: "female" },
  { id: "james", name: "James", language: "en-GB", gender: "male" },
];

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const SARAH_API_URL = Deno.env.get("SARAH_MEMORY_API_URL") || "https://api.sarahmemory.com";
    const { action, text, voice, audio } = await req.json();
    
    console.log(`[voice] Action: ${action}, Voice: ${voice || "default"}`);
    
    // =========================================================================
    // LIST AVAILABLE VOICES - GET /get_available_voices
    // =========================================================================
    if (action === "list_voices") {
      try {
        console.log("[voice] Fetching voices from /get_available_voices");
        const response = await fetch(`${SARAH_API_URL}/get_available_voices`, {
          method: "GET",
          headers: { 
            "Content-Type": "application/json",
            "Accept": "application/json",
          },
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log("[voice] Got voices:", JSON.stringify(data).slice(0, 200));
          
          // Handle different response formats
          let voices = Array.isArray(data) ? data : (data.voices || []);
          
          const normalizedVoices = voices.map((v: any) => ({
            id: v.id || v.name?.toLowerCase().replace(/\s+/g, "_") || "unknown",
            name: v.name || v.id || "Unknown Voice",
            language: v.language || "en-US",
            gender: v.gender || "female",
            preview_url: v.preview_url,
          }));
          
          return new Response(JSON.stringify({ 
            success: true,
            voices: normalizedVoices.length > 0 ? normalizedVoices : DEFAULT_VOICES,
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
        console.log(`[voice] Get voices returned ${response.status}`);
      } catch (e) {
        console.error("[voice] List voices error:", e);
      }
      
      // Return default voices as fallback
      return new Response(JSON.stringify({ 
        success: true,
        voices: DEFAULT_VOICES,
        fallback: true,
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // SET ACTIVE VOICE - POST /set_user_setting
    // =========================================================================
    if (action === "set_voice") {
      if (!voice) {
        return new Response(JSON.stringify({ 
          success: false,
          error: "Voice ID is required",
        }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      try {
        console.log(`[voice] Setting voice to ${voice} via /set_user_setting`);
        const response = await fetch(`${SARAH_API_URL}/set_user_setting`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: "voice_profile", value: voice }),
        });
        
        if (response.ok) {
          return new Response(JSON.stringify({ 
            success: true,
            voice_id: voice,
            message: "Voice updated successfully",
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[voice] Set voice error:", e);
      }
      
      // Even if backend fails, return success for local persistence
      return new Response(JSON.stringify({ 
        success: true,
        voice_id: voice,
        fallback: true,
        message: "Voice preference saved locally",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // TEXT-TO-SPEECH (Backend handles TTS internally via SarahMemoryVoice.py)
    // =========================================================================
    if (action === "speak" || action === "tts") {
      // TTS is handled by the backend when chat responses are generated
      // The audio_url comes back with the chat response
      // This endpoint is for explicit TTS requests
      return new Response(JSON.stringify({ 
        success: false,
        error: "TTS is handled via chat responses",
        fallback: true,
        voice_id: voice || "sarah",
        message: "Use browser TTS for explicit speak requests",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // PREVIEW VOICE (Use browser TTS since backend doesn't have preview endpoint)
    // =========================================================================
    if (action === "preview") {
      return new Response(JSON.stringify({ 
        success: false,
        fallback: true,
        voice_id: voice || "sarah",
        message: "Use browser TTS for voice preview",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // SPEECH-TO-TEXT (If backend has STT endpoint)
    // =========================================================================
    if (action === "transcribe" || action === "stt") {
      return new Response(JSON.stringify({ 
        success: false,
        error: "STT not available via this endpoint",
        fallback: true,
        message: "Use browser speech recognition",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // UNKNOWN ACTION
    // =========================================================================
    return new Response(JSON.stringify({ 
      success: false,
      error: `Unknown action: ${action}`,
    }), {
      status: 400,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
    
  } catch (error) {
    console.error("[voice] Error:", error);
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

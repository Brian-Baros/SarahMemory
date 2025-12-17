import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

/**
 * Dialer Edge Function
 * 
 * Handles phone/VoIP dialing operations by proxying to
 * the SarahMemory Flask backend at https://api.sarahmemory.com
 * 
 * SECURITY: Sensitive operations (initiate, end) require authentication
 */

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Helper to verify authentication for sensitive operations
const verifyAuth = async (req: Request): Promise<{ authenticated: boolean; userId?: string }> => {
  const authHeader = req.headers.get('authorization');
  if (!authHeader) {
    return { authenticated: false };
  }
  
  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } }
    });
    
    const { data: { user }, error } = await supabase.auth.getUser();
    if (error || !user) {
      return { authenticated: false };
    }
    return { authenticated: true, userId: user.id };
  } catch {
    return { authenticated: false };
  }
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const SARAH_API_URL = Deno.env.get("SARAH_MEMORY_API_URL") || "https://api.sarahmemory.com";
    const { action, number, ip_address, room_id } = await req.json();
    
    console.log(`[dialer] Action: ${action}, Target: ${number || ip_address || room_id || 'none'}`);
    
    // =========================================================================
    // CHECK CALL ACTIVE - GET /check_call_active (Public - read-only status check)
    // =========================================================================
    if (action === "check_availability" || action === "check_active") {
      try {
        const response = await fetch(`${SARAH_API_URL}/check_call_active`, {
          method: "GET",
          headers: { "Accept": "application/json" },
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            available: true,
            active: data.active || data.call_active || false,
            call_id: data.call_id,
            status: data.status,
            message: data.message || "VoIP status retrieved",
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.error("[dialer] Check call active error:", e);
      }
      
      return new Response(JSON.stringify({ 
        available: false,
        active: false,
        message: "VoIP/Video calling feature unavailable",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // INITIATE CALL - POST /initiate_call (REQUIRES AUTHENTICATION)
    // =========================================================================
    if (action === "initiate") {
      // SECURITY: Require authentication to initiate calls
      const auth = await verifyAuth(req);
      if (!auth.authenticated) {
        console.log("[dialer] Unauthorized attempt to initiate call");
        return new Response(JSON.stringify({ 
          success: false,
          error: "Authentication required to initiate calls",
        }), {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      if (!number && !ip_address && !room_id) {
        return new Response(JSON.stringify({ 
          success: false,
          error: "Number, IP address, or room ID required",
        }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      try {
        console.log(`[dialer] User ${auth.userId} initiating call via /initiate_call`);
        const response = await fetch(`${SARAH_API_URL}/initiate_call`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            number: number || ip_address || room_id,
            user_id: auth.userId,
          }),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            success: true,
            call_id: data.call_id,
            status: data.status || "initiated",
            message: data.message,
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
        
        const errorData = await response.json().catch(() => ({}));
        return new Response(JSON.stringify({ 
          success: false,
          error: errorData.error || `Call initiation failed (${response.status})`,
        }), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      } catch (e) {
        console.error("[dialer] Initiate call error:", e);
      }
      
      return new Response(JSON.stringify({ 
        success: false,
        error: "VoIP not available",
        message: "Call feature is currently unavailable",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
    // =========================================================================
    // END CALL (REQUIRES AUTHENTICATION)
    // =========================================================================
    if (action === "end") {
      // SECURITY: Require authentication to end calls
      const auth = await verifyAuth(req);
      if (!auth.authenticated) {
        console.log("[dialer] Unauthorized attempt to end call");
        return new Response(JSON.stringify({ 
          success: false,
          error: "Authentication required to end calls",
        }), {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      console.log(`[dialer] User ${auth.userId} ending call`);
      return new Response(JSON.stringify({ success: true, message: "Call ended" }), {
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
    console.error("[dialer] Error:", error);
    return new Response(
      JSON.stringify({ error: "An error occurred processing your request" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

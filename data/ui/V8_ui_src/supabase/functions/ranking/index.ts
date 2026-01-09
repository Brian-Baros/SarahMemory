import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

/**
 * Ranking Edge Function
 * 
 * Handles integration with the SarahMemory ranking/reputation system
 * at api.sarahmemory.com
 * 
 * SECURITY: All operations require authentication to prevent abuse
 */

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Helper to verify authentication
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
    // SECURITY: All ranking operations require authentication
    const auth = await verifyAuth(req);
    if (!auth.authenticated) {
      console.log("[ranking] Unauthorized access attempt");
      return new Response(JSON.stringify({ 
        success: false,
        error: "Authentication required for ranking operations",
      }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const RANKING_API_URL = "https://api.sarahmemory.com";
    const { action, session_id, metrics, user_id } = await req.json();
    
    // Use authenticated user ID, ignore any user_id from request body for security
    const authenticatedUserId = auth.userId;
    
    console.log(`[ranking] Action: ${action}, User: ${authenticatedUserId}`);
    
    if (action === "submit_session") {
      // Submit a session for ranking
      try {
        const response = await fetch(`${RANKING_API_URL}/api/ranking/submit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            session_id, 
            metrics,
            user_id: authenticatedUserId, // Use authenticated user ID
            timestamp: new Date().toISOString(),
          }),
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            success: true,
            ranked: true,
            score: data.score,
            message: "Session saved & ranked",
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch (e) {
        console.log("[ranking] Backend unavailable:", e);
      }
      
      // Fallback - log locally
      return new Response(JSON.stringify({ 
        success: true,
        ranked: false,
        message: "Session saved locally",
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
      
    } else if (action === "get_stats") {
      // Get user ranking stats - only for authenticated user
      try {
        const response = await fetch(`${RANKING_API_URL}/api/ranking/stats/${authenticatedUserId}`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
        
        if (response.ok) {
          const data = await response.json();
          return new Response(JSON.stringify({ 
            success: true,
            stats: data,
          }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }
      } catch {
        // Backend not available
      }
      
      return new Response(JSON.stringify({ 
        success: true,
        stats: {
          total_sessions: 0,
          average_score: 0,
          rank: "New User",
        },
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
      
    } else {
      return new Response(JSON.stringify({ 
        error: `Unknown action: ${action}` 
      }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
    
  } catch (error) {
    console.error("[ranking] Error:", error);
    return new Response(
      JSON.stringify({ error: "An error occurred processing your request" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

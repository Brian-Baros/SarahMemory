// SARAHMEMORY_PATCH_NOTE 2026-06-24:
// Supabase is an optional cloud helper, not a V9 local runtime dependency.
// The old generated client could throw at import-time if env values were
// missing, producing a blank white UI before React rendered. This wrapper keeps
// imports safe offline and only allows cloud function calls when explicitly
// enabled by VITE_ENABLE_SUPABASE_FALLBACK=true.
import { createClient } from '@supabase/supabase-js';
import type { Database } from './types';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
const SUPABASE_PUBLISHABLE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || '';
const SUPABASE_ALLOWED = String(import.meta.env.VITE_ENABLE_SUPABASE_FALLBACK || 'false').toLowerCase() === 'true';

function createOfflineSupabaseStub() {
  const invoke = async (name: string) => {
    throw new Error(`Supabase cloud fallback disabled for local-first SarahMemory runtime: ${name}`);
  };
  return {
    functions: { invoke },
    auth: {
      getSession: async () => ({ data: { session: null }, error: null }),
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }),
      signOut: async () => ({ error: null }),
    },
  } as any;
}

export const supabase = (SUPABASE_ALLOWED && SUPABASE_URL && SUPABASE_PUBLISHABLE_KEY)
  ? createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
      auth: {
        storage: typeof localStorage !== 'undefined' ? localStorage : undefined,
        persistSession: true,
        autoRefreshToken: true,
      }
    })
  : createOfflineSupabaseStub();

/*--==The SarahMemory Project==--
File: BASE_DIR/data/ui/app.js
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-06
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
===============================================================================
*/

/* === SM_BRIDGE_BASE_AUTODETECT_V1 === */
(function(){
  try {
    var apiBase = location.origin;
    var host = (location.hostname || "").toLowerCase();
    var port = (location.port || "");

    // Cloud deployment: any sarahmemory.com host should talk to the PythonAnywhere API host.
    // This keeps GoogieHost (static UI) and PythonAnywhere (Flask API) in sync while letting
    // us serve the SAME files everywhere.
    if (/sarahmemory\.com$/i.test(host)) {
      apiBase = "https://api.sarahmemory.com";
    }

    // If you're running the UI on 8080 locally, point the API to Flask on 5000
    if ((host === "127.0.0.1" || host === "localhost") && (port === "8080")) {
      apiBase = "http://127.0.0.1:5000";
    }

    // If opened from file:// or any non-http origin, default to local Flask
    if (location.protocol !== "http:" && location.protocol !== "https:") {
      apiBase = "http://127.0.0.1:5000";
    }

    // Normalize to use the HTTP API under /api on whatever host we're on.
    // Example:
    //   file://                → http://127.0.0.1:5000/api
    //   http://127.0.0.1:8080  → http://127.0.0.1:5000/api
    //   https://ai.sarahmemory.com → https://ai.sarahmemory.com/api
    apiBase = (apiBase || '').replace(/\/+$/, '');
    if (!/\/api$/i.test(apiBase)) {
      apiBase = apiBase + '/api';
    }

    // Expose for other scripts
    window.SM_API_BASE = apiBase;

    // If the page already defines API_BASE, align it
    try { window.API_BASE = apiBase; } catch (e) {}

// ============================================================================
// Phase B: Authentication State
// ============================================================================
let authToken = localStorage.getItem('sarahmemory_token') || null;
let currentUser = JSON.parse(localStorage.getItem('sarahmemory_user') || 'null');


    // Helper to build API URLs safely (keeps existing callers intact if they adopt it)
    window.SM_getApi = function (path) {
      path = String(path || "");
      if (!path.startsWith("/")) path = "/" + path;
      return apiBase + path;
    };
  } catch (err) {
    console.warn("SM bridge autodetect failed:", err);
  }
})();
/* === /SM_BRIDGE_BASE_AUTODETECT_V1 === */


// pywebview bridge helpers
const hasBridge = () => typeof window.pywebview !== 'undefined' && window.pywebview.api;
const api = () => window.pywebview.api;

// === ENV & Path Resolver (single-file, works local + server) ==================
var API_BASE = (typeof window !== 'undefined' && window.SM_API_BASE)
  ? window.SM_API_BASE
  : '/api';

// --- Health polling for boot readiness ---
async function SM_pollHealth(timeoutMs = 20000, intervalMs = 800) {
  const t0 = Date.now();

  // Prefer the unified helper if available
  const healthUrl = (typeof window !== 'undefined' && window.SM_getApi)
    ? window.SM_getApi('/health')
    : (
        typeof API_BASE !== 'undefined'
          ? String(API_BASE).replace(/\/+$/, '') + '/health'
          : '/api/health'
      );

  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(healthUrl, { credentials: 'same-origin' });
      if (r.ok) {
        const j = await r.json();
        if (j && (j.running || j.ok || j.ok === true)) {
          return true;
        }
      }
    } catch (e) {
      // ignore and keep polling
    }
    await new Promise(r => setTimeout(r, intervalMs));
  }

  return false;
}


const ENV = (() => {
  const host = (location.hostname||'').toLowerCase();
  const isWeb = /^https?:/.test(location.protocol);
  const isServer = /sarahmemory\.com$/.test(host) || host === 'api.sarahmemory.com' || host === 'www.sarahmemory.com';
  const onAPI = host === 'api.sarahmemory.com';
  const base = isServer ? (onAPI ? 'https://api.sarahmemory.com' : 'https://www.sarahmemory.com/api') : '';
  const uiBase = isServer ? (onAPI ? 'https://api.sarahmemory.com/data/ui' : 'https://www.sarahmemory.com/api/data/ui') : '';
  // app.js
var API_BASE = (typeof window !== 'undefined' && window.SM_API_BASE) ? window.SM_API_BASE : '/api';
// fetch(`${API_BASE}/health`) ...
  return { isServer, onAPI, base, uiBase };
})();

function SM_urlFor(pathLike){
  if (!pathLike) return '';
  if (/^https?:\/\//i.test(pathLike)) return pathLike;
  if (ENV.isServer) return (ENV.base + '/' + String(pathLike).replace(/^\/+/, ''));
  // Local: relative path works under file:// or localhost with pywebview
  return String(pathLike);
}

function SM_ui(file){
  if (!file) return '';
  if (ENV.isServer) return (ENV.uiBase + '/' + String(file).replace(/^\/+/, ''));
  // Local: UI assets live alongside index.html (C:\SarahMemory\data\ui\)
  return String(file);
}

// Expose minimal resolver to the window for other scripts (non-breaking):
window.SM_RESOLVE = Object.freeze({
  apiBase: () => ENV.base,
  ui: (f) => SM_ui(f),
  url: (p) => SM_urlFor(p),
  env: ENV
});
// ==============================================================================





// --- Auto-launch SarahMemoryMain.py (server-only, safe & idempotent) ----------
function SM_shouldAutoLaunch(){
  if (!ENV.isServer) return false;
  const p = (location.pathname || '').toLowerCase();
  // target only /data/ui/SarahMemory.html or /data/ui/index.html (any host you specified)
  return /\/data\/ui\/(sarahmemory\.html|index\.html)$/.test(p);
}

async function SM_tryServerLaunch(){
  // Allow manual override with ?autostart=0
  try {
    const sp = new URLSearchParams(location.search);
    if (sp.get('autostart') === '0') return;

    // Avoid repeated triggers
    const GUARD_KEY = 'sm_autolaunch_v2';
    if (localStorage.getItem(GUARD_KEY) === '1') return;

    // Prefer native bridge if this runs in the desktop wrapper
    if (typeof window.pywebview !== 'undefined' && window.pywebview.api){
      try{
        const b = window.pywebview.api;
        // Try several likely bridge methods (non-breaking)
        if (b.launch_sarah_main) { await b.launch_sarah_main(); }
        else if (b.run_python)   { await b.run_python('SarahMemoryMain.py'); }
        else if (b.run_process)  { await b.run_process('python', ['SarahMemoryMain.py']); }
        localStorage.setItem(GUARD_KEY, '1');
        console.log('[SM] Launched via native bridge.');
        return;
      }catch(e){ console.warn('[SM] Bridge launch not available:', e); }
    }

    // Server-side attempts (no new server files; best-effort)
    // ENV.base resolves to:
    //   www host → https://www.sarahmemory.com/api
    //   api host → https://api.sarahmemory.com
    const base = ENV.base;

    // STRICT payload: only runs "python|python3|venv/bin/python" with SarahMemoryMain.py
    // Your server must already expose one of these endpoints.
    const payloads = [
      { cmd: 'python',  args: ['SarahMemoryMain.py'] },
      { cmd: 'python3', args: ['SarahMemoryMain.py'] },
      { cmd: './venv/bin/python', args: ['SarahMemoryMain.py'] },   // linux venv (if mapped)
      { cmd: 'venv\\Scripts\\python.exe', args: ['SarahMemoryMain.py'] } // windows-style if applicable
    ];

    const endpoints = [
      `${base}/launch`,
      `${base}/run`,
      `${base}/start`,
      `${base}/_start`,
      `${base}/wsgi/launch`,
      `${base}/api/launch`
    ];

    // Try all combinations; mark done on first 2xx
    for (const url of endpoints){
      for (const body of payloads){
        try{
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Add auth header here if you secure these routes:
            // headers: { 'Content-Type':'application/json', 'X-API-Key': '...' }
            body: JSON.stringify(body),
            credentials: 'same-origin' // keep cookies if any
          });
          if (res.ok) {
            localStorage.setItem(GUARD_KEY, '1');
            console.log('[SM] Server launch requested via', url, '→', body.cmd);
            return;
          }
        }catch(e){
          // Try next combo silently
        }
      }
    }

    // Last nudge: wake Passenger/WSGI (lazy spawn) by touching base
    try { await fetch(base + '/', { method:'GET', mode:'no-cors' }); } catch {}
    // Do not set the guard here; we didn't get an affirmative OK
  } catch(e){
    console.warn('[SM] server autolaunch skipped:', e);
  }
}
// --- /Auto-launch block ---

// === Rail State Helpers (persisted) ===========================================
function SM_setRailCollapsed(whichId, collapsed){
  try{
    const app = document.getElementById('app');
    const rail = document.getElementById(whichId);
    if (!rail || !app) return;
    if (collapsed){
      rail.classList.add('collapsed');
      if (whichId==='left-rail') app.classList.add('left-collapsed');
      if (whichId==='right-rail') app.classList.add('right-collapsed');
    }else{
      rail.classList.remove('collapsed');
      if (whichId==='left-rail') app.classList.remove('left-collapsed');
      if (whichId==='right-rail') app.classList.remove('right-collapsed');
    }
    localStorage.setItem(whichId+'_collapsed', collapsed ? '1':'0');
  }catch{}
}
function SM_getRailCollapsed(whichId, defVal=false){
  try{
    const v = localStorage.getItem(whichId+'_collapsed');
    if (v===null || v===undefined) return !!defVal;
    return v==='1';
  }catch{ return !!defVal; }
}
// ==============================================================================

// DOM refs

// [ADDED] One-time audio unlock to satisfy autoplay policies
(function audioUnlockOnce(){
  if (window.__audio_unlocked__) return;
  function unlock(){
    try{
      if (window.speechSynthesis) { /* gesture occurred; synth is allowed */ }
      window.__audio_unlocked__ = true;
      document.removeEventListener('click', unlock);
      document.removeEventListener('keydown', unlock);
      document.removeEventListener('touchstart', unlock);
    }catch{}
  }
  document.addEventListener('click', unlock, {once:true});
  document.addEventListener('keydown', unlock, {once:true});
  document.addEventListener('touchstart', unlock, {once:true});
})();
const messages = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const fileEl = document.getElementById('file');
const fileChip = document.getElementById('fileChip');
const form = document.getElementById('composer');
const threadDate = document.getElementById('thread-date');
const threadList = document.getElementById('thread-list');
const snapEl = document.getElementById('snapshot');

// Reply/Compare toggles (support right-rail IDs or compact settings IDs)
const toggleReply = document.getElementById('toggle-reply-status') || document.getElementById('optReply');
const toggleCompare = document.getElementById('toggle-compare') || document.getElementById('optCompare');

const remList = document.getElementById('reminders');
const remForm = document.getElementById('new-reminder');
const micBtn = document.getElementById('mic');



// === [ADDED] Web Speech TTS Integration (non-destructive) ===
const voiceToggleBtn = document.getElementById('voiceToggle');
/* ==========================================================================
   /* TTS ENGINE (Web Speech API) - v1.0
   - One-click enable/disable via "Voice: On" button
   - Remembers voice & rate in localStorage
   - Pauses SR (speech recognition) while speaking to avoid echo
   ========================================================================== */
const TTS = (() => {
  const store = {
    get enabled(){ return localStorage.getItem('voice_enabled') !== 'false'; },
    set enabled(v){ localStorage.setItem('voice_enabled', v ? 'true' : 'false'); },
    get voice(){ return localStorage.getItem('voice_name') || ''; },
    set voice(name){ localStorage.setItem('voice_name', name||''); },
    get rate(){ const r = parseFloat(localStorage.getItem('voice_rate') || '1'); return (r>0 && r<=2)?r:1; },
    set rate(v){ localStorage.setItem('voice_rate', String(v)); }
  };

  let voices = [];
  let speaking = false;
  const q = []; // simple FIFO queue

  function loadVoices() {
    voices = window.speechSynthesis.getVoices() || [];
    // Auto-pick a sensible default the first time
    if (!store.voice && voices.length) {
      const pref = voices.find(v => /en(-|_)?US/i.test(v.lang) && /female|zira|sarah|jenny/i.test(v.name)) ||
                   voices.find(v => /en(-|_)?US/i.test(v.lang)) ||
                   voices[0];
      if (pref) store.voice = pref.name;
    }
  }
  if ('speechSynthesis' in window) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    document.addEventListener('visibilitychange', () => {
      // some browsers only populate voices after visibility changes
      if (document.visibilityState === 'visible') loadVoices();
    });
  }

  function setButtonLabel() {
    if (!voiceToggleBtn) return;
    voiceToggleBtn.textContent = 'Voice: ' + (store.enabled ? 'On' : 'Off');
  }

  async function ensureReady() {
    if (!('speechSynthesis' in window)) throw new Error('Speech Synthesis API not supported');
    // on some browsers voices populate async
    if (!voices.length) {
      loadVoices();
      for (let i=0;i<10 && !voices.length;i++) {
        await new Promise(r=>setTimeout(r,100));
        voices = window.speechSynthesis.getVoices() || [];
      }
    }
  }

  function _pickVoice() {
    if (!voices.length) return null;
    if (store.voice) {
      const match = voices.find(v => v.name === store.voice);
      if (match) return match;
    }
    return voices[0] || null;
  }

  function _speakNow(utter) {
    return new Promise(resolve => {
      utter.onend = () => resolve();
      utter.onerror = () => resolve();
      try { window.speechSynthesis.speak(utter); } catch { resolve(); }
    });
  }

  async function speak(text, opts={}) {
    if (!text || !text.trim()) return;
    try { await ensureReady(); } catch { return; }
    if (!store.enabled) return;

    // Abort SR during TTS to avoid feedback/echo
    try { if (_sr && _srOn) { _sr.abort(); _srOn = false; micBtn?.classList.remove('active'); } } catch {}

    const voice = _pickVoice();
    const u = new SpeechSynthesisUtterance(text);
    if (voice) u.voice = voice;
    u.rate = Math.max(0.5, Math.min(2, opts.rate ?? store.rate));
    u.pitch = Math.max(0.5, Math.min(2, opts.pitch ?? 1));
    u.volume = Math.max(0, Math.min(1, opts.volume ?? 1));

    // queue handling
    q.push(u);
    if (speaking) return;
    speaking = true;
    while (q.length) {
      const next = q.shift();
      await _speakNow(next);
    }
    speaking = false;
  }

  function stop() {
    try { window.speechSynthesis.cancel(); } catch {}
    speaking = false;
    q.length = 0;
  }

  function toggleEnabled() {
    store.enabled = !store.enabled;
    if (!store.enabled) stop();
    setButtonLabel();
  }

  // Button wiring is managed by host app to avoid double listeners.
  // Host should call TTS.enable(true/false) and TTS.setButtonLabel() inside its own toggle handler.

  return { speak, stop, setButtonLabel, enable:(v)=>{ store.enabled=!!v; if(!store.enabled) stop(); }, get enabled(){return store.enabled;}, set voiceName(n){store.voice=n;}, get voiceName(){return store.voice;}, set rate(v){store.rate=v;}, get rate(){return store.rate;} };
})();
window.TTS = TTS;
// === [END ADDED TTS] ===
// Collapsible rails
document.querySelectorAll('.collapse').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.getAttribute('data-target');
    const rail = document.getElementById(id);
    if (rail) {
      rail.classList.toggle('collapsed');
      toggleAppColumn(id);
      // persist new state
      try {
        const collapsed = rail.classList.contains('collapsed');
        localStorage.setItem(id + '_collapsed', collapsed ? '1' : '0');
      } catch (e) { /* ignore storage errors */ }
    }
  });
});

// Collapsible sections (Contacts/Reminders)
document.querySelectorAll('.section-toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    const sec = btn.closest('.rail-section');
    if (sec) {
      sec.classList.toggle('collapsed');
    }
  });
});

function toggleAppColumn(targetId){
  const app = document.getElementById('app');
  if(!app) return;
  if(targetId==='left-rail'){ app.classList.toggle('left-collapsed'); }
  if(targetId==='right-rail'){ app.classList.toggle('right-collapsed'); }
}
// File chip count
fileEl.addEventListener('change', () => {
  const n = fileEl.files?.length || 0;
  fileChip.textContent = n ? `${n} file${n>1?'s':''} selected` : 'Choose files';
});

// Add a message bubble
function addMsg(role, text, meta) {
  const row = document.createElement('div');
  row.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text || '';
  row.appendChild(bubble);
  messages.appendChild(row);

  if (meta && meta.source) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = `[Source: ${meta.source}]` + (meta.intent ? ` (Intent: ${meta.intent})` : '');
    bubble.appendChild(m);
    if (toggleReply && !toggleReply.checked) m.style.display = 'none';
  }
  messages.scrollTop = messages.scrollHeight;

  // [ADDED] Speak assistant replies via TTS
  try { if (role === 'assistant' && (typeof TTS !== 'undefined') && TTS.enabled) { TTS.speak(String(text||'')); } } catch(e) { /* no-tts */ }
}

// Convert files to base64 payloads
async function filesToPayloads(files) {
  const out = [];
  for (const f of files) {
    const b = await f.arrayBuffer();
    const base64 = btoa(String.fromCharCode(...new Uint8Array(b)));
    out.push({ name: f.name, type: f.type, size: f.size, data: base64 });
  }
  return out;
}

// Status LEDs
function setLED(stateText, route) {
  document.getElementById('status-text').textContent = stateText || '';
  const L = document.getElementById('led-local');
  const W = document.getElementById('led-web');
  const A = document.getElementById('led-api');
  const N = document.getElementById('led-net');
  [L,W,A,N].forEach(d => d.classList.remove('blink'));
  if (route === 'local') L.classList.add('blink');
  else if (route === 'web') W.classList.add('blink');
  else if (route === 'api') A.classList.add('blink');
  else [L,W,A,N].forEach(d => d.classList.remove('blink'));
}

// Composer submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
  const text = promptEl.value.trim();
  if (!text) return;
  addMsg('user', text);
  promptEl.value = '';
  const blobs = await filesToPayloads(fileEl.files || []);
  fileEl.value = '';
  fileChip.textContent = 'Choose files';

  // [ADDED] Visual color query fast-path (local + backend)
  // [ADDED] Visual OCR query fast-path (local via pywebview)
  try {
    const handledOCR = await __SM_visualOCRQuery(text);
    if (handledOCR) { setLED('Done','local'); setTimeout(()=>setLED('Ready',''), 900); return; }
  } catch(e) { console.warn(e); }

  try {
    const handled = await __SM_visualColorQuery(text, blobs);
    if (handled) { setLED('Done','local'); setTimeout(()=>setLED('Ready',''), 900); return; }
  } catch(e) { console.warn(e); }
setLED('Thinking...', 'api');
  try {
    if (hasBridge()) {
      const bridge = api();
      const fn = bridge.send_message || bridge.send_text;
      const out = await fn.call(bridge, text, blobs);
      const meta = (out && out.meta) || {};
      const src = (meta.source || '').toLowerCase();
      addMsg('assistant', (out && (out.response||out.reply)) ? (out.response||out.reply) : '(no response)', meta);
      setLED('Done', src.includes('local')?'local':src.includes('web')?'web':'api');
      setTimeout(()=>setLED('Ready',''), 900);
    } else {
      // HTTP fallback for pure Web UI (no native pywebview bridge)
      (async () => {
        try {
          // First try the health endpoint so we can keep the old
          // "Bridge not ready (Preview mode)" behaviour when the
          // PythonAnywhere API really is offline.
          let up = false;
          try {
            up = await SM_pollHealth(2500, 500);
          } catch (e) {
            // If /health is missing or CORS-blocked, we'll still try
            // a direct chat call below.
          }
          if (!up) {
            addMsg('assistant', 'Bridge not ready. (Preview mode)');
            setLED('Ready', '');
            return;
          }

          const url = (typeof window !== 'undefined' && window.SM_getApi)
            ? window.SM_getApi('/chat')
            : '/api/chat';

          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, files: blobs }),
            credentials: 'include'
          });

          if (!res.ok) {
            addMsg('assistant', `API error (${res.status}) while contacting SarahMemory.`);
            setLED('Error', 'api');
            return;
          }

          const out = await res.json();
          const meta = (out && out.meta) || {};
          const src = (meta.source || '').toLowerCase();

          addMsg(
            'assistant',
            (out && (out.response || out.reply || out.text))
              ? (out.response || out.reply || out.text)
              : '(no response)',
            meta
          );

          setLED('Done', src.includes('local') ? 'local'
                          : (src.includes('web') ? 'web' : 'api'));
          setTimeout(() => setLED('Ready', ''), 900);
        } catch (e) {
          console.error('HTTP chat error', e);
          addMsg('assistant', 'Could not reach SarahMemory API from browser.');
          setLED('Error', 'api');
        }
      })();
    }
  } catch (err) {
    addMsg('assistant', 'Error: ' + err);
    setLED('Error','');
  }
});

// Toggle bindings
function bindReplyToggle(el){
  if(!el) return;
  el.addEventListener('change', () => {
    [...document.querySelectorAll('.meta')].forEach(m => m.style.display = el.checked ? '' : 'none');
    if (hasBridge()) api().set_flag('REPLY_STATUS', el.checked);
  });
}
bindReplyToggle(toggleReply);
function bindCompareToggle(el){
  if(!el) return;
  el.addEventListener('change', () => {
    if (hasBridge()) api().set_flag('API_RESPONSE_CHECK_TRAINER', el.checked);
  });
}
bindCompareToggle(toggleCompare);

// Boot state & initial load
(async function boot(){
  try { if (SM_shouldAutoLaunch()) { SM_tryServerLaunch(); } } catch(e) { console.warn(e); }
  // Default: collapse LEFT rail on first load (persisted)
  try{
    const leftPref = localStorage.getItem('left-rail_collapsed');
    if (leftPref===null){ SM_setRailCollapsed('left-rail', true); }
    else { SM_setRailCollapsed('left-rail', leftPref==='1'); }
    const rightPref = localStorage.getItem('right-rail_collapsed');
    if (rightPref!==null){ SM_setRailCollapsed('right-rail', rightPref==='1'); }
  }catch{}
  try{ if(window.TTS && typeof voiceEnabled !== 'undefined'){ TTS.enable(voiceEnabled); TTS.setButtonLabel && TTS.setButtonLabel(); } }catch(e){}

  try {
    if (hasBridge()) {
      const state = await api().get_boot_state?.() || {};
      if(toggleReply) toggleReply.checked = !!(state.REPLY_STATUS || state.reply_status);
      if(toggleCompare) toggleCompare.checked = !!(state.API_RESPONSE_CHECK_TRAINER || state.compare_trainer);

      const listFn = api().list_threads_for_date || api().list_threads || api().listThreads;
      const th = listFn ? await listFn.call(api(), state.today || null) : [];
      threadList.innerHTML = '';
      renderThreadGroups(th || []);

      const snap = await api().get_snapshot?.();
      if (snap && snap.data_url) snapEl.src = snap.data_url;
    }
  } catch (e) { console.warn(e); }

  addMsg('assistant', "Hi! I'm Sarah — ready when you are. Try asking me anything.");

  // [ADDED] Sync voice toggle label to persisted state
  try { if (window.TTS && TTS.setButtonLabel) TTS.setButtonLabel(); } catch(e) {}
})();

// Periodic snapshot refresh
setInterval(async () => {
  if (!hasBridge()) return;
  try {
    const snap = await api().get_snapshot?.();
    if (snap && snap.data_url) snapEl.src = snap.data_url;
  } catch {}
}, 4000);

// Reminders
remForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
  const title = document.getElementById('rem-title').value.trim();
  const when = document.getElementById('rem-when').value;
  const note = document.getElementById('rem-note').value.trim();
  if (!title || !when) return;
  if (hasBridge()) {
    try{ await api().create_reminder?.(title, when, note); }catch{}
    try{
      const items = await api().list_reminders?.();
      renderRem(items || []);
    }catch{}
  }
  remForm.reset();
});
function renderRem(items) {
  remList.innerHTML = '';
  (items || []).forEach(r => {
    const div = document.createElement('div');
    div.className = 'rem';
    div.textContent = `${r.when} — ${r.title}` + (r.note ? ` (${r.note})` : '');
    remList.appendChild(div);
  });
}

// Webcam handling
let _camStream=null;
async function ensureWebcam(){
  if(_camStream && _camStream.active) return _camStream;
  try{
    _camStream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    document.getElementById('webcam').srcObject=_camStream;
    document.getElementById('webcamToggle').textContent='Webcam: On';
  }catch(e){
    console.warn('Webcam not available', e);
    document.getElementById('webcamToggle').textContent='Webcam: Off';
  }
  return _camStream;
}
async function toggleWebcam(){
  const s = await ensureWebcam();
  if(!s) return;
  const tracks = s.getVideoTracks();
  const enabled = !(tracks[0]?.enabled);
  tracks.forEach(t => t.enabled = enabled);
  document.getElementById('webcamToggle').textContent='Webcam: ' + (enabled?'On':'Off');
}
document.getElementById('webcamToggle')?.addEventListener('click', toggleWebcam);
ensureWebcam();

// Mic / SR: one permission, click-to-capture (no constant loops)
let _micStream=null, _sr=null, _srOn=false;
async function ensureMicOnce(){
  if(_micStream && _micStream.active) return _micStream;
  try{
    _micStream = await navigator.mediaDevices.getUserMedia({audio:true, video:false});
    return _micStream;
  }catch(e){ console.warn('mic permission failed', e); return null; }
}
async function startSROnceAndSend(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){ return; }
  await ensureMicOnce();
  _sr = new SR(); _sr.lang='en-US'; _sr.continuous=false; _sr.interimResults=false;
  _sr.onstart=()=>{ _srOn=true; micBtn?.classList.add('active'); };
  _sr.onend=()=>{ _srOn=false; micBtn?.classList.remove('active'); };
  _sr.onresult=(e)=>{ const t=e.results[0]?.[0]?.transcript||''; if(!t) return; promptEl.value=t.trim(); try{ form.requestSubmit(); }catch{} };
  try{ _sr.start(); }catch{}
}
micBtn?.addEventListener('click', startSROnceAndSend);
document.getElementById('micToggle')?.addEventListener('click', async ()=>{
  const s = await ensureMicOnce();
  if(!s) return;
  const a = s.getAudioTracks();
  const enabled = !(a[0]?.enabled);
  a.forEach(t => t.enabled = enabled);
  document.getElementById('micToggle').textContent='Microphone: ' + (enabled?'On':'Muted');
});

// Network state poller
async function pollNetwork(){
  if(!hasBridge() || !api().network_state) return;
  try{
    const s = await api().network_state();
    if (s && typeof s.mode === 'string') { const mt=document.getElementById('mode-text'); if(mt) mt.textContent = s.mode; }
    const remote = document.getElementById('remote');
    const snap = document.getElementById('snapshot');
    if (s && s.call_active){ remote.hidden=false; snap.hidden=true; }
    else { remote.hidden=true; snap.hidden=false; }
    document.getElementById('led-net')?.classList.toggle('blink', !!(s && s.busy));
  }catch{}
}
setInterval(pollNetwork, 1200);

// Contacts
const contactsEl = document.getElementById('contacts');
function loadContactsLocal(){ try{ return JSON.parse(localStorage.getItem('sm_contacts')||'[]'); }catch(e){ return []; } }
function saveContactsLocal(list){ localStorage.setItem('sm_contacts', JSON.stringify(list)); }
function renderContacts(list){
  contactsEl.innerHTML='';
  list.forEach((c,idx)=>{
    const div=document.createElement('div'); div.className='contact';
    div.innerHTML=`<div><strong>${c.name||'Contact'}</strong><br/><small>${c.addr||''}</small></div>`;
    const act=document.createElement('div'); act.className='actions';
    const call=document.createElement('button'); call.textContent='Call'; call.addEventListener('click', async ()=>{ try{ if(hasBridge() && api().start_network_chat) await api().start_network_chat(c.addr); }catch{} });
    const del=document.createElement('button'); del.textContent='Remove'; del.addEventListener('click', ()=>{ const arr=loadContactsLocal(); arr.splice(idx,1); saveContactsLocal(arr); renderContacts(arr); });
    act.appendChild(call); act.appendChild(del); div.appendChild(act);
    contactsEl.appendChild(div);
  });
}
renderContacts(loadContactsLocal());
document.getElementById('new-contact')?.addEventListener('submit', async (e)=>{
  e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
  const name=document.getElementById('c-name').value.trim();
  const addr=document.getElementById('c-address').value.trim();
  if(!name||!addr) return;
  const arr=loadContactsLocal(); arr.unshift({name,addr}); saveContactsLocal(arr); renderContacts(arr);
  try{ if(hasBridge() && api().save_contact) await api().save_contact(name,addr); }catch{}
  e.target.reset();
});

// Settings open/close
const settingsBtn = document.getElementById('settingsBtn');
const settingsDlg = document.getElementById('settings');
const settingsClose = document.getElementById('settingsClose');
settingsBtn?.addEventListener('click', ()=>{ settingsDlg.hidden = false; });
settingsClose?.addEventListener('click', ()=>{ settingsDlg.hidden = true; });

// Settings: theme + voice
const themeSelect = document.getElementById('themeSelect');
const voiceSelect = document.getElementById('voiceSelect');
async function bootSettings(){
  try{
    if(hasBridge() && api().get_themes){ const list=await api().get_themes(); themeSelect.innerHTML=''; (list||[]).forEach(n=>{ const o=document.createElement('option'); o.value=n; o.textContent=n; themeSelect.appendChild(o); }); }
    if(hasBridge() && api().list_voices){ const voices=await api().list_voices(); voiceSelect.innerHTML=''; (voices||[]).forEach(v=>{ const o=document.createElement('option'); o.value=v.id||v.name; o.textContent=v.name||v.id; voiceSelect.appendChild(o); }); }
  }catch{}
}
themeSelect?.addEventListener('change', async ()=>{
  try{
    if(hasBridge() && api().set_theme) await api().set_theme(themeSelect.value);
    document.getElementById('themeLink')?.setAttribute('href','unified-theme.css?ts='+Date.now());
  }catch{}
});
document.getElementById('voiceRate')?.addEventListener('input', async (e)=>{ try{ if(hasBridge() && api().set_voice) await api().set_voice({rate:parseFloat(e.target.value)}); }catch{} });
document.getElementById('voicePitch')?.addEventListener('input', async (e)=>{ try{ if(hasBridge() && api().set_voice) await api().set_voice({pitch:parseFloat(e.target.value)}); }catch{} });
document.getElementById('voiceVol')?.addEventListener('input', async (e)=>{ try{ if(hasBridge() && api().set_voice) await api().set_voice({volume:parseFloat(e.target.value)}); }catch{} });
voiceSelect?.addEventListener('change', async (e)=>{ try{ if(hasBridge() && api().set_voice) await api().set_voice({voice:e.target.value}); }catch{} });
document.getElementById('openAvatar')?.addEventListener('click', async ()=>{ try{ if(hasBridge() && api().open_avatar_panel) await api().open_avatar_panel(); }catch{} });
bootSettings();

// Route mode
const routeSel = document.getElementById('routeMode');
routeSel?.addEventListener('change', async ()=>{
  const v = routeSel.value || 'Any';
  const mt=document.getElementById('mode-text'); if(mt) mt.textContent = v;
  try{ if(hasBridge()) await api().set_flag('route_mode', v); }catch{}
});

function updateStatusbarSpace(){
  const sb=document.getElementById('statusbar');
  if(!sb) return;
  const h = sb.offsetHeight || 40;
  document.documentElement.style.setProperty('--sb-h', (h+6)+'px');
}
window.addEventListener('load', updateStatusbarSpace);
window.addEventListener('resize', updateStatusbarSpace);

// Thread grouping
function groupByDate(items){
  const now=new Date(); const todayStr=now.toISOString().slice(0,10);
  const lastWeek = new Date(now); lastWeek.setDate(now.getDate()-7);
  const groups={today:[], week:[], older:[]};
  (items||[]).forEach(t=>{
    const ts = t.timestamp ? new Date(t.timestamp) : now;
    if (ts.toISOString().slice(0,10)===todayStr) groups.today.push(t);
    else if (ts>=lastWeek) groups.week.push(t);
    else groups.older.push(t);
  });
  return groups;
}
function renderThreadGroups(items){
  const list=document.getElementById('thread-list');
  list.classList.add('thread-groups');
  list.innerHTML='';
  const g=groupByDate(items);
  function addSection(label, arr){
    if(!arr.length) return;
    const hdr=document.createElement('div'); hdr.className='group-hdr'; hdr.textContent=label;
    list.appendChild(hdr);
    arr.forEach(t=>{
      const d=document.createElement('div'); d.className='thread';
      d.innerHTML = `<div class="title">${t.title||t.id||'Thread'}</div><small>${t.timestamp||''}</small>`;
      d.addEventListener('click', ()=>{ try{ __sarah_open_thread_fallback(t.id); }catch(e){} });
      list.appendChild(d);
    });
  }
  addSection('Today', g.today);
  addSection('Previous week', g.week);
  addSection('Older', g.older);
}


// Voice toggle
const voiceToggle = document.getElementById('voiceToggle');
let voiceEnabled = true;
function toggleVoice(){
  try{ if(window.TTS && typeof voiceEnabled !== 'undefined'){ TTS.enable(!voiceEnabled); TTS.setButtonLabel && TTS.setButtonLabel(); } }catch(e){}
  try{ if(window.TTS){ localStorage.setItem('voice_enabled', (!voiceEnabled).toString()); } }catch(e){} voiceEnabled = !voiceEnabled;
  if (voiceToggle) voiceToggle.textContent = 'Voice: ' + (voiceEnabled ? 'On' : 'Silent');
  try { if (hasBridge() && api().set_flag) api().set_flag('VOICE_FEEDBACK_ENABLED', voiceEnabled); } catch {}
}
voiceToggle?.addEventListener('click', toggleVoice);


// ===== UnifiedComms bridge (contacts) =====
async function refreshContactsUI(){
  const box = document.getElementById('contacts'); if(!box) return;
  box.innerHTML = '';
  try{
    const items = hasBridge() ? (await api().telecom_get_contacts()) : [];
    (items||[]).forEach(c=>{
      const row = document.createElement('div');
      row.className = 'thread';
      row.innerHTML = `<strong>${c.name || ''}</strong><small>${c.address || c.email || c.phone || ''}</small>`;
      const del = document.createElement('button'); del.textContent='Delete'; del.className='icon';
      del.addEventListener('click', async()=>{ if(hasBridge()) await api().telecom_delete_contact({id:c.id}); refreshContactsUI(); });
      const call = document.createElement('button'); call.textContent='Video'; call.className='icon';
      call.addEventListener('click', async()=>{ if(hasBridge()) await api().telecom_start_call({peer:c.addr || c.address || c.phone || c.email, video:true}); });
      row.appendChild(document.createElement('br'));
      row.appendChild(call); row.appendChild(del);
      box.appendChild(row);
    });
  }catch(e){ console.warn(e); }
}

document.getElementById('new-contact')?.addEventListener('submit', async (e)=>{
  e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
  const name = document.getElementById('c-name')?.value||'';
  const addr = document.getElementById('c-address')?.value||'';
  if(!name && !addr) return;
  if(hasBridge()) await api().telecom_add_contact({name:name, addr:addr, address:addr});
  e.target.reset();
  refreshContactsUI();
});

refreshContactsUI();


// ===== Calendar + Chat History (HTTP fallback) =====
// If the bridge is unavailable, support server endpoints used by app-nr.js.
async function fetchThreadsByDateHTTP(dateStr){
  const path = dateStr ? `/get_chat_threads_by_date?date=${encodeURIComponent(dateStr)}` : '/get_chat_threads_by_date';
  const url = (typeof window !== 'undefined' && window.SM_getApi) ? window.SM_getApi(path) : path;
  try{
    const res = await fetch(url);
    const data = await res.json();
    const items = (data && data.threads) || [];
    renderThreadGroups(items);
  }catch(e){ console.warn('thread fetch failed', e); }
}

async function fetchConversationByIdHTTP(id){
  try{
    const url = (typeof window !== 'undefined' && window.SM_getApi)
      ? window.SM_getApi(`/get_conversation_by_id?id=${encodeURIComponent(id)}`)
      : `/get_conversation_by_id?id=${encodeURIComponent(id)}`;
    const res = await fetch(url);
    const rows = await res.json();
    // rows expected: [{role:'user'|'assistant', text:'...', meta:'...'}]
    const prevScroll = messages.scrollHeight;
    (rows||[]).forEach(r=> addMsg((r.role==='user')?'user':'assistant', r.text||'', {source: r.meta||''}));
    // Scroll to newest, respecting existing behavior
    messages.scrollTop = messages.scrollHeight || prevScroll;
  }catch(e){ console.warn('conversation fetch failed', e); }
}

// Wire calendar change to either bridge or HTTP fallback
(function bindCalendarChange(){
  const dateEl = document.getElementById('thread-date');
  if(!dateEl) return;
  dateEl.addEventListener('change', async ()=>{
    const d = dateEl.value || null;
    try{
      if (hasBridge() && api().list_threads_for_date){
        const th = await api().list_threads_for_date(d);
        renderThreadGroups(th||[]);
      } else {
        await fetchThreadsByDateHTTP(d||'');
      }
    }catch(e){
      console.warn(e);
      await fetchThreadsByDateHTTP(d||'');
    }
  });
})();

// On first load, if no bridge list available, use HTTP to load threads
window.addEventListener('DOMContentLoaded', async ()=>{
  try{
    if (!(hasBridge() && (api().list_threads_for_date || api().list_threads || api().listThreads))) {
      await fetchThreadsByDateHTTP('');
    }
  }catch(e){ /* noop */ }
});

// When renderThreadGroups creates clickable items, if the bridge can't open, use HTTP detail loader
function __sarah_open_thread_fallback(id){
  if (hasBridge() && api().open_thread){
    try{ api().open_thread(id); return; }catch(e){ /* fallback */ }
  }
  fetchConversationByIdHTTP(id);
}



// ===== System Tools (optional server endpoints) =====
(function bindSystemTools(){
  const btnBackup = document.getElementById('btn-backup');
  const btnRestore = document.getElementById('btn-restore');
  const btnClear = document.getElementById('btn-clear');
  const btnTidy = document.getElementById('btn-tidylogs');
  const cleanRange = document.getElementById('cleanup-range');
  if(btnBackup) btnBackup.addEventListener('click', async ()=>{ try{ await fetch(window.SM_getApi ? window.SM_getApi('/cleanup/backup_all') : '/cleanup/backup_all'); }catch{} });
  if(btnRestore) btnRestore.addEventListener('click', async ()=>{ try{ await fetch(window.SM_getApi ? window.SM_getApi('/cleanup/restore_latest') : '/cleanup/restore_latest'); }catch{} });
  if(btnClear) btnClear.addEventListener('click', async ()=>{
    try{
      const seconds = parseInt((cleanRange && cleanRange.value)||'0',10);
      await fetch(window.SM_getApi ? window.SM_getApi('/cleanup/clear_range') : '/cleanup/clear_range',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({db:'context_history.db', seconds})});
    }catch{}
  });
  if(btnTidy) btnTidy.addEventListener('click', async ()=>{ try{ await fetch(window.SM_getApi ? window.SM_getApi('/cleanup/tidy_logs') : '/cleanup/tidy_logs'); }catch{} });
})();


// ===== Contact Manager (server fallback) =====
async function __contacts_refresh_server(){
  const box = document.getElementById('contacts'); if(!box) return;
  try{
    const res = await fetch(window.SM_getApi ? window.SM_getApi('/get_all_contacts') : '/get_all_contacts');
    const data = await res.json();
    box.innerHTML='';
    (data.contacts||[]).forEach(c=>{
      const row = document.createElement('div'); row.className='contact';
      row.innerHTML = `<div><strong>${c.name||'Contact'}</strong><br/><small>${c.number||c.addr||''}</small></div>`;
      const act=document.createElement('div'); act.className='actions';
      const del=document.createElement('button'); del.textContent='Remove';
      del.addEventListener('click', async ()=>{
        try{ await fetch(window.SM_getApi ? window.SM_getApi('/delete_contact') : '/delete_contact',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id:c.id})}); __contacts_refresh_server(); }catch{}
      });
      act.appendChild(del); row.appendChild(act); box.appendChild(row);
    });
  }catch(e){ /* ignore */ }
}
(function contactsFallbackInit(){
  // Only enable HTTP contacts if bridge contact APIs are missing
  const noBridgeContacts = !(hasBridge() && (api().telecom_get_contacts || api().save_contact));
  if (!noBridgeContacts) return;
  window.addEventListener('DOMContentLoaded', __contacts_refresh_server);
  const addForm = document.getElementById('new-contact');
  if (addForm){
    addForm.addEventListener('submit', async (e)=>{
      e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
      const name=document.getElementById('c-name').value.trim();
      const addr=document.getElementById('c-address').value.trim();
      if(!name||!addr) return;
      try{ await fetch(window.SM_getApi ? window.SM_getApi('/add_contact') : '/add_contact',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, number:addr})}); }catch{}
      try{ __contacts_refresh_server(); }catch{}
      try{ e.target.reset(); }catch{}
    });
  }
})();


// ---- Compat extensions merged from app-nr.js (non-breaking) ----

(function(){
  const bridge = () => (typeof window.pywebview !== 'undefined' && window.pywebview.api) ? window.pywebview.api : null;

  // Optional elements that may exist in alternate layouts
  const voiceDropdown = document.getElementById('voice-dropdown');
  const themeSelectAlt = document.getElementById('themeSelect') || document.getElementById('theme-select');
  const contactsDiv = document.getElementById('contacts');
  const remindersList = document.getElementById('reminders-list') || document.getElementById('remList');
  const threadsListAlt = document.getElementById('thread-list');

  async function applyTheme(themeName){
    try{
      if(!themeName) return;
      const api = bridge();
      if(api && api.set_theme) await api.set_theme(themeName);
      // Ensure CSS link is refreshed in either case
      const link = document.getElementById('themeLink');
      if(link){
        const base = link.getAttribute('href')?.split('?')[0] || 'unified-theme.css';
        link.setAttribute('href', base + '?ts=' + Date.now());
      }
    }catch(e){ console.warn('[applyTheme]', e); }
  }

  // Compatible thread loader (falls back to existing renderer)
  async function loadThreads(isoDate){
    try{
      const api = bridge();
      if(!api) return;
      const fn = api.list_threads_for_date || api.list_threads || api.listThreads;
      if(!fn) return;
      const th = await fn.call(api, isoDate || null);
      if(Array.isArray(th) && threadsListAlt){
        // use existing renderer if present
        if(typeof renderThreadGroups === 'function'){ renderThreadGroups(th); return; }
        // minimal fallback renderer
        threadsListAlt.innerHTML = '';
        (th||[]).forEach(t=>{
          const d = document.createElement('div');
          d.className = 'thread';
          d.textContent = (t.title || t.id || 'Conversation');
          d.addEventListener('click', ()=> loadConversation(t.id));
          threadsListAlt.appendChild(d);
        });
      }
    }catch(e){ console.warn('[loadThreads]', e); }
  }

  async function loadConversation(threadId){
    try{
      const api = bridge();
      if(api && api.open_thread) await api.open_thread(threadId);
    }catch(e){ console.warn('[loadConversation]', e); }
  }

  async function loadReminders(){
    try{
      const api = bridge();
      if(!(api && api.list_reminders)) return;
      const items = await api.list_reminders();
      if(!remindersList) return;
      remindersList.innerHTML = '';
      (items||[]).forEach(r=>{
        const li = document.createElement('li');
        li.textContent = `${r.title || r.message || 'Reminder'} — ${r.when || r.remind_time || ''}`;
        remindersList.appendChild(li);
      });
    }catch(e){ console.warn('[loadReminders]', e); }
  }

  async function refreshContacts(){
    try{
      const api = bridge();
      if(!(api && api.telecom_get_contacts) || !contactsDiv) return;
      const data = await api.telecom_get_contacts();
      contactsDiv.innerHTML = '';
      (data || []).forEach(c=>{
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<span>${(c.name||'').toString()} — ${(c.number||c.phone||'').toString()}</span> <button data-id="${c.id||''}">Delete</button>`;
        row.querySelector('button').onclick = async ()=>{
          try{
            if(api.telecom_delete_contact) await api.telecom_delete_contact({id: c.id});
            await refreshContacts();
          }catch(e){ console.warn('[delete_contact]', e); }
        };
        contactsDiv.appendChild(row);
      });
    }catch(e){ console.warn('[refreshContacts]', e); }
  }

  // Optional voice dropdown (alternate id). Keep in sync with core voiceSelect if both exist.
  async function ensureVoices(){
    const api = bridge();
    const voiceSelectCore = document.getElementById('voiceSelect');
    if(!(voiceDropdown || voiceSelectCore) || !(api && api.list_voices)) return;
    try{
      const voices = await api.list_voices();
      const fill = (sel)=>{
        if(!sel) return;
        sel.innerHTML = '';
        (voices||[]).forEach(v=>{
          const o = document.createElement('option');
          o.value = v.id || v.name || '';
          o.textContent = v.name || v.id || '';
          sel.appendChild(o);
        });
      };
      fill(voiceDropdown);
      // Keep both selects synced
      if(voiceDropdown){
        voiceDropdown.addEventListener('change', async ()=>{
          try{ if(api.set_voice) await api.set_voice(voiceDropdown.value); }catch(e){}
          if(voiceSelectCore && voiceSelectCore.value !== voiceDropdown.value){
            voiceSelectCore.value = voiceDropdown.value;
            voiceSelectCore.dispatchEvent(new Event('change'));
          }
        });
      }
    }catch(e){ console.warn('[ensureVoices]', e); }
  }

  // Wire theme dropdown alt handler
  themeSelectAlt?.addEventListener('change', ()=> applyTheme(themeSelectAlt.value));

  // Kick off optional loaders without interfering with existing startup
  window.addEventListener('load', ()=>{
    try{ ensureVoices(); }catch(e){}
    try{ refreshContacts(); }catch(e){}
    try{ loadReminders(); }catch(e){}
    try{ loadThreads(); }catch(e){}
  });

  // Expose helpers (optional)
  window.SarahWebUI = Object.assign(window.SarahWebUI||{}, {
    applyTheme, loadConversation, loadThreads, loadReminders, refreshContacts
  });
})();


// ===== CRK Feature Pack (defensive, non-breaking) =====
(function(){
  // Guard: don't run twice
  if (window.__CRK_FEATURES_ATTACHED__) return;
  window.__CRK_FEATURES_ATTACHED__ = true;

  // Bridge helper with graceful fallbacks to multiple names
  function callBridge(methods, ...args){
    try{
      if(!hasBridge()) throw new Error('Bridge unavailable');
      const b = api();
      for(const m of methods){
        if(typeof b[m] === 'function'){
          return b[m](...args);
        }
      }
      throw new Error('No matching bridge method: '+methods.join(','));
    }catch(e){ console.warn('[callBridge]', e); throw e; }
  }

  // ----- Collapsible rails/sections (only bind if classes exist) -----
  try{
    document.querySelectorAll('.collapse').forEach(btn => {
      if (btn.__crk_bound__) return; btn.__crk_bound__ = true;
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-target');
        const tgt = id && document.getElementById(id);
        if(tgt) tgt.classList.toggle('collapsed');
        if (typeof toggleAppColumn === 'function') toggleAppColumn(id);
      });
    });
    document.querySelectorAll('.section-toggle').forEach(btn => {
      if (btn.__crk_bound__) return; btn.__crk_bound__ = true;
      btn.addEventListener('click', () => {
        const sec = btn.closest('.rail-section');
        if(sec) sec.classList.toggle('collapsed');
      });
    });
  }catch(e){}

  // ----- Contacts / Comms (enhanced) -----
  const el = {
    contacts: document.getElementById('contacts'),
    cForm: document.getElementById('new-contact'),
    cName: document.getElementById('c-name'),
    cAddr: document.getElementById('c-address'),
    cPhone: document.getElementById('c-phone'),
    cNotes: document.getElementById('c-notes'),
    cClear: document.getElementById('c-clear'),
    cImport: document.getElementById('c-import'),
    cExport: document.getElementById('c-export'),
    cSearch: document.getElementById('c-search'),
    detail: document.getElementById('contact-detail'),
    threadView: document.getElementById('thread'),
    thComposer: document.getElementById('thread-composer'),
    thInput: document.getElementById('thread-input'),
    callStatus: document.getElementById('call-status'),
    btnCall: document.getElementById('btn-call'),
    btnVideo: document.getElementById('btn-video'),
    btnEnd: document.getElementById('btn-end'),
    callMic: document.getElementById('call-mic'),
    callCam: document.getElementById('call-cam'),
    callSpk: document.getElementById('call-spk'),
    keypad: document.getElementById('keypad'),
    webcam: document.getElementById('webcam'),
    webcamToggle: document.getElementById('webcamToggle'),
    micToggle: document.getElementById('micToggle'),
    voiceToggle: document.getElementById('voiceToggle'),
    remote: document.getElementById('remote'),
    snapshot: document.getElementById('snapshot'),
    modeText: document.getElementById('mode-text'),
  };

  let CONTACTS = [];
  let SELECTED = null;
  const THREADS = {}; // {addr: [{me:boolean,text,ts}]}

  function loadContactsLocal(){ try{ return JSON.parse(localStorage.getItem('SM_contacts')||'[]'); }catch{ return []; } }
  function saveContactsLocal(list){ try{ localStorage.setItem('SM_contacts', JSON.stringify(list)); }catch{} }

  async function loadContacts(){
    try{
      let res = [];
      if(hasBridge()){
        res = await callBridge(['ucp_list_contacts','contacts_list','telecom_get_contacts']);
      }else{
        res = loadContactsLocal();
      }
      CONTACTS = Array.isArray(res) ? res : [];
    }catch(e){
      // preview
      CONTACTS = CONTACTS.length ? CONTACTS : [{name:'Brian', address:'brian@sarah'}];
    }
    renderContacts();
  }

  function renderContacts(filter=''){
    if(!el.contacts) return;
    el.contacts.innerHTML='';
    const q = (filter||'').toLowerCase();
    CONTACTS.filter(c => !q || (c.name?.toLowerCase().includes(q) || (c.address||c.addr||'').toLowerCase().includes(q)))
    .forEach(c => {
      const row = document.createElement('div');
      row.className = 'contact-item'+(SELECTED && (SELECTED.address||SELECTED.addr)===(c.address||c.addr)?' active':'');
      row.dataset.addr = c.address||c.addr||'';
      row.innerHTML = `<div class="avatar">${(c.name||'?').slice(0,1).toUpperCase()}</div>
                       <div class="title"><div>${c.name||'(no name)'}</div>
                       <div class="meta">${c.address||c.addr||''}</div></div>`;
      row.addEventListener('click', ()=> selectContact(c.address||c.addr||''));
      el.contacts.appendChild(row);
    });
  }

  function renderDetail(){
    if(!el.detail) return;
    if(!SELECTED){
      el.detail.innerHTML = `<div class='empty'>Select a contact to view details.</div>`;
      return;
    }
    const c = SELECTED;
    el.detail.innerHTML = `
      <div class="contact-header">
        <div class="avatar">${(c.name||'?').slice(0,1).toUpperCase()}</div>
        <div class="title">
          <div style="font-weight:700">${c.name||'(no name)'}</div>
          <small class="meta">${c.address||c.addr||''}</small>
        </div>
        <div class="contact-actions">
          <button class="btn" id="act-edit">Edit</button>
          <button class="btn danger" id="act-del">Delete</button>
        </div>
      </div>
      <div class="meta">Phone: ${c.phone||'—'} | Notes: ${c.notes||'—'}</div>`;
    const edit = el.detail.querySelector('#act-edit');
    const del = el.detail.querySelector('#act-del');
    if (edit) edit.addEventListener('click', ()=>{
      if(el.cName) el.cName.value = c.name||'';
      if(el.cAddr) el.cAddr.value = c.address||c.addr||'';
      if(el.cPhone) el.cPhone.value = c.phone||'';
      if(el.cNotes) el.cNotes.value = c.notes||'';
      const cs = document.getElementById('c-save'); if(cs) cs.textContent = 'Update contact';
    });
    if (del) del.addEventListener('click', async ()=>{
      if(!confirm(`Delete ${c.name||c.address||c.addr}?`)) return;
      try{
        if(hasBridge()){
          await callBridge(['ucp_delete_contact','contacts_delete','telecom_delete_contact'], c.address||c.addr);
        }
        CONTACTS = CONTACTS.filter(x => (x.address||x.addr)!==(c.address||c.addr));
        saveContactsLocal(CONTACTS);
        SELECTED = null; renderContacts(el.cSearch?.value); renderDetail(); if(el.threadView) el.threadView.innerHTML='';
      }catch(e){ alert('Delete failed: '+e.message); }
    });
  }

  function selectContact(addr){
    const c = CONTACTS.find(x => (x.address||x.addr)===addr);
    SELECTED = c || null;
    renderContacts(el.cSearch?.value);
    renderDetail();
    openSubsectionsForUse();
    loadThread(addr);
  }

  function openSubsectionsForUse(){
    const s1=document.getElementById('ss-chat'), s2=document.getElementById('ss-call'), s3=document.getElementById('ss-keypad');
    if(s1) s1.open=true; if(s2) s2.open=true; if(s3) s3.open=false;
  }

  function renderThread(addr){
    if(!el.threadView) return;
    el.threadView.innerHTML='';
    const list = THREADS[addr] || [];
    list.forEach(m => {
      const row = document.createElement('div');
      row.className = 'thread-msg'+(m.me?' me':'');
      row.innerHTML = `<div class="bub ${m.me?'me':''}">${m.text}</div>`;
      el.threadView.appendChild(row);
    });
    el.threadView.scrollTop = el.threadView.scrollHeight;
  }

  async function loadThread(addr){
    if(!addr) return;
    try{
      if(hasBridge()){
        const res = await callBridge(['ucp_fetch_thread','comms_fetch_thread','ucp_get_thread'], addr);
        if(Array.isArray(res)) THREADS[addr] = res.map(x => ({me: !!x.me, text: x.text, ts: x.ts||Date.now()}));
      }
    }catch(e){ /* preview */ }
    if(!THREADS[addr]) THREADS[addr] = [];
    renderThread(addr);
  }

  // Composer for per-contact messaging
  if (el.thComposer && !el.thComposer.__crk_bound__) {
    el.thComposer.__crk_bound__ = true;
    el.thComposer.addEventListener('submit', async (e)=>{
      e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
      if(!SELECTED) return;
      const text = (el.thInput && el.thInput.value.trim()) || '';
      if(!text) return;
      if(el.thInput) el.thInput.value='';
      const addr = SELECTED.address||SELECTED.addr;
      (THREADS[addr] = THREADS[addr] || []).push({me:true,text,ts:Date.now()});
      renderThread(addr);
      try{
        if(hasBridge()){
          await callBridge(['ucp_send_direct','ucp_send_text','comms_send_text'], addr, text);
        }
      }catch(e){
        (THREADS[addr] = THREADS[addr] || []).push({me:false,text:'(send failed: '+e.message+')',ts:Date.now()});
        renderThread(addr);
      }
    });
  }

  // Contact form save/clear/search
  if (el.cForm && !el.cForm.__crk_bound__) {
    el.cForm.__crk_bound__ = true;
    el.cForm.addEventListener('submit', async (e)=>{
      e.preventDefault();
  try{ if(window.TTS){ if (typeof voiceEnabled !== 'undefined') TTS.enable(voiceEnabled); else if (localStorage.getItem('voice_enabled')===null) TTS.enable(true); } }catch(e){}
      const rec = {
        name: el.cName?.value.trim() || '',
        address: el.cAddr?.value.trim() || '',
        phone: el.cPhone?.value.trim() || '',
        notes: el.cNotes?.value.trim() || ''
      };
      if(!rec.name || !rec.address) return;
      const idx = CONTACTS.findIndex(x => (x.address||x.addr)===rec.address);
      try{
        if(hasBridge()){
          const method = idx>=0 ? ['ucp_update_contact','contacts_update','telecom_add_contact'] : ['ucp_add_contact','contacts_add','telecom_add_contact'];
          await callBridge(method, rec);
        }
        if(idx>=0) CONTACTS[idx] = rec; else CONTACTS.push(rec);
        saveContactsLocal(CONTACTS);
        renderContacts(el.cSearch?.value);
        el.cForm.reset(); const saveBtn = document.getElementById('c-save'); if(saveBtn) saveBtn.textContent='Add contact';
      }catch(e){ alert('Save failed: '+e.message); }
    });
  }
  el.cClear && el.cClear.addEventListener('click', ()=>{ el.cForm?.reset(); const saveBtn=document.getElementById('c-save'); if(saveBtn) saveBtn.textContent='Add contact'; });
  el.cSearch && el.cSearch.addEventListener('input', ()=> renderContacts(el.cSearch.value));

  // Import / Export JSON
  el.cExport && el.cExport.addEventListener('click', ()=>{
    const blob = new Blob([JSON.stringify(CONTACTS,null,2)], {type:'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'contacts.json'; a.click();
  });
  el.cImport && el.cImport.addEventListener('click', async ()=>{
    const inp = document.createElement('input'); inp.type='file'; inp.accept='application/json';
    inp.onchange = async ()=>{
      const f = inp.files?.[0]; if(!f) return;
      const txt = await f.text();
      try{
        const arr = JSON.parse(txt);
        if(Array.isArray(arr)){
          CONTACTS = arr; saveContactsLocal(CONTACTS); renderContacts(el.cSearch?.value);
        }
      }catch{ alert('Invalid JSON'); }
    };
    inp.click();
  });

  // ----- Calls + media (on-demand permissions to avoid repetitive prompts) -----
  let localStream = null;
  async function getMedia(constraints){ return await navigator.mediaDevices.getUserMedia(constraints); }
  async function startCall(video=false){
    if(!SELECTED || !(el.callStatus)) return;
    try{
      el.callStatus.textContent = 'Connecting...';
      if(hasBridge()){
        await callBridge(['ucp_start_call','comms_start_call','telecom_start_call'],
          SELECTED.address||SELECTED.addr, {video, mic:!!(el.callMic && el.callMic.checked), cam:!!(el.callCam && el.callCam.checked), spk:!!(el.callSpk && el.callSpk.checked)});
      }
      if(video){
        if(!localStream) localStream = await getMedia({audio: !!(el.callMic && el.callMic.checked), video: !!(el.callCam && el.callCam.checked)});
        if(el.webcam) el.webcam.srcObject = localStream;
      }else{
        if(!localStream && (el.callMic && el.callMic.checked)) localStream = await getMedia({audio:true, video:false});
      }
      el.callStatus.textContent = video ? 'Video call active' : 'Call active';
    }catch(e){ el.callStatus.textContent = 'Error: '+e.message; }
  }
  async function endCall(){
    try{ if(hasBridge()) await callBridge(['ucp_end_call','comms_end_call','telecom_end_call'], SELECTED?.address||SELECTED?.addr); }catch(e){}
    if(el.callStatus) el.callStatus.textContent = 'Idle';
    if(localStream){
      localStream.getTracks().forEach(t=>t.stop()); localStream=null; if(el.webcam) el.webcam.srcObject=null;
    }
  }
  el.btnCall && el.btnCall.addEventListener('click', ()=> startCall(false));
  el.btnVideo && el.btnVideo.addEventListener('click', ()=> startCall(true));
  el.btnEnd && el.btnEnd.addEventListener('click', endCall);

  // DTMF keypad
  if (el.keypad) {
    el.keypad.querySelectorAll('button').forEach(b=>{
      b.addEventListener('click', async ()=>{
        const tone = b.dataset.tone;
        try{ if(hasBridge()) await callBridge(['ucp_dtmf','comms_dtmf','telecom_dtmf'], SELECTED?.address||SELECTED?.addr, tone); }catch(e){}
      });
    });
  }

  // Webcam/Mic/Voice toggles (request permission once)
  let previewOn = true, micOn = true, voiceOn = true;
  el.webcamToggle && el.webcamToggle.addEventListener('click', async ()=>{
    previewOn = !previewOn;
    el.webcamToggle.textContent = 'Webcam: ' + (previewOn ? 'On':'Off');
    if(previewOn){
      if(!localStream) { try{ localStream = await getMedia({video:true, audio:false}); }catch(e){} }
      if(localStream && el.webcam) el.webcam.srcObject = localStream;
    }else{
      if(localStream){ localStream.getTracks().forEach(t=> t.kind==='video' && t.stop()); if(el.webcam) el.webcam.srcObject = null; localStream = null; }
    }
  });
  el.micToggle && el.micToggle.addEventListener('click', async ()=>{
    micOn = !micOn;
    el.micToggle.textContent = 'Microphone: ' + (micOn ? 'On':'Off');
    if(localStream){ localStream.getAudioTracks().forEach(t=> t.enabled = micOn); }
  });
  el.voiceToggle && el.voiceToggle.addEventListener('click', async ()=>{
    voiceOn = !voiceOn;
    el.voiceToggle.textContent = 'Voice: ' + (voiceOn ? 'On':'Off');
    try{ if(hasBridge()) await callBridge(['ucp_set_flag','set_flag'], 'VOICE_ENABLED', voiceOn); }catch(e){}
  });

  // ----- Network state poller (only if API exists) -----
  async function pollNetwork(){
    if(!hasBridge() || !api().network_state) return;
    try{
      const s = await api().network_state();
      if (s && typeof s.mode === 'string' && el.modeText) el.modeText.textContent = s.mode;
      if (el.remote && el.snapshot){
        if (s && s.call_active){ el.remote.hidden=false; el.snapshot.hidden=true; }
        else { el.remote.hidden=true; el.snapshot.hidden=false; }
      }
      const ledNet = document.getElementById('led-net');
      if(ledNet) ledNet.classList.toggle('blink', !!(s && s.busy));
    }catch{}
  }
  setInterval(pollNetwork, 1200);

  // Initial boot for this pack (non-intrusive)
  loadContacts();
  window.SarahWebUI = Object.assign(window.SarahWebUI||{}, {
    callBridge, loadContacts, selectContact: (addr)=>{ try{ selectContact(addr); }catch(e){} }
  });
})();


// ===== Overall Enhancements Pack (voices, themes, tones, UI polish) =====
(function(){
  if (window.__ENHANCEMENTS_ATTACHED__) return;
  window.__ENHANCEMENTS_ATTACHED__ = true;

  // --- Helpers ---
  const hasBridge = () => typeof window.pywebview !== 'undefined' && window.pywebview.api;
  const api = () => window.pywebview.api;

  function $(id){ return document.getElementById(id); }
  function addOption(sel, value, label){
    if(!sel) return;
    const opt = document.createElement('option');
    opt.value = value; opt.textContent = label || value;
    sel.appendChild(opt);
  }

  // --- THEMES: populate + apply (bridge-first, HTTP fallback) ---
  async function loadThemes(){
    const sels = [ $('themeSelect'), $('theme-select') ].filter(Boolean);
    if(!sels.length) return;
    let items = [];
    try{
      if (hasBridge() && api().list_themes){
        items = await api().list_themes();
      }else{
        const r = await fetch(window.SM_getApi ? window.SM_getApi('/get_theme_files') : '/get_theme_files');
        items = await r.json();
      }
    }catch(e){ console.warn('loadThemes', e); }
    items = Array.isArray(items) ? items : (items?.themes||items?.files||[]);
    sels.forEach(s => { s.innerHTML = ''; addOption(s, '', '-- Select theme --'); });
    (items||[]).forEach(t => {
      const name = t.name || t.filename || String(t);
      sels.forEach(s => addOption(s, name, name));
    });
  }

  async function applyTheme(name){
    if(!name) return;
    try{ if (hasBridge() && api().set_theme) await api().set_theme(name); }catch{}
    const link = $('themeLink') || document.querySelector('link[data-theme]');
    if (link){
      const baseHref = link.getAttribute('data-base') || link.getAttribute('href') || '';
      const parts = baseHref.split('?')[0];
      const file = parts.includes('.css') ? parts : `/api/static/themes/${name}.css`;
      link.setAttribute('href', file + `?v=${Date.now()}`);
      link.setAttribute('data-base', file);
    }
  }

  // Wire theme dropdowns
  [ $('themeSelect'), $('theme-select') ].forEach(sel => {
    if(!sel || sel.__bound__) return;
    sel.__bound__ = true;
    sel.addEventListener('change', e => applyTheme(e.target.value));
  });

  // --- VOICES: populate + select (bridge-first, HTTP fallback) ---
  async function loadVoices(){
    const sels = [ $('voiceSelect'), $('voice-dropdown') ].filter(Boolean);
    if(!sels.length) return;
    let voices = [];
    try{
      if (hasBridge() && api().list_voices){ voices = await api().list_voices(); }
      else { const r = await fetch(window.SM_getApi ? window.SM_getApi('/get_available_voices') : '/get_available_voices'); voices = await r.json(); }
    }catch(e){ console.warn('loadVoices', e); }
    voices = Array.isArray(voices) ? voices : [];
    sels.forEach(s => { s.innerHTML = ''; addOption(s, '', '-- Select voice --'); });
    voices.forEach(v => {
      const id = v.id || v.value || v.name; const label = v.name || v.label || id;
      sels.forEach(s => addOption(s, id, label));
    });
  }

  async function setVoice(id){
    if(!id) return;
    try{ if (hasBridge() && api().set_voice) await api().set_voice(id); }catch{}
  }

  [ $('voiceSelect'), $('voice-dropdown') ].forEach(sel => {
    if(!sel || sel.__bound__) return;
    sel.__bound__ = true;
    sel.addEventListener('change', e => setVoice(e.target.value));
  });

  // Load on boot
  window.addEventListener('DOMContentLoaded', () => { loadVoices(); loadThemes(); });

  // --- TOUCH TONES (WebAudio overlay on keypad clicks) ---
  const Tone = (function(){
    const ctx = (window.AudioContext || window.webkitAudioContext) ? new (window.AudioContext||window.webkitAudioContext)() : null;
    const map = {
      '1':[1209,697], '2':[1336,697], '3':[1477,697],
      '4':[1209,770], '5':[1336,770], '6':[1477,770],
      '7':[1209,852], '8':[1336,852], '9':[1477,852],
      '*':[1209,941], '0':[1336,941], '#':[1477,941]
    };
    function play(symbol, ms=120){
      if(!ctx) return;
      const freqs = map[symbol]; if(!freqs) return;
      const g = ctx.createGain(); g.gain.value = 0.1; g.connect(ctx.destination);
      const osc1 = ctx.createOscillator(); osc1.frequency.value = freqs[0]; osc1.type='sine'; osc1.connect(g);
      const osc2 = ctx.createOscillator(); osc2.frequency.value = freqs[1]; osc2.type='sine'; osc2.connect(g);
      try{ osc1.start(); osc2.start(); }catch{}
      setTimeout(()=>{ try{osc1.stop(); osc2.stop(); g.disconnect();}catch{} }, ms);
    }
    return { play };
  })();

  const keypad = $('keypad');
  if (keypad && !keypad.__tones__) {
    keypad.__tones__ = true;
    keypad.querySelectorAll('button').forEach(b => {
      b.addEventListener('click', ()=>{
        const tone = b.dataset?.tone || b.textContent?.trim();
        if(tone) Tone.play(tone);
      });
    });
  }

  // --- UI Polish: replace Speaker checkbox with button; hide checkboxes block; remove IDLE button ---
  (function fixCallControls(){
    const btnBar = document.getElementById('call-buttons') || document.getElementById('callBar') || document.getElementById('callbar');
    const btnEnd = document.getElementById('btn-end');
    // Hide checkbox block under End Call
    const cbxBlock = document.getElementById('call-cbx-block') || document.querySelector('.call-checkboxes');
    if (cbxBlock) cbxBlock.style.display = 'none';

    // Remove/hide IDLE button
    const idleBtn = document.getElementById('btn-idle') || Array.from(document.querySelectorAll('button')).find(x => x.textContent?.trim().toUpperCase() === 'IDLE');
    if (idleBtn) idleBtn.style.display = 'none';

    // Create SPEAKER button aligned with others
    const spkCbx = document.getElementById('call-spk'); // likely the old checkbox
    if (btnBar && !document.getElementById('btn-speaker')){
      const btn = document.createElement('button');
      btn.id = 'btn-speaker';
      btn.className = 'btn';
      btn.textContent = 'Speaker';
      let on = spkCbx ? !!spkCbx.checked : true;
      const setUI = ()=>{ btn.classList.toggle('active', !!on); };
      setUI();
      btn.addEventListener('click', async ()=>{
        on = !on; setUI();
        try{
          if(hasBridge()){
            const setFlag = api().set_flag || api().ucp_set_flag;
            if (setFlag) await setFlag.call(api(), 'SPEAKER_ENABLED', on);
          }
        }catch{}
      });
      if (btnEnd && btnEnd.parentElement){
        btnEnd.parentElement.insertBefore(btn, btnEnd); // place before End button to align
      }else if(btnBar){
        btnBar.appendChild(btn);
      }
    }
  })();
})();


/* =========================
   SarahMemory WebUI — Final Merge Pack
   - Keep existing Settings logic from primary app.js
   - Pull in keypad letters/DTMF behavior (from app-3.js)
   - Remove ONLY the three checkboxes under call bar (from app-4.js)
   - Fix side rails to collapse cleanly while keeping the ☰ button visible to reopen
   ========================= */
(function(){
  if (window.__FINAL_MERGE_PACK__) return;
  window.__FINAL_MERGE_PACK__ = true;

  // -------- Helpers --------
  const hasBridge = () => typeof window.pywebview !== 'undefined' && window.pywebview.api;
  const api = () => window.pywebview.api;

  // -------- 1) Rails: collapse without hiding the hamburger --------
  // Rebind collapse buttons to toggle 'collapsed' class and app flags only.
  document.querySelectorAll('.collapse').forEach(btn => {
    if (btn.__final_bound__) return;
    btn.__final_bound__ = true;
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-target');
      const rail = id && document.getElementById(id);
      if (!rail) return;
      rail.classList.toggle('collapsed');
      const app = document.getElementById('app');
      if (app) {
        if (id==='left-rail') app.classList.toggle('left-collapsed');
        if (id==='right-rail') app.classList.toggle('right-collapsed');
      }
    });
  });

  // CSS: collapse rails to a slim bar, keep the ☰ button visible for reopen
  (function injectRailCss(){
    if (document.getElementById('final-rails-css')) return;
    const css = document.createElement('style');
    css.id = 'final-rails-css';
    css.textContent = `
      /* Slim bar when collapsed, not display:none so the toggle remains available */
      #left-rail.collapsed, #right-rail.collapsed {
        width: 12px !important; min-width: 12px !important; max-width: 12px !important;
        flex: 0 0 12px !important;
        overflow: visible !important;
        position: relative;
      }
      /* Hide inner content when collapsed, except the header area */
      #left-rail.collapsed .rail-section > :not(header),
      #right-rail.collapsed .rail-section > :not(header) { display: none !important; }

      /* Ensure the hamburger stays clickable and visible */
      #left-rail.collapsed .rail-section > header .collapse,
      #right-rail.collapsed .rail-section > header .collapse {
        position: absolute; right: 2px; top: 2px;
        z-index: 5;
        opacity: 1;
        pointer-events: auto;
      }
    `;
    document.head.appendChild(css);
  })();

  // -------- 2) Keypad letters (non-destructive) + touch tones stay intact --------
  (function keypadLetters(){
    const keypad = document.getElementById('keypad');
    if (!keypad || keypad.__letters_applied__) return;
    keypad.__letters_applied__ = true;
    const map = { '1':'', '2':'ABC', '3':'DEF', '4':'GHI', '5':'JKL', '6':'MNO', '7':'PQRS', '8':'TUV', '9':'WXYZ', '*':'', '0':'+', '#':'' };
    keypad.querySelectorAll('button').forEach(b => {
      const val = (b.dataset?.tone || b.textContent?.trim() || '');
      if (!val) return;
      if (!b.querySelector('.sub')) {
        const sub = document.createElement('div');
        sub.className = 'sub';
        sub.textContent = map[val] || '';
        // light, compact presentation; do not alter main text
        sub.style.display = 'block';
        sub.style.fontSize = '10px';
        sub.style.opacity = '0.7';
        sub.style.lineHeight = '1';
        sub.style.marginTop = '2px';
        b.appendChild(sub);
      }
      // slight size tune (no impact on layout grids)
      if (!b.classList.contains('btn-sm')) b.classList.add('btn-sm');
    });
    // Minimal CSS in case theme lacks .btn-sm
    if (!document.getElementById('final-keypad-css')){
      const css = document.createElement('style'); css.id='final-keypad-css';
      css.textContent = `#keypad button.btn-sm{padding:10px 12px;min-width:48px;line-height:1.1;font-size:18px}`;
      document.head.appendChild(css);
    }
  })();

  // -------- 3) Remove ONLY the three checkboxes under call bar + "IDLE" word --------
  function removeCheckboxRowOnce(){
    const mic = document.getElementById('call-mic');
    const cam = document.getElementById('call-cam');
    const spk = document.getElementById('call-spk');
    const parts = [mic, cam, spk].filter(Boolean);
    if (parts.length){
      const row = parts[0].closest('.row') || parts[0].parentElement;
      if (row && row.parentElement) row.parentElement.removeChild(row);
    }
    const status = document.getElementById('call-status');
    if (status && status.textContent && status.textContent.trim().toUpperCase()==='IDLE'){
      status.textContent = '';
    }
  }
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', removeCheckboxRowOnce);
  } else {
    removeCheckboxRowOnce();
  }
  // also retry shortly if keypad renders late
  setTimeout(removeCheckboxRowOnce, 300);
})();

// === Contacts ↔ Reminders layout sync (surgical, no other changes) ===
(function () {
  if (window.__CONTACTS_REMINDERS_SYNC__) return;
  window.__CONTACTS_REMINDERS_SYNC__ = true;

  const $ = (sel, scope = document) => scope.querySelector(sel);

  // Elements
  const reminders = document.getElementById('reminders');
  const contactsRoot = document.getElementById('contacts');

  // Find the Contacts section container (the thing that gets .collapsed)
  const contactsSection =
    document.getElementById('contacts-section') ||
    (contactsRoot && (contactsRoot.closest('.rail-section') || contactsRoot.closest('section')));

  // Find a Status Bar container (top-level element that contains #status-text)
  const statusBar =
    document.getElementById('statusbar') ||
    document.getElementById('status-bar') ||
    ($('#status-text') && $('#status-text').closest('.statusbar')) ||
    ($('#status-text') && $('#status-text').closest('.rail-section')) ||
    ($('#status-text') && $('#status-text').parentElement);

  if (!reminders || !contactsSection || !statusBar) return;

  // Keep a placeholder to restore original position under Contacts
  let placeholder = document.getElementById('reminders-home-placeholder');
  if (!placeholder) {
    placeholder = document.createElement('div');
    placeholder.id = 'reminders-home-placeholder';
    // place the placeholder where Reminders currently lives
    reminders.parentElement && reminders.parentElement.insertBefore(placeholder, reminders);
  }

  function moveRemindersToStatusTop() {
    if (!statusBar.contains(reminders)) {
      statusBar.insertBefore(reminders, statusBar.firstChild || null);
      statusBar.scrollTop = 0;
    }
  }

  function restoreRemindersUnderContacts() {
    if (placeholder && placeholder.parentElement && reminders.previousSibling !== placeholder) {
      placeholder.parentElement.insertBefore(reminders, placeholder.nextSibling);
    }
  }

  function syncReminders() {
    const isCollapsed = contactsSection.classList.contains('collapsed');
    if (isCollapsed) {
      // Contacts collapsed → put Reminders back under Contacts
      restoreRemindersUnderContacts();
    } else {
      // Contacts expanded → put Reminders at top of Status Bar
      moveRemindersToStatusTop();
    }
  }

  // Bind to the Contacts section toggle (if present)
  const toggleBtn =
    contactsSection.querySelector('.section-toggle') || document.getElementById('contacts-toggle');
  if (toggleBtn && !toggleBtn.__cr_sync__) {
    toggleBtn.__cr_sync__ = true;
    toggleBtn.addEventListener('click', () => {
      // Wait for the section's class to update
      requestAnimationFrame(syncReminders);
    });
  }

  // Also observe class changes in case collapse/expand is triggered elsewhere
  const obs = new MutationObserver(syncReminders);
  obs.observe(contactsSection, { attributes: true, attributeFilter: ['class'] });

  // Initial placement
  syncReminders();
})();
// === Contacts ↔ Reminders layout sync (final, surgical) ===
(function () {
  if (window.__CR_SYNC_FINAL__) return;
  window.__CR_SYNC_FINAL__ = true;

  // Helpers
  const $ = (sel, scope = document) => scope.querySelector(sel);
  const $$ = (sel, scope = document) => Array.from(scope.querySelectorAll(sel));

  // Base elements
  const reminders = document.getElementById('reminders');
  const contactsRoot = document.getElementById('contacts');
  const statusBar =
    document.getElementById('statusbar') ||
    document.getElementById('status-bar') ||
    ($('#status-text') && $('#status-text').closest('.statusbar')) ||
    ($('#status-text') && $('#status-text').parentElement);

  if (!reminders || !contactsRoot || !statusBar) return;

  // The actual Contacts "section" is the .rail-section that contains #contacts.
  const contactsSection =
    contactsRoot.closest('.rail-section') ||
    contactsRoot.closest('section');

  if (!contactsSection) return;

  // Placeholder to restore original location under Contacts
  let placeholder = document.getElementById('reminders-home-placeholder');
  if (!placeholder) {
    placeholder = document.createElement('div');
    placeholder.id = 'reminders-home-placeholder';
    // Insert placeholder right where Reminders currently lives
    reminders.parentElement && reminders.parentElement.insertBefore(placeholder, reminders);
  }

  function moveRemindersToStatusTop() {
    // Put Reminders at very top of the Status bar
    if (!statusBar.contains(reminders)) {
      statusBar.insertBefore(reminders, statusBar.firstChild || null);
    } else if (statusBar.firstChild !== reminders) {
      statusBar.insertBefore(reminders, statusBar.firstChild || null);
    }
  }

  function restoreRemindersUnderContacts() {
    // Return Reminders just after the placeholder (original spot under Contacts)
    if (placeholder && placeholder.parentElement && reminders.previousSibling !== placeholder) {
      placeholder.parentElement.insertBefore(reminders, placeholder.nextSibling);
    }
  }

  function syncReminders() {
    // Contacts is expanded when it DOES NOT have .collapsed (your code toggles this class)
    const isCollapsed = contactsSection.classList.contains('collapsed');
    if (isCollapsed) restoreRemindersUnderContacts();
    else moveRemindersToStatusTop();
  }

  // Bind to the existing "section-toggle" (your code toggles .collapsed on closest .rail-section)
  const toggleBtn = contactsSection.querySelector('.section-toggle');
  if (toggleBtn && !toggleBtn.__cr_sync_final__) {
    toggleBtn.__cr_sync_final__ = true;
    toggleBtn.addEventListener('click', () => {
      // wait a frame so the class is updated by your handler, then sync
      requestAnimationFrame(syncReminders);
    });
  }

  // Also observe class changes in case the section is toggled elsewhere
  const obs = new MutationObserver(() => syncReminders());
  obs.observe(contactsSection, { attributes: true, attributeFilter: ['class'] });

  // Final: initial placement
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', syncReminders, { once: true });
  } else {
    syncReminders();
  }
})();


// === [SM-VISUAL COLOR PIPELINE] ===
/*
  Adds webcam bootstrap, capture, local color naming, and a fast-path intent handler.
  Non-destructive: no renames, no removals. Works offline and speaks replies.
*/
(function(){
  const videoEl = document.getElementById('webcam');
  let _camReady = false, _hiddenCanvas, _ctx;

  async function ensureWebcam(){
    if (!videoEl) return false;
    if (_camReady && videoEl.srcObject) return true;
    try{
      const s = await (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? navigator.mediaDevices.getUserMedia({video:true, audio:false}) : Promise.reject('getUserMedia unavailable'));
      if (!s) return false;
      videoEl.srcObject = s;
      try {
        const track = s.getVideoTracks?.()[0];
        await track.applyConstraints({ advanced: [
          { focusMode: "continuous" },
          { exposureMode: "continuous" },
          { whiteBalanceMode: "continuous" }
        ]});
      } catch(e) { /* ignore if unsupported */ }

      if (videoEl.play) { try { await videoEl.play(); } catch(e){} }
      _camReady = true;
      return true;
    }catch(e){
      console.warn('[SM][vision] webcam unavailable:', e);
      return false;
    }
  }

  function captureWebcamFrame(){
    if (!videoEl || !videoEl.videoWidth || !videoEl.videoHeight) return null;
    _hiddenCanvas = _hiddenCanvas || document.createElement('canvas');
    _ctx = _ctx || _hiddenCanvas.getContext('2d', { willReadFrequently: true });
    const W = videoEl.videoWidth, H = videoEl.videoHeight;
    _hiddenCanvas.width = W; _hiddenCanvas.height = H;
    _ctx.drawImage(videoEl, 0, 0, W, H);

    // Probable shirt area: lower 40% / middle 50% width (fast heuristic)
    const x0 = Math.floor(W*0.25), y0 = Math.floor(H*0.55);
    const w  = Math.floor(W*0.50),  h  = Math.floor(H*0.35);
    let data = null;
    try { data = _ctx.getImageData(x0, y0, w, h); } catch(e) { try { data = _ctx.getImageData(0, 0, W, H); } catch(_){} }
    const dataURL = _hiddenCanvas.toDataURL && _hiddenCanvas.toDataURL('image/jpeg', 0.75);
    return { dataURL, data };
  }

  function colorFromImageData(imgData){
    if (!imgData) return null;
    const a = imgData.data;
    let r=0,g=0,b=0,c=0;
    // sample every 5th pixel (fast, stable)
    for (let i=0;i<a.length;i+=20){
      const rr=a[i], gg=a[i+1], bb=a[i+2], aa=a[i+3];
      if (aa<8) continue;
      r+=rr; g+=gg; b+=bb; c++;
    }
    if (!c) return null;
    r=Math.round(r/c); g=Math.round(g/c); b=Math.round(b/c);
    return { r, g, b, hex:'#'+[r,g,b].map(v=>v.toString(16).padStart(2,'0')).join('') };
  }

  // â‰ˆ150 color names (CSS/xkcd-ish) for human-friendly output
  const COLOR_TABLE=(function(){
    const entries = [
      ["black","#000000"],["white","#ffffff"],["gray","#808080"],["silver","#c0c0c0"],["dark gray","#a9a9a9"],
      ["light gray","#d3d3d3"],["red","#ff0000"],["maroon","#800000"],["crimson","#dc143c"],["brown","#8b4513"],
      ["sienna","#a0522d"],["peru","#cd853f"],["tan","#d2b48c"],["burlywood","#deb887"],["peach","#ffcba4"],
      ["salmon","#fa8072"],["coral","#ff7f50"],["tomato","#ff6347"],["orange","#ffa500"],["dark orange","#ff8c00"],
      ["chocolate","#d2691e"],["gold","#ffd700"],["khaki","#f0e68c"],["beige","#f5f5dc"],["yellow","#ffff00"],
      ["olive","#808000"],["lime","#00ff00"],["green","#008000"],["forest green","#228b22"],["sea green","#2e8b57"],
      ["spring green","#00ff7f"],["mint","#98ff98"],["teal","#008080"],["aqua","#00ffff"],["turquoise","#40e0d0"],
      ["light sea green","#20b2aa"],["cyan","#00ffff"],["steel blue","#4682b4"],["sky blue","#87ceeb"],
      ["deep sky blue","#00bfff"],["dodger blue","#1e90ff"],["royal blue","#4169e1"],["blue","#0000ff"],
      ["navy","#000080"],["indigo","#4b0082"],["violet","#ee82ee"],["purple","#800080"],["orchid","#da70d6"],
      ["magenta","#ff00ff"],["fuchsia","#ff00ff"],["plum","#dda0dd"],["pink","#ffc0cb"],["hot pink","#ff69b4"],
      ["deeppink","#ff1493"],["lavender","#e6e6fa"],["thistle","#d8bfd8"],["midnight blue","#191970"],
      ["slate blue","#6a5acd"],["slategray","#708090"],["dark slate gray","#2f4f4f"],["light slate gray","#778899"],
      ["gainsboro","#dcdcdc"],["linen","#faf0e6"],["honeydew","#f0fff0"],["azure","#f0ffff"],["alice blue","#f0f8ff"],
      ["ghost white","#f8f8ff"],["snow","#fffafa"],["bisque","#ffe4c4"],["moccasin","#ffe4b5"],["navajo white","#ffdead"],
      ["wheat","#f5deb3"],["pale goldenrod","#eee8aa"],["goldenrod","#daa520"],["dark goldenrod","#b8860b"],
      ["pale green","#98fb98"],["light green","#90ee90"],["medium sea green","#3cb371"],["dark green","#006400"],
      ["pale turquoise","#afeeee"],["cadet blue","#5f9ea0"],["cornflower blue","#6495ed"],["light blue","#add8e6"],
      ["light steel blue","#b0c4de"],["powder blue","#b0e0e6"],["medium blue","#0000cd"],["dark blue","#00008b"],
      ["rebeccapurple","#663399"],["pale violet red","#db7093"],["rosy brown","#bc8f8f"],["indian red","#cd5c5c"],
      ["firebrick","#b22222"],["dark red","#8b0000"],["dark salmon","#e9967a"],["light salmon","#ffa07a"],
      ["sandy brown","#f4a460"],["light coral","#f08080"],["light pink","#ffb6c1"],["misty rose","#ffe4e1"],
      ["old lace","#fdf5e6"],["cornsilk","#fff8dc"],["ivory","#fffff0"],["floral white","#fffaf0"],
      ["lemon chiffon","#fffacd"],["pale yellow","#ffffcc"],["olive drab","#6b8e23"],["dark olive green","#556b2f"],
      ["yellow green","#9acd32"],["chartreuse","#7fff00"],["lawngreen","#7cfc00"],["medium spring green","#00fa9a"],
      ["medium aquamarine","#66cdaa"],["aquamarine","#7fffd4"],["dark cyan","#008b8b"],["light cyan","#e0ffff"],
      ["medium turquoise","#48d1cc"],["dark turquoise","#00ced1"],["light sky blue","#87cefa"],
      ["medium slate blue","#7b68ee"],["blue violet","#8a2be2"],["medium purple","#9370db"],
      ["dark orchid","#9932cc"],["dark violet","#9400d3"],["medium violet red","#c71585"],
      ["brown chocolate","#7b3f00"],["camel","#c19a6b"],["coffee","#6f4e37"],["denim","#1560bd"]
    ];
    function hexToRgb(h){h=h.replace('#','');return{r:parseInt(h.slice(0,2),16),g:parseInt(h.slice(2,4),16),b:parseInt(h.slice(4,6),16)}}
    return entries.map(([name,hex])=>({name,hex,...hexToRgb(hex)}));
  })();
  function dist2(a,b){const dr=a.r-b.r,dg=a.g-b.g,db=a.b-b.b;return dr*dr+dg*dg+db*db}
  function nameNearestColor(rgb){let best=COLOR_TABLE[0],bestD=1e12;for(const c of COLOR_TABLE){const d=dist2(rgb,c);if(d<bestD){bestD=d;best=c}}return best?best.name:'unknown'}

  async function __SM_visualColorQuery(text, blobs){
    try{
      const rx=/\bwhat\s+(?:color|colour)\s+(?:is|are)\s+(?:my|the)\s+(shirt|top|tee|t-?shirt|hoodie|sweater|jacket|coat|pants|trousers|jeans|shorts|skirt|hat|cap|beanie|shoes?)\b/i;
      if(!rx.test(text)) return false;
      const ok=await ensureWebcam();
      if(!ok){ addMsg('assistant',"I can't access the webcam yet to analyze color."); return true; }
      const snap=captureWebcamFrame();
      if(!snap){ addMsg('assistant',"Camera not ready. Try again in a second."); return true; }
      const avg=colorFromImageData(snap.data);
      if(!avg){ addMsg('assistant',"I couldn't read enough pixels to determine the color."); return true; }
      const name=nameNearestColor(avg);
      const answer=`Looks ${name} to me.`;
      addMsg('assistant', answer);

      // Optional: log to backend for learning (best-effort, no-throw)
      if (typeof window.pywebview !== 'undefined' && window.pywebview.api){
        try{
          const payload={role:'webcam',kind:'visual_color',prompt:text,hex:avg.hex,rgb:avg,name,data_url:snap.dataURL};
          const fn=window.pywebview.api.log_visual_observation||window.pywebview.api.save_visual||window.pywebview.api.store_visual;
          if (fn) await fn.call(window.pywebview.api, payload);
        }catch(e){}
      }
      return true;
    }catch(e){ console.warn('[SM][vision] error:',e); return false; }
  }

  // expose
  window.__SM_visualColorQuery = __SM_visualColorQuery;
  window.__SM_captureWebcamFrame = captureWebcamFrame;
  window.__SM_nameNearestColor = nameNearestColor;
  window.__SM_ensureWebcam = ensureWebcam;
})();
// === [END SM-VISUAL COLOR PIPELINE] ===
// === [SM-VISUAL OCR PIPELINE] ===
(function(){
  async function __SM_visualOCRQuery(text){
    try{
      const rx = /(what\s+does\s+(this|it)\s+say\??|read\s+(this|it)\b|what\s+does\s+my\s+shirt\s+say\b)/i;
      if(!rx.test(text)) return false;
      if(!hasBridge()) return false;
      // Try optical zoom if supported
  try{
    const tr = document.getElementById('webcam')?.srcObject?.getVideoTracks?.()[0];
    const caps = tr?.getCapabilities?.();
    if (caps && caps.zoom && caps.zoom.max > caps.zoom.min) {
      const mid = Math.min(caps.zoom.max, (caps.zoom.min + caps.zoom.max) * 0.6);
      await tr.applyConstraints({ advanced:[{ zoom: mid }] });
      await new Promise(r=>setTimeout(r, 80));
    }
  }catch(e){}
  const out = await api().visual_query(text);
      const resp = (out && out.response) ? out.response : "I tried to read it, but couldn't.";
      addMsg('assistant', resp, out && out.meta ? out.meta : {source:'local-vision'});
      return true;
    }catch(e){ console.warn(e); return false; }
  }
  window.__SM_visualOCRQuery = __SM_visualOCRQuery;
})();





// === MOBILE RESPONSIVE LAYOUT MANAGER — v7.7.5 (safe, non-destructive) ===
(function SM_Responsive() {
  // Breakpoints you can tune
  const BP = { phone: 680, tablet: 1024 };

  // Cache refs + original layout anchors so we can restore precisely
  const app = document.getElementById('app');

  // Try several common IDs for center pane to be robust to older markup
  function getCenterNode() {
    return (
      document.querySelector('#center') ||
      document.querySelector('#center-rail') ||
      document.querySelector('#center-column') ||
      document.querySelector('#middle') ||
      document.querySelector('main') ||
      document.getElementById('messages')?.closest('.center') ||
      document.getElementById('messages')?.parentElement
    );
  }

  const leftRail  = document.getElementById('left-rail');
  const rightRail = document.getElementById('right-rail');
  const center    = getCenterNode();

  // Guard: if we can't find the required elements, bail quietly
  if (!app || !center || !rightRail) return;

  // Remember original placement so we can restore on wide screens
  const orig = {
    rightParent: rightRail.parentNode,
    rightNext:   rightRail.nextSibling,
  };

  // One-time "mobile default collapse" for left rail (but respect user choice later)
  (function collapseLeftRailOnce() {
    try {
      const key = 'sm_mobile_left_defaulted_v1';
      const already = localStorage.getItem(key) === '1';
      const userPref = localStorage.getItem('left-rail_collapsed'); // '1' or '0' or null
      const isNarrow = Math.min(window.innerWidth, window.innerHeight) <= BP.phone;
      if (!already && isNarrow && leftRail) {
        // Only set default if user never set a preference
        if (userPref === null) {
          SM_setRailCollapsed('left-rail', true);
        }
        localStorage.setItem(key, '1');
      }
    } catch {}
  })();

  // Utility: move rightRail after center (vertical stack)
  function stackRightBelowCenter() {
    try {
      if (rightRail.parentNode !== center.parentNode || center.nextSibling !== rightRail) {
        center.insertAdjacentElement('afterend', rightRail);
      }
      app.classList.add('stacked-mobile');
    } catch {}
  }

  // Utility: restore rightRail to original grid slot
  function restoreRightToGrid() {
    try {
      if (orig.rightParent) {
        if (orig.rightNext && orig.rightNext.parentNode === orig.rightParent) {
          orig.rightParent.insertBefore(rightRail, orig.rightNext);
        } else {
          orig.rightParent.appendChild(rightRail);
        }
      }
      app.classList.remove('stacked-mobile');
    } catch {}
  }

  // Messages viewport fit (avoid inner scrollbars on phones)
  const messages = document.getElementById('messages');
  const composer = document.getElementById('composer');
  function fitMessagesHeight() {
    if (!messages) return;
    try {
      const rect = app.getBoundingClientRect();
      const head = document.querySelector('#topbar, header, .topbar');
      const headH = head ? head.getBoundingClientRect().height : 0;
      const compH = composer ? composer.getBoundingClientRect().height : 0;
      const pad   = 10; // breathing room
      const target = window.innerHeight - headH - compH - pad - (rect.top > 0 ? rect.top : 0);
      if (target > 120) {
        messages.style.maxHeight = `${Math.floor(target)}px`;
        messages.style.height    = `${Math.floor(target)}px`;
      } else {
        // Fallback: let CSS handle it
        messages.style.maxHeight = '';
        messages.style.height    = '';
      }
    } catch {}
  }

  // Main layout decider
  function applyResponsiveLayout() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const portrait = h >= w;
    // Phone portrait: stack (Right under Center)
    if (w <= BP.phone && portrait) {
      stackRightBelowCenter();
    } else if (w <= BP.tablet && portrait) {
      // Tablet portrait: also stack, but leave left rail user-controlled
      stackRightBelowCenter();
    } else {
      // Wide or landscape: restore classic 3-column
      restoreRightToGrid();
    }
    fitMessagesHeight();
  }

  // Run once DOM is ready (in case this block loads before body)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyResponsiveLayout, { once: true });
  } else {
    applyResponsiveLayout();
  }

  // React to resizes & rotations
  window.addEventListener('resize',  debounce(applyResponsiveLayout, 120), { passive: true });
  window.addEventListener('orientationchange', debounce(applyResponsiveLayout, 120), { passive: true });

  // Simple debounce to avoid thrashing
  function debounce(fn, wait) {
    let t = 0;
    return function() {
      clearTimeout(t);
      t = setTimeout(fn, wait);
    };
  }
})();


// === SM_THEME_LOADER_V1 ===
async function SM_listThemes(){
  const url = (typeof window !== 'undefined' && window.SM_getApi)
    ? window.SM_getApi('/get_theme_files')
    : (location.origin + '/get_theme_files');
  const res = await fetch(url, { credentials: 'same-origin' });
  if (!res.ok) throw new Error('get_theme_files failed');
  const data = await res.json();
  const root = String(data.root || '/api/data/themes').replace(/\/$/, '');
  const files = Array.isArray(data.files) ? data.files : [];
  const out = files.map(f => `${location.origin}${root}/${encodeURIComponent(f)}`);
  return { root, files, urls: out, count: data.count || out.length };
}
window.SM_listThemes = SM_listThemes;


// === [SM_UI_SPLIT_MESSENGER_KEYPAD_V2] — moves subsections into standalone collapsibles ===
(function SM_splitMessengerKeypadV2(){
  if (window.__SM_MK_SPLIT_V2__) return; window.__SM_MK_SPLIT_V2__ = true;

  function $(sel, scope=document){ return scope.querySelector(sel); }
  function text(node){ return (node && (node.textContent||'')).trim().toLowerCase(); }

  function findSectionByTitle(root, title){
    title = String(title||'').trim().toLowerCase();
    const sections = Array.from(root.querySelectorAll(':scope > section.rail-section'));
    for (const sec of sections){
      const h = sec.querySelector(':scope > header > h2');
      if (h && text(h) === title) return sec;
    }
    return null;
  }

  function makeSection(title){
    const sec = document.createElement('section');
    sec.className = 'rail-section collapsed';
    const header = document.createElement('header');
    const h2 = document.createElement('h2'); h2.textContent = title;
    const btn = document.createElement('button'); btn.className = 'section-toggle'; btn.title = 'Collapse/Expand'; btn.textContent = '▾';
    header.appendChild(h2); header.appendChild(btn);
    const body = document.createElement('div'); body.className = 'section-body';
    sec.appendChild(header); sec.appendChild(body);
    // Bind a toggle just for this newly created button (existing code won't catch dynamic nodes)
    btn.addEventListener('click', () => { sec.classList.toggle('collapsed'); });
    return {sec, body, btn};
  }

  function moveChildrenExcept(container, selectorToSkip, into){
    const nodes = Array.from(container.childNodes);
    for (const n of nodes){
      if (n.nodeType === 1 && n.matches && n.matches(selectorToSkip)) continue;
      into.appendChild(n); // move preserves listeners
    }
  }

  function run(){
    const right = document.getElementById('right-rail');
    if (!right) return;

    const contactsSec  = findSectionByTitle(right, 'contacts');
    const remindersSec = findSectionByTitle(right, 'reminders');
    if (!contactsSec || !remindersSec) return;

    // If Messenger/Keypad already exist, bail
    if (findSectionByTitle(right, 'messenger') || findSectionByTitle(right, 'keypad')) return;

    // Look inside Contacts' body for subsections
    const body = contactsSec.querySelector(':scope > .section-body') || contactsSec;
    const subs = Array.from(body.querySelectorAll(':scope .subsection'));
    let msgSub=null, padSub=null;
    for (const s of subs){
      const h3 = s.querySelector(':scope > h3');
      const t = text(h3);
      if (t === 'messages') msgSub = s;
      else if (t === 'keypad') padSub = s;
    }

    // Build and insert Messenger (before Reminders)
    if (msgSub){
      const {sec, body: msgBody} = makeSection('Messenger');
      // Move #thread and #thread-composer etc. (everything except the <h3> label)
      moveChildrenExcept(msgSub, 'h3', msgBody);
      right.insertBefore(sec, remindersSec);
    }
    // Build and insert Keypad (before Reminders; after Messenger if any)
    if (padSub){
      const {sec, body: kpBody} = makeSection('Keypad');
      moveChildrenExcept(padSub, 'h3', kpBody);
      right.insertBefore(sec, remindersSec);
    }
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', run, {once:true});
  } else {
    run();
  }
})();
// === [/SM_UI_SPLIT_MESSENGER_KEYPAD_V2] ===


// === [SM_DIALER_V1] Compact dial screen + keypad shrink + mobile-friendly toggles ===
(function SM_DialerV1(){
  if (window.__SM_DIALER_V1__) return; window.__SM_DIALER_V1__ = true;

  // --- Utility to find a right-rail section by title (e.g., "Keypad") ---
  function text(n){ return (n && (n.textContent||'')).trim().toLowerCase(); }
  function findSectionByTitle(title){
    const right = document.getElementById('right-rail'); if(!right) return null;
    const secs = Array.from(right.querySelectorAll(':scope > section.rail-section'));
    title = String(title||'').trim().toLowerCase();
    for (const sec of secs){
      const h = sec.querySelector(':scope > header > h2');
      if (h && text(h) === title) return sec;
    }
    return null;
  }

  // --- CSS injection: shrink keypad, bigger toggles, no-scroll for keypad section ---
  (function injectDialerCss(){
    if (document.getElementById('sm-dialer-css')) return;
    const css = document.createElement('style');
    css.id = 'sm-dialer-css';
    css.textContent = `
      /* Touch-friendly header: make whole header tappable and enlarge chevron */
      .rail-section > header { padding: 12px 14px; cursor: pointer; }
      .rail-section > header h2 { font-size: 15px; }
      .rail-section > header .section-toggle { font-size: 22px; min-width: 44px; min-height: 44px; }
      @media (hover: none) {
        .rail-section > header .section-toggle { font-size: 24px; }
      }

      /* Compact keypad + dialer so both fit without overflow */
      .dialer { display:flex; align-items:center; gap:8px; padding: 6px 6px 8px; }
      .dialer input#dial-display {
        flex:1; padding:8px 10px; border-radius:10px; border:1px solid #2a3240;
        background:#0f141b; color: var(--text); font-size:16px; letter-spacing:0.5px;
      }
      .dialer .btn {
        padding:8px 10px; border-radius:10px; border:1px solid var(--border);
        background:#111822; color: var(--text); font-weight:600;
        min-width:44px; min-height:40px;
      }
      #keypad.keypad { grid-template-columns: repeat(3, 1fr); gap:8px; padding: 6px; }
      #keypad.keypad button {
        padding: 10px 0; border-radius:12px; font-size:18px; line-height:1.1; min-height:46px;
      }
      /* Ensure the Keypad section body never scrolls (dialer+grid visible) */
      .rail-section:has(> header > h2:nth-child(1):where(:is(h2))) {}
      /* Target the Keypad section body more robustly */
      #right-rail > section.rail-section:has(> header > h2:where(:not(:empty)):contains("Keypad")) > .section-body {
        max-height: none; overflow: visible;
      }
      /* Fallback: if :has/:contains unsupported, still give some breathing room globally */
      @supports not (selector(:has(*))) {
        #right-rail .rail-section > .section-body { max-height: 240px; }
      }

      /* Slightly reduce right rail width to fit on phones without inner scrollbars */
      :root { --right-col: 300px; }
      @media (max-width: 680px) {
        :root { --right-col: 280px; }
      }
    `;
    // `:contains()` is not standard; for broad support we won't rely on it at runtime.
    document.head.appendChild(css);
  })();

  // --- Make entire header a toggle (not just the tiny triangle) ---
  (function widenHeaderToggles(){
    document.querySelectorAll('#right-rail .rail-section > header').forEach(hdr => {
      if (hdr.__sm_hdr_bound__) return;
      hdr.__sm_hdr_bound__ = true;
      hdr.addEventListener('click', (e)=>{
        // Don't double-trigger when actual button is clicked; but treat header as target
        const sec = hdr.closest('.rail-section');
        if (!sec) return;
        sec.classList.toggle('collapsed');
      });
      // Keep the triangle button working; stop event bubbling to avoid double toggle flip
      const btn = hdr.querySelector('.section-toggle');
      if (btn && !btn.__sm_stop_bubble__) {
        btn.__sm_stop_bubble__ = true;
        btn.addEventListener('click', (e)=> e.stopPropagation());
      }
    });
  })();

  // --- Build/attach Dialer UI above keypad ---
  function ensureDialer(){
    const keypad = document.getElementById('keypad');
    if (!keypad) return null;
    if (keypad.__sm_dialer__) return keypad.__sm_dialer__;

    const body = keypad.closest('.section-body') || keypad.parentElement;
    if (!body) return null;

    const wrap = document.createElement('div');
    wrap.className = 'dialer';

    const display = document.createElement('input');
    display.type = 'text'; display.id = 'dial-display';
    display.placeholder = 'Enter number or IP...';

    const btnBack = document.createElement('button');
    btnBack.type = 'button'; btnBack.className = 'btn'; btnBack.title='Backspace'; btnBack.textContent = '⌫';

    const btnDot = document.createElement('button');
    btnDot.type = 'button'; btnDot.className = 'btn'; btnDot.title='Dot'; btnDot.textContent = '.';

    wrap.appendChild(display);
    wrap.appendChild(btnBack);
    wrap.appendChild(btnDot);

    // Insert dialer at the very top of the keypad section body
    body.insertBefore(wrap, body.firstChild);

    // Wire keypad buttons to append to display (supports digits, *, #, +)
    keypad.querySelectorAll('button').forEach(b=>{
      b.addEventListener('click', ()=>{
        const ch = (b.dataset.tone || b.textContent || '').trim();
        if (!ch) return;
        // Allow digits, '*', '#', '+', and '.' (period)
        if (/^[0-9*#+]$/.test(ch)) {
          display.value += ch;
        }
      }, { capture: true }); // capture to run even if someone stops propagation later
    });

    // Backspace button
    btnBack.addEventListener('click', ()=>{
      display.value = display.value.slice(0, -1);
      display.focus();
    });

    // Dot button for IP address
    btnDot.addEventListener('click', ()=>{
      display.value += '.';
      display.focus();
    });

    // Keep a reference for other hooks
    keypad.__sm_dialer__ = { wrap, display, btnBack, btnDot };
    return keypad.__sm_dialer__;
  }

  // --- Rebind Call/Video buttons to prefer dialer value when present ---
  async function dialCall(peer, video=false){
    const status = document.getElementById('call-status');
    try{
      if (status) status.textContent = 'Connecting...';
      if (typeof window.pywebview !== 'undefined' && window.pywebview.api){
        const b = window.pywebview.api;
        const opts = { video, mic:true, cam:video, spk:true };
        if (b.telecom_start_call) await b.telecom_start_call({ peer, video });
        else if (b.ucp_start_call) await b.ucp_start_call(peer, { video });
        else if (b.comms_start_call) await b.comms_start_call(peer, { video });
      }
      if (status) status.textContent = video ? 'Video call active' : 'Call active';
    }catch(e){
      if (status) status.textContent = 'Error: ' + e.message;
    }
  }

  function rebindCallButtons(){
    const dialer = ensureDialer();
    const display = dialer && dialer.display;

    const btnCall = document.getElementById('btn-call');
    if (btnCall && !btnCall.__sm_dial_bind__) {
      btnCall.__sm_dial_bind__ = true;
      btnCall.addEventListener('click', function(e){
        if (!display) return;
        const peer = (display.value||'').trim();
        if (peer) {
          e.preventDefault(); e.stopImmediatePropagation();
          dialCall(peer, false);
        }
      }, true); // use capture to supersede earlier listeners
    }

    const btnVideo = document.getElementById('btn-video');
    if (btnVideo && !btnVideo.__sm_dial_bind__) {
      btnVideo.__sm_dial_bind__ = true;
      btnVideo.addEventListener('click', function(e){
        if (!display) return;
        const peer = (display.value||'').trim();
        if (peer) {
          e.preventDefault(); e.stopImmediatePropagation();
          dialCall(peer, true);
        }
      }, true);
    }
  }

  function init(){
    ensureDialer();
    rebindCallButtons();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
// === [/SM_DIALER_V1] ===


// === [SM_RAIL_COMPACT_V1] Compact right-rail sections + mobile accordion ===
(function SM_RailCompactV1(){
  if (window.__SM_RAIL_COMPACT_V1__) return; window.__SM_RAIL_COMPACT_V1__ = true;

  // Inject compact styling for rail sections to reduce vertical footprint
  (function injectCompactCss(){
    if (document.getElementById('sm-rail-compact-css')) return;
    const css = document.createElement('style');
    css.id = 'sm-rail-compact-css';
    css.textContent = `
      /* Compact section headers & bodies */
      #right-rail > .rail-section { margin: 6px 0; }
      #right-rail > .rail-section > header { padding: 8px 10px; min-height: 36px; }
      #right-rail > .rail-section > header h2 { font-size: 14px; line-height: 1.1; }
      #right-rail > .rail-section > header .section-toggle { font-size: 18px; min-width: 36px; min-height: 36px; }

      #right-rail > .rail-section > .section-body { padding: 8px 8px 10px; }
      #right-rail > .rail-section.collapsed > .section-body { display: none; }

      /* Lists inside right rail more compact */
      #right-rail .list, #right-rail ul, #right-rail ol { margin: 4px 0; padding: 0; }
      #right-rail .list li, #right-rail li { margin: 2px 0; }

      /* Messenger thread composer compact */
      #thread-composer { margin-top: 6px; }
      #thread-composer textarea { min-height: 56px; padding: 8px; }

      /* Compact reminders form */
      #new-reminder input, #new-reminder button { padding: 6px 8px; }

      /* Ensure keypad section shows all content without inner scrollbars */
      #right-rail .rail-section:has(> header > h2):where(:defined) {}
      /* Fallback target by index: look for Keypad title text */
      /* We can't rely on :contains() widely; ensure general overflow rules: */
      #right-rail .rail-section > .section-body { overflow: visible; max-height: none; }

      /* Slightly narrower right rail on phones to avoid horizontal scroll */
      :root { --right-col: 300px; }
      @media (max-width: 680px) {
        :root { --right-col: 270px; }
        #right-rail > .rail-section > header { padding: 10px 12px; } /* keep header tap-friendly */
      }
    `;
    document.head.appendChild(css);
  })();

  // Make the entire header clickable as a toggle (if not already bound)
  (function widenHeaderToggles(){
    document.querySelectorAll('#right-rail .rail-section > header').forEach(hdr => {
      if (hdr.__sm_hdr_compact_bound__) return;
      hdr.__sm_hdr_compact_bound__ = true;
      hdr.addEventListener('click', (e)=>{
        // Do not toggle when clicking on interactive controls inside header except the chevron
        const target = e.target;
        if (target && target.closest && target.closest('button, input, select, .section-toggle') && !target.classList.contains('section-toggle')) return;
        const sec = hdr.closest('.rail-section');
        if (!sec) return;
        sec.classList.toggle('collapsed');

        // Mobile accordion: collapse siblings on narrow screens
        if (window.matchMedia('(max-width: 680px)').matches && !sec.classList.contains('collapsed')){
          const sibs = Array.from(sec.parentElement.querySelectorAll(':scope > .rail-section')).filter(s=>s!==sec);
          sibs.forEach(s => s.classList.add('collapsed'));
        }
      });
      const btn = hdr.querySelector('.section-toggle');
      if (btn && !btn.__sm_stop_bubble2__) {
        btn.__sm_stop_bubble2__ = true;
        btn.addEventListener('click', (e)=> e.stopPropagation());
      }
    });
  })();

  // If Keypad exists, ensure its dialer + grid are compact enough (cooperate with previous patch)
  (function tightenKeypad(){
    const kp = document.getElementById('keypad');
    if (!kp) return;
    kp.classList.add('keypad'); // ensure grid styles apply
    // Reduce spacing if custom gaps previously set
    kp.style.setProperty('gap','6px');
  })();
})();
// === [/SM_RAIL_COMPACT_V1] ===


/* ===== APPEND PATCH BLOCK BELOW ===== */

/* === SM_v7.7.5 Mobile Dialer + VoIP + Pin + PermCache PATCH (non-destructive) === */
(function(){
  "use strict";

  // ---- Mobile detection + global flag ----
  function isMobileDevice(){
    try{
      var ua = (navigator.userAgent||"").toLowerCase();
      var touch = (('ontouchstart' in window) || (navigator.maxTouchPoints||0) > 0);
      var small = Math.min(window.innerWidth||0, screen.width||0) <= 820;
      return /(iphone|ipad|ipod|android|mobile)/.test(ua) || (touch && small);
    }catch(e){ return false; }
  }
  window.SM_isMobile = isMobileDevice();

  // ---- Phone/IP heuristics ----
  function normalizePhone(s){
    s = String(s||"");
    var plus = s.trim().startsWith("+") ? "+" : "";
    var digits = s.replace(/[^\d]/g,"");
    return plus + digits;
  }
  function looksLikePhone(s){
    s = String(s||"");
    // allow digits, spaces, (), -, +; must have 7-15 digits after stripping
    var digits = s.replace(/[^\d]/g,"");
    if (digits.length < 7 || digits.length > 15) return false;
    // reject if clearly like an IP or contains letters
    if (/[a-z]/i.test(s)) return false;
    if (isIPLike(s)) return false;
    return true;
  }
  function isIPv4(s){
    var m = /^(\d{1,3}\.){3}\d{1,3}$/.test(s);
    if (!m) return false;
    var parts = s.split(".");
    for (var i=0;i<4;i++){
      var n = +parts[i];
      if (n<0 || n>255) return false;
    }
    return true;
  }
  function isIPv6(s){
    // light heuristic (not exhaustive): groups of hex separated by :
    return /^([0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}$/i.test(s);
  }
  function isIPLike(s){
    s = String(s||"").trim();
    if (isIPv4(s) || isIPv6(s)) return true;
    // hostnames like mybox.lan or 10.0.0.5:5060
    if (/^[a-z0-9.-]+\.[a-z]{2,}$/i.test(s)) return true;
    if (/^\d{1,3}(\.\d{1,3}){3}:\d{2,5}$/.test(s)) return true;
    return false;
  }

  // ---- Permission cache helpers ----
  function getStoredPerm(name){
    try{ return localStorage.getItem("sm_perm_"+name) || ""; } catch(e){ return ""; }
  }
  function setStoredPerm(name, val){
    try{ localStorage.setItem("sm_perm_"+name, String(val)); }catch(e){}
  }
  window.SM_getPerm = getStoredPerm;
  window.SM_setPerm = setStoredPerm;

  // Wrap ensureMicOnce / ensureWebcam if present
  (function(){
    function wrap(name, fnName){
      try{
        var orig = window[fnName];
        if (typeof orig !== "function") return;
        window[fnName] = async function(){
          try{
            var res = await orig.apply(this, arguments);
            setStoredPerm(name, "granted");
            return res;
          }catch(e){
            setStoredPerm(name, "denied");
            throw e;
          }
        };
      }catch(e){/*no-op*/}
    }
    wrap("mic", "ensureMicOnce");
    wrap("cam", "ensureWebcam");
  })();

  // ---- Native dial handoff (mobile) ----
  async function nativeDialIfMobile(peer, opts){
    try{
      opts = opts || {};
      if (!window.SM_isMobile) return false;
      if (!looksLikePhone(peer)) return false;
      var tel = "tel:" + normalizePhone(peer);
      // Reliable way: temporary anchor and programmatic click
      var a = document.createElement("a");
      a.setAttribute("href", tel);
      a.style.display = "none";
      document.body.appendChild(a);
      a.click();
      setTimeout(function(){ try{ document.body.removeChild(a); }catch(e){}; }, 2000);
      return true;
    }catch(e){
      console.warn("[SM] nativeDialIfMobile failed:", e);
      return false;
    }
  }
  window.SM_nativeDialIfMobile = nativeDialIfMobile;

  // ---- Network VoIP (SMNP) handoff to backend (graceful if API missing) ----
  async function SM_netCall(peer, options){
    options = options || {};
    var base = (typeof window.SM_getApi === "function") ? window.SM_getApi("/net/call")
              : ((window.SM_API_BASE || "/api") + "/net/call");
    var body = {
      peer: String(peer||""),
      video: !!options.video,
      protocol: "SMNP",
      meta: { source: "webui", ts: Date.now() }
    };
    try{
      var r = await fetch(base, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        credentials: "same-origin"
      });
      if (!r.ok){
        console.warn("[SM] netCall backend returned", r.status);
      }
      return r.ok;
    }catch(e){
      console.warn("[SM] netCall fetch failed:", e);
      return false;
    }
  }
  window.SM_netCall = SM_netCall;

  // ---- dialCall wrapper (non-destructive) ----
  var _orig_dialCall = (typeof window.dialCall === "function") ? window.dialCall : null;
  window.dialCall = async function(peer, video){
    try{
      // prefer explicit param; else read from dial input if present
      if (!peer){
        var input = document.getElementById("dial-input") || document.querySelector('input[name="dial"]');
        if (input && input.value) peer = input.value.trim();
      }
      var val = String(peer||"").trim();
      var isPhone = looksLikePhone(val);
      var isIP = isIPLike(val);

      // 1) Mobile native handoff for phone-like
      if (isPhone && window.SM_isMobile){
        var handed = await nativeDialIfMobile(val, { video: !!video });
        if (handed) return true;
      }

      // 2) Call original implementation if present
      if (_orig_dialCall){
        return _orig_dialCall.apply(this, arguments);
      }

      // 3) Fallbacks
      if (isPhone){
        try{
          // Safe tel: fallback even on desktop—OS may hand to softphone
          window.location.href = "tel:" + normalizePhone(val);
          return true;
        }catch(e){ /* continue */ }
      }

      // 4) Network VoIP via SMNP for IP/SIP/host
      if (isIP || /^sip:/i.test(val)){
        return await SM_netCall(val, { video: !!video });
      }

      // 5) Last resort: try backend anyway
      return await SM_netCall(val, { video: !!video });
    }catch(e){
      console.warn("[SM] dialCall wrapper error:", e);
      try{ if (_orig_dialCall) return _orig_dialCall.apply(this, arguments); }catch(_){}
      return false;
    }
  };

  // ---- Long-press pin/unpin on right-rail headers ----
  function initPinHeaders(){
    var rail = document.getElementById("right-rail");
    if (!rail) return;
    var PIN_KEY = "sm_pinned_section";
    function getIdForSection(sec){
      // use first <h2> text as id fallback
      var id = sec.getAttribute("id");
      if (id) return id;
      var h2 = sec.querySelector("header h2");
      var name = h2 ? h2.textContent.trim().toLowerCase().replace(/\s+/g,"-") : "section";
      // assign a stable synthetic id if missing
      id = "rr-" + name;
      if (!sec.hasAttribute("id")) sec.setAttribute("id", id);
      return id;
    }
    function readPinned(){ try{ return localStorage.getItem(PIN_KEY) || ""; }catch(e){ return ""; } }
    function writePinned(v){ try{ localStorage.setItem(PIN_KEY, v||""); }catch(e){} }

    var pinned = readPinned();
    var secs = rail.querySelectorAll(".rail-section");
    secs.forEach(function(sec){
      var id = getIdForSection(sec);
      if (pinned && id === pinned) sec.classList.add("pinned");
      var header = sec.querySelector("header");
      if (!header) return;

      var pressTimer = null, pressed = false;
      function startPress(evt){
        pressed = true;
        pressTimer = setTimeout(function(){
          if (!pressed) return;
          // toggle pin
          var currentPinned = readPinned();
          if (sec.classList.contains("pinned")){
            sec.classList.remove("pinned");
            if (currentPinned === id) writePinned("");
          }else{
            // remove other pins
            secs.forEach(function(s){ s.classList.remove("pinned"); });
            sec.classList.add("pinned");
            writePinned(id);
          }
          // subtle visual hint
          try{ header.classList.add("pin-flash"); setTimeout(function(){ header.classList.remove("pin-flash"); }, 600); }catch(e){}
        }, 650); // long-press threshold
      }
      function cancelPress(){ pressed = false; if (pressTimer){ clearTimeout(pressTimer); pressTimer=null; } }

      header.addEventListener("mousedown", startPress);
      header.addEventListener("touchstart", startPress, {passive:true});
      header.addEventListener("mouseup", cancelPress);
      header.addEventListener("mouseleave", cancelPress);
      header.addEventListener("touchend", cancelPress);
      header.addEventListener("touchcancel", cancelPress);
    });
  }
  initPinHeaders();

  // ---- Mobile accordion behavior (respects pinned) ----
  (function mobileAccordionGuard(){
    var rail = document.getElementById("right-rail");
    if (!rail) return;
    var PIN_KEY = "sm_pinned_section";
    function readPinned(){ try{ return localStorage.getItem(PIN_KEY) || ""; }catch(e){ return ""; } }

    function applyAccordion(){
      if (!window.SM_isMobile) return;
      var narrow = (Math.min(window.innerWidth||0, screen.width||0) <= 680);
      if (!narrow) return;
      var pinned = readPinned();
      var sections = Array.from(rail.querySelectorAll(".rail-section"));
      // ensure only one open (not collapsed), never auto-collapse pinned
      var open = sections.filter(s => !s.classList.contains("collapsed"));
      // if multiple open and a pinned exists, collapse others
      if (pinned){
        sections.forEach(function(s){
          var id = s.getAttribute("id");
          if (id && id === pinned) return; // keep open
          if (!s.classList.contains("collapsed")) s.classList.add("collapsed");
        });
        return;
      }
      // otherwise keep the first open, collapse others
      for (var i=1;i<open.length;i++){ open[i].classList.add("collapsed"); }
    }

    // Also intercept header clicks to accordion others when one opens
    rail.addEventListener("click", function(e){
      var header = e.target.closest(".rail-section > header");
      if (!header) return;
      var sec = header.parentElement;
      // Toggle already handled by existing code; delay to read final state
      setTimeout(function(){
        if (!(window.SM_isMobile && (Math.min(window.innerWidth||0, screen.width||0)<=680))) return;
        var pinned = readPinned();
        if (pinned && sec.id !== pinned) return; // don't auto-change when pin rules apply
        if (!sec.classList.contains("collapsed")){
          // collapse siblings (except pinned)
          rail.querySelectorAll(".rail-section").forEach(function(s){
            if (s===sec) return;
            if (pinned && s.id===pinned) return;
            s.classList.add("collapsed");
          });
        }
      }, 0);
    });

    applyAccordion();
    window.addEventListener("resize", applyAccordion);
  })();

  // ---- Wire Call/Video buttons to wrapper (safe) ----
  (function wireNativeHandoff(){
    function pickPeer(){
      var input = document.getElementById("dial-input") || document.querySelector('input[name="dial"]');
      if (input && input.value) return input.value.trim();
      // try selection in contacts
      var sel = document.querySelector('[data-selected-peer]');
      if (sel) return (sel.getAttribute('data-selected-peer')||"").trim();
      return "";
    }
    function bind(btn, video){
      if (!btn) return;
      btn.addEventListener("click", function(ev){
        ev.preventDefault();
        var peer = btn.getAttribute("data-peer") || pickPeer();
        if (!peer){ console.warn("[SM] No peer to dial"); return; }
        window.dialCall(peer, !!video);
      });
    }
    bind(document.getElementById("callBtn"), false);
    bind(document.getElementById("videoBtn"), true);
    // Delegated binds for dynamic elements with data-action
    document.addEventListener("click", function(e){
      var a = e.target.closest('[data-action="call"],[data-action="video"]');
      if (!a) return;
      e.preventDefault();
      var peer = a.getAttribute("data-peer") || pickPeer();
      var isVid = a.getAttribute("data-action")==="video";
      if (!peer){ console.warn("[SM] No peer to dial"); return; }
      window.dialCall(peer, isVid);
    });
  })();

})();

// ============================================================================
//  Authentication Functions
// ============================================================================

function initAuth() {
  const loginBtn = document.getElementById('loginBtn');
  const authModal = document.getElementById('authModal');
  const authModalClose = document.getElementById('authModalClose');
  const registerForm = document.getElementById('registerForm');
  const loginForm = document.getElementById('loginForm');
  const verifyForm = document.getElementById('verifyForm');
  const showLogin = document.getElementById('showLogin');
  const showRegister = document.getElementById('showRegister');
  const userInfo = document.getElementById('userInfo');

  if (!loginBtn || !authModal) {
    console.warn('[Phase B] Authentication UI elements not found');
    return;
  }

  // Ensure modal is hidden on initialization (safety check)
  authModal.hidden = true;
  
  // Ensure forms are in correct initial state
  if (loginForm) loginForm.hidden = false;
  if (registerForm) registerForm.hidden = true;
  if (verifyForm) verifyForm.hidden = true;

  // Check if already logged in
  if (authToken && currentUser) {
    updateUIForLoggedInUser();
  }

  // Login button click - only show modal when explicitly clicked
  loginBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (authToken && currentUser) {
      // Already logged in, show user menu or logout
      showUserMenu();
    } else {
      // Show login modal with login form visible
      authModal.hidden = false;
      if (loginForm) loginForm.hidden = false;
      if (registerForm) registerForm.hidden = true;
      if (verifyForm) verifyForm.hidden = true;
      clearAuthMessage();
    }
  });

  // Close modal - multiple event bindings for reliability
  function closeModal() {
    authModal.hidden = true;
    clearAuthMessage();
  }

  // Close button click
  if (authModalClose) {
    authModalClose.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeModal();
    });
  }

  // Close modal when clicking on the backdrop (outside modal content)
  authModal.addEventListener('click', (e) => {
    // Only close if clicking directly on the modal backdrop, not the content
    if (e.target === authModal) {
      closeModal();
    }
  });

  // Close modal on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !authModal.hidden) {
      closeModal();
    }
  });

  // Switch between login/register
  showLogin.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm.hidden = false;
    registerForm.hidden = true;
    verifyForm.hidden = true;
    clearAuthMessage();
  });

  showRegister.addEventListener('click', (e) => {
    e.preventDefault();
    registerForm.hidden = false;
    loginForm.hidden = true;
    verifyForm.hidden = true;
    clearAuthMessage();
  });

  // Register form submit
  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('regEmail').value;
    const name = document.getElementById('regName').value;
    const password = document.getElementById('regPassword').value;
    const pin = document.getElementById('regPin').value;

    try {
      const response = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ email, display_name: name, password, pin })
      });

      const data = await response.json();

      if (response.ok) {
        showAuthMessage('Registration successful! Please check your email for verification code.', 'success');
        registerForm.hidden = true;
        verifyForm.hidden = false;
        document.getElementById('verifyForm').dataset.email = email;
      } else {
        showAuthMessage(data.error || 'Registration failed', 'error');
      }
    } catch (error) {
      console.error('[Phase B] Registration error:', error);
      showAuthMessage('Network error. Please try again.', 'error');
    }
  });

  // Login form submit
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const pin = document.getElementById('loginPin').value;

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ email, password, pin })
      });

      const data = await response.json();

      if (response.ok) {
        authToken = data.token;
        currentUser = data.user;
        localStorage.setItem('sarahmemory_token', authToken);
        localStorage.setItem('sarahmemory_user', JSON.stringify(currentUser));

        authModal.hidden = true;
        updateUIForLoggedInUser();
        showAuthMessage('Login successful!', 'success');

        // Load user's conversations
        loadUserConversations();
      } else {
        showAuthMessage(data.error || 'Login failed', 'error');
      }
    } catch (error) {
      console.error('[Phase B] Login error:', error);
      showAuthMessage('Network error. Please try again.', 'error');
    }
  });

  // Verify form submit
  verifyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const code = document.getElementById('verifyCode').value;
    const email = verifyForm.dataset.email;

    try {
      const response = await fetch(`${API_BASE}/auth/verify-email`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ email, code })
      });

      const data = await response.json();

      if (response.ok) {
        showAuthMessage('Email verified! You can now log in.', 'success');
        setTimeout(() => {
          verifyForm.hidden = true;
          loginForm.hidden = false;
        }, 2000);
      } else {
        showAuthMessage(data.error || 'Verification failed', 'error');
      }
    } catch (error) {
      console.error('[Phase B] Verification error:', error);
      showAuthMessage('Network error. Please try again.', 'error');
    }
  });

  console.log('[Phase B] Authentication initialized');
}

function updateUIForLoggedInUser() {
  const loginBtn = document.getElementById('loginBtn');
  const userInfo = document.getElementById('userInfo');

  if (currentUser && loginBtn && userInfo) {
    loginBtn.textContent = currentUser.display_name || currentUser.email;
    userInfo.textContent = `👤 ${currentUser.display_name}`;
    userInfo.style.display = 'block';
  }
}

function showUserMenu() {
  // TODO: Implement user dropdown menu with:
  // - View profile
  // - Settings
  // - Logout
  if (confirm('Logout?')) {
    logout();
  }
}

function logout() {
  authToken = null;
  currentUser = null;
  localStorage.removeItem('sarahmemory_token');
  localStorage.removeItem('sarahmemory_user');

  const loginBtn = document.getElementById('loginBtn');
  const userInfo = document.getElementById('userInfo');

  if (loginBtn) loginBtn.textContent = 'Login';
  if (userInfo) userInfo.style.display = 'none';

  // Clear conversations
  const threadList = document.getElementById('thread-list');
  if (threadList) threadList.innerHTML = '';

  console.log('[Phase B] User logged out');
}

function showAuthMessage(message, type) {
  const authMessage = document.getElementById('authMessage');
  if (authMessage) {
    authMessage.textContent = message;
    authMessage.className = `auth-message ${type}`;
    authMessage.style.display = 'block';
  }
}

function clearAuthMessage() {
  const authMessage = document.getElementById('authMessage');
  if (authMessage) {
    authMessage.style.display = 'none';
  }
}

async function loadUserConversations() {
  if (!authToken) return;

  try {
    const response = await fetch(`${API_BASE}/user/conversations`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });

    if (response.ok) {
      const conversations = await response.json();
      // TODO: Display conversations in left rail
      console.log('[Phase B] Loaded conversations:', conversations);
    }
  } catch (error) {
    console.error('[Phase B] Error loading conversations:', error);
  }
}

// Initialize authentication on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initAuth();
  console.log('[Phase B] Authentication system ready');
});
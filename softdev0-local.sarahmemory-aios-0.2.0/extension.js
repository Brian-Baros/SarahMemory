const vscode = require('vscode');
const http = require('http');
const https = require('https');
const { URL } = require('url');

const SARAH_MEMORY_CONTAINER_ID = 'sarahMemorySidebar';
const SARAH_MEMORY_VIEW_ID = 'sarahMemory.chatView';

let outputChannel;
let statusBarItem;
let currentPanel;
let currentSidebarView;
let lastReply = '';
let healthTimer;

function activate(context) {
  outputChannel = vscode.window.createOutputChannel('SarahMemory AiOS');
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'sarahMemory.focusChat';
  statusBarItem.text = '$(comment-discussion) SarahMemory';
  statusBarItem.tooltip = 'Focus SarahMemory Chat';
  statusBarItem.show();

  context.subscriptions.push(outputChannel, statusBarItem);

  const sidebarProvider = new SarahMemorySidebarProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SARAH_MEMORY_VIEW_ID, sidebarProvider, {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.focusChat', async () => {
    await focusChatSidebar();
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.openPanel', async () => {
    await openPanel(context);
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.ask', async () => {
    const prompt = await vscode.window.showInputBox({
      prompt: 'Ask SarahMemory',
      placeHolder: 'Type a prompt for SarahMemory AiOS'
    });
    if (!prompt) {
      return;
    }
    await focusChatSidebar();
    await sendPromptAndPresent(prompt, { revealPanel: false, revealSidebar: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.sendSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active editor found.');
      return;
    }
    const selection = editor.document.getText(editor.selection).trim();
    if (!selection) {
      vscode.window.showWarningMessage('No selected text found.');
      return;
    }
    await focusChatSidebar();
    await sendPromptAndPresent(selection, { revealPanel: false, revealSidebar: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.checkHealth', async () => {
    await runHealthCheck({ announce: true, revealOutputOnError: true, postToViews: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.startAiOS', async () => {
    await startDebugLaunch(getConfig('launchConfigMain'));
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.startApi', async () => {
    await startDebugLaunch(getConfig('launchConfigApi'));
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.insertLastReply', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active editor found.');
      return;
    }
    if (!lastReply) {
      vscode.window.showWarningMessage('There is no SarahMemory reply to insert yet.');
      return;
    }
    await editor.edit((editBuilder) => editBuilder.insert(editor.selection.active, lastReply));
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.setApiBaseUrl', async () => {
    const baseUrl = await promptForApiBaseUrl(true);
    if (baseUrl) {
      broadcastPayload({ type: 'config', baseUrl, lastReply });
    }
  }));

  if (getConfig('autoHealthCheck')) {
    runHealthCheck({ announce: false, revealOutputOnError: false, postToViews: true }).catch(() => {});
    healthTimer = setInterval(() => {
      runHealthCheck({ announce: false, revealOutputOnError: false, postToViews: true }).catch(() => {});
    }, 30000);
    context.subscriptions.push({
      dispose: () => {
        clearInterval(healthTimer);
      }
    });
  }

  if (getConfig('autoFocusSidebarOnStartup')) {
    setTimeout(() => {
      focusChatSidebar().catch(() => {});
    }, 800);
  }
}

function deactivate() {
  if (healthTimer) {
    clearInterval(healthTimer);
    healthTimer = undefined;
  }
}

class SarahMemorySidebarProvider {
  constructor(context) {
    this.context = context;
  }

  resolveWebviewView(webviewView) {
    currentSidebarView = webviewView;
    webviewView.title = 'SarahMemory Chat';
    webviewView.description = 'Primary SarahMemory workspace';
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.context.extensionUri]
    };
    webviewView.webview.html = getWebviewHtml(webviewView.webview, { mode: 'sidebar' });
    attachWebviewMessageHandler(webviewView.webview, { source: 'sidebar' });

    webviewView.onDidDispose(() => {
      if (currentSidebarView === webviewView) {
        currentSidebarView = undefined;
      }
    });
  }
}

function getConfig(key) {
  return vscode.workspace.getConfiguration('sarahMemory').get(key);
}

async function setConfig(key, value) {
  return vscode.workspace.getConfiguration('sarahMemory').update(key, value, vscode.ConfigurationTarget.Global);
}

async function focusChatSidebar() {
  await vscode.commands.executeCommand(`workbench.view.extension.${SARAH_MEMORY_CONTAINER_ID}`);
  if (currentSidebarView) {
    try {
      currentSidebarView.show?.(true);
    } catch {
      // Ignore best-effort focus failure.
    }
    return currentSidebarView;
  }
  return undefined;
}

async function promptForApiBaseUrl(forcePrompt = false) {
  let baseUrl = String(getConfig('apiBaseUrl') || '').trim();
  if (baseUrl && !forcePrompt) {
    return normalizeBaseUrl(baseUrl);
  }

  const value = await vscode.window.showInputBox({
    prompt: 'Enter the base URL for the running SarahMemory API',
    placeHolder: 'http://127.0.0.1:5000',
    value: baseUrl || 'http://127.0.0.1:5000',
    ignoreFocusOut: true,
    validateInput: (input) => {
      const test = input.trim();
      if (!test) {
        return 'A base URL is required.';
      }
      try {
        const url = new URL(test);
        if (!/^https?:$/i.test(url.protocol)) {
          return 'Only http:// or https:// URLs are supported.';
        }
        return null;
      } catch {
        return 'Enter a valid URL, for example http://127.0.0.1:5000';
      }
    }
  });

  if (!value) {
    return undefined;
  }

  baseUrl = normalizeBaseUrl(value);
  await setConfig('apiBaseUrl', baseUrl);
  return baseUrl;
}

function normalizeBaseUrl(input) {
  return String(input || '').trim().replace(/\/+$/, '');
}

function buildEndpoints(baseUrl) {
  const base = normalizeBaseUrl(baseUrl);
  return {
    base,
    health: `${base}/api/health`,
    chat: `${base}/api/chat`
  };
}

async function startDebugLaunch(configName) {
  const launchName = String(configName || '').trim();
  if (!launchName) {
    vscode.window.showErrorMessage('Launch configuration name is empty. Update the SarahMemory extension settings.');
    return;
  }
  const started = await vscode.debug.startDebugging(undefined, launchName);
  if (!started) {
    vscode.window.showErrorMessage(`Unable to start launch configuration '${launchName}'. Check launch.json.`);
  }
}

async function runHealthCheck(options = {}) {
  const announce = Boolean(options.announce);
  const revealOutputOnError = Boolean(options.revealOutputOnError);
  const postToViews = Boolean(options.postToViews);
  const baseUrl = await promptForApiBaseUrl(false);
  if (!baseUrl) {
    updateStatus('disconnected', 'SarahMemory: API URL not set');
    if (announce) {
      vscode.window.showWarningMessage('SarahMemory API base URL is not configured.');
    }
    if (postToViews) {
      broadcastPayload({ type: 'health', body: { ok: false, error: 'API base URL not configured' } });
    }
    return { ok: false, error: 'API base URL not configured' };
  }

  const endpoints = buildEndpoints(baseUrl);
  try {
    const result = await requestJson('GET', endpoints.health, undefined, Number(getConfig('requestTimeoutMs') || 45000));
    const body = result.body || {};
    const ok = Boolean(body.ok !== false);
    if (ok) {
      const version = body.version ? ` v${body.version}` : '';
      const model = body.routing && body.routing.model ? ` · ${body.routing.model}` : '';
      updateStatus('connected', `SarahMemory${version}${model}`);
      if (announce) {
        vscode.window.showInformationMessage(`SarahMemory connected${version}${model}`);
      }
    } else {
      updateStatus('warning', 'SarahMemory: health degraded');
      if (announce) {
        vscode.window.showWarningMessage('SarahMemory health check returned a degraded state.');
      }
    }
    if (postToViews) {
      broadcastPayload({ type: 'health', body });
    }
    return body;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`[health] ${message}`);
    updateStatus('disconnected', 'SarahMemory: offline');
    if (announce) {
      vscode.window.showErrorMessage(`SarahMemory health check failed: ${message}`);
    }
    if (revealOutputOnError) {
      outputChannel.show(true);
    }
    if (postToViews) {
      broadcastPayload({ type: 'health', body: { ok: false, error: message } });
    }
    return { ok: false, error: message };
  }
}

function updateStatus(state, label) {
  switch (state) {
    case 'connected':
      statusBarItem.text = '$(radio-tower) SarahMemory';
      break;
    case 'warning':
      statusBarItem.text = '$(warning) SarahMemory';
      break;
    default:
      statusBarItem.text = '$(circle-slash) SarahMemory';
      break;
  }
  statusBarItem.tooltip = label;
}

async function sendPromptAndPresent(prompt, options = {}) {
  const revealPanel = Boolean(options.revealPanel);
  const revealSidebar = Boolean(options.revealSidebar);
  if (revealSidebar) {
    await focusChatSidebar();
  }
  const response = await sendPrompt(prompt);
  if (response && response.reply) {
    if (revealPanel) {
      await openPanel();
    }
    broadcastPayload({
      type: 'reply',
      prompt,
      reply: response.reply,
      meta: response.raw.meta || response.raw.routing || {}
    });
    vscode.window.showInformationMessage('SarahMemory reply received.');
  }
  return response;
}

async function sendPrompt(prompt) {
  const baseUrl = await promptForApiBaseUrl(false);
  if (!baseUrl) {
    throw new Error('SarahMemory API base URL is not configured.');
  }

  const endpoints = buildEndpoints(baseUrl);
  outputChannel.appendLine(`[chat] -> ${prompt}`);
  const body = {
    text: prompt,
    source: 'vscode_extension',
    ui: 'vscode_extension'
  };
  const result = await requestJson('POST', endpoints.chat, body, Number(getConfig('requestTimeoutMs') || 45000));
  const raw = result.body || {};
  const reply = String(raw.presentation_reply || raw.reply || raw.response || raw.text || '').trim();
  if (!reply) {
    throw new Error(raw.error || 'SarahMemory returned an empty reply.');
  }
  lastReply = reply;
  outputChannel.appendLine(`[chat] <- ${reply}`);
  return { reply, raw };
}

async function openPanel(context) {
  const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : vscode.ViewColumn.One;
  if (currentPanel) {
    currentPanel.reveal(column);
    return currentPanel;
  }

  currentPanel = vscode.window.createWebviewPanel(
    'sarahMemoryPanel',
    'SarahMemory Full Chat',
    column || vscode.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true
    }
  );

  currentPanel.webview.html = getWebviewHtml(currentPanel.webview, { mode: 'panel' });
  attachWebviewMessageHandler(currentPanel.webview, { source: 'panel' });
  postToWebview(currentPanel.webview, {
    type: 'config',
    baseUrl: normalizeBaseUrl(String(getConfig('apiBaseUrl') || '')),
    lastReply
  });

  currentPanel.onDidDispose(() => {
    currentPanel = undefined;
  });

  return currentPanel;
}

function attachWebviewMessageHandler(webview, { source }) {
  webview.onDidReceiveMessage(async (message) => {
    try {
      switch (message.type) {
        case 'ready': {
          const baseUrl = normalizeBaseUrl(String(getConfig('apiBaseUrl') || ''));
          postToWebview(webview, { type: 'config', baseUrl, lastReply, source });
          break;
        }
        case 'sendPrompt': {
          const prompt = String(message.prompt || '').trim();
          if (!prompt) {
            return;
          }
          broadcastPayload({ type: 'pending', prompt });
          const response = await sendPrompt(prompt);
          broadcastPayload({
            type: 'reply',
            prompt,
            reply: response.reply,
            meta: response.raw.meta || response.raw.routing || {}
          });
          break;
        }
        case 'checkHealth': {
          await runHealthCheck({ announce: false, revealOutputOnError: true, postToViews: true });
          break;
        }
        case 'setBaseUrl': {
          const incoming = String(message.baseUrl || '').trim();
          if (!incoming) {
            return;
          }
          const normalized = normalizeBaseUrl(incoming);
          new URL(normalized);
          await setConfig('apiBaseUrl', normalized);
          broadcastPayload({ type: 'config', baseUrl: normalized, lastReply });
          vscode.window.showInformationMessage(`SarahMemory API base URL set to ${normalized}`);
          break;
        }
        case 'startMain': {
          await startDebugLaunch(getConfig('launchConfigMain'));
          break;
        }
        case 'startApi': {
          await startDebugLaunch(getConfig('launchConfigApi'));
          break;
        }
        case 'insertLastReply': {
          await vscode.commands.executeCommand('sarahMemory.insertLastReply');
          break;
        }
        case 'openFullPanel': {
          await openPanel();
          break;
        }
      }
    } catch (error) {
      const messageText = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[${source}-error] ${messageText}`);
      broadcastPayload({ type: 'error', error: messageText });
      vscode.window.showErrorMessage(`SarahMemory ${source} error: ${messageText}`);
    }
  });
}

function broadcastPayload(payload) {
  if (currentSidebarView) {
    postToWebview(currentSidebarView.webview, payload);
  }
  if (currentPanel) {
    postToWebview(currentPanel.webview, payload);
  }
}

function postToWebview(webview, payload) {
  if (!webview) {
    return;
  }
  webview.postMessage(payload);
}

function getWebviewHtml(webview, options = {}) {
  const mode = String(options.mode || 'sidebar');
  const nonce = String(Date.now()) + String(Math.random()).slice(2);
  const isSidebar = mode === 'sidebar';
  const title = isSidebar ? 'SarahMemory Chat' : 'SarahMemory Full Chat';
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title}</title>
  <style>
    :root {
      color-scheme: light dark;
      --gap: 10px;
      --radius: 8px;
      --border: rgba(127, 127, 127, 0.35);
      --surface: var(--vscode-editorWidget-background);
      --surfaceAlt: color-mix(in srgb, var(--surface) 88%, transparent);
      --accent: var(--vscode-button-background);
      --accentFg: var(--vscode-button-foreground);
      --muted: var(--vscode-descriptionForeground);
    }
    * { box-sizing: border-box; }
    body {
      font-family: var(--vscode-font-family);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
      margin: 0;
      padding: ${isSidebar ? '10px' : '14px'};
    }
    .root {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: var(--gap);
      height: calc(100vh - ${isSidebar ? '20px' : '28px'});
    }
    .banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
    }
    .bannerTitle {
      font-weight: 600;
    }
    .toolbar, .settings, .composer {
      display: grid;
      gap: var(--gap);
    }
    .toolbar {
      grid-template-columns: repeat(${isSidebar ? '3' : '5'}, max-content);
      align-items: center;
      flex-wrap: wrap;
    }
    .toolbarWrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .settings {
      grid-template-columns: 1fr auto;
      align-items: center;
    }
    .log {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px;
      overflow: auto;
      background: var(--surface);
      display: flex;
      flex-direction: column-reverse;
      gap: 10px;
    }
    .entry {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surfaceAlt);
    }
    .promptLabel, .replyLabel {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .prompt, .reply { white-space: pre-wrap; word-break: break-word; }
    .reply { margin-top: 8px; }
    .meta { font-size: 11px; opacity: 0.75; margin-top: 8px; white-space: pre-wrap; }
    input, textarea, button { font: inherit; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
    }
    textarea { min-height: ${isSidebar ? '86px' : '120px'}; resize: vertical; }
    button {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 6px 12px;
      cursor: pointer;
      color: var(--accentFg);
      background: var(--accent);
    }
    button.secondary {
      color: var(--vscode-button-secondaryForeground);
      background: var(--vscode-button-secondaryBackground);
    }
    .status {
      font-size: 12px;
      opacity: 0.85;
      color: var(--muted);
    }
    .composerFooter {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>
  <div class="root">
    <div class="banner">
      <div>
        <div class="bannerTitle">${title}</div>
        <div class="status" id="status">Idle</div>
      </div>
      ${isSidebar ? '<button id="openFullPanel" class="secondary">Full Panel</button>' : ''}
    </div>
    <div class="toolbarWrap">
      <button id="startMain">Start AiOS Main</button>
      <button id="startApi">Start Flask API</button>
      <button id="health" class="secondary">Check Health</button>
      <button id="insert" class="secondary">Insert Last Reply</button>
    </div>
    <div class="settings">
      <input id="baseUrl" type="text" placeholder="http://127.0.0.1:5000" />
      <button id="saveBase" class="secondary">Save URL</button>
    </div>
    <div id="log" class="log" aria-live="polite"></div>
    <div class="composer">
      <textarea id="prompt" placeholder="Ask SarahMemory..."></textarea>
      <div class="composerFooter">
        <button id="send">Send</button>
        <span class="status">SarahMemory owns this workspace surface.</span>
      </div>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const log = document.getElementById('log');
    const promptInput = document.getElementById('prompt');
    const statusNode = document.getElementById('status');
    const baseUrlInput = document.getElementById('baseUrl');

    function setStatus(text) {
      statusNode.textContent = text;
    }

    function addEntry(kind, prompt, reply, meta) {
      const wrapper = document.createElement('div');
      wrapper.className = 'entry';

      const promptLabel = document.createElement('div');
      promptLabel.className = 'promptLabel';
      promptLabel.textContent = kind === 'health' ? 'System' : 'You';
      wrapper.appendChild(promptLabel);

      const promptNode = document.createElement('div');
      promptNode.className = 'prompt';
      promptNode.textContent = prompt;
      wrapper.appendChild(promptNode);

      if (reply) {
        const replyLabel = document.createElement('div');
        replyLabel.className = 'replyLabel';
        replyLabel.textContent = 'SarahMemory';
        wrapper.appendChild(replyLabel);

        const replyNode = document.createElement('div');
        replyNode.className = 'reply';
        replyNode.textContent = reply;
        wrapper.appendChild(replyNode);
      }

      if (meta) {
        const metaNode = document.createElement('div');
        metaNode.className = 'meta';
        metaNode.textContent = typeof meta === 'string' ? meta : JSON.stringify(meta, null, 2);
        wrapper.appendChild(metaNode);
      }

      log.prepend(wrapper);
    }

    document.getElementById('send').addEventListener('click', () => {
      const prompt = promptInput.value.trim();
      if (!prompt) {
        return;
      }
      vscode.postMessage({ type: 'sendPrompt', prompt });
      promptInput.value = '';
      setStatus('Sending...');
    });

    document.getElementById('health').addEventListener('click', () => {
      vscode.postMessage({ type: 'checkHealth' });
      setStatus('Checking health...');
    });

    document.getElementById('saveBase').addEventListener('click', () => {
      vscode.postMessage({ type: 'setBaseUrl', baseUrl: baseUrlInput.value.trim() });
    });

    document.getElementById('startMain').addEventListener('click', () => vscode.postMessage({ type: 'startMain' }));
    document.getElementById('startApi').addEventListener('click', () => vscode.postMessage({ type: 'startApi' }));
    document.getElementById('insert').addEventListener('click', () => vscode.postMessage({ type: 'insertLastReply' }));

    const openFullPanel = document.getElementById('openFullPanel');
    if (openFullPanel) {
      openFullPanel.addEventListener('click', () => vscode.postMessage({ type: 'openFullPanel' }));
    }

    promptInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        document.getElementById('send').click();
      }
    });

    window.addEventListener('message', (event) => {
      const message = event.data || {};
      switch (message.type) {
        case 'config':
          if (message.baseUrl) {
            baseUrlInput.value = message.baseUrl;
          }
          if (message.lastReply) {
            setStatus('Last reply cached');
          }
          break;
        case 'pending':
          addEntry('pending', message.prompt, '', 'Pending...');
          setStatus('Waiting for SarahMemory...');
          break;
        case 'reply':
          addEntry('reply', message.prompt, message.reply, message.meta || null);
          setStatus('Reply received');
          break;
        case 'health': {
          const body = message.body || {};
          const ok = Boolean(body.ok);
          addEntry('health', 'Health Check', '', body);
          setStatus(ok ? 'Connected' : 'Health check failed');
          break;
        }
        case 'error':
          addEntry('error', 'Error', '', message.error || 'Unknown error');
          setStatus('Error');
          break;
      }
    });

    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
}

function requestJson(method, targetUrl, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    const url = new URL(targetUrl);
    const transport = url.protocol === 'https:' ? https : http;
    const payload = body ? Buffer.from(JSON.stringify(body), 'utf8') : undefined;

    const req = transport.request({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: `${url.pathname}${url.search}`,
      method,
      timeout: timeoutMs,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'Content-Length': payload ? payload.length : 0
      }
    }, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        raw += chunk;
      });
      res.on('end', () => {
        let parsed;
        try {
          parsed = raw ? JSON.parse(raw) : {};
        } catch {
          parsed = { raw };
        }
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
          resolve({ statusCode: res.statusCode, body: parsed });
        } else {
          const message = parsed && parsed.error ? parsed.error : raw || `HTTP ${res.statusCode}`;
          reject(new Error(message));
        }
      });
    });

    req.on('timeout', () => {
      req.destroy(new Error(`Request timed out after ${timeoutMs}ms`));
    });
    req.on('error', reject);
    if (payload) {
      req.write(payload);
    }
    req.end();
  });
}

module.exports = {
  activate,
  deactivate
};

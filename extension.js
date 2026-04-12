const vscode = require('vscode');
const http = require('http');
const https = require('https');
const { URL } = require('url');

let outputChannel;
let statusBarItem;
let currentPanel;
let lastReply = '';
let healthTimer;

function activate(context) {
  outputChannel = vscode.window.createOutputChannel('SarahMemory AiOS');
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'sarahMemory.openPanel';
  statusBarItem.text = '$(pulse) SarahMemory';
  statusBarItem.tooltip = 'SarahMemory AiOS';
  statusBarItem.show();

  context.subscriptions.push(outputChannel, statusBarItem);

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.openPanel', () => openPanel(context)));
  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.ask', async () => {
    const prompt = await vscode.window.showInputBox({
      prompt: 'Ask SarahMemory',
      placeHolder: 'Type a prompt for SarahMemory AiOS'
    });
    if (!prompt) return;
    await sendPromptAndPresent(prompt, { revealPanel: true });
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
    await sendPromptAndPresent(selection, { revealPanel: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.checkHealth', async () => {
    await runHealthCheck({ announce: true, revealOutputOnError: true });
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
    await editor.edit(editBuilder => editBuilder.insert(editor.selection.active, lastReply));
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.setApiBaseUrl', async () => {
    await promptForApiBaseUrl(true);
  }));

  if (getConfig('autoHealthCheck')) {
    runHealthCheck({ announce: false, revealOutputOnError: false }).catch(() => {});
    healthTimer = setInterval(() => {
      runHealthCheck({ announce: false, revealOutputOnError: false }).catch(() => {});
    }, 30000);
    context.subscriptions.push({ dispose: () => clearInterval(healthTimer) });
  }
}

function deactivate() {
  if (healthTimer) {
    clearInterval(healthTimer);
    healthTimer = undefined;
  }
}

function getConfig(key) {
  return vscode.workspace.getConfiguration('sarahMemory').get(key);
}

async function setConfig(key, value) {
  return vscode.workspace.getConfiguration('sarahMemory').update(key, value, vscode.ConfigurationTarget.Global);
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
      if (!test) return 'A base URL is required.';
      try {
        const u = new URL(test);
        if (!/^https?:$/i.test(u.protocol)) {
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
  const baseUrl = await promptForApiBaseUrl(false);
  if (!baseUrl) {
    updateStatus('disconnected', 'SarahMemory: API URL not set');
    if (announce) {
      vscode.window.showWarningMessage('SarahMemory API base URL is not configured.');
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
  const response = await sendPrompt(prompt);
  if (response && response.reply) {
    if (revealPanel) {
      const panel = openPanel();
      postToPanel(panel, {
        type: 'reply',
        prompt,
        reply: response.reply,
        meta: response.raw.meta || response.raw.routing || {}
      });
    }
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

function openPanel() {
  const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : vscode.ViewColumn.One;
  if (currentPanel) {
    currentPanel.reveal(column);
    return currentPanel;
  }

  currentPanel = vscode.window.createWebviewPanel(
    'sarahMemoryPanel',
    'SarahMemory AiOS',
    column || vscode.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true
    }
  );

  currentPanel.webview.html = getWebviewHtml(currentPanel.webview);

  currentPanel.webview.onDidReceiveMessage(async (message) => {
    try {
      switch (message.type) {
        case 'ready': {
          const baseUrl = normalizeBaseUrl(String(getConfig('apiBaseUrl') || ''));
          postToPanel(currentPanel, { type: 'config', baseUrl, lastReply });
          break;
        }
        case 'sendPrompt': {
          const prompt = String(message.prompt || '').trim();
          if (!prompt) return;
          postToPanel(currentPanel, { type: 'pending', prompt });
          const response = await sendPrompt(prompt);
          postToPanel(currentPanel, {
            type: 'reply',
            prompt,
            reply: response.reply,
            meta: response.raw.meta || response.raw.routing || {}
          });
          break;
        }
        case 'checkHealth': {
          const body = await runHealthCheck({ announce: false, revealOutputOnError: true });
          postToPanel(currentPanel, { type: 'health', body });
          break;
        }
        case 'setBaseUrl': {
          const incoming = String(message.baseUrl || '').trim();
          if (!incoming) return;
          const normalized = normalizeBaseUrl(incoming);
          new URL(normalized);
          await setConfig('apiBaseUrl', normalized);
          postToPanel(currentPanel, { type: 'config', baseUrl: normalized, lastReply });
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
      }
    } catch (error) {
      const messageText = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[panel-error] ${messageText}`);
      postToPanel(currentPanel, { type: 'error', error: messageText });
      vscode.window.showErrorMessage(`SarahMemory panel error: ${messageText}`);
    }
  });

  currentPanel.onDidDispose(() => {
    currentPanel = undefined;
  });

  return currentPanel;
}

function postToPanel(panel, payload) {
  if (!panel) return;
  panel.webview.postMessage(payload);
}

function getWebviewHtml(webview) {
  const nonce = String(Date.now()) + String(Math.random()).slice(2);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SarahMemory AiOS</title>
  <style>
    :root {
      color-scheme: light dark;
      --gap: 10px;
      --radius: 8px;
      --border: rgba(127, 127, 127, 0.35);
    }
    body {
      font-family: var(--vscode-font-family);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
      margin: 0;
      padding: 14px;
    }
    .root {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: var(--gap);
      height: calc(100vh - 28px);
    }
    .toolbar, .settings, .composer {
      display: grid;
      gap: var(--gap);
    }
    .toolbar {
      grid-template-columns: repeat(5, max-content);
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
      background: var(--vscode-editorWidget-background);
    }
    .entry {
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .entry:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
    .prompt { color: var(--vscode-textLink-foreground); white-space: pre-wrap; }
    .reply { white-space: pre-wrap; margin-top: 6px; }
    .meta { font-size: 11px; opacity: 0.75; margin-top: 6px; }
    input, textarea, button {
      font: inherit;
    }
    input, textarea {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 10px;
      color: var(--vscode-input-foreground);
      background: var(--vscode-input-background);
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 6px 12px;
      cursor: pointer;
      color: var(--vscode-button-foreground);
      background: var(--vscode-button-background);
    }
    button.secondary {
      color: var(--vscode-button-secondaryForeground);
      background: var(--vscode-button-secondaryBackground);
    }
    .status {
      font-size: 12px;
      opacity: 0.85;
    }
  </style>
</head>
<body>
  <div class="root">
    <div class="toolbar">
      <button id="startMain">Start AiOS Main</button>
      <button id="startApi">Start Flask API</button>
      <button id="health" class="secondary">Check Health</button>
      <button id="insert" class="secondary">Insert Last Reply</button>
      <span id="status" class="status">Idle</span>
    </div>
    <div class="settings">
      <input id="baseUrl" type="text" placeholder="http://127.0.0.1:5000" />
      <button id="saveBase" class="secondary">Save URL</button>
    </div>
    <div id="log" class="log" aria-live="polite"></div>
    <div class="composer">
      <textarea id="prompt" placeholder="Ask SarahMemory..."></textarea>
      <div>
        <button id="send">Send</button>
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
      const promptNode = document.createElement('div');
      promptNode.className = 'prompt';
      promptNode.textContent = kind === 'reply' ? 'You: ' + prompt : prompt;
      wrapper.appendChild(promptNode);
      if (reply) {
        const replyNode = document.createElement('div');
        replyNode.className = 'reply';
        replyNode.textContent = 'SarahMemory: ' + reply;
        wrapper.appendChild(replyNode);
      }
      if (meta) {
        const metaNode = document.createElement('div');
        metaNode.className = 'meta';
        metaNode.textContent = typeof meta === 'string' ? meta : JSON.stringify(meta);
        wrapper.appendChild(metaNode);
      }
      log.prepend(wrapper);
    }

    document.getElementById('send').addEventListener('click', () => {
      const prompt = promptInput.value.trim();
      if (!prompt) return;
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
          break;
        case 'reply':
          addEntry('reply', message.prompt, message.reply, message.meta || null);
          setStatus('Reply received');
          break;
        case 'health':
          addEntry('health', 'Health Check', '', message.body || {});
          setStatus(message.body && message.body.ok ? 'Connected' : 'Health check failed');
          break;
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
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Content-Length': payload ? payload.length : 0
      }
    }, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', chunk => raw += chunk);
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

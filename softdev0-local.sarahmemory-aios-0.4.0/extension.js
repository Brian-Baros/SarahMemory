const vscode = require('vscode');
const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const SARAH_MEMORY_CONTAINER_ID = 'sarahMemorySidebar';
const SARAH_MEMORY_VIEW_ID = 'sarahMemory.chatView';
const CHAT_PARTICIPANT_ID = 'softdev0-local.sarahmemory';
const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';
const DEFAULT_MODELS_ROOT = 'C:\\SarahMemory\\data\\models';
const DEFAULT_TERMINAL_NAME = 'SarahMemory AiOS';
const PROVIDER_KEY_MAP = {
  openai: ['OPENAI_API_KEY'],
  claude: ['CLAUDE_API_KEY', 'ANTHROPIC_API_KEY'],
  anthropic: ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
  mistral: ['MISTRAL_API_KEY'],
  gemini: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
  huggingface: ['HF_API_KEY', 'HF_TOKEN', 'HUGGINGFACE_API_KEY'],
  deepseek: ['DEEPSEEK_API_KEY'],
  groq: ['GROQ_API_KEY'],
  cohere: ['COHERE_API_KEY', 'CO_API_KEY'],
  local: ['LOCAL_BRAIN'],
  local_llm: ['LOCAL_LLM_API'],
  mesh: ['MESH_API']
};

let extensionContext;
let outputChannel;
let statusBarItem;
let currentPanel;
let currentSidebarView;
let lastReply = '';
let healthTimer;
let apiStateTimer;
let sarahTerminal;
let discoveredModelState = [];
let discoveredKeyState = {};
let currentDiagnostics = {};
let startupInFlight = false;
let chatParticipant;

function activate(context) {
  extensionContext = context;
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

  registerCommands(context);
  registerChatParticipant(context);

  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(async (event) => {
    if (event.affectsConfiguration('sarahMemory.modelsRoot')) {
      await refreshModelCatalog({ announce: false, postToViews: true });
    }
    if (event.affectsConfiguration('sarahMemory.healthPollMs')) {
      await restartPolling();
    }
  }));

  initializeExtension().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`[activate-error] ${message}`);
  });
}

function deactivate() {
  if (healthTimer) {
    clearInterval(healthTimer);
    healthTimer = undefined;
  }
  if (apiStateTimer) {
    clearInterval(apiStateTimer);
    apiStateTimer = undefined;
  }
}

async function initializeExtension() {
  await seedSettings();
  await refreshDiscoveredSecrets({ announce: false, postToViews: false });
  await refreshModelCatalog({ announce: false, postToViews: false });

  if (getConfig('autoFocusSidebarOnStartup')) {
    setTimeout(() => {
      focusChatSidebar().catch(() => {});
    }, 900);
  }

  if (getConfig('autoStartAiOSOnStartup')) {
    await autoStartSarahMemory();
  }

  if (getConfig('autoOpenVsCodeChatOnStartup')) {
    setTimeout(() => {
      vscode.commands.executeCommand('workbench.action.chat.open').catch(() => {});
    }, 1400);
  }

  await restartPolling();
  broadcastPayload({
    type: 'startup',
    ok: true,
    baseUrl: normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL)),
    models: discoveredModelState,
    keys: discoveredKeyState,
    diagnostics: currentDiagnostics,
    selectedModel: String(getConfig('selectedModel') || ''),
    selectedProvider: String(getConfig('selectedProvider') || 'local_llm')
  });
}

function registerCommands(context) {
  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.focusChat', async () => {
    await focusChatSidebar();
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.openPanel', async () => {
    await openPanel();
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.openVsCodeChat', async () => {
    await vscode.commands.executeCommand('workbench.action.chat.open');
    vscode.window.showInformationMessage('VS Code Chat opened. Use @sarahmemory, or rely on participant detection after the first use.');
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.ask', async () => {
    const prompt = await vscode.window.showInputBox({
      prompt: 'Ask SarahMemory',
      placeHolder: 'Type a prompt for SarahMemory AiOS'
    });
    if (!prompt) return;
    await focusChatSidebar();
    await sendPromptAndPresent(prompt, { revealSidebar: true });
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
    await sendPromptAndPresent(selection, { revealSidebar: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.checkHealth', async () => {
    await runHealthCheck({ announce: true, revealOutputOnError: true, postToViews: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.startAiOS', async () => {
    await startAiOSTerminal({ announce: true, restart: false });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.stopAiOS', async () => {
    await stopAiOSTerminal({ announce: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.restartAiOS', async () => {
    await startAiOSTerminal({ announce: true, restart: true });
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
      broadcastConfig();
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.refreshModels', async () => {
    await refreshModelCatalog({ announce: true, postToViews: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.runTerminalTask', async () => {
    const command = await vscode.window.showInputBox({
      prompt: 'Run a terminal task through SarahMemory workspace terminal',
      placeHolder: 'python SarahMemoryMain.py or npm test'
    });
    if (!command) return;
    await runTerminalTask(command, { announce: true, revealSidebar: true });
  }));

  context.subscriptions.push(vscode.commands.registerCommand('sarahMemory.launchAgentTask', async () => {
    const task = await vscode.window.showInputBox({
      prompt: 'Launch an agent-style SarahMemory task',
      placeHolder: 'Create a website and test it'
    });
    if (!task) return;
    await focusChatSidebar();
    await sendPromptAndPresent(task, { revealSidebar: true, agentTask: true });
  }));
}

function registerChatParticipant(context) {
  if (!vscode.chat || typeof vscode.chat.createChatParticipant !== 'function') {
    outputChannel.appendLine('[chat-participant] VS Code Chat Participant API is unavailable in this build. Sidebar chat remains available.');
    return;
  }

  const handler = async (request, chatContext, stream, token) => {
    try {
      await ensureRuntimeReady(stream);

      if (request.command === 'health') {
        stream.progress('Checking SarahMemory runtime health...');
        const body = await runHealthCheck({ announce: false, revealOutputOnError: true, postToViews: true });
        stream.markdown(renderHealthMarkdown(body, currentDiagnostics));
        addCommonChatButtons(stream);
        return { metadata: { kind: 'health', ok: Boolean(body && body.ok !== false) } };
      }

      if (request.command === 'models') {
        stream.progress('Refreshing local model catalog...');
        const models = await refreshModelCatalog({ announce: false, postToViews: true });
        stream.markdown(renderModelsMarkdown(models, String(getConfig('selectedModel') || '')));
        addCommonChatButtons(stream);
        return { metadata: { kind: 'models', count: models.length } };
      }

      if (request.command === 'terminal') {
        const command = String(request.prompt || '').trim();
        if (!command) {
          stream.markdown('Provide a shell command after `/terminal`, for example `/terminal pytest -q`.');
          return { metadata: { kind: 'terminal', ok: false } };
        }
        stream.progress(`Running terminal task: ${command}`);
        await runTerminalTask(command, { announce: false, revealSidebar: false });
        stream.markdown(`Started terminal task:\n\n\
\
${command}\n\
\
`);
        addCommonChatButtons(stream);
        return { metadata: { kind: 'terminal', ok: true, command } };
      }

      const agentTask = request.command === 'agent';
      const prompt = String(request.prompt || '').trim();
      if (!prompt) {
        stream.markdown('Enter a prompt for SarahMemory.');
        addCommonChatButtons(stream);
        return { metadata: { kind: 'empty', ok: false } };
      }

      stream.progress('Routing prompt into SarahMemory...');
      const response = await sendPrompt(prompt, { agentTask, chatContext });
      const diagnostics = currentDiagnostics || {};
      stream.markdown(response.reply);

      const routingMeta = response.raw && (response.raw.meta || response.raw.routing) ? (response.raw.meta || response.raw.routing) : {};
      if (routingMeta && Object.keys(routingMeta).length) {
        stream.markdown(`\n\n---\n**Routing**\n\n\
\
${safeJson(routingMeta)}\n\
\
`);
      }
      if (diagnostics && Object.keys(diagnostics).length) {
        stream.markdown(`\n\n**Runtime Diagnostics**\n\n\
\
${safeJson(diagnostics)}\n\
\
`);
      }

      addCommonChatButtons(stream);
      return {
        metadata: {
          kind: agentTask ? 'agent' : 'chat',
          ok: true,
          model: String(getConfig('selectedModel') || ''),
          provider: String(getConfig('selectedProvider') || 'local_llm')
        }
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[chat-participant-error] ${message}`);
      stream.markdown(`SarahMemory chat participant error:\n\n\
\
${message}\n\
\
`);
      stream.button({ command: 'sarahMemory.startAiOS', title: 'Start SarahMemory Main' });
      stream.button({ command: 'sarahMemory.openPanel', title: 'Open SarahMemory Panel' });
      return { metadata: { kind: 'error', ok: false, error: message } };
    }
  };

  chatParticipant = vscode.chat.createChatParticipant(CHAT_PARTICIPANT_ID, handler);
  chatParticipant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'resources', 'sarahmemory.svg');
  chatParticipant.followupProvider = {
    provideFollowups(result, chatContext, token) {
      return [
        { prompt: 'Show SarahMemory runtime health and routing details', label: 'Show health' },
        { prompt: 'Review the active file and suggest governed fixes', label: 'Review active file' },
        { prompt: 'List my local models and tell me which one is selected', label: 'List models' }
      ];
    }
  };
  context.subscriptions.push(chatParticipant);
}

async function ensureRuntimeReady(stream) {
  const baseUrl = normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL));
  const up = await quickHealthProbe(baseUrl);
  if (up) return true;
  stream.progress('SarahMemory runtime is not responding. Launching SarahMemoryMain.py...');
  await startAiOSTerminal({ announce: false, restart: false });
  for (let i = 0; i < 12; i += 1) {
    await sleep(1000);
    if (await quickHealthProbe(baseUrl)) {
      return true;
    }
  }
  throw new Error(`SarahMemory runtime did not become healthy at ${baseUrl}.`);
}

function addCommonChatButtons(stream) {
  try {
    stream.button({ command: 'sarahMemory.focusChat', title: 'Open SarahMemory Sidebar' });
    stream.button({ command: 'sarahMemory.checkHealth', title: 'Check Health' });
    stream.button({ command: 'sarahMemory.refreshModels', title: 'Refresh Models' });
  } catch {
    // best effort
  }
}

function renderHealthMarkdown(body, diagnostics) {
  const version = body && body.version ? `- Version: ${body.version}` : '';
  const routing = body && body.routing ? safeJson(body.routing) : '{}';
  const diag = diagnostics ? safeJson(diagnostics) : '{}';
  return [
    '## SarahMemory Runtime Health',
    '',
    `- API Base URL: ${normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL))}`,
    `- Healthy: ${Boolean(body && body.ok !== false)}`,
    version,
    '',
    '**Routing**',
    '',
    '```json',
    routing,
    '```',
    '',
    '**Diagnostics**',
    '',
    '```json',
    diag,
    '```'
  ].filter(Boolean).join('\n');
}

function renderModelsMarkdown(models, selectedModel) {
  const lines = ['## SarahMemory Local Models', ''];
  lines.push(`- Models root: ${String(getConfig('modelsRoot') || DEFAULT_MODELS_ROOT)}`);
  lines.push(`- Selected model: ${selectedModel || '(auto/default)'}`);
  lines.push('');
  if (!models || !models.length) {
    lines.push('No model folders were discovered.');
    return lines.join('\n');
  }
  for (const model of models) {
    const marker = (selectedModel && model.repo === selectedModel) ? ' **(selected)**' : '';
    lines.push(`- ${model.label}${marker}`);
  }
  return lines.join('\n');
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
    webviewView.webview.html = getWebviewHtml({ mode: 'sidebar' });
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

async function setConfig(key, value, target) {
  return vscode.workspace.getConfiguration('sarahMemory').update(key, value, target ?? preferredConfigurationTarget());
}

function preferredConfigurationTarget() {
  try {
    if (vscode.workspace.workspaceFile || (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0)) {
      return vscode.ConfigurationTarget.Workspace;
    }
  } catch {
    // fall through
  }
  return vscode.ConfigurationTarget.Global;
}

async function seedSettings() {
  const target = preferredConfigurationTarget();
  const desired = {
    apiBaseUrl: DEFAULT_API_BASE_URL,
    requestTimeoutMs: 45000,
    launchConfigMain: 'SarahMemory AiOS Main',
    launchConfigApi: 'SarahMemory Flask API',
    autoHealthCheck: true,
    autoFocusSidebarOnStartup: true,
    autoOpenVsCodeChatOnStartup: false,
    autoStartAiOSOnStartup: true,
    healthPollMs: 15000,
    modelsRoot: DEFAULT_MODELS_ROOT,
    autoInjectActiveFile: true,
    autoInjectWorkspaceContext: true,
    workspaceFileListLimit: 200,
    maxActiveFileChars: 60000,
    autoDiscoverApiKeys: true,
    selectedProvider: 'local_llm',
    selectedModel: '',
    aiosTerminalName: DEFAULT_TERMINAL_NAME,
    autoStartDelayMs: 1200
  };

  for (const [key, value] of Object.entries(desired)) {
    const existing = getConfig(key);
    if (existing === undefined || existing === null || existing === '') {
      await setConfig(key, value, target);
    }
  }

  try {
    const chatConfig = vscode.workspace.getConfiguration('chat');
    const disabled = chatConfig.get('disableAIFeatures');
    if (disabled === true) {
      await chatConfig.update('disableAIFeatures', false, target);
    }
  } catch {
    // best effort only
  }
}

async function focusChatSidebar() {
  await vscode.commands.executeCommand(`workbench.view.extension.${SARAH_MEMORY_CONTAINER_ID}`);
  if (currentSidebarView) {
    try {
      currentSidebarView.show?.(true);
    } catch {
      // ignore
    }
    return currentSidebarView;
  }
  return undefined;
}

async function promptForApiBaseUrl(forcePrompt = false) {
  let baseUrl = String(getConfig('apiBaseUrl') || '').trim();
  if (!baseUrl) {
    baseUrl = DEFAULT_API_BASE_URL;
    await setConfig('apiBaseUrl', baseUrl);
  }
  if (baseUrl && !forcePrompt) {
    return normalizeBaseUrl(baseUrl);
  }

  const value = await vscode.window.showInputBox({
    prompt: 'Enter the base URL for the running SarahMemory API',
    placeHolder: DEFAULT_API_BASE_URL,
    value: baseUrl || DEFAULT_API_BASE_URL,
    ignoreFocusOut: true,
    validateInput: (input) => {
      const test = input.trim();
      if (!test) return 'A base URL is required.';
      try {
        const url = new URL(test);
        if (!/^https?:$/i.test(url.protocol)) {
          return 'Only http:// or https:// URLs are supported.';
        }
        return null;
      } catch {
        return `Enter a valid URL, for example ${DEFAULT_API_BASE_URL}`;
      }
    }
  });

  if (!value) return undefined;
  baseUrl = normalizeBaseUrl(value);
  await setConfig('apiBaseUrl', baseUrl);
  return baseUrl;
}

function normalizeBaseUrl(input) {
  return String(input || '').trim().replace(/\/+$/, '');
}

function buildEndpoints(baseUrl) {
  const base = normalizeBaseUrl(baseUrl || DEFAULT_API_BASE_URL);
  return {
    base,
    health: `${base}/api/health`,
    chat: `${base}/api/chat`,
    state: `${base}/api/state`
  };
}

async function autoStartSarahMemory() {
  if (startupInFlight) return;
  startupInFlight = true;
  try {
    const baseUrl = normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL));
    const alreadyUp = await quickHealthProbe(baseUrl);
    if (!alreadyUp) {
      const delayMs = Number(getConfig('autoStartDelayMs') || 1200);
      if (delayMs > 0) {
        await sleep(delayMs);
      }
      await startAiOSTerminal({ announce: false, restart: false });
    }
  } finally {
    startupInFlight = false;
  }
}

async function startAiOSTerminal(options = {}) {
  const announce = Boolean(options.announce);
  const restart = Boolean(options.restart);
  const cwd = getWorkspaceRoot() || process.cwd();
  const terminalName = String(getConfig('aiosTerminalName') || DEFAULT_TERMINAL_NAME);

  if (restart) {
    await stopAiOSTerminal({ announce: false });
  } else {
    const existing = findSarahTerminal(terminalName);
    if (existing) {
      sarahTerminal = existing;
      if (announce) {
        vscode.window.showInformationMessage('SarahMemory AiOS terminal is already running or available.');
      }
      broadcastPayload({ type: 'terminal', action: 'attached', name: terminalName });
      return;
    }
  }

  const launchEnv = await buildLaunchEnvironment();
  sarahTerminal = vscode.window.createTerminal({
    name: terminalName,
    cwd,
    env: launchEnv
  });
  sarahTerminal.show(false);
  sarahTerminal.sendText('python SarahMemoryMain.py', true);
  outputChannel.appendLine(`[launch] python SarahMemoryMain.py (cwd=${cwd})`);
  if (announce) {
    vscode.window.showInformationMessage('SarahMemory AiOS launch issued: python SarahMemoryMain.py');
  }
  broadcastPayload({ type: 'terminal', action: 'start', name: terminalName, cwd, env: summarizeEnvironment(launchEnv) });
}

async function stopAiOSTerminal(options = {}) {
  const announce = Boolean(options.announce);
  const terminalName = String(getConfig('aiosTerminalName') || DEFAULT_TERMINAL_NAME);
  const term = sarahTerminal || findSarahTerminal(terminalName);
  if (!term) {
    if (announce) {
      vscode.window.showWarningMessage('SarahMemory AiOS terminal is not running.');
    }
    return;
  }
  term.dispose();
  sarahTerminal = undefined;
  outputChannel.appendLine('[launch] SarahMemory AiOS terminal disposed');
  if (announce) {
    vscode.window.showInformationMessage('SarahMemory AiOS terminal stopped.');
  }
  broadcastPayload({ type: 'terminal', action: 'stop', name: terminalName });
}

function findSarahTerminal(terminalName) {
  const target = String(terminalName || DEFAULT_TERMINAL_NAME).trim();
  return vscode.window.terminals.find((terminal) => terminal.name === target);
}

async function buildLaunchEnvironment() {
  const envSources = await discoverEnvironmentSources();
  const merged = { ...process.env, ...envSources.env };
  const selectedModel = String(getConfig('selectedModel') || '').trim();
  const selectedProvider = String(getConfig('selectedProvider') || 'local_llm').trim();
  merged.ACTIVE_LLM_PROVIDER = selectedProvider || merged.ACTIVE_LLM_PROVIDER || 'local_llm';
  if (selectedModel) {
    merged.ACTIVE_LLM_MODEL = selectedModel;
  }
  return merged;
}

function summarizeEnvironment(env) {
  return {
    ACTIVE_LLM_PROVIDER: env.ACTIVE_LLM_PROVIDER || '',
    ACTIVE_LLM_MODEL: env.ACTIVE_LLM_MODEL || '',
    OPENAI_API_KEY: env.OPENAI_API_KEY ? '[loaded]' : '',
    ANTHROPIC_API_KEY: env.ANTHROPIC_API_KEY ? '[loaded]' : '',
    GEMINI_API_KEY: env.GEMINI_API_KEY ? '[loaded]' : ''
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

async function restartPolling() {
  if (healthTimer) clearInterval(healthTimer);
  if (apiStateTimer) clearInterval(apiStateTimer);

  if (getConfig('autoHealthCheck')) {
    await runHealthCheck({ announce: false, revealOutputOnError: false, postToViews: true }).catch(() => {});
    const healthPollMs = Math.max(5000, Number(getConfig('healthPollMs') || 15000));
    healthTimer = setInterval(() => {
      runHealthCheck({ announce: false, revealOutputOnError: false, postToViews: true }).catch(() => {});
    }, healthPollMs);
    apiStateTimer = setInterval(() => {
      refreshRuntimeState().catch(() => {});
    }, healthPollMs);
    if (extensionContext) {
      extensionContext.subscriptions.push({
        dispose: () => {
          if (healthTimer) clearInterval(healthTimer);
          if (apiStateTimer) clearInterval(apiStateTimer);
        }
      });
    }
  }
}

async function runHealthCheck(options = {}) {
  const announce = Boolean(options.announce);
  const revealOutputOnError = Boolean(options.revealOutputOnError);
  const postToViews = Boolean(options.postToViews);
  const baseUrl = await promptForApiBaseUrl(false);
  if (!baseUrl) {
    updateStatus('disconnected', 'SarahMemory: API URL not set');
    if (announce) vscode.window.showWarningMessage('SarahMemory API base URL is not configured.');
    const body = { ok: false, error: 'API base URL not configured' };
    if (postToViews) broadcastPayload({ type: 'health', body, diagnostics: currentDiagnostics });
    return body;
  }

  const endpoints = buildEndpoints(baseUrl);
  try {
    const result = await requestJson('GET', endpoints.health, undefined, Number(getConfig('requestTimeoutMs') || 45000));
    const body = result.body || {};
    currentDiagnostics.health = body;
    const ok = Boolean(body.ok !== false);
    if (ok) {
      const version = body.version ? ` v${body.version}` : '';
      const routing = body.routing || {};
      const routeLabel = routing.model ? ` · ${routing.provider || 'provider'} / ${routing.model}` : '';
      updateStatus('connected', `SarahMemory${version}${routeLabel}`);
      if (announce) {
        vscode.window.showInformationMessage(`SarahMemory connected${version}${routeLabel}`);
      }
    } else {
      updateStatus('warning', 'SarahMemory: health degraded');
      if (announce) {
        vscode.window.showWarningMessage('SarahMemory health check returned a degraded state.');
      }
    }
    await refreshRuntimeState();
    if (postToViews) {
      broadcastPayload({ type: 'health', body, diagnostics: currentDiagnostics });
    }
    return body;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`[health] ${message}`);
    currentDiagnostics.health = { ok: false, error: message };
    updateStatus('disconnected', 'SarahMemory: offline');
    if (announce) {
      vscode.window.showErrorMessage(`SarahMemory health check failed: ${message}`);
    }
    if (revealOutputOnError) {
      outputChannel.show(true);
    }
    if (postToViews) {
      broadcastPayload({ type: 'health', body: { ok: false, error: message }, diagnostics: currentDiagnostics });
    }
    return { ok: false, error: message };
  }
}

async function refreshRuntimeState() {
  const baseUrl = normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL));
  const endpoints = buildEndpoints(baseUrl);
  try {
    const stateRes = await requestJson('GET', endpoints.state, undefined, Number(getConfig('requestTimeoutMs') || 45000));
    currentDiagnostics.state = stateRes.body || {};
  } catch {
    currentDiagnostics.state = {};
  }
  currentDiagnostics.models = discoveredModelState;
  currentDiagnostics.keys = discoveredKeyState;
  broadcastPayload({ type: 'diagnostics', diagnostics: currentDiagnostics });
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
  const revealSidebar = Boolean(options.revealSidebar);
  const revealPanel = Boolean(options.revealPanel);
  const agentTask = Boolean(options.agentTask);
  if (revealSidebar) {
    await focusChatSidebar();
  }
  const response = await sendPrompt(prompt, { agentTask, chatContext: options.chatContext });
  if (response && response.reply) {
    if (revealPanel) {
      await openPanel();
    }
    broadcastPayload({
      type: 'reply',
      prompt,
      reply: response.reply,
      meta: response.raw.meta || response.raw.routing || {},
      diagnostics: currentDiagnostics
    });
  }
  return response;
}

async function sendPrompt(prompt, options = {}) {
  const baseUrl = await promptForApiBaseUrl(false);
  if (!baseUrl) {
    throw new Error('SarahMemory API base URL is not configured.');
  }

  const payload = await buildChatPayload(prompt, options);
  const endpoints = buildEndpoints(baseUrl);
  outputChannel.appendLine(`[chat] -> ${prompt}`);
  const result = await requestJson('POST', endpoints.chat, payload, Number(getConfig('requestTimeoutMs') || 45000));
  const raw = result.body || {};
  const reply = String(raw.presentation_reply || raw.reply || raw.response || raw.text || '').trim();
  if (!reply) {
    throw new Error(raw.error || 'SarahMemory returned an empty reply.');
  }
  lastReply = reply;
  currentDiagnostics.lastMeta = raw.meta || raw.routing || {};
  outputChannel.appendLine(`[chat] <- ${reply}`);
  return { reply, raw };
}

async function buildChatPayload(prompt, options = {}) {
  const selectedModel = String(getConfig('selectedModel') || '').trim();
  const selectedProvider = String(getConfig('selectedProvider') || 'local_llm').trim() || 'local_llm';
  const contextBlock = await gatherContextForPrompt();
  const files = [];
  if (contextBlock.activeFile) {
    files.push({
      path: contextBlock.activeFile.path,
      language: contextBlock.activeFile.languageId,
      selection: contextBlock.activeFile.selection,
      content: contextBlock.activeFile.content
    });
  }

  const history = extractChatHistory(options.chatContext);

  return {
    text: prompt,
    source: 'vscode_extension',
    ui: 'vscode_extension',
    provider: selectedProvider,
    provider_hint: selectedProvider,
    requested_provider: selectedProvider,
    model: selectedModel || undefined,
    requested_model: selectedModel || undefined,
    local_model_override: selectedModel || undefined,
    files,
    workspace: contextBlock.workspace,
    active_file: contextBlock.activeFile,
    agent_task: Boolean(options.agentTask),
    history,
    meta: {
      panel: 'vscode_chat',
      addon: 'vscode_extension',
      driver: 'vscode',
      diagnostics_ping: false,
      force_neuron: true,
      display_requested: true,
      files,
      workspace: contextBlock.workspace,
      active_file: contextBlock.activeFile,
      selected_model: selectedModel || '',
      selected_provider: selectedProvider,
      api_keys_available: Object.keys(discoveredKeyState).filter((key) => discoveredKeyState[key]),
      models_available: discoveredModelState.map((item) => item.repo || item.folder),
      vscode: {
        workspaceName: vscode.workspace.name || '',
        machineId: vscode.env.machineId || '',
        sessionId: await getOrCreateSessionId()
      }
    }
  };
}

function extractChatHistory(chatContext) {
  if (!chatContext || !Array.isArray(chatContext.history)) {
    return [];
  }
  return chatContext.history.slice(-6).map((item) => {
    try {
      if (item && typeof item.prompt === 'string') {
        return { role: 'user', content: item.prompt };
      }
      if (item && Array.isArray(item.response)) {
        return { role: 'assistant', content: item.response.map((part) => String(part.value || '')).join('\n') };
      }
    } catch {
      // ignore malformed items
    }
    return null;
  }).filter(Boolean);
}

async function gatherContextForPrompt() {
  const activeFile = await getActiveFileContext();
  const workspace = await getWorkspaceContext();
  return { activeFile, workspace };
}

async function getActiveFileContext() {
  if (!getConfig('autoInjectActiveFile')) {
    return null;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }
  const document = editor.document;
  const maxChars = Math.max(1000, Number(getConfig('maxActiveFileChars') || 60000));
  const fullText = document.getText();
  const selection = document.getText(editor.selection).trim();
  return {
    path: document.uri.fsPath,
    fileName: path.basename(document.uri.fsPath),
    languageId: document.languageId,
    selection,
    content: truncateString(fullText, maxChars),
    lineCount: document.lineCount
  };
}

async function getWorkspaceContext() {
  if (!getConfig('autoInjectWorkspaceContext')) {
    return null;
  }
  const folders = vscode.workspace.workspaceFolders || [];
  if (!folders.length) {
    return null;
  }
  const folderPaths = folders.map((folder) => folder.uri.fsPath);
  const limit = Math.max(1, Number(getConfig('workspaceFileListLimit') || 200));
  const fileUris = await vscode.workspace.findFiles('**/*', '{**/node_modules/**,**/.git/**,**/.venv/**,**/venv/**,**/__pycache__/**,**/.next/**,**/dist/**,**/build/**}', limit);
  const files = fileUris.map((uri) => ({
    path: uri.fsPath,
    relativePath: getRelativeWorkspacePath(uri.fsPath)
  }));
  return {
    workspaceName: vscode.workspace.name || '',
    folders: folderPaths,
    files,
    fileCount: files.length
  };
}

function getRelativeWorkspacePath(fsPath) {
  const folders = vscode.workspace.workspaceFolders || [];
  for (const folder of folders) {
    const root = folder.uri.fsPath;
    if (fsPath.startsWith(root)) {
      return path.relative(root, fsPath);
    }
  }
  return fsPath;
}

async function refreshModelCatalog(options = {}) {
  const announce = Boolean(options.announce);
  const postToViews = Boolean(options.postToViews);
  const modelsRoot = String(getConfig('modelsRoot') || DEFAULT_MODELS_ROOT).trim() || DEFAULT_MODELS_ROOT;
  let models = [];
  try {
    const entries = await fs.promises.readdir(modelsRoot, { withFileTypes: true });
    models = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => buildModelRecord(entry.name, modelsRoot))
      .sort((a, b) => a.label.localeCompare(b.label));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outputChannel.appendLine(`[models] ${message}`);
    if (announce) {
      vscode.window.showWarningMessage(`Unable to read models folder: ${message}`);
    }
  }

  discoveredModelState = models;
  const selectedModel = String(getConfig('selectedModel') || '').trim();
  if (!selectedModel && models.length) {
    await setConfig('selectedModel', models[0].repo);
  }
  if (postToViews) {
    broadcastPayload({ type: 'models', models: discoveredModelState, selectedModel: String(getConfig('selectedModel') || ''), selectedProvider: String(getConfig('selectedProvider') || 'local_llm') });
  }
  if (announce) {
    vscode.window.showInformationMessage(`SarahMemory discovered ${models.length} local model folder(s).`);
  }
  return models;
}

function buildModelRecord(folderName, modelsRoot) {
  const repoGuess = folderName.includes('_') ? folderName.replace('_', '/') : folderName;
  const vendor = repoGuess.includes('/') ? repoGuess.split('/')[0] : folderName.split(/[_-]/)[0];
  return {
    folder: folderName,
    repo: repoGuess,
    label: repoGuess,
    vendor,
    path: path.join(modelsRoot, folderName),
    isLocal: true
  };
}

async function refreshDiscoveredSecrets(options = {}) {
  const announce = Boolean(options.announce);
  const postToViews = Boolean(options.postToViews);
  const envSources = await discoverEnvironmentSources();
  const available = {};
  for (const [provider, keys] of Object.entries(PROVIDER_KEY_MAP)) {
    available[provider] = keys.some((name) => Boolean(envSources.env[name] || process.env[name]));
  }
  discoveredKeyState = available;
  if (postToViews) {
    broadcastPayload({ type: 'keys', keys: discoveredKeyState });
  }
  if (announce) {
    vscode.window.showInformationMessage('SarahMemory refreshed API key discovery from .env and process environment.');
  }
  return available;
}

async function discoverEnvironmentSources() {
  const workspaceRoot = getWorkspaceRoot();
  const envPaths = [];
  if (workspaceRoot) {
    envPaths.push(path.join(workspaceRoot, '.env'));
  }
  const parsed = {};
  for (const envPath of envPaths) {
    try {
      const content = await fs.promises.readFile(envPath, 'utf8');
      Object.assign(parsed, parseDotEnv(content));
    } catch {
      // ignore missing .env files
    }
  }
  return { env: parsed, paths: envPaths };
}

function parseDotEnv(content) {
  const result = {};
  for (const rawLine of String(content || '').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#') || !line.includes('=')) continue;
    const idx = line.indexOf('=');
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    result[key] = value;
  }
  return result;
}

function getWorkspaceRoot() {
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  return folder ? folder.uri.fsPath : undefined;
}

async function quickHealthProbe(baseUrl) {
  try {
    const endpoints = buildEndpoints(baseUrl);
    const result = await requestJson('GET', endpoints.health, undefined, 3500);
    return Boolean(result.body && result.body.ok !== false);
  } catch {
    return false;
  }
}

async function runTerminalTask(command, options = {}) {
  const announce = Boolean(options.announce);
  const revealSidebar = Boolean(options.revealSidebar);
  const terminalName = `${String(getConfig('aiosTerminalName') || DEFAULT_TERMINAL_NAME).trim()} Tasks`;
  const cwd = getWorkspaceRoot() || process.cwd();
  const terminal = vscode.window.createTerminal({ name: terminalName, cwd });
  terminal.show(false);
  terminal.sendText(command, true);
  outputChannel.appendLine(`[task] ${command}`);
  if (announce) {
    vscode.window.showInformationMessage(`SarahMemory terminal task launched: ${command}`);
  }
  if (revealSidebar) {
    await focusChatSidebar();
  }
  broadcastPayload({ type: 'terminalTask', command, terminal: terminalName, cwd });
}

async function openPanel() {
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

  currentPanel.webview.html = getWebviewHtml({ mode: 'panel' });
  attachWebviewMessageHandler(currentPanel.webview, { source: 'panel' });
  broadcastConfig();

  currentPanel.onDidDispose(() => {
    currentPanel = undefined;
  });

  return currentPanel;
}

function attachWebviewMessageHandler(webview, { source }) {
  webview.onDidReceiveMessage(async (message) => {
    try {
      switch (message.type) {
        case 'ready':
          broadcastConfig(webview, source);
          break;
        case 'sendPrompt': {
          const prompt = String(message.prompt || '').trim();
          if (!prompt) return;
          broadcastPayload({ type: 'pending', prompt });
          const response = await sendPrompt(prompt, { agentTask: Boolean(message.agentTask) });
          broadcastPayload({
            type: 'reply',
            prompt,
            reply: response.reply,
            meta: response.raw.meta || response.raw.routing || {},
            diagnostics: currentDiagnostics
          });
          break;
        }
        case 'checkHealth':
          await runHealthCheck({ announce: false, revealOutputOnError: true, postToViews: true });
          break;
        case 'setBaseUrl': {
          const incoming = String(message.baseUrl || '').trim();
          if (!incoming) return;
          const normalized = normalizeBaseUrl(incoming);
          new URL(normalized);
          await setConfig('apiBaseUrl', normalized);
          broadcastConfig();
          vscode.window.showInformationMessage(`SarahMemory API base URL set to ${normalized}`);
          break;
        }
        case 'selectModel': {
          const selectedModel = String(message.model || '').trim();
          await setConfig('selectedModel', selectedModel);
          broadcastConfig();
          break;
        }
        case 'selectProvider': {
          const selectedProvider = String(message.provider || 'local_llm').trim() || 'local_llm';
          await setConfig('selectedProvider', selectedProvider);
          broadcastConfig();
          break;
        }
        case 'startMain':
          await startAiOSTerminal({ announce: true, restart: false });
          break;
        case 'stopMain':
          await stopAiOSTerminal({ announce: true });
          break;
        case 'restartMain':
          await startAiOSTerminal({ announce: true, restart: true });
          break;
        case 'startApi':
          await startDebugLaunch(getConfig('launchConfigApi'));
          break;
        case 'insertLastReply':
          await vscode.commands.executeCommand('sarahMemory.insertLastReply');
          break;
        case 'openFullPanel':
          await openPanel();
          break;
        case 'runTerminalTask': {
          const command = String(message.command || '').trim();
          if (!command) return;
          await runTerminalTask(command, { announce: true, revealSidebar: true });
          break;
        }
        case 'refreshModels':
          await refreshModelCatalog({ announce: true, postToViews: true });
          break;
        case 'openVsCodeChat':
          await vscode.commands.executeCommand('workbench.action.chat.open');
          break;
      }
    } catch (error) {
      const messageText = error instanceof Error ? error.message : String(error);
      outputChannel.appendLine(`[${source}-error] ${messageText}`);
      broadcastPayload({ type: 'error', error: messageText });
      vscode.window.showErrorMessage(`SarahMemory ${source} error: ${messageText}`);
    }
  });
}

function broadcastConfig(targetWebview, source) {
  const payload = {
    type: 'config',
    source,
    baseUrl: normalizeBaseUrl(String(getConfig('apiBaseUrl') || DEFAULT_API_BASE_URL)),
    lastReply,
    models: discoveredModelState,
    keys: discoveredKeyState,
    diagnostics: currentDiagnostics,
    selectedModel: String(getConfig('selectedModel') || ''),
    selectedProvider: String(getConfig('selectedProvider') || 'local_llm')
  };
  if (targetWebview) {
    postToWebview(targetWebview, payload);
  } else {
    broadcastPayload(payload);
  }
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
  if (!webview) return;
  webview.postMessage(payload);
}

function getWebviewHtml(options = {}) {
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
      grid-template-rows: auto auto auto 1fr auto auto;
      gap: var(--gap);
      height: calc(100vh - ${isSidebar ? '20px' : '28px'});
    }
    .banner, .diag, .modelRow, .terminalRow, .settings, .composer {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--surface);
      padding: 10px;
    }
    .bannerTop, .toolbarWrap, .modelGrid, .terminalGrid, .composerFooter {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .bannerTitle { font-weight: 700; }
    .status { font-size: 12px; opacity: 0.85; color: var(--muted); }
    .toolbarWrap { margin-top: 8px; justify-content: flex-start; }
    .settings { display: grid; gap: 8px; }
    .settingsGrid { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }
    .diag pre, .meta, .reply, .prompt {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .diag pre {
      margin: 8px 0 0;
      max-height: 180px;
      overflow: auto;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 12px;
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
      min-height: 220px;
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
    input, textarea, button, select {
      font: inherit;
    }
    input, textarea, select {
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
    .rowLabel {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
  </style>
</head>
<body>
  <div class="root">
    <div class="banner">
      <div class="bannerTop">
        <div>
          <div class="bannerTitle">${title}</div>
          <div class="status" id="status">Idle</div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button id="openVsCodeChat" class="secondary">VS Code Chat</button>
          ${isSidebar ? '<button id="openFullPanel" class="secondary">Full Panel</button>' : ''}
        </div>
      </div>
      <div class="toolbarWrap">
        <button id="startMain">Start Main</button>
        <button id="stopMain" class="secondary">Stop Main</button>
        <button id="restartMain" class="secondary">Restart Main</button>
        <button id="health" class="secondary">Health</button>
        <button id="insert" class="secondary">Insert Reply</button>
      </div>
    </div>

    <div class="settings">
      <div class="rowLabel">Runtime Endpoint</div>
      <div class="settingsGrid">
        <input id="baseUrl" type="text" placeholder="${DEFAULT_API_BASE_URL}" />
        <button id="saveBase" class="secondary">Save URL</button>
      </div>
    </div>

    <div class="modelRow">
      <div class="rowLabel">Model / Provider Control</div>
      <div class="modelGrid">
        <select id="providerSelect"></select>
        <select id="modelSelect"></select>
        <button id="refreshModels" class="secondary">Refresh Models</button>
      </div>
      <div class="status" id="keyStatus">API keys: scanning...</div>
    </div>

    <div class="diag">
      <div class="rowLabel">Diagnostics / Routing</div>
      <pre id="diagText">Waiting for runtime diagnostics...</pre>
    </div>

    <div id="log" class="log" aria-live="polite"></div>

    <div class="terminalRow">
      <div class="rowLabel">Terminal Task / Agent Launcher</div>
      <div class="terminalGrid">
        <input id="terminalCommand" type="text" placeholder="npm test or python SarahMemoryMain.py" />
        <button id="runTerminalTask" class="secondary">Run Terminal Task</button>
      </div>
    </div>

    <div class="composer">
      <div class="rowLabel">SarahMemory Chat</div>
      <textarea id="prompt" placeholder="Ask SarahMemory..."></textarea>
      <div class="composerFooter">
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button id="send">Send</button>
          <button id="sendAgent" class="secondary">Launch Agent Task</button>
        </div>
        <span class="status">Active file and workspace context are injected automatically.</span>
      </div>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const log = document.getElementById('log');
    const promptInput = document.getElementById('prompt');
    const statusNode = document.getElementById('status');
    const baseUrlInput = document.getElementById('baseUrl');
    const providerSelect = document.getElementById('providerSelect');
    const modelSelect = document.getElementById('modelSelect');
    const diagText = document.getElementById('diagText');
    const keyStatus = document.getElementById('keyStatus');
    const terminalCommand = document.getElementById('terminalCommand');

    let latestModels = [];

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

    function renderProviders(selectedProvider, keys) {
      const options = [
        { value: 'local_llm', label: 'Local LLM' },
        { value: 'local', label: 'Local API' },
        { value: 'openai', label: 'OpenAI' },
        { value: 'claude', label: 'Claude/Anthropic' },
        { value: 'gemini', label: 'Gemini' },
        { value: 'mistral', label: 'Mistral' },
        { value: 'deepseek', label: 'DeepSeek' },
        { value: 'groq', label: 'Groq' },
        { value: 'cohere', label: 'Cohere' },
        { value: 'huggingface', label: 'HuggingFace' }
      ];
      providerSelect.innerHTML = '';
      for (const option of options) {
        const node = document.createElement('option');
        const available = !keys || typeof keys[option.value] === 'undefined' ? true : Boolean(keys[option.value]);
        node.value = option.value;
        node.textContent = available ? option.label : option.label + ' (key missing)';
        providerSelect.appendChild(node);
      }
      providerSelect.value = selectedProvider || 'local_llm';
    }

    function renderModels(models, selectedModel) {
      latestModels = Array.isArray(models) ? models : [];
      modelSelect.innerHTML = '';
      const blank = document.createElement('option');
      blank.value = '';
      blank.textContent = 'Auto / Default';
      modelSelect.appendChild(blank);
      for (const model of latestModels) {
        const node = document.createElement('option');
        node.value = model.repo || model.folder;
        node.textContent = model.label || model.repo || model.folder;
        modelSelect.appendChild(node);
      }
      modelSelect.value = selectedModel || '';
    }

    function renderKeys(keys) {
      const names = Object.keys(keys || {}).filter((name) => keys[name]);
      keyStatus.textContent = names.length ? 'API keys detected: ' + names.join(', ') : 'API keys detected: none';
    }

    function renderDiagnostics(diag) {
      diagText.textContent = JSON.stringify(diag || {}, null, 2);
    }

    document.getElementById('send').addEventListener('click', () => {
      const prompt = promptInput.value.trim();
      if (!prompt) return;
      vscode.postMessage({ type: 'sendPrompt', prompt, agentTask: false });
      promptInput.value = '';
      setStatus('Sending...');
    });

    document.getElementById('sendAgent').addEventListener('click', () => {
      const prompt = promptInput.value.trim();
      if (!prompt) return;
      vscode.postMessage({ type: 'sendPrompt', prompt, agentTask: true });
      promptInput.value = '';
      setStatus('Launching agent task...');
    });

    document.getElementById('health').addEventListener('click', () => {
      vscode.postMessage({ type: 'checkHealth' });
      setStatus('Checking health...');
    });

    document.getElementById('saveBase').addEventListener('click', () => {
      vscode.postMessage({ type: 'setBaseUrl', baseUrl: baseUrlInput.value.trim() });
    });

    providerSelect.addEventListener('change', () => {
      vscode.postMessage({ type: 'selectProvider', provider: providerSelect.value });
    });

    modelSelect.addEventListener('change', () => {
      vscode.postMessage({ type: 'selectModel', model: modelSelect.value });
    });

    document.getElementById('refreshModels').addEventListener('click', () => {
      vscode.postMessage({ type: 'refreshModels' });
    });

    document.getElementById('startMain').addEventListener('click', () => vscode.postMessage({ type: 'startMain' }));
    document.getElementById('stopMain').addEventListener('click', () => vscode.postMessage({ type: 'stopMain' }));
    document.getElementById('restartMain').addEventListener('click', () => vscode.postMessage({ type: 'restartMain' }));
    document.getElementById('insert').addEventListener('click', () => vscode.postMessage({ type: 'insertLastReply' }));
    document.getElementById('runTerminalTask').addEventListener('click', () => {
      const command = terminalCommand.value.trim();
      if (!command) return;
      vscode.postMessage({ type: 'runTerminalTask', command });
      terminalCommand.value = '';
    });
    document.getElementById('openVsCodeChat').addEventListener('click', () => vscode.postMessage({ type: 'openVsCodeChat' }));

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
        case 'startup':
        case 'config':
          if (message.baseUrl) baseUrlInput.value = message.baseUrl;
          renderModels(message.models || [], message.selectedModel || '');
          renderProviders(message.selectedProvider || 'local_llm', message.keys || {});
          renderKeys(message.keys || {});
          renderDiagnostics(message.diagnostics || {});
          if (message.lastReply) setStatus('Last reply cached');
          break;
        case 'models':
          renderModels(message.models || [], message.selectedModel || '');
          setStatus('Models refreshed');
          break;
        case 'keys':
          renderKeys(message.keys || {});
          break;
        case 'diagnostics':
          renderDiagnostics(message.diagnostics || {});
          break;
        case 'terminal':
          addEntry('health', 'Terminal', '', message);
          setStatus('Terminal update received');
          break;
        case 'terminalTask':
          addEntry('health', 'Terminal Task', '', message);
          setStatus('Terminal task launched');
          break;
        case 'pending':
          addEntry('pending', message.prompt, '', 'Pending...');
          setStatus('Waiting for SarahMemory...');
          break;
        case 'reply':
          addEntry('reply', message.prompt, message.reply, message.meta || message.diagnostics || null);
          renderDiagnostics(message.diagnostics || {});
          setStatus('Reply received');
          break;
        case 'health':
          addEntry('health', 'Health Check', '', message.body || {});
          renderDiagnostics(message.diagnostics || message.body || {});
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

function truncateString(value, maxChars) {
  const text = String(value || '');
  if (!maxChars || text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n\n[TRUNCATED ${text.length - maxChars} CHARS]`;
}

async function getOrCreateSessionId() {
  let sessionId = extensionContext?.globalState.get('sarahMemory.sessionId');
  if (!sessionId) {
    sessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    await extensionContext?.globalState.update('sarahMemory.sessionId', sessionId);
  }
  return sessionId;
}

function safeJson(value) {
  try {
    return JSON.stringify(value || {}, null, 2);
  } catch {
    return '{}';
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

module.exports = {
  activate,
  deactivate
};

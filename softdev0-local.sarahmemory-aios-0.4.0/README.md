# SarahMemory AiOS VS Code Extension

SarahMemory-first VS Code chat, workspace, runtime, and built-in Chat participant integration for a running SarahMemory AiOS instance.

## What this build does

This revision makes SarahMemory a first-class part of VS Code in two surfaces:

- a dedicated **SarahMemory** Activity Bar sidebar chat/runtime surface
- a built-in **VS Code Chat participant** available as `@sarahmemory`

It also:

- launches `python SarahMemoryMain.py` automatically when VS Code starts
- seeds VS Code settings automatically with SarahMemory defaults
- defaults the SarahMemory API base URL to `http://127.0.0.1:8000`
- sends active file content and workspace file context automatically with chat requests
- surfaces SarahMemory health, routing, notes, and runtime diagnostics live in the sidebar UI
- discovers local model folders under `C:\SarahMemory\data\models` and exposes them in a quick-swap selector
- discovers API key presence from `.env` and OS environment variables so you do not need to re-enter them in the extension
- provides a terminal-task launcher and an agent-task chat launcher inside the SarahMemory chat surface
- contributes a sticky chat participant with slash commands:
  - `@sarahmemory /health`
  - `@sarahmemory /models`
  - `@sarahmemory /agent`
  - `@sarahmemory /terminal`

## Important runtime note

This extension uses **your SarahMemory runtime** as the chat backend. The sidebar chat replies through SarahMemory, and the built-in VS Code Chat participant routes prompts into SarahMemory as well.

## Expected SarahMemory routes

- `GET /api/health`
- `GET /api/state`
- `POST /api/chat`

## Local startup contract

This build assumes your local SarahMemory runtime is available at:

- `http://127.0.0.1:8000`

and that launching SarahMemory locally is done with:

```bash
python SarahMemoryMain.py
```

## How to use built-in VS Code Chat

Open the built-in Chat view and use:

```text
@sarahmemory <your prompt>
```

Because the participant is registered as sticky, after the first use it tends to remain selected in that chat input.

The extension also provides participant detection metadata so VS Code can try routing suitable SarahMemory-oriented prompts automatically.

## Local model discovery

The extension scans:

- `C:\SarahMemory\data\models`

Every top-level model directory is surfaced in the sidebar UI. Folder names such as:

- `Qwen_Qwen2.5-7B-Instruct`
- `google_gemma-3-4b-it`

are converted into display labels such as:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-4b-it`
- `NOTE THIS SYSTEM CAN ACCEPT PRETTY MUCH ANY MODEL INCLUDING GEMMA 4, GPT, ETC`

The selected model is attached to SarahMemory requests and also exported into the SarahMemory launch environment as `ACTIVE_LLM_MODEL` when the extension starts `SarahMemoryMain.py`.

## API key discovery

The extension reads key presence from:

- `${workspaceFolder}\.env`
- the VS Code process environment / OS environment variables

It surfaces availability only. It does not render secret values in the UI.

## Install locally

### Option A: unzip into your VS Code extensions folder

Windows:

`%USERPROFILE%\.vscode\extensions\softdev0-local.sarahmemory-aios-0.4.0`

Then reload VS Code.

### Option B: package into a VSIX

From the extension folder:

```bash
npm install -g @vscode/vsce
vsce package
```

Then in VS Code choose **Extensions → ... → Install from VSIX...**

## Commands

- `SarahMemory: Focus Sidebar Chat`
- `SarahMemory: Open Full Chat Panel`
- `SarahMemory: Open VS Code Chat`
- `SarahMemory: Ask`
- `SarahMemory: Send Selection`
- `SarahMemory: Check Health`
- `SarahMemory: Start AiOS Main`
- `SarahMemory: Stop AiOS Main`
- `SarahMemory: Restart AiOS Main`
- `SarahMemory: Start Flask API`
- `SarahMemory: Insert Last Reply`
- `SarahMemory: Set API Base URL`
- `SarahMemory: Refresh Local Models`
- `SarahMemory: Run Terminal Task`
- `SarahMemory: Launch Agent Task`

## Settings seeded automatically

On activation, the extension seeds these settings when missing:

- `sarahMemory.apiBaseUrl = http://127.0.0.1:8000`
- `sarahMemory.autoStartAiOSOnStartup = true`
- `sarahMemory.autoFocusSidebarOnStartup = true`
- `sarahMemory.modelsRoot = C:\SarahMemory\data\models`
- `sarahMemory.selectedProvider = local_llm`

It also makes a best-effort attempt to ensure `chat.disableAIFeatures = false` if that setting had been disabled.

## Operational model

This build is designed so SarahMemory can function as:

- the VS Code sidebar chat surface
- a built-in VS Code Chat participant (`@sarahmemory`)
- the runtime launcher for local SarahMemory
- the workspace-aware context bridge into SarahMemory
- a model-swap front-end for locally installed models
- a diagnostics and routing console for SarahMemory runtime state

## Important platform limitation

This build makes SarahMemory a first-class participant in VS Code Chat, but it does **not** replace Microsoft's built-in Copilot backend globally. In the built-in Chat surface, SarahMemory is used through the supported participant model (`@sarahmemory`) rather than by taking ownership of Copilot itself.

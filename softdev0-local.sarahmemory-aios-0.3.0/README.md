# SarahMemory AiOS VS Code Extension

SarahMemory-first VS Code chat, workspace, and runtime integration for a running SarahMemory AiOS instance.

## What this build does

This revision moves SarahMemory from a simple panel into a workspace-owned chat/runtime surface:

- launches `python SarahMemoryMain.py` automatically when VS Code starts
- seeds VS Code settings automatically with SarahMemory defaults
- defaults the SarahMemory API base URL to `http://127.0.0.1:8000`
- sends active file content and workspace file context automatically with chat requests
- surfaces SarahMemory health, routing, notes, and runtime diagnostics live in the chat UI
- discovers local model folders under `C:\SarahMemory\data\models` and exposes them in a quick-swap selector
- discovers API key presence from `.env` and OS environment variables so you do not need to re-enter them in the extension
- provides a terminal-task launcher and an agent-task chat launcher inside the SarahMemory chat surface

## Important runtime note

This extension uses SarahMemory as the chat backend. The SarahMemory chat view replies through your own SarahMemory runtime rather than VS Code's built-in AI surfaces.

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

## Local model discovery

The extension scans:

- `C:\SarahMemory\data\models`

Every top-level model directory is surfaced in the chat UI. Folder names such as:

- `Qwen_Qwen2.5-7B-Instruct`
- `google_gemma-3-4b-it`

are converted into display labels such as:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-4b-it`

The selected model is attached to SarahMemory requests and also exported into the SarahMemory launch environment as `ACTIVE_LLM_MODEL` when the extension starts `SarahMemoryMain.py`.

## API key discovery

The extension reads key presence from:

- `${workspaceFolder}\.env`
- the VS Code process environment / OS environment variables

It surfaces availability only. It does not render secret values in the UI.

## Install locally

### Option A: unzip into your VS Code extensions folder

Windows:

`%USERPROFILE%\.vscode\extensions\softdev0-local.sarahmemory-aios-0.3.0`

Then reload VS Code.

### Option B: package into a VSIX

From the extension folder:

```bash
npm install -g @vscode/vsce
vsce package
```

Then in VS Code choose **Extensions → ... → Install from VSIX...**

## Commands

- `SarahMemory: Focus Chat`
- `SarahMemory: Open Full Chat Panel`
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

## Operational model

This build is designed so SarahMemory can function as:

- the VS Code chat surface for the user
- the runtime launcher for local SarahMemory
- the workspace-aware context bridge into SarahMemory
- a model-swap front-end for locally installed models
- a diagnostics and routing console for SarahMemory runtime state

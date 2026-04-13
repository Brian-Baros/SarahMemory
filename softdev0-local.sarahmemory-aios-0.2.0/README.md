# SarahMemory AiOS VS Code Extension

SarahMemory-first VS Code integration with a dedicated **sidebar chat interface** for a running SarahMemory AiOS instance.

## What changed

This revision moves SarahMemory beyond a simple command bridge and makes it the **primary SarahMemory chat surface** inside VS Code:

- Adds a dedicated **SarahMemory Activity Bar icon**.
- Adds a persistent **SarahMemory Chat** sidebar view.
- Keeps the optional **full chat panel** for larger conversations.
- Retains launch helpers for:
  - `SarahMemory AiOS Main`
  - `SarahMemory Flask API`
- Retains API routing through:
  - `GET /api/health`
  - `POST /api/chat`

## Important limitation

This extension makes SarahMemory the primary **SarahMemory chat UI** inside VS Code, but it does **not** replace or override VS Code's built-in Copilot or built-in Chat surfaces. VS Code extensions can contribute their own **View Containers**, **Views**, **Webview Views**, and **Commands**, but they do not take ownership of Microsoft's built-in Copilot UI.

Operationally, the intended workflow is:

1. Use the **SarahMemory** Activity Bar item as your main AI chat workspace.
2. Optionally hide Copilot in VS Code if you do not want to use it.
3. Keep SarahMemory's launch, health, and prompt flow fully local through your own SarahMemory runtime.

## Expected SarahMemory API routes

- `GET /api/health`
- `POST /api/chat`

## Install locally

### Option A: unzip into your VS Code extensions folder

Windows:

`%USERPROFILE%\\.vscode\\extensions\\softdev0-local.sarahmemory-aios-0.2.0`

Then reload VS Code.

### Option B: package into a VSIX later

From the extension folder:

```bash
npm install -g @vscode/vsce
vsce package
```

Then in VS Code choose **Extensions → ... → Install from VSIX...**

## Configure

Open VS Code settings and set:

- `SarahMemory AiOS: Api Base Url`

Example:

- `http://127.0.0.1:5000`
- `http://127.0.0.1:8000`

Use the URL for the SarahMemory API process that exposes `/api/health` and `/api/chat`.

## Commands

- `SarahMemory: Focus Chat`
- `SarahMemory: Open Full Chat Panel`
- `SarahMemory: Ask`
- `SarahMemory: Send Selection`
- `SarahMemory: Check Health`
- `SarahMemory: Start AiOS Main`
- `SarahMemory: Start Flask API`
- `SarahMemory: Insert Last Reply`
- `SarahMemory: Set API Base URL`

## Notes

This build is chat-first and sidebar-first. It is designed to make SarahMemory feel like your primary AI workspace inside VS Code while still staying inside the supported VS Code extension model.

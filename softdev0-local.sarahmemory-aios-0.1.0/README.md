# SarahMemory AiOS VS Code Extension

Local VS Code bridge for a running SarahMemory AiOS instance.

## What it does

- Opens a SarahMemory panel inside VS Code.
- Sends prompts to the running SarahMemory API.
- Sends the current editor selection to SarahMemory.
- Checks `/api/health`.
- Starts your existing `launch.json` profiles for:
  - `SarahMemory AiOS Main`
  - `SarahMemory Flask API`
- Inserts the last SarahMemory reply into the active editor.

## Expected SarahMemory API routes

- `GET /api/health`
- `POST /api/chat`

## Install locally

### Option A: unzip into your VS Code extensions folder

Windows:

`%USERPROFILE%\\.vscode\\extensions\\softdev0-local.sarahmemory-aios-0.1.0`

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

- `SarahMemory: Open Panel`
- `SarahMemory: Ask`
- `SarahMemory: Send Selection`
- `SarahMemory: Check Health`
- `SarahMemory: Start AiOS Main`
- `SarahMemory: Start Flask API`
- `SarahMemory: Insert Last Reply`
- `SarahMemory: Set API Base URL`

## Notes

This is a local integration bridge. It makes **SarahMemory running inside VS Code** accessible from the editor. It does not replace VS Code's native AI/Copilot provider stack.

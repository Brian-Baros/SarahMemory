# SarahMemory AiOS V9 Workstation UI Source

This is a source-code package, not a hosted-site shortcut and not a compiled-only export.

It is based on the older V9 UI because that version contains the working panels, backend wiring, avatar controls, NAILDE workbench, file browser, research lane, DL Engine, SarahNet, media, settings, and terminal surfaces.

The revamp keeps those working screens and adds a cleaner source registry, better local build scripts, backend URL fixes, responsive shell improvements, and a professional workstation desktop/mobile layer.

This revision also adds universal panel-to-Chat context handoff, governed file upload with path/hash receipts, explicit local ingestion controls, microphone permission/device testing and recognition-language options, plus an MCP workbench under SarahNet for JSON-RPC initialization, tools, resources, prompts, receipts, and passive evidence.

This workstation revision adds:

- Real SarahMemory logo in the launcher and mobile header.
- Desktop shortcuts for the main panels.
- Workspace presets for Chat, Research, Operator, Engineer, and Media modes.
- Start/AiOS launcher polish with app search and quick settings.
- Background brightness, dim overlay, blur, and panel opacity sliders.
- Improved top/bottom/left/right taskbar handling.
- Mobile portrait quick actions for Chat, Camera Vision, Voice, Files, Avatar, NAILDE, Models, and Settings.
- Camera Vision mobile route using the existing `/vision` HUD and existing `/api/vision/*` backend contracts.

## Run on Windows

From this folder:

```bat
INSTALL_WINDOWS.bat
START_LOCAL_UI.bat
```

Or manually:

```bat
npm ci --include=dev --legacy-peer-deps
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Build on Windows

```bat
REBUILD_V9_UI.bat
```

Manual equivalent:

```bat
npm ci --include=dev --legacy-peer-deps
npm run build
```

If you want TypeScript checking too:

```bat
npm run check
npm run build:checked
```

## Why `tsc is not recognized` happened

`tsc` is the TypeScript compiler. It is installed locally by `npm ci --include=dev` because it lives in `devDependencies`.

Do not install random global TypeScript just to fix this. Install the project dependencies from the source folder first. The npm scripts call the local compiler through:

```text
node .\node_modules\typescript\bin\tsc --noEmit
```

## Backend connection

Default local API Bridge:

```text
http://127.0.0.1:8000
```

If your bridge runs elsewhere, copy `.env.example` to `.env.local` and set:

```text
VITE_SARAH_API_URL=http://127.0.0.1:8000
```

The frontend routes NAILDE, avatar, webcam, DL Engine, Camera Vision, chat, and ingestion calls through the configured backend instead of hardcoding same-origin `/api` calls.

## Source map

Start here:

- `docs/UI_SOURCE_MAP.md`
- `src/features/featureRegistry.tsx`
- `docs/FUNCTIONAL_FIXES.md`
- `docs/LOCAL_BUILD_WINDOWS.md`
- `docs/VALIDATION_REPORT.md`

## Important runtime truth

The 3D avatar loads only when the SarahMemory backend or local public asset provides a real GLB/model endpoint. If no model exists, the UI falls back to the working V9 2D/morph avatar and visibly animates during speaking/listening instead of pretending the 3D model loaded.

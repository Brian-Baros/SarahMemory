# SarahMemory AiOS V9 Workstation UI Source

This is a source-code package, not a hosted-site shortcut and not a compiled-only export.

It is based on the older V9 UI because that version contains the working panels, backend wiring, avatar controls, NAILDE workbench, file browser, research lane, DL Engine, SarahNet, media, settings, and terminal surfaces.

The revamp keeps those working screens and adds a cleaner source registry, better local build scripts, backend URL fixes, responsive shell improvements, and a professional workstation desktop/mobile layer.

This revision also adds universal panel-to-Chat context handoff, governed file upload with path/hash receipts, explicit local ingestion controls, microphone permission/device testing and recognition-language options, plus an MCP workbench under SarahNet for JSON-RPC initialization, tools, resources, prompts, receipts, and passive evidence.

This workstation revision adds:

- Real SarahMemory logo in the launcher and mobile header.
- Movable persisted desktop shortcuts for main panels.
- User-created desktop shortcuts for registered apps or URLs.
- User-created shortcuts can be deleted from the shortcut itself or by dragging them onto Recovery Bin.
- Desktop Trash icon that opens the Files panel Trash surface and activates real trash support when the API Bridge advertises it.
- Audio mixer panel from the toolbar speaker icon with master/output/input volume, bass, treble, balance, voice, microphone, spatial audio, and noise suppression controls.
- Nexus Power options for Power Down, Reboot, and Sleep Mode requests.
- Sleep Mode requests REM / DL mode, blanks the screen, and wakes on keyboard, mouse, or touch input.
- Clock/date popover from the command rail clock with editable date, time, and timezone/locality context.
- Clock timezone control is now a dropdown with selected-timezone preview and constrained popover layout.
- WiFi/network command rail button that opens Device Manager directly to network devices.
- Device Manager screen for boot-detected governed drivers, network/audio/printer/storage/input/camera grouping, registry enable/autoload/trust controls, and driver configuration.
- Device Manager dynamically normalizes driver bridge, manifest audit, self/body hardware, vision, and browser-visible inventory sources.
- Ask Sarah context handoff moved into each window title bar so it no longer covers screen content.
- Workspace presets for Chat, Research, Operator, Engineer, and Media modes.
- Start/AiOS launcher polish with app search and quick settings.
- Background brightness, dim overlay, blur, and panel opacity sliders.
- All shipped themes now drive the current React shell tokens, material opacity, borders, status colors, popovers, inputs, sidebar, and glow variables.
- Improved top/bottom/left/right taskbar handling.
- Mobile portrait quick actions for Chat, Camera Vision, Voice, Audio, Files, Avatar, NAILDE, Models, Apps, and Settings.
- Mobile All Apps launcher so every registered screen remains touch-accessible in V-View.
- Camera Vision mobile route using the existing `/vision` HUD and existing `/api/vision/*` backend contracts.
- Pointer-based window movement/resizing so desktop windows work with mouse, pen, and touch.
- Screen and panel owner directories under `src/components/screens/*` and `src/components/panels/*`.

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
- `src/components/screens/README.md`
- `src/components/panels/README.md`
- `docs/FUNCTIONAL_FIXES.md`
- `docs/LOCAL_BUILD_WINDOWS.md`
- `docs/VALIDATION_REPORT.md`

## Important runtime truth

The 3D avatar loads only when the SarahMemory backend or local public asset provides a real GLB/model endpoint. If no model exists, the UI falls back to the working V9 2D/morph avatar and visibly animates during speaking/listening instead of pretending the 3D model loaded.

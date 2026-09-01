# Functional Fixes in This Source Package

## Preserved from old V9

- Real old V9 screen components were retained.
- Chat, Avatar, NAILDE, Files, Research, Studios, Media, SarahNet, DL Engine, Addons, Terminal, History, and Settings remain separate owner files.
- Existing API client and state stores remain the core integration path.

## Fixed backend routing

Several components called raw same-origin `/api/...` endpoints. That breaks when the frontend runs from Vite on `127.0.0.1:5173` while the SarahMemory API Bridge runs on `127.0.0.1:8000`.

Fixed files:

- `src/components/chat/ChatPanel.tsx`
- `src/components/avatar/WebcamOverlay.tsx`
- `src/components/screens/nailde/NAILDEScreen.tsx`
- `src/components/screens/dl-engine/DLEngineScreen.tsx`

Those now route through `src/lib/config.ts` and `apiFetch`, or use `config.apiBaseUrl` for multipart upload.

## Voice and avatar fixes

- `src/lib/api.ts` now tries multiple voice speak routes:
  - `/api/tts/speak`
  - `/api/voice/speak`
  - `/api/voice`
  - `/voice/speak`
- Duplicate voice response keys were removed.
- `src/components/avatar/AvatarPanel.tsx` now animates the 2D fallback during speaking/listening, so the avatar does not become a dead static card when the canvas or 3D model is unavailable.
- Real 3D remains honest: it requires a real GLB/model endpoint or local asset.

## NAILDE revamp

- NAILDE now opens in a cleaner default IDE layout instead of scattered oversized floating panels.
- The surface is styled as VS Code navigation plus VB6 toolbox/properties/forms.
- Core panels open by default:
  - Battle Plan
  - Project Explorer
  - Natural Language Prompt
  - Code Editor
  - Output / Evidence
  - VB Toolbox
  - Properties
  - Governed Terminal
- Advanced panels remain available through the activity rail, menu, and command palette.

## Source organization

Added:

- `src/features/featureRegistry.tsx`
- `src/components/screens/README.md`
- `src/components/panels/README.md`
- `docs/UI_SOURCE_MAP.md`
- `docs/LOCAL_BUILD_WINDOWS.md`
- `docs/FUNCTIONAL_FIXES.md`
- `docs/VALIDATION_REPORT.md`
- `INSTALL_WINDOWS.bat`
- `START_LOCAL_UI.bat`
- rebuilt `REBUILD_V9_UI.bat`

Desktop and mobile shells now use one feature registry instead of duplicated import/switch logic.

Registered screens now live in owner directories under `src/components/screens/*`.
Shared panels now live in owner directories under `src/components/panels/*`.

## Workstation UI revamp

- Added the supplied SarahMemory logo as `src/assets/smaios-logo.jpg`.
- Reworked the desktop shell into a branded workstation surface with desktop shortcuts and workspace presets.
- Added persisted sliders for:
  - background brightness
  - background dim overlay
  - background blur
  - panel opacity
- Routed the slider values through `useSarahStore`, `DesktopShell`, `SettingsScreen`, and `AiOSShellCenter`.
- Added persisted draggable desktop shortcuts in `DesktopShell`.
- Added a desktop shortcut manager for user-created app and URL shortcuts.
- Added a Trash desktop shortcut wired through the UI control bus to the Files Trash surface.
- Added a shared Audio Mixer panel under `src/components/panels/audio-mixer`.
- The audio mixer emits `sarah:audio` events so a future driver/AppSys backend bridge can bind to the same frontend contract.
- Improved taskbar behavior for bottom/top/left/right docks.
- Fixed the window manager taskbar double-subtraction issue so maximized windows fill the real workspace area.
- Added pointer-based window movement/resizing for mouse, pen, and touch.
- Added mobile portrait quick actions for Chat, Camera Vision, Voice, Audio, Files, Avatar, NAILDE, Models, Apps, and Settings.
- Added a mobile All Apps launcher so every registered panel is accessible in V-View.
- Updated Camera Vision HUD layout for mobile portrait/landscape readability while keeping the existing `/api/vision/*` contracts.

## Build/package fixes

- The package name/version now identifies this as the V9 functional source package.
- `npm run build` builds with Vite. TypeScript checking remains available through `npm run check` and `npm run build:checked`.
- `INSTALL_WINDOWS.bat`, `START_LOCAL_UI.bat`, and `REBUILD_V9_UI.bat` install dev dependencies before running local commands.
- `package-lock.json` was repaired for stale `emoji-regex@9.0.0` and `file-entry-cache@9.0.0` integrity values that blocked clean installs.
- The Tailwind scanner issue caused by a timestamp regex pattern was fixed so it no longer generates invalid CSS.

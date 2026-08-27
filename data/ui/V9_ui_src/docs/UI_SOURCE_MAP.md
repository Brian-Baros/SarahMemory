# UI Source Map

This package keeps the functional old V9 UI as the base. The source is organized by runtime responsibility so each screen/panel can be traced quickly.

## Bootstrap

| Area | File | Purpose |
| --- | --- | --- |
| React entry | `src/main.tsx` | Mounts the React app. |
| App providers | `src/App.tsx` | Query client, router, theme/toast wrappers. |
| Main route | `src/pages/Index.tsx` | Chooses desktop/mobile shell. |
| Global styles | `src/index.css` | Theme tokens, viewport fixes, avatar fallback motion. |
| Backend config | `src/lib/config.ts` | Resolves SarahMemory API base URL and provides `apiFetch`. |
| API client | `src/lib/api.ts` | Chat, voice, avatar, files, NAILDE, DL Engine, SarahNet, and other backend adapters. |

## Shell and navigation

| Area | File | Purpose |
| --- | --- | --- |
| Feature registry | `src/features/featureRegistry.tsx` | Single source of truth mapping app/screen IDs to components and owner files. |
| Panel-to-Chat bridge | `src/components/shell/PanelChatBridge.tsx` | Adds the universal Ask Sarah context handoff to every registered non-Chat screen. |
| Desktop shell | `src/components/shell/DesktopShell.tsx` | Desktop workspace, wallpaper, dock/taskbar placement. |
| Window manager | `src/components/shell/WindowManager.tsx` | Opens registered panels in draggable/resizable windows. |
| Window chrome | `src/components/shell/Window.tsx` | Generic AiOS window frame. |
| Mobile shell | `src/components/shell/MobileShell.tsx` | Mobile screen host with swipe navigation and portrait quick actions. |
| Bottom navigation | `src/components/shell/BottomNav.tsx` | Primary mobile nav and desktop app shortcuts. |
| Dock/control center | `src/components/shell/Dock.tsx`, `src/components/shell/ControlCenter.tsx` | Desktop dock and quick controls. |
| Status/taskbar | `src/components/StatusBar.tsx` | Clock, readiness, launcher, app buttons, and system status. |
| Window state | `src/stores/useWindowStore.ts` | Window IDs, defaults, positions, presets, and UI control bus. |
| Mobile nav state | `src/stores/useNavigationStore.ts` | Mobile screens, swipe order, active desktop app, shell mode. |

## Main screens

| Screen | Owner file | Backend/API path |
| --- | --- | --- |
| Chat | `src/components/chat/ChatPanel.tsx` | `src/lib/api.ts` chat/voice/avatar APIs |
| Chat composer | `src/components/chat/ChatComposer.tsx` | Sends text/files into ChatPanel handlers |
| Chat message | `src/components/chat/ChatMessage.tsx` | Renders response cards/actions |
| History | `src/components/screens/HistoryScreen.tsx` | Conversation/session history |
| Files | `src/components/screens/FilesScreen.tsx` | Filesystem and local file APIs |
| Research | `src/components/screens/ResearchScreen.tsx` | Research/evidence APIs through `apiFetch` |
| Studios | `src/components/screens/StudiosScreen.tsx` | Creative modules in `src/components/modules/*` |
| Avatar | `src/components/screens/AvatarScreen.tsx` | Avatar surface wrapper |
| SarahNet | `src/components/screens/SarahNetScreen.tsx` | Broker/node/interop/readiness endpoints |
| MCP connections | `src/components/network/MCPConnectionsPanel.tsx` | MCP initialize, tools/resources/prompts discovery, governed calls, receipts, and passive interop evidence. |
| Media | `src/components/screens/MediaScreen.tsx` | Media player/library |
| DL Engine | `src/components/screens/DLEngineScreen.tsx` | DL runtime, REM, weights, jobs, traces |
| NAILDE | `src/components/screens/NAILDEScreen.tsx` | Governed development workbench |
| Terminal | `src/components/screens/TerminalScreen.tsx` | Governed terminal request UI |
| Addons | `src/components/screens/AddonsScreen.tsx` | Addon registry and install visibility |
| Settings | `src/components/screens/SettingsScreen.tsx` | Runtime/theme/voice/model/device settings |
| Camera Vision HUD | `src/components/screens/VisionScreen.tsx` | Mobile/desktop camera HUD, frame submit, analysis trigger, target overlays |

## Workstation shell controls

| Control | Owner file | Notes |
| --- | --- | --- |
| Real logo asset | `src/assets/smaios-logo.jpg` | Used by desktop shell, Start/AiOS center, and mobile header. |
| Desktop shortcuts | `src/components/shell/DesktopShell.tsx` | Opens registered panels through `useWindowStore.openWindow`. |
| Workspace presets | `src/components/shell/DesktopShell.tsx`, `src/components/shell/AiOSShellCenter.tsx` | Opens curated panel groups without changing backend authority. |
| Start/AiOS center | `src/components/shell/AiOSShellCenter.tsx` | App search, permissions, runtime activity, workspaces, appearance controls. |
| Taskbar/tray | `src/components/StatusBar.tsx` | Mode switch, app dock, clock, sound/network indicators, API readiness. |
| Background sliders | `src/components/screens/SettingsScreen.tsx`, `src/components/shell/AiOSShellCenter.tsx` | Persisted through `useSarahStore`; applied in `DesktopShell`. |
| Mobile portrait quick actions | `src/components/shell/MobileShell.tsx` | Chat, Camera Vision, Voice, Files, Avatar, NAILDE, Models, Settings. |
| Camera Vision mobile HUD | `src/components/screens/VisionScreen.tsx` | Uses `/api/vision/frame/submit`, `/api/vision/analyze`, `/api/vision/hud/packet`, and `/api/vision/frame/status`. |

## Avatar stack

| Area | File | Purpose |
| --- | --- | --- |
| Avatar panel | `src/components/avatar/AvatarPanel.tsx` | 2D/3D mode switching, voice/listening state, fallback motion. |
| 3D renderer | `src/components/avatar/Avatar3D.tsx` | Three.js GLB loader. Requires a real model URL. |
| Preview host | `src/components/avatar/PreviewSurface.tsx` | Hosts `AvatarPanel` in screen layouts. |
| Webcam overlay | `src/components/avatar/WebcamOverlay.tsx` | Camera preview and governed frame submit. |
| Background | `src/components/avatar/AvatarBackground.tsx` | Avatar background visuals. |
| Assets | `src/assets/sarah-*.png`, `src/assets/sarah-*.webp` | Local 2D fallback images. |

## NAILDE workbench map

NAILDE is intentionally modeled as a Visual Studio Code plus Visual Basic 6.0 workbench.

| Panel | NAILDE window ID | Purpose |
| --- | --- | --- |
| Activity rail | `ACTIVITY_TO_WINDOW` in `src/components/screens/NAILDEScreen.tsx` | VS Code-style side launcher. |
| Menu bar | `menus` state loaded from backend SDK/status | File/Edit/View/Run-style command access. |
| Project Explorer | `explorer` | Sandbox workspace files. |
| Natural Language Prompt | `prompt` | Top-level app/build intent. |
| Code Editor | `editor` | Monaco-like text/code editor area with line numbers and drag/drop snippet support. |
| Output / Evidence | `output` | Build, validation, compare, and backend result stream. |
| Governed Terminal | `terminal` | Terminal-like output. No raw shell authority from the browser. |
| VB Toolbox | `toolbox` | VB6-style draggable UI/control/data/code snippets. |
| Properties | `properties` | VB6-style selected object/file properties panel. |
| Problems | `problems` | Bug/task/validation problem list. |
| Validation | `validation` | Sandbox validation results. |
| SDK Library | `sdk` | Available NAILDE adapters/capabilities. |
| Database Builder | `database_builder` | Access/VB-style tables, fields, forms, queries, reports. |
| Form Designer | `form_designer` | Form/control design notes and snippets. |
| BlockForge | `blockforge` | Block/object graph editing concept surface. |
| HoloForge / XR | `holoforge` | Future XR spatial coding model, sandbox-only. |
| Device Bay | `device_bay` | Read-only device discovery and safety boundary. |
| Governance Gates | `governance` | Required approvals and denial surfaces. |
| Ledger Receipts | `receipts` | Audit/receipt output. |
| GitHub Sandbox Bridge | `github` | Planned repository bridge actions under sandbox governance. |

## Stores

| File | Purpose |
| --- | --- |
| `src/stores/useSarahStore.ts` | Main app state, settings, chat state, avatar state, media state, voice bootstrap. |
| `src/stores/useWindowStore.ts` | Desktop window state and workspace presets. |
| `src/stores/useNavigationStore.ts` | Mobile/desktop navigation state and control bus. |
| `src/stores/usePreviewStore.ts` | Preview state. |
| `src/stores/useCreativeCacheStore.ts` | Creative module cache. |

## UI component library

Reusable shadcn/Radix primitives live in:

```text
src/components/ui/
```

Do not edit these first when changing product behavior. Edit screen/panel owner files first.

# Validation Report

Date: 2026-09-01

Package: `sarahmemory-aios-v9-functional-ui` version `9.0.0-alpha.workstation.6`

## SarahMemory UI Workmode Pass

- Preserved the existing React/Vite frontend architecture and stable route/window IDs.
- Refreshed `src/assets/smaios-logo.jpg` from the attached `SMAIOSLOGO.jpg` file and verified the bytes match.
- Reworked visible shell terminology toward SarahMemory-native UI language: Command Nexus, Command Rail, Workspace Compass, Fabric Surface, File Cortex, Evidence Lens, Model Forge, Creation Bay, and System Tuning.
- Removed the legacy development tagger dependency and Vite import path from the active source/config.
- Replaced the old fallback source label in `src/lib/api.ts` with `governed_remote_fallback`.
- Added missing `src/components/screens/README.md` and `src/components/panels/README.md` owner maps required by the UI contract verifier.
- Added visible delete controls for user-created desktop shortcuts and drag-to-Recovery-Bin removal.
- Added Nexus Power options, Sleep Mode blank/wake behavior, command rail clock/date controls, WiFi-to-Device-Manager launch, Device Manager screen, and title-bar Ask Sarah.
- Expanded Device Manager into a dynamic inventory surface that normalizes `/api/drivers`, driver manifest audit, self/body hardware topology, vision device reports, and browser-visible devices.
- Replaced the clock timezone text field with a bounded timezone dropdown, selected-timezone preview, local date math, and popover containment fixes.
- Upgraded every shipped theme CSS file to provide the full React shell token set, including material, status, input, border, popover, sidebar, glow, and `--primary-rgb` variables.
- Added local-first material variables, focused window depth, higher-contrast handling, forced-colors handling, workspace ribbon styling, and stronger focus-ring behavior.
- Did not read or copy external UI-framework source into the production tree.
- Did not modify API bridge, CORE, drivers, DATAMODS, or backend ownership zones.

## Passed

| Check | Command used | Result |
| --- | --- | --- |
| Lockfile refresh | `npm install --package-lock-only --ignore-scripts --legacy-peer-deps --offline` | Passed. |
| JSON validation | Node JSON parse for `package.json` and `package-lock.json` | Passed. |
| Logo asset verification | `cmp` against attached `SMAIOSLOGO.jpg` | Passed. |
| UI contract checks | `node ./scripts/verify-ui-contracts.mjs` | Passed: 41/41 contracts including title-bar Ask Sarah, Nexus Power, Sleep overlay, timezone dropdown, popover containment, WiFi Device Manager launch, dynamic Device Manager inventory, theme token coverage, panel-to-Chat context, upload, ingest, microphone, MCP, workstation shell, movable desktop shortcuts, custom shortcuts, custom shortcut deletion, Trash handoff, audio mixer, Settings audio, mobile launcher, mobile audio, Camera Vision, source organization, and local bridge target. |
| Theme CSS integrity | Node CSS token/balance check over `src/index.css` and `public/themes/*.css` | Passed for all shipped themes. |
| Legacy label scan | Source/docs/public/package search | Passed across active source/docs/public/package files, excluding `dist`. |

## Expected warnings

- Vite reports `/api/ui/runtime-config.js` as a non-bundled runtime script. That is intentional because the backend can serve runtime configuration outside the static bundle.
- Vite reports large chunks. This comes from the old V9 all-in-one app surface plus Three.js/avatar/media dependencies. It is a performance cleanup item, not a build failure.
- ESLint reports React dependency-array warnings in legacy screens. These are warnings, not errors. They should be addressed incrementally with behavior tests instead of mass-edited blindly.
- NPM may print an inherited `http-proxy` warning in this container. It did not block the offline lockfile refresh or UI contract verifier.

## Not Verified In This Workspace

| Check | Command used | Result |
| --- | --- | --- |
| Dependency install | `npm ci --include=dev --legacy-peer-deps --offline` | Failed: npm cache is empty and `zustand-5.0.9.tgz` was not available offline. Network package fetch was blocked by workspace policy. |
| TypeScript | `npm run check` / `tsc --version` | Failed because `node_modules/typescript/bin/tsc` is not installed in this workspace and no global `tsc` is available. |
| Production build | `npm run build` | Failed because `node_modules/vite/bin/vite.js` is not installed in this workspace. |
| ESLint | `npm run lint` | Not run because dependencies are not installed in this workspace. |
| Browser preview / screenshot QA | Vite preview / Playwright | Not run because the production build could not be regenerated here. |

## Runtime evidence boundary

- File upload is wired to the real API Bridge route `/api/files/upload`; explicit durable learning is wired to `/api/ingest/eat_this` and remains local-only/governance-gated.
- Browser microphone permission, device enumeration/test, recognition language, backend listening state, server TTS, and browser speech fallback are wired. Physical microphone capture was not hardware-tested in this workspace.
- Mobile Camera Vision is wired through the existing `/vision` route and `/api/vision/*` contracts. Physical camera/object-recognition with the running CORE was not hardware-tested in this workspace.
- Movable desktop shortcuts, custom app/URL shortcuts, and desktop shortcut positions are frontend-persisted through `useSarahStore`.
- Desktop Trash opens the Files Trash surface and attempts advertised backend trash/dumpster list routes. The current bridge must return `canTrash: true` and provide trash endpoints before live filesystem Trash listing/restoration can be verified.
- The audio mixer stores master/output/input/EQ preferences and emits `sarah:audio` events. Live driver/AppSys volume binding is not verified because this pass intentionally does not change backend/API bridge code.
- Device Manager reads existing `/api/drivers*`, `/api/self/*`, and `/api/vision/devices` inventory signals, supplements them with safe browser-visible devices, and requests registry/config/session changes only through existing driver governance. Real WiFi password/profile application depends on the corresponding authorized network driver.
- Nexus Sleep Mode calls `/api/avatar/rem/start`, blanks the UI locally, wakes on keyboard/mouse/touch, and calls `/api/avatar/rem/stop`. Power Down/Reboot are queued as governed UI requests because no verified system-power execution endpoint exists in the inspected API bridge.
- MCP initialize, tools, resources, prompts, calls, receipts, and passive evidence are implemented through a configurable local JSON-RPC gateway. The supplied API Bridge currently reports `mcp: adapter_only` and blocks direct remote tool execution, so successful external MCP execution requires a governed gateway implementation at the configured endpoint.
- Playwright package was available, but no browser executable was installed at `/root/.cache/ms-playwright/...`; screenshot QA is therefore not verified in this container.
- Vite preview printed `http://127.0.0.1:4173/`, but an independent `curl`/`wget` connection check was refused in this workspace. Runtime preview is therefore not claimed as verified for this pass.

## Install/build notes

The prior Windows error:

```text
'tsc' is not recognized as an internal or external command
```

means `node_modules` had not been installed with dev dependencies. Run `INSTALL_WINDOWS.bat` first, then `REBUILD_V9_UI.bat`.

The source package should exclude:

- `node_modules/`
- `dist/`
- `.env`
- npm logs/cache

The included `.env.example` is safe to copy to `.env.local`.

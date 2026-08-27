# Validation Report

Date: 2026-08-27

Package: `sarahmemory-aios-v9-functional-ui` version `9.0.0-alpha.workstation.4`

## Passed

| Check | Command used | Result |
| --- | --- | --- |
| Dependency install | `npm ci` | Passed with local `.npmrc` using dev dependencies and legacy peer resolution. |
| TypeScript | `npm run check` | Passed. |
| ESLint | `npm run lint` | Passed with legacy warnings only; zero errors. |
| Production build | `npm run build` | Passed. |
| UI contract checks | `npm run test:contracts` | Passed: 12/12 contracts including panel-to-Chat, upload, ingest, microphone, MCP, workstation shell, appearance sliders, mobile Camera Vision, and local bridge target. |
| Same-origin API scan | `rg "fetch\\((['\\\"])/api"` | Passed; raw local `/api` fetches were routed through config/api helpers. |
| Tailwind malformed utility scan | `rg "\\[-:\\.TZ\\]\|-:\\.TZ|\\.TZ"` | Passed after replacing the timestamp regex pattern. |

## Expected warnings

- Vite reports `/api/ui/runtime-config.js` as a non-bundled runtime script. That is intentional because the backend can serve runtime configuration outside the static bundle.
- Vite reports large chunks. This comes from the old V9 all-in-one app surface plus Three.js/avatar/media dependencies. It is a performance cleanup item, not a build failure.
- ESLint reports React dependency-array warnings in legacy screens. These are warnings, not errors. They should be addressed incrementally with behavior tests instead of mass-edited blindly.
- NPM may print an inherited `http-proxy` warning in this container. It did not block install/build.

## Runtime evidence boundary

- File upload is wired to the real API Bridge route `/api/files/upload`; explicit durable learning is wired to `/api/ingest/eat_this` and remains local-only/governance-gated.
- Browser microphone permission, device enumeration/test, recognition language, backend listening state, server TTS, and browser speech fallback are wired. Physical microphone capture was not hardware-tested in this workspace.
- Mobile Camera Vision is wired through the existing `/vision` route and `/api/vision/*` contracts. Physical camera/object-recognition with the running CORE was not hardware-tested in this workspace.
- MCP initialize, tools, resources, prompts, calls, receipts, and passive evidence are implemented through a configurable local JSON-RPC gateway. The supplied API Bridge currently reports `mcp: adapter_only` and blocks direct remote tool execution, so successful external MCP execution requires a governed gateway implementation at the configured endpoint.
- Playwright package was available, but no browser executable was installed at `/root/.cache/ms-playwright/...`; screenshot QA is therefore not verified in this container.

## Install/build notes

The prior Windows error:

```text
'tsc' is not recognized as an internal or external command
```

means `node_modules` had not been installed with dev dependencies. Run `INSTALL_WINDOWS.bat` first, then `REBUILD_V9_UI.bat`.

The source package excludes:

- `node_modules/`
- `dist/`
- `.env`
- npm logs/cache

The included `.env.example` is safe to copy to `.env.local`.

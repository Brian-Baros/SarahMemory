# Local Windows Build Guide

## Prerequisites

- Node.js 20.19 or newer
- npm 10 or newer
- SarahMemory API Bridge running locally for full backend functionality

Check versions:

```bat
node --version
npm --version
```

## Install

Run from the extracted source folder:

```bat
INSTALL_WINDOWS.bat
```

Or manually:

```bat
npm ci --include=dev --legacy-peer-deps
```

If `npm ci` reports a lock/cache integrity issue, run:

```bat
npm install --include=dev --legacy-peer-deps
```

This installs `vite`, `typescript`, React, Tailwind, Three.js, and every source dependency into `node_modules`.

## Start development UI

```bat
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Build production files

```bat
npm run build
```

Output:

```text
dist/
```

## TypeScript check

```bat
npm run check
```

## If `tsc is not recognized`

That means dependencies are missing or only production dependencies were installed.

Fix:

```bat
INSTALL_WINDOWS.bat
```

Then run:

```bat
npm run check
```

Do not rely on a global `tsc`; this package calls the project-local compiler through `node ./node_modules/typescript/bin/tsc`.

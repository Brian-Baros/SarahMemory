# Screen Component Layout

SarahMemory UI screen owners live under this directory. Screens are full-window or full-surface application views opened from the desktop, mobile shell, Nexus menu, or governed Chat UI actions.

Canonical screen owners:

- `addons/AddonsScreen.tsx` — ADDONS surface.
- `avatar/AvatarScreen.tsx` — avatar runtime surface.
- `device-manager/DeviceManagerScreen.tsx` — hardware/device manager surface.
- `dl-engine/DLEngineScreen.tsx` — model/DL engine surface.
- `files/FilesScreen.tsx` — file manager and Trash handoff surface.
- `history/HistoryScreen.tsx` — chat/history surface.
- `media/MediaScreen.tsx` — media tools surface.
- `nailde/NaildeScreen.tsx` — NAILDE build/workbench surface.
- `research/ResearchScreen.tsx` — research surface.
- `sarah-net/SarahNetScreen.tsx` — SarahNet/MCP/fabric surface.
- `settings/SettingsScreen.tsx` — system settings surface.
- `studios/StudiosScreen.tsx` — creative studio surface.
- `terminal/TerminalScreen.tsx` — terminal surface.
- `vision/VisionScreen.tsx` — camera vision/object-recognition surface.

Rules:

- Screens own large application workflows, not small popovers.
- Chat UI may open a screen only from an explicit UI/action command.
- Hardware fact questions must answer in Chat and must not auto-open a screen.

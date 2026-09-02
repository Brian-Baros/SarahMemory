# Screen Component Layout

Top-level routed workspace screens live in this directory. Each complex screen
keeps its own owner folder so frontend work stays zoned and reviewable.

| Screen | Owner file | Responsibility |
| --- | --- | --- |
| Addons | `addons/AddonsScreen.tsx` | Addon catalog and governed lifecycle actions. |
| Avatar | `avatar/AvatarScreen.tsx` | Avatar presentation surface. |
| DL Engine | `dl-engine/DLEngineScreen.tsx` | Model download and engine controls. |
| Device Manager | `device-manager/DeviceManagerScreen.tsx` | Boot-detected governed driver/device inventory with class-specific configuration profiles. |
| Files | `files/FilesScreen.tsx` | File Cortex, upload, selection, and Trash surface. |
| History | `history/HistoryScreen.tsx` | Memory Trail and prior conversations. |
| Media | `media/MediaScreen.tsx` | Media Deck controls. |
| NAILDE | `nailde/NAILDEScreen.tsx` | NAILDE workbench and addon generation. |
| Research | `research/ResearchScreen.tsx` | Evidence Lens and research routing. |
| SarahNet | `sarah-net/SarahNetScreen.tsx` | SarahNet fabric, MCP, realtime, and worlds views. |
| Settings | `settings/SettingsScreen.tsx` | System Tuning, appearance, audio, backend, and policy controls. |
| Studios | `studios/StudiosScreen.tsx` | Creation Bay studio launcher. |
| Terminal | `terminal/TerminalScreen.tsx` | Operator Terminal shell bridge. |
| Vision | `vision/VisionScreen.tsx` | Camera Vision and VR HUD route. |

# Panel Component Layout

SarahMemory UI panel owners live under this directory. Panels are bounded supporting controls used inside the shell, status bar, drawers, or screen contexts.

Canonical panel owners:

- `audio-mixer/AudioMixerPanel.tsx` — volume, bass, treble, channel mode controls.
- `contacts/ContactsPanel.tsx` — contact quick panel.
- `creative-tools/CreativeToolsPanel.tsx` — creative workflow support panel.
- `dialer/DialerPanel.tsx` — communication keypad panel.
- `reminders/RemindersPanel.tsx` — reminder quick panel.
- `settings-modal/SettingsModal.tsx` — modal settings surface.
- `system-clock/SystemClockPanel.tsx` — date, time, timezone, and clock-court controls.
- `terminal/TerminalPanel.tsx` — embedded terminal/control panel.

Rules:

- Panels must remain composable and bounded.
- Screens may embed panels when ownership is clear.
- Chat response actions must use the single UI action queue and avoid duplicate `sarah:ui` dispatch for the same returned action batch.

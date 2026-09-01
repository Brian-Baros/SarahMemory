# Panel Component Layout

Reusable window panels live in this directory. Each panel keeps a dedicated
folder when it has state, backend contracts, or non-trivial controls.

| Panel | Owner file | Responsibility |
| --- | --- | --- |
| Audio Mixer | `audio-mixer/AudioMixerPanel.tsx` | Master volume, input/output volume, EQ, and audio event dispatch. |
| Contacts | `contacts/ContactsPanel.tsx` | Contact list and contact editing. |
| Creative Tools | `creative-tools/CreativeToolsPanel.tsx` | Creative tool controls used by studio surfaces. |
| Dialer | `dialer/DialerPanel.tsx` | Phone/keypad style communication controls. |
| Reminders | `reminders/RemindersPanel.tsx` | Reminder creation and completion controls. |
| Settings Modal | `settings-modal/SettingsModal.tsx` | Modal settings entry point. |
| Terminal | `terminal/TerminalPanel.tsx` | Embedded terminal panel controls. |

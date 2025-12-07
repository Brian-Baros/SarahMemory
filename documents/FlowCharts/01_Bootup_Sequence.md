# Bootup Sequence

```mermaid
flowchart TD
    A[User runs: python SarahMemoryMain.py] --> B[Load SarahMemoryGlobals.py config]
    B --> D[Load core modules]
    D --> E[Show Main Menu]
    E -->|1| F[Launch GUI]
    E -->|2| G[Shutdown / Exit]
```

**Notes:**

- Main does not appear to call DB initialization early; ensure DBs exist before diagnostics/personality subsystems read them.
- Could not statically detect a GUI launcher symbol in SarahMemoryMain.py; verify the menu option triggers SarahMemoryGUI.py correctly.

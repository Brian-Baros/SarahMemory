# Right Hemisphere: Logic/Control/Adaptation

```mermaid
flowchart TD
    A[System context monitor] --> B[Platform detect (Windows/Linux/macOS/Android/iOS)]
    B --> C[Hardware info: SarahMemoryHi.py]
    B --> D[Software info: SarahMemorySi.py]
    C --> E[Diagnostics: SarahMemoryDiagnostics.py]
    D --> E
    E --> F[Optimization: SarahMemoryOptimization.py]
    F --> G[Apply safe changes + rollback plan]
    G --> H[Research missing capabilities: SarahMemorySoftwareResearch.py]
    H --> I[Integrate findings, update config, notify user]
    I --> J[Continuous loop with rate limits]
```

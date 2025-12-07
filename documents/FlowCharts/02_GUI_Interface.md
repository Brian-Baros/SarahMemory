# GUI Interface Flow

```mermaid
flowchart TD
    A[GUI Loaded] --> B[Init widgets: tabs/panels/status lights/mini-browser]
    B --> C[User input (text/voice) -> generate_response()]
    C --> D[Routing: local/web/api pipelines]
    D --> E[Personality + Emotion synthesis]
    E --> F[Reply output + Avatar cues]
    F --> G[Logs/DB updates]
    A --> H[Exit Button / Close Window] --> I[Graceful shutdown: stop threads, save state, release devices]
```

**Notes:**

- No button/click handler functions were detected; UI wiring may be inline or named differently.

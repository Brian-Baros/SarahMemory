# Agent Task Execution Flow

```mermaid
flowchart TD
    A[Operator command (voice/text)] --> B[Intent classification & entity extraction]
    B --> C{Permission & safety checks}
    C -->|approved| D[Executor selects toolchain: OS control / web / API / codegen]
    C -->|denied| Z[Politely refuse / ask clarification]
    D --> E[Plan steps & generate actions]
    E --> F[Execute step-by-step with retries]
    F --> G[Capture outputs, errors, and logs]
    G --> H[Summarize result for user + update learning DBs]
    H --> I[Trigger Avatar gestures / TTS / UI notifications]
```

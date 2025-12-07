# Reply & Compare Flow

```mermaid
flowchart TD
    A[Candidate responses (N)] --> B[SarahMemoryCompare.py: compare/rank/grade]
    B --> C{COMPARE_VOTE?}
    C -->|True| D[Ask user to choose/approve]
    C -->|False| E[Auto-select best candidate]
    D --> F[Record vote decision + feedback]
    E --> F
    F --> G[Update learning DBs & confidence weights]
    G --> H[SarahMemoryReply.py: finalize & store]
```

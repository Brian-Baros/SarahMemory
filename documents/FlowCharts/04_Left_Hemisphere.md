# Left Hemisphere: Emotion + 3-Layer Pipeline

```mermaid
flowchart TD
    A[User input in GUI] --> B[generate_response()]
    B --> C[Parser: topic/sentiment/emotion cues]
    C --> D{Routing policy: local vs web vs api}
    D -->|local| E[SarahMemoryResearch.py: local KB/vector search]
    D -->|web| F[SarahMemoryWebsym.py: scraping/search APIs]
    D -->|api| G[External LLM/API calls]
    E --> H[Candidate answers]
    F --> H
    G --> H
    H --> I[SarahMemoryPersonality.py: style, tone]
    I --> J[Emotion model: emojis, expressive cues]
    J --> K[SarahMemoryReply.py: assemble final message]
    K --> L[Display in GUI + Avatar + Logs]
```

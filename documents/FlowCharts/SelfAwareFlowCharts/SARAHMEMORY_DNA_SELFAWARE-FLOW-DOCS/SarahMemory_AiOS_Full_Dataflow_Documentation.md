# SarahMemory AiOS Full Chat/Input Dataflow Map

## Scope

This document maps the current SarahMemory AiOS input-to-output route from **Frontend Chat input** through backend governance, cognitive routing, lane execution, validation, presentation, and frontend output.

The source trace states that the backend-confirmed ingress point is `POST /api/chat` in `app.py`, and that the governed flow is:

```text
Ingress
→ Context Packet
→ Governor
→ AdvCU/Neuron
→ Compare
→ Presentation
→ Reply Bundle
```

## Lettered Flow Index

| Letter | Layer | Role |
|---|---|---|
| A | Input Surfaces | User/environment/device signals enter the organism |
| B | Backend Ingress | `app.py /api/chat` payload normalization |
| C | Canonical Query | Normalize once / classify once / fast-path gates |
| D | Context + Runtime | Flags, caller, proposed action, mode state |
| E | Cognitive Tri-Force | Self, Services, Thinker, Compass |
| F | Neuron Router | Primary lane selection and tier routing |
| G | SelfAware / Evidence Court | Body/system fact verification |
| H | General Answer | Logic, symbolic reasoning, database, model helper |
| I | Research / Browser | Web/research evidence and offline fallback |
| J | Creative Studio | Image, music, lyrics, video artifacts |
| K | Vision / Sensory | Vision, SOBJE, facial recognition, scene facts |
| L | Action / Driver / SMGET | OperatorCore, drivers, actions, rollback |
| M | SarahNet / Network | Remote node and network-governed operations |
| N | REM / DL | Learning, REM candidates, consolidation |
| O | Safety / Trust / Assurance | SafetyPolicies, TrustRegistry, SecurityGovernor, AssuranceGate |
| P | Compare / Compass | Validation, anti-drift, reanchor, release gate |
| Q | Reply / Frontend Output | Presentation bundle, chat render, avatar/voice output |
| R | Storage / Audit | Logs, memory, approved metadata, REM candidate storage |

## Primary Flow

```text
A. Input Surface
→ B. Backend Ingress
→ C. Canonical Query Packet
→ D. Context Packet / Runtime Flags
→ E. Cognitive Tri-Force
→ O. Safety / Trust / Assurance
→ F. Neuron Primary Router
→ G/H/I/J/K/L/M/N Lane Owner
→ P. Compare / Compass Validation
→ Q. Reply Bundle / Frontend Output
→ R. Optional Memory / Audit
```

## A. Input Surfaces

SarahMemory is treated as a multi-body AiOS organism. Inputs may come from:

- Classic UI
- Web / React / Flask UI
- Voice / Avatar / TTS-STT
- Vision / Sensors / Drivers
- SarahNet / remote nodes
- Browser / Research panel
- Addon / Driver UI
- REM / Deep Learning scheduled internal events

All paths converge into backend ingress.

## B. Backend Ingress

Backend ingress begins at `app.py /api/chat`.

Payload includes:

- `text`
- `intent`
- `tone`
- `complexity`
- `avatar_request`
- `diagnostics_ping`
- runtime mode flags
- caller/surface/session metadata

`app.py` builds a context packet and a virtual ingress route, then attaches the proposed action metadata.

## C. Canonical Query Layer

This layer prevents query drift.

It preserves:

- raw text
- normalized text
- domain
- intent
- requested component
- requested metric
- fact kind
- target
- answer shape
- read-only / no-write flags

This allows the system to answer factual self/body questions as evidence cases instead of generic chatbot prompts.

## D. Runtime Context

Runtime context carries:

- `LOCAL_ONLY_MODE`
- `SAFE_MODE`
- `NEOSKYMATRIX`
- `DEVELOPERSMODE`
- public/cloud/local mode
- user present / user consented
- caller authority
- proposed action metadata
- selected helper/model hints

## E. Cognitive Tri-Force

The Cognitive Tri-Force is mapped as:

| Component | File | Function |
|---|---|---|
| CognitiveSelf | `SarahMemoryCognitiveSelf.py` | What exists / what body am I in / what can I do now |
| CognitiveServices | `SarahMemoryCognitiveServices.py` | What may proceed / ALLOW, DENY, DEFER, REQUIRE_USER |
| CognitiveThinker | `SarahMemoryCognitiveThinker.py` | Meaning, possibility, compassion, sandbox imagination |
| CognitiveCompass | `SarahMemoryCognitiveCompass.py` | Anti-drift, anti-loop, original-goal bearing |

## F. Neuron Router

`SarahMemoryNeuron.py` selects the primary lane and helper family.

Main tier logic:

| Tier | Meaning |
|---|---|
| Tier-0 | Deterministic local answer |
| Tier-1 | Symbolic logic/science/math support |
| Tier-2 | Research / evidence |
| Tier-3 | Optional API/model fallback |

## G. SelfAware / Evidence Court Lane

Used for system/body/capability questions.

Evidence sources include:

- `SarahMemoryHi.py`
- `SarahMemoryDiagnostics.py`
- `SarahNetMCP_Diagnostics.py`
- `appself.py`

Evidence court logic:

| Quorum | Result |
|---|---|
| 0/3 | Denied: no evidence |
| 1/3 | Denied: weak evidence |
| 2/3 | Escalate high review |
| 3/3 | Approved fact |

## H. General Answer Lane

General reasoning may use:

- `SarahMemoryLogicCalc.py`
- `SarahMemoryWebSYM.py`
- `SarahMemoryDatabase.py`
- `SarahMemoryAPI.py`

Third-party models remain helper tools only.

## I. Research / Browser Lane

Research requests pass through online/offline governance.

The lane may use:

- local cache
- research modules
- browser surface
- online fetch only if permitted
- redaction/privacy minimization

## J. Creative Studio Lane

Creative requests route to:

- `SarahMemoryCanvasStudio.py`
- `SarahMemoryMusicGenerator.py`
- `SarahMemoryLyricsToSong.py`
- `SarahMemoryVideoEditorCore.py`

Output is an artifact packet plus preview/download metadata.

## K. Vision / Sensory Lane

Vision route supports:

- frontend live frame input
- backend vision process
- facial recognition
- SOBJE object analysis
- color / scene / object summary
- cloud/local capability distinction

Cloud may analyze provided frames. Local runtime may additionally detect local device model/hardware.

## L. Action / Driver / SMGET Lane

Action requests route through SMGET and OperatorCore.

Core flow:

```text
Action request
→ _sm_try_operatorcore_request()
→ SarahMemoryOperatorCore
→ ActionContract
→ SafetyPolicies / Trust / Security / Assurance
→ simulate / draft / apply / rollback
→ execution result or rollback packet
```

No action should bypass this contract path.

## M. SarahNet / Network Lane

SarahNet/network requests require:

- network policy review
- caller trust
- node identity
- consent
- exposure minimization
- result packet validation

## N. REM / Deep Learning Lane

REM/DL is not direct execution authority.

Current doctrine:

- HIGH REM tickets are discarded/quarantined by default
- MID tickets are review-only
- LOW tickets may proceed only through AssuranceGate and governance
- Deep Learning updates may go to memory only when allowed

## O. Safety / Trust / Assurance

Safety layer includes:

- `SarahMemorySafetyPolicies.py`
- `SarahMemoryTrustRegistry.py`
- `SarahMemorySecurityGovernor.py`
- `SarahMemoryAssuranceGate.py`

It enforces:

- no raw model authority
- no silent mutation
- consent requirements
- rollback preference
- simulate/dry-run preference
- safe-mode/local-only restrictions

## P. Compare / Compass

Before final output:

```text
Candidate output
→ SarahMemoryCompare.py
→ validation / guarddogs
→ CognitiveCompass reanchor if needed
→ release only when Compare passes
```

If Compare fails, the system reanchors to the original goal or returns safe clarification/denial.

## Q. Reply / Frontend Output

Final presentation path:

```text
_sm_present_text()
→ SarahMemoryReply.py fallback/bundle layer
→ _sm_make_outward_bundle()
→ JSON response
→ Frontend Chat render
→ optional voice/avatar/browser/artifact output
```

JSON bundle may include:

- `ok`
- `reply`
- `response`
- `presentation_reply`
- `meta`
- `artifacts`
- `actions`
- `errors`
- `raw_answer`

## R. Storage / Audit / Memory

Storage is optional and governed.

Possible sinks:

- approved metadata / database
- audit logs
- cognitive governor events
- cognitive self / thinker / compass databases
- REM/DL candidate store
- user-approved memory

Rejected or unproven facts must not be persisted as truth.

## Operational Doctrine

```text
First prove what exists.
Then reason what it means.
Then govern what may be claimed or acted on.
Then keep the task on course.
Then present it cleanly.
Then validate the final output.
Then store only what is allowed.
```

## Render Instructions

Use any Mermaid-compatible renderer:

```powershell
npx @mermaid-js/mermaid-cli -i SarahMemory_AiOS_Full_Dataflow.mmd -o SarahMemory_AiOS_Full_Dataflow.svg
```

Or open the included HTML file in a browser:

```text
SarahMemory_AiOS_Full_Dataflow_Render.html
```

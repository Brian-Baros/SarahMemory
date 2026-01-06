# SarahMemoryAiOS ShellSpec v0.0.1
**Project:** SarahMemory (v8.0.0 → Future Evolution Track)  
**Spec Version:** v0.0.1 (GenX Track / “v10” future baseline)  
**Author:** Brian Lee Baros (SOFTDEV0 LLC)  
**Status:** Draft – Living Document (to be refined continuously)

---

## 0) Purpose & Scope

SarahMemoryAiOS ShellSpec defines the **unified UI shell + runtime contract** for a decentralized, local-first AI Operating System that can run:

1) **On top of** Windows / Linux / macOS (Application Mode)  
2) **Portably** from USB/SSD while using the host OS (Portable Mode)  
3) **As its own bootable environment** (Boot Mode / True OS)

This spec is **not a Windows/Mac/Android clone**.  
It is a new “AI-native” shell with familiar usability traits.

**Non-goals (v0.0.1):**
- Replace all OS features (file manager parity, system settings parity, etc.)
- Centralized cloud “brain”
- Background data harvesting of private user memory

---

## 1) Core Principles (Non-Negotiables)

### P1 — Sovereign Node Intelligence
Each installation (node) is its own **AGI centerpoint**:
- local reasoning
- local memory
- local control
- offline-first functionality

### P2 — Internet as Public Library
Public knowledge is **read**, filtered, and compiled into **small, provenance-rich artifacts**.  
The system does not “copy the internet” into centralized storage.

### P3 — SarahNet is a Mesh Broker, Not a Mind
SarahNet does not execute commands, does not own intelligence, and does not store private memory by default.  
It brokers presence, signaling, public artifacts, and reputation signals.

### P4 — Safety by Design
- explicit consent for risk actions
- safe mode defaults
- visible indicators for mic/cam/control
- reversible learning and rollback

### P5 — One UI Contract Across All Devices
The shell must be:
- mobile-friendly
- notebook/laptop-friendly
- desktop-friendly
- touch + mouse + keyboard + voice friendly

---

## 2) Runtime Modes (Deployment Targets)

### Mode A — Application Mode (Host OS)
SarahMemoryAiOS runs as an application on Windows/Linux/macOS.
- UI is served locally (loopback HTTP)
- shown via browser or embedded webview
- OS integrations are optional and consent-gated

### Mode B — Portable Mode (Host OS, Portable Identity)
Runs from USB/SSD with data stored on the removable device.
- portable identity & memory vault (user-controlled)
- minimal host footprint
- consistent UI experience

### Mode C — Boot Mode (True OS)
Boot from USB/SSD into SarahMemoryAiOS environment.
- no dependency on installed Windows/Linux
- same UI contract and API contract
- drivers loaded only with detection + consent rules

---

## 3) Shell UI Model (SarahOS “Desktop” Without Cloning)

### 3.1 The Shell Layout (Canonical)
**Always Present**
- **Main Chat Timeline** (the “OS console” and history)
- **Top Status Strip** (network, mic, cam, safe mode, node identity)
- **Dock / Taskbar** (apps/panels + running tasks + quick toggles)
- **Side Panel** (collapsible; becomes bottom drawer on mobile)

**Panels/Windows (Dock-launchable)**
- Avatar Panel (2D default, toggle 3D)
- Desktop Mirror Panel (optional, consent-gated)
- Research/Browser Panel
- Files/Artifacts Panel
- SarahNet Panel (presence/messages/groups)
- Settings Panel (global + per-addon/per-driver)

### 3.2 Behavior Across Devices
**Desktop/Laptop**
- draggable resizable windows/panels
- split views allowed
- multi-panel mode supported

**Mobile**
- panels become stacked “cards”
- dock becomes bottom bar
- side panel becomes a drawer
- input optimized for thumb reach
- microphone mode is user-configurable:
  - wake-word + push-to-talk
  - wake-word + continuous transcription (opt-in)

### 3.3 Accessibility (Required)
- scalable font sizing
- high contrast mode
- keyboard navigation
- captions for voice output if enabled
- reduced-motion mode

---

## 4) The API Contract (Stable Shell Interface)

The shell UI must communicate with the runtime through **a stable, versioned local API** (even in Boot Mode).

### 4.1 Contract Principles
- UI never directly manipulates system internals
- everything goes through consent + policy checks
- endpoints degrade gracefully when offline/headless

### 4.2 Core API Categories (Must Exist)
1) **/api/health** – runtime status, mode, version
2) **/api/chat** – message → response (with routing metadata)
3) **/api/voice** – TTS controls, mic mode, speaking state
4) **/api/research** – public-library lookup and summarization (opt-in)
5) **/api/memory** – store/recall/forget/rollback (local-first)
6) **/api/addons** – list/enable/disable/configure addons
7) **/api/drivers** – detect/validate/dry-run/apply/start/stop
8) **/api/net** – SarahNet presence/signaling/messages (broker-safe)

---

## 5) Storage Model (Local-First, Portable, Reversible)

### 5.1 Storage Classes
**Private Memory (Default Local)**
- conversation history
- personal preferences
- identity/private keys
- local embeddings

**Public Knowledge Artifacts**
- curated packs with provenance
- reversible installs/removals
- signed and versioned

**Optional Continuity Vault (User-Owned)**
- per-user remote sync (if enabled)
- encrypted end-to-end
- wipeable by the user at any time

### 5.2 Reversibility Requirements
Every learned artifact must be:
- traceable to source
- uninstallable
- rollbackable
- auditable (when enabled)

---

## 6) Consent & Safety Gates (Hard Requirements)

### 6.1 Consent Domains
Actions requiring explicit consent:
- system control / automation
- file deletion or movement
- installing addons/drivers/models
- connecting to SarahNet
- enabling desktop mirror or control modes
- uploading content outside the node

### 6.2 Safe Mode
Safe Mode must:
- disable automation/control by default
- restrict network to read-only or off (user chooses)
- keep chat functional
- display a visible Safe Mode indicator

### 6.3 Interrupt Commands
While TTS is speaking, the user must be able to interrupt with:
- STOP, QUIT, SHUTUP, OKAY, ENOUGH (configurable)

---

## 7) SarahNet Rules (Mesh Without Central Brain)

### 7.1 What SarahNet May Carry
- presence/heartbeat
- signaling for sessions (chat/video negotiation)
- public knowledge packs
- patch/fix packs (metadata + tests)
- reputation signals and receipts

### 7.2 What SarahNet Must Not Carry (Default)
- private user conversation logs
- private memory databases
- raw unreviewed scraped dumps
- executable commands intended to run remotely

---

## 8) Knowledge Ingestion (Public Library → Compiled Packs)

The system may read public sources under ethical constraints and compile:
- how-to cards
- troubleshooting trees
- code patterns and snippets (license-aware)
- concept graphs
- verified Q/A templates

The system must store:
- source URL / dataset ID
- retrieval timestamp
- license metadata (when available)
- confidence score
- user-visible citations where possible

---

## 9) Naming & Branding Track (Future Evolution)

### 9.1 Current Name
**SarahMemory** remains the v8.x identity.

### 9.2 Future Name (GenX Track)
A future rename is planned for the “v10 = v0.0.1” era.
Requirements for the new name:
- not already heavily used
- unique and memorable
- compatible with a decentralized OS/AGI identity
- brand-safe across domains and social handles
- not tied to a single persona name unless intended

This spec is name-agnostic: the shell contract survives branding changes.

---

## 10) Roadmap Milestones (Spec-Driven)

### Milestone S1 — Shell Parity
- Web UI behaves as a true shell (dock, panels, mode indicators)
- consistent across mobile + desktop
- local API contract stable

### Milestone S2 — Portable Identity
- portable mode with encrypted vault
- reversible artifacts
- host-minimal footprint

### Milestone S3 — Bootable SarahMemoryAiOS
- boot mode proof-of-concept
- device detection + driver contract
- same shell + same API contract

---

## 11) Open Questions (To Resolve in v0.0.2+)

- Canonical format for “Knowledge Packs” (schema + signing)
- Reputation math & decay (formal scoring)
- Model-pack distribution rules (open weights vs user-provided)
- Licensing + compliance automation (artifact provenance enforcement)
- UI customization limits (themes, layouts, accessibility profiles)

---

## 12) Summary (One Sentence)

**SarahMemoryAiOS is a sovereign, local-first intelligence platform where every node is its own AGI centerpoint, the internet is a public library, and the shell UI remains consistent whether running as an app, portable environment, or a true bootable OS.**

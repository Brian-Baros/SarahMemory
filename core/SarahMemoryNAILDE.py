"""--==The SarahMemory Project==--
File: SarahMemoryNAILDE.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0-alpha-qsml-0.3-addons
Date: 2026-08-16
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

NAILDE — Natural AI Language Developer Environment

Governed AI-native development sandbox / visual cockpit for SarahMemory AiOS.
This module exposes selected SarahMemory CORE/API capabilities as a bounded
internal SDK for NAILDE. It does not modify live CORE, API, UI, driver, device,
model, checkpoint, or DLScreen state.

Primary doctrine:
- NAILDE is a sandbox-first development cockpit, not an execution authority.
- Live SarahMemory files are read-only to NAILDE.
- New or changed software must be created in a NAILDE workspace or staged zone.
- Live CORE/API/UI apply remains owned by existing governed patch / DevBridge flows.
- A validated application may be installed into SarahMemoryGlobals.ADDONS_DIR only after explicit user confirmation; installation never auto-runs the addon.
- WeightLab may adjust sandbox learning weights only; production tensor weights
  and global DLScreen/Panel values remain outside AI authority.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "nailde_runtime"
# CATEGORY = "governed_development_sandbox"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "nailde"
# HARDWARE_DOMAIN = "display_desktop_vr_xr_device_readonly_future"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "nailde"
# FAMILY = "development_sandbox"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-08-07"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "NAILDE visual governed SDK/sandbox runtime. No live file writes, no device mutation, no production weight/tensor edits, no self-approval."
# --- SARAHMETA END ---

import ast
import copy
import hashlib
import json
import os
import random
import re
import shutil
import threading
import zipfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

MODULE_NAME = "SarahMemoryNAILDE"
MODULE_VERSION = "9.0.0-alpha-qsml-0.3-addons"
SDK_SCHEMA = "SarahMemory.nailde.internal_sdk.v1"
STATE_SCHEMA = "SarahMemory.nailde.self_state.v1"
THOUGHT_SCHEMA = "SarahMemory.nailde.thought_loop.v1"
WEIGHTLAB_SCHEMA = "SarahMemory.nailde.weightlab.v1"

# Universal QSML synthesis budgets. These are safety/resource rails only; they
# never select application content or hardcode a user's requested solution.
_MAX_UNIVERSAL_TEXT_FILE_BYTES = max(16 * 1024, min(2 * 1024 * 1024, int(os.getenv("NAILDE_UNIVERSAL_MAX_TEXT_FILE_BYTES", "524288") or 524288)))
_MAX_UNIVERSAL_PROJECT_BYTES = max(_MAX_UNIVERSAL_TEXT_FILE_BYTES, min(64 * 1024 * 1024, int(os.getenv("NAILDE_UNIVERSAL_MAX_PROJECT_BYTES", "16777216") or 16777216)))
_MAX_UNIVERSAL_EXPORT_FILES = max(1, min(512, int(os.getenv("NAILDE_UNIVERSAL_MAX_EXPORT_FILES", "160") or 160)))


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _base_dir() -> str:
    try:
        value = getattr(config, "BASE_DIR", None) if config else None
        if value:
            return os.path.abspath(os.path.expanduser(str(value)))
    except Exception:
        pass
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(here).lower() == "core":
            return os.path.abspath(os.path.join(here, ".."))
        return here
    except Exception:
        return os.getcwd()


def _data_dir() -> str:
    try:
        value = getattr(config, "DATA_DIR", None) if config else None
        if value:
            return os.path.abspath(os.path.expanduser(str(value)))
    except Exception:
        pass
    return os.path.join(_base_dir(), "data")


def _safe_name(value: Any, default: str = "workspace") -> str:
    raw = str(value or default).strip()
    raw = re.sub(r"[^A-Za-z0-9._:-]+", "_", raw).strip("._-")
    return (raw or default)[:140]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256_text(value: Any) -> str:
    try:
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = str(value)
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return copy.deepcopy(value)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_only_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def _strip_code_fence(text: Any) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def _extract_first_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from a local-model response."""
    raw = _strip_code_fence(text)
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(raw[start:idx + 1])
                    return value if isinstance(value, dict) else None
                except Exception:
                    return None
    return None


def _language_from_path(path: str, fallback: str = "text") -> str:
    ext = os.path.splitext(str(path or "").lower())[1]
    return {
        ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript", ".html": "html",
        ".css": "css", ".json": "json", ".md": "markdown", ".txt": "text",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".xml": "xml", ".svg": "svg",
        ".sql": "sql", ".java": "java", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp",
        ".c": "c", ".h": "c_header", ".hpp": "cpp_header", ".cs": "csharp", ".go": "go",
    }.get(ext, fallback)


# -----------------------------------------------------------------------------
# Static contracts
# -----------------------------------------------------------------------------
NAILDE_WRITE_ZONES = [
    "data/nailde/workspaces/<task_id>",
    "data/nailde/packages",
    "data/nailde/exports",
    "data/addons/pending",
    "data/addons/<addon_id> (explicit_user_authorized_install_only)",
    "data/devbridge/staged",
]

NAILDE_LIVE_DENY_ZONES = [
    "core",
    "api",
    "data/ui",
    "drivers",
    "data/memory/datasets",
]

NAILDE_ALLOWED_ACTIONS = [
    "observe",
    "analyze",
    "plan",
    "generate_sandbox_code",
    "synthesize_arbitrary_application",
    "compile_qsml_blueprint",
    "repair_generated_code",
    "simulate",
    "preview_media",
    "rank_candidates",
    "stage_proposal",
    "package_addon_pending_review",
    "install_validated_addon_after_explicit_user_confirmation",
]

NAILDE_DENIED_ACTIONS = [
    "live_file_write",
    "modify_core_directly",
    "overwrite_api_directly",
    "activate_addon",
    "push_firmware",
    "write_plc_logic",
    "control_devices",
    "shell_execution",
    "credential_access",
    "production_tensor_edit",
    "global_dlpanel_write",
    "self_approval",
]


@dataclass
class CapabilitySpec:
    name: str
    source_file: str
    role: str
    family: str
    risk: str
    ui_surface: str
    allowed_uses: List[str]
    denied_uses: List[str]
    adapter: str
    execution_authority: bool = False
    sandbox_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self.source_file,
            "role": self.role,
            "family": self.family,
            "risk": self.risk,
            "ui_surface": self.ui_surface,
            "allowed_uses": list(self.allowed_uses),
            "denied_uses": list(self.denied_uses),
            "adapter": self.adapter,
            "execution_authority": bool(self.execution_authority),
            "sandbox_only": bool(self.sandbox_only),
        }


SDK_CAPABILITIES: List[CapabilitySpec] = [
    CapabilitySpec(
        "Workbench Awareness",
        "SarahMemoryCognitiveSelf.py",
        "identity_capability_body_desktop_witness",
        "cognitive_triforce",
        "critical",
        "internal_state_panel",
        ["summarize_identity", "summarize_capabilities", "witness_body_desktop_state"],
        ["authorize_execution", "activate_discovered_modules", "mutate_identity_silently"],
        "CognitiveSelfAdapter",
    ),
    CapabilitySpec(
        "Governance Judge",
        "SarahMemoryCognitiveServices.py",
        "allow_deny_defer_require_user_judge",
        "cognitive_triforce",
        "critical",
        "gate_overlay",
        ["classify_intent", "score_risk", "produce_governance_packet"],
        ["execute", "patch", "schedule_background_work", "bypass_user"],
        "CognitiveServicesAdapter",
    ),
    CapabilitySpec(
        "Possibility Thinker",
        "SarahMemoryCognitiveThinker.py",
        "sandbox_possibility_generator",
        "cognitive_triforce",
        "critical",
        "thought_loop_panel",
        ["generate_possibilities", "rank_meaning", "recommend_sandbox_trials"],
        ["claim_speculation_as_fact", "rewrite_runtime", "hot_patch_core"],
        "CognitiveThinkerAdapter",
    ),
    CapabilitySpec(
        "WeightLab",
        "SarahMemoryDL.py",
        "sandbox_learning_weight_explorer",
        "learning",
        "bounded",
        "weightlab_panel",
        ["simulate_weight_profiles", "read_dl_status", "rank_repair_candidates"],
        ["production_tensor_edit", "global_dlpanel_write", "download_models", "train_live_models"],
        "DLWeightLabAdapter",
    ),
    CapabilitySpec(
        "Filesystem Workbench",
        "SarahMemoryFilesystem.py",
        "sandbox_file_map_backup_integrity_witness",
        "filesystem",
        "critical",
        "filesystem_settings_panel",
        ["map_sandbox_paths", "read_file_info", "calculate_sandbox_checksums", "create_sandbox_backup_plan"],
        ["write_live_files", "delete_live_files", "move_live_files", "bypass_workspace_containment"],
        "FilesystemSandboxAdapter",
    ),
    CapabilitySpec(
        "Desktop Witness",
        "SarahMemoryDesktop.py",
        "read_only_desktop_observation",
        "desktop",
        "critical_private_screen",
        "workbench_awareness_panel",
        ["capture_read_only_frame", "observe_visible_workspace"],
        ["mouse_control", "keyboard_control", "file_write", "os_control"],
        "DesktopReadOnlyAdapter",
    ),
    CapabilitySpec(
        "Vision Objects",
        "SarahMemorySOBJE.py",
        "bounded_object_detection",
        "vision",
        "restricted",
        "xr_scene_overlay",
        ["detect_objects_from_approved_frame", "create_scene_proxy_nodes"],
        ["physical_action", "silent_learning", "device_control"],
        "SOBJEAdapter",
    ),
    CapabilitySpec(
        "Face Frame Analysis",
        "SarahMemoryFacialRecognition.py",
        "bounded_face_presence_analysis",
        "vision",
        "restricted_sensitive",
        "approved_frame_overlay",
        ["detect_face_presence", "support_user_approved_frame_analysis"],
        ["silent_identity_learning", "background_monitoring", "authorization_by_face"],
        "FacialRecognitionAdapter",
    ),
    CapabilitySpec(
        "Deterministic Logic",
        "SarahMemoryLogicCalc.py",
        "math_simulation_validation",
        "reasoning",
        "bounded",
        "simulation_panel",
        ["formula_reasoning", "dimension_check", "engineering_estimate"],
        ["execute_code", "mutate_files", "claim_unverified_measurements"],
        "LogicCalcAdapter",
    ),
    CapabilitySpec(
        "QuantumSafe Ranking",
        "SarahMemoryQuantumSafe.py",
        "bounded_option_ranking",
        "reasoning",
        "bounded",
        "candidate_rank_panel",
        ["rank_options", "bounded_search", "score_paths"],
        ["quantum_hardware_claim", "unbounded_search", "device_control"],
        "QuantumSafeAdapter",
    ),
    CapabilitySpec(
        "Research Evidence",
        "SarahMemoryResearch.py",
        "local_web_api_evidence_gathering",
        "research",
        "bounded_network_sensitive",
        "evidence_panel",
        ["local_research", "approved_web_research", "source_trace"],
        ["unbounded_scrape", "silent_network", "credential_access"],
        "ResearchAdapter",
    ),
    CapabilitySpec(
        "Avatar Presentation",
        "SarahMemoryAvatarPanel.py + UnifiedAvatarController.py",
        "shared_avatar_voice_media_surface",
        "presentation",
        "bounded",
        "nailde_avatar_channel",
        ["speak_nailde_status", "show_simulation_media", "set_visual_state"],
        ["create_second_avatar_panel", "self_authorize", "control_devices"],
        "AvatarPanelAdapter",
    ),
    CapabilitySpec(
        "Simulation Media",
        "SarahMemoryCanvasStudio.py + SarahMemoryVideoEditorCore.py + SarahMemoryMusicGenerator.py",
        "sandbox_visual_audio_video_preview",
        "media",
        "bounded",
        "simulation_media_panel",
        ["generate_preview_frames", "assemble_micro_video_manifest", "queue_media_preview"],
        ["present_simulation_as_runtime_proof", "write_live_files"],
        "SimulationMediaAdapter",
    ),
    CapabilitySpec(
        "Agent Task Bay",
        "SarahMemoryTerminal.py + SarahMemoryAgentFirewall.py + SarahMemoryTrustRegistry.py",
        "passported_agent_task_interface",
        "agent_governance",
        "critical",
        "agent_mission_panel",
        ["create_agent_mission_proposal", "show_passport_scope", "capture_return"],
        ["agent_self_spawn", "shared_passport", "direct_shell", "direct_live_write"],
        "AgentTaskAdapter",
    ),
    CapabilitySpec(
        "Ledger Proof",
        "SarahMemoryLedger.py",
        "immutable_receipt_writer",
        "governance",
        "critical",
        "receipt_panel",
        ["record_receipt", "verify_receipt_chain", "surface_receipt_ids"],
        ["authorize_execution", "delete_receipts", "rewrite_history"],
        "LedgerAdapter",
    ),
]


# -----------------------------------------------------------------------------
# NAILDE Runtime
# -----------------------------------------------------------------------------
class SarahMemoryNAILDERuntime:
    """Process-local NAILDE runtime facade used by app.py and future UI routes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.base_dir = _base_dir()
        self.data_dir = _data_dir()
        self.nailde_dir = os.path.join(self.data_dir, "nailde")
        self.workspaces_dir = os.path.join(self.nailde_dir, "workspaces")
        self.packages_dir = os.path.join(self.nailde_dir, "packages")
        self.exports_dir = os.path.join(self.nailde_dir, "exports")
        self.addons_dir = self._addons_root()
        self._last_status: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Audit / proof helpers
    # ------------------------------------------------------------------
    def _record_receipt(self, event_type: str, subject_id: str, summary: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ledger = _read_only_import("SarahMemoryLedger")
        fn = getattr(ledger, "record_governance_receipt", None) if ledger else None
        if not callable(fn):
            return {"ok": False, "skipped": True, "reason": "SarahMemoryLedger unavailable"}
        try:
            return fn(
                domain="nailde",
                event_type=str(event_type),
                subject_id=str(subject_id or "nailde"),
                task_id=str((metadata or {}).get("workspace_id") or subject_id or ""),
                lane="development_sandbox",
                verdict=str((metadata or {}).get("verdict") or "OBSERVED"),
                risk=str((metadata or {}).get("risk") or "low"),
                summary=str(summary)[:800],
                metadata=metadata or {},
                retention_class="standard",
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "execution_authority": False}

    # ------------------------------------------------------------------
    # Public API packets
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            capability_summary = self._cognitive_self_summary()
            dl_status = self._dl_status()
            desktop_status = self._desktop_status()
            filesystem_status = self.filesystem_status()
            payload = {
                "ok": True,
                "schema": "SarahMemory.nailde.status.v1",
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "ts": _now_iso(),
                "base_dir": self.base_dir,
                "data_dir": self.data_dir,
                "nailde_dir": self.nailde_dir,
                "write_zones": list(NAILDE_WRITE_ZONES),
                "live_deny_zones": list(NAILDE_LIVE_DENY_ZONES),
                "allowed_actions": list(NAILDE_ALLOWED_ACTIONS),
                "denied_actions": list(NAILDE_DENIED_ACTIONS),
                "sandbox_first": True,
                "execution_authority": False,
                "core_live_files_read_only": True,
                "weight_isolation": {
                    "sandbox_learning_weights_only": True,
                    "production_weight_access": False,
                    "raw_tensor_edit": False,
                    "global_dlpanel_write": False,
                    "user_ui_required_for_global_change": True,
                    "outside_sandbox_values_static": True,
                },
                "surfaces": {
                    "triple_panel": True,
                    "block_graph": True,
                    "form_designer": True,
                    "device_bay_read_only": True,
                    "holo_forge_vr_xr": True,
                    "simulation_media": True,
                    "workbench_awareness": True,
                    "internal_sdk": True,
                    "weightlab": True,
                    "dynamic_addon_install": True,
                    "runtime_addon_icon_generation": True,
                },
                "capability_summary": capability_summary,
                "dl_status": dl_status,
                "desktop_status": desktop_status,
                "filesystem_status": filesystem_status,
                "sdk_count": len(SDK_CAPABILITIES),
                "last_status_hash": _sha256_text(self._last_status) if self._last_status else "",
            }
            self._last_status = _json_clone(payload)
            return payload

    def sdk_catalog(self) -> Dict[str, Any]:
        capabilities = [spec.to_dict() for spec in SDK_CAPABILITIES]
        families: Dict[str, int] = {}
        for item in capabilities:
            families[item["family"]] = families.get(item["family"], 0) + 1
        return {
            "ok": True,
            "schema": SDK_SCHEMA,
            "module": MODULE_NAME,
            "ts": _now_iso(),
            "capabilities": capabilities,
            "families": families,
            "execution_authority": False,
            "doctrine": {
                "core_files_are_organs": True,
                "nailde_uses_adapters": True,
                "discovery_is_not_activation": True,
                "sdk_calls_are_scoped": True,
                "sandbox_first": True,
            },
        }

    def create_workspace(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        goal = str(payload.get("goal") or payload.get("title") or "NAILDE Workspace").strip() or "NAILDE Workspace"
        workspace_id = _safe_name(payload.get("workspace_id") or f"nailde-{int(time.time())}-{uuid.uuid4().hex[:8]}", "nailde-workspace")
        root = os.path.join(self.workspaces_dir, workspace_id)
        dirs = {
            "root": root,
            "sandbox": os.path.join(root, "sandbox"),
            "simulation": os.path.join(root, "simulation"),
            "media": os.path.join(root, "simulation", "media"),
            "reports": os.path.join(root, "reports"),
            "staged": os.path.join(root, "staged"),
        }
        for path in dirs.values():
            _ensure_dir(path)
        manifest = {
            "schema": "SarahMemory.nailde.workspace_manifest.v1",
            "workspace_id": workspace_id,
            "goal": goal,
            "created_at": _now_iso(),
            "mode": str(payload.get("mode") or "PLAN"),
            "sandbox_first": True,
            "live_files_read_only": True,
            "execution_authority": False,
            "write_zones": dirs,
            "denied_actions": list(NAILDE_DENIED_ACTIONS),
            "input_hash": _sha256_text(payload),
        }
        manifest_path = os.path.join(root, "nailde_workspace_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        receipt = self._record_receipt(
            "NAILDE_WORKSPACE_CREATED",
            workspace_id,
            f"NAILDE workspace created: {goal}",
            {"workspace_id": workspace_id, "goal": goal, "risk": "low", "verdict": "CREATED_SANDBOX"},
        )
        return {"ok": True, "workspace": manifest, "manifest_path": manifest_path, "receipt": receipt}

    def workbench_awareness(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        include_desktop = bool(payload.get("include_desktop") or payload.get("desktop") or False)
        workspace_id = str(payload.get("workspace_id") or "")
        selected = str(payload.get("selected_object") or payload.get("selected_file") or "")
        state = {
            "ok": True,
            "schema": STATE_SCHEMA,
            "module": MODULE_NAME,
            "ts": _now_iso(),
            "workspace_id": workspace_id,
            "selected_object": selected,
            "active_project": str(payload.get("active_project") or payload.get("goal") or "NAILDE"),
            "selected_surface": str(payload.get("selected_surface") or "Workbench"),
            "chat_panel_bridge": "not_echoing",
            "avatar_channel": "nailde_development_assistant",
            "execution_authority": False,
            "capability_summary": self._cognitive_self_summary(),
            "desktop_observation": {"requested": include_desktop, "available": False, "observe_only": True},
            "validation_state": str(payload.get("validation_state") or "not_started"),
            "sandbox_state": str(payload.get("sandbox_state") or "read_only_live_files"),
        }
        if include_desktop:
            state["desktop_observation"] = self._desktop_observe(include_image=False)
        return state

    def thought_loop(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        problem = str(payload.get("problem") or payload.get("text") or payload.get("goal") or "Review NAILDE sandbox issue.").strip()
        workspace_id = str(payload.get("workspace_id") or "")
        selected = str(payload.get("selected_object") or payload.get("selected_file") or "")
        max_ideas = max(1, min(8, int(payload.get("max_ideas") or 5)))
        seed_value = str(payload.get("seed") or f"{workspace_id}|{selected}|{problem}|{int(time.time() // 60)}")
        rng = random.Random(hashlib.sha256(seed_value.encode("utf-8", errors="replace")).hexdigest())

        governance = self._govern_sandbox_problem(problem, workspace_id=workspace_id, selected_object=selected)
        thinker_context = self._thinker_context(problem, workspace_id=workspace_id)
        strategies = [
            ("minimal_patch", "Use the smallest deterministic sandbox edit that preserves existing contracts."),
            ("reuse_sdk_organ", "Reuse an existing SarahMemory SDK organ instead of duplicating logic."),
            ("split_ui_backend_contract", "Separate UI display state from backend proof state."),
            ("add_validation_gate", "Add a visible sandbox validation gate before staging."),
            ("simplify_state", "Reduce moving parts and prefer one source of truth."),
            ("compare_two_candidates", "Generate two bounded candidates and compare diffs."),
            ("media_simulation", "Generate a micro simulation preview before runtime claims."),
            ("weightlab_rank", "Use WeightLab sandbox weights to rank repair paths."),
            ("device_readonly", "Keep device/desktop/vision observations read-only until explicit approval."),
        ]
        rng.shuffle(strategies)
        ideas: List[Dict[str, Any]] = []
        for idx, (key, text) in enumerate(strategies[:max_ideas], start=1):
            risk = "low" if key in {"minimal_patch", "reuse_sdk_organ", "simplify_state", "device_readonly"} else "medium"
            idea = {
                "idea_id": f"nailde-idea-{idx}-{uuid.uuid4().hex[:6]}",
                "strategy": key,
                "title": key.replace("_", " ").title(),
                "summary": text,
                "risk": risk,
                "sandbox_only": True,
                "execution_authority": False,
                "requires": ["visible_diff", "sandbox_validation", "user_review"],
                "denied": ["live_file_write", "self_approval", "device_write"],
                "score": self._score_idea(key, problem, rng),
            }
            ideas.append(idea)
        ideas.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        best = ideas[0] if ideas else None
        packet = {
            "ok": True,
            "schema": THOUGHT_SCHEMA,
            "module": MODULE_NAME,
            "ts": _now_iso(),
            "workspace_id": workspace_id,
            "trigger": str(payload.get("trigger") or "manual_think_on_this"),
            "problem": problem,
            "selected_object": selected,
            "mode": "sandbox_only",
            "cognitive_self": self._cognitive_self_summary(),
            "cognitive_services": governance,
            "cognitive_thinker": thinker_context,
            "ideas": ideas,
            "recommended": best,
            "execution_authority": False,
            "live_apply_allowed": False,
        }
        receipt = self._record_receipt(
            "NAILDE_THOUGHT_LOOP",
            workspace_id or "nailde-thought-loop",
            f"NAILDE thought loop produced {len(ideas)} sandbox ideas.",
            {"workspace_id": workspace_id, "idea_count": len(ideas), "risk": "low", "verdict": "PROPOSED_SANDBOX"},
        )
        packet["receipt"] = receipt
        return packet

    def weightlab_simulate(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        problem = str(payload.get("problem") or payload.get("text") or "Rank sandbox repair candidates.")
        workspace_id = str(payload.get("workspace_id") or "")
        category = str(payload.get("category") or "coder")
        model_id = str(payload.get("model_id") or "")
        max_candidates = max(1, min(10, int(payload.get("max_candidates") or 5)))
        seed_value = str(payload.get("seed") or f"{workspace_id}|{category}|{model_id}|{problem}|weightlab")
        rng = random.Random(hashlib.sha256(seed_value.encode("utf-8", errors="replace")).hexdigest())
        baseline = self._dl_weight_profile(category=category, model_id=model_id)
        base_weights = self._normalize_weights((baseline.get("weights") or {}))

        candidates: List[Dict[str, Any]] = []
        for idx in range(max_candidates):
            profile = dict(base_weights)
            profile["coding"] = self._clamp(profile["coding"] + rng.randint(-8, 20))
            profile["precision"] = self._clamp(profile["precision"] + rng.randint(0, 18))
            profile["safety"] = self._clamp(max(profile["safety"], 88) + rng.randint(-2, 8))
            profile["creativity"] = self._clamp(profile["creativity"] + rng.randint(-20, 12))
            profile["speed"] = self._clamp(profile["speed"] + rng.randint(-12, 10))
            profile["autonomy"] = self._clamp(min(profile["autonomy"], 35) + rng.randint(-10, 4))
            profile["research"] = self._clamp(profile["research"] + rng.randint(-5, 12))
            profile["memory"] = self._clamp(profile["memory"] + rng.randint(-5, 10))
            profile["reasoning"] = self._clamp(profile["reasoning"] + rng.randint(-4, 14))
            score = self._score_weight_profile(profile, problem)
            candidates.append({
                "candidate_id": f"weightlab-{idx + 1}-{uuid.uuid4().hex[:6]}",
                "title": f"Sandbox Weight Candidate {idx + 1}",
                "weights": profile,
                "score": score,
                "reason": self._weight_reason(profile, base_weights),
                "sandbox_only": True,
                "raw_tensor_edit": False,
                "production_weight_access": False,
                "global_dlpanel_write": False,
                "execution_authority": False,
            })
        candidates.sort(key=lambda item: float((item.get("score") or {}).get("total", 0.0)), reverse=True)
        packet = {
            "ok": True,
            "schema": WEIGHTLAB_SCHEMA,
            "module": MODULE_NAME,
            "ts": _now_iso(),
            "workspace_id": workspace_id,
            "problem": problem,
            "category": category,
            "model_id": model_id,
            "baseline": baseline,
            "candidates": candidates,
            "recommended": candidates[0] if candidates else None,
            "sandbox_only": True,
            "raw_tensor_edit": False,
            "production_weight_access": False,
            "global_dlpanel_write": False,
            "user_ui_required_for_global_change": True,
            "execution_authority": False,
            "note": "These are temporary NAILDE sandbox weights only. They do not modify production tensors, checkpoints, DLScreen values, or live model behavior.",
        }
        receipt = self._record_receipt(
            "NAILDE_WEIGHTLAB_SIMULATION",
            workspace_id or "nailde-weightlab",
            f"WeightLab simulated {len(candidates)} sandbox weight candidates.",
            {"workspace_id": workspace_id, "candidate_count": len(candidates), "risk": "low", "verdict": "SIMULATED_SANDBOX"},
        )
        packet["receipt"] = receipt
        return packet

    def avatar_message(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        message = str(payload.get("message") or payload.get("text") or "NAILDE status available.")[:1200]
        speak = bool(payload.get("speak", False))
        level = str(payload.get("level") or "info")
        packet = {
            "ok": True,
            "schema": "SarahMemory.nailde.avatar_message.v1",
            "source": "NAILDE",
            "target": "AvatarPanel",
            "channel": "nailde_development_assistant",
            "message": message,
            "level": level,
            "speak": speak,
            "display_in_chat_panel": False,
            "display_in_nailde_panel": True,
            "avatar_state": "speaking" if speak else "ready",
            "execution_authority": False,
            "delivered": False,
            "delivery_note": "AvatarPanel delivery is attempted best-effort through existing panel APIs; ChatPanel echo is disabled.",
        }
        if speak:
            delivered = self._deliver_avatar_speech(message)
            packet["delivered"] = bool(delivered.get("ok"))
            packet["delivery"] = delivered
        receipt = self._record_receipt(
            "NAILDE_AVATAR_MESSAGE",
            str(payload.get("workspace_id") or "nailde-avatar"),
            f"NAILDE avatar message queued: {message[:120]}",
            {"risk": "low", "verdict": "PRESENTATION_ONLY", "speak": speak},
        )
        packet["receipt"] = receipt
        return packet

    # ------------------------------------------------------------------
    # Filesystem / editor / settings integrations
    # ------------------------------------------------------------------
    def filesystem_status(self) -> Dict[str, Any]:
        """Return a bounded filesystem capability/status packet for NAILDE.

        This is a witness/map only. It does not grant live filesystem mutation.
        """
        fs = _read_only_import("SarahMemoryFilesystem")
        paths = {
            "base_dir": self.base_dir,
            "data_dir": self.data_dir,
            "nailde_dir": self.nailde_dir,
            "workspaces_dir": self.workspaces_dir,
            "packages_dir": self.packages_dir,
            "exports_dir": self.exports_dir,
            "allowed_write_zones": list(NAILDE_WRITE_ZONES),
            "live_deny_zones": list(NAILDE_LIVE_DENY_ZONES),
        }
        module_info = {
            "available": bool(fs),
            "module": "SarahMemoryFilesystem",
            "source": getattr(fs, "__file__", "") if fs else "",
            "governance_level": "critical",
        }
        if fs:
            for name in ("BASE_DIR", "DATA_DIR", "BACKUP_DIR", "SANDBOX_DIR", "QUARANTINE_DIR"):
                try:
                    module_info[name.lower()] = str(getattr(fs, name))
                except Exception:
                    pass
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.filesystem.status.v1",
            "ts": _now_iso(),
            "paths": paths,
            "module": module_info,
            "sandbox_only": True,
            "live_file_write": False,
            "execution_authority": False,
            "note": "NAILDE uses SarahMemoryFilesystem as a bounded witness/helper. Writes remain inside NAILDE workspaces unless DevBridge/user approval applies live changes.",
        }

    def filesystem_map(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a bounded map of NAILDE workspace filesystem state."""
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        include_checksums = bool(payload.get("include_checksums", False))
        max_files = max(1, min(500, int(payload.get("max_files") or 200)))
        status = self.filesystem_status()
        roots = [self.nailde_dir, self.workspaces_dir, self.packages_dir, self.exports_dir]
        if workspace_id:
            roots.insert(0, self._workspace_root(workspace_id))
        entries: List[Dict[str, Any]] = []
        fs = _read_only_import("SarahMemoryFilesystem")
        file_ops = getattr(fs, "FileOperations", None) if fs else None
        for root in roots:
            root_abs = os.path.abspath(root)
            if not os.path.exists(root_abs):
                entries.append({"path": root_abs, "exists": False, "type": "missing"})
                continue
            if os.path.isfile(root_abs):
                paths = [root_abs]
            else:
                paths = []
                for dirpath, dirnames, filenames in os.walk(root_abs):
                    dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
                    for name in sorted(filenames):
                        paths.append(os.path.join(dirpath, name))
                        if len(paths) >= max_files:
                            break
                    if len(paths) >= max_files:
                        break
            for path in paths[:max_files]:
                try:
                    rel = os.path.relpath(path, self.data_dir).replace("\\", "/") if path.startswith(os.path.abspath(self.data_dir)) else path
                    info = None
                    if file_ops is not None and hasattr(file_ops, "get_file_info"):
                        try:
                            info = file_ops.get_file_info(path)
                        except Exception:
                            info = None
                    if not isinstance(info, dict):
                        st = os.stat(path)
                        info = {"path": path, "exists": True, "size_bytes": int(st.st_size), "modified": datetime.fromtimestamp(st.st_mtime).isoformat(), "is_file": True, "extension": os.path.splitext(path)[1]}
                    if not include_checksums:
                        info.pop("checksum", None)
                    entries.append({"relative": rel, "info": info, "sandbox_contained": self._is_nailde_path(path)})
                except Exception as exc:
                    entries.append({"path": path, "error": str(exc)})
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.filesystem.map.v1",
            "workspace_id": workspace_id,
            "status": status,
            "entries": entries[:max_files],
            "entry_count": len(entries[:max_files]),
            "include_checksums": include_checksums,
            "live_file_write": False,
            "execution_authority": False,
        }

    def editor_validate(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate active editor text and return line-level diagnostics/tasks."""
        payload = payload if isinstance(payload, dict) else {}
        content = str(payload.get("content") if payload.get("content") is not None else "")
        path = str(payload.get("path") or payload.get("file_path") or "inline.txt")
        workspace_id = str(payload.get("workspace_id") or "")
        result = self._validate_text(content, path)
        tasks = self._tasks_from_validation(result, workspace_id=workspace_id)
        return {
            "ok": bool(result.get("ok")),
            "schema": "SarahMemory.nailde.editor.validate.v1",
            "workspace_id": workspace_id,
            "path": path,
            "result": result,
            "problems": result.get("problems", []),
            "tasks": tasks,
            "line_count": result.get("line_count", 0),
            "indent_grid": self._indent_grid(content),
            "execution_authority": False,
        }

    def create_application_from_editor(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create/update a sandbox application package from visible editor text.

        This does not execute the code and does not stage a live apply.
        """
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        content = str(payload.get("content") if payload.get("content") is not None else "")
        path = str(payload.get("path") or payload.get("file_path") or "sandbox/app/main.py")
        goal = str(payload.get("goal") or payload.get("prompt") or "Application created from NAILDE code editor.")
        app_name = _safe_name(payload.get("app_name") or self._title_from_prompt(goal), "NAILDEApp")
        if not workspace_id:
            created = self.create_workspace({"goal": goal, "mode": "BUILD_SANDBOX"})
            workspace_id = str((created.get("workspace") or {}).get("workspace_id") or "")
        saved_main = self.save_workspace_file({"workspace_id": workspace_id, "path": path, "content": content})
        validation = self.editor_validate({"workspace_id": workspace_id, "path": path, "content": content})
        app_manifest = {
            "schema": "SarahMemory.nailde.sandbox_application.v1",
            "app_name": app_name,
            "workspace_id": workspace_id,
            "created_from_editor_path": path,
            "goal": goal,
            "status": "draft_sandbox_only",
            "validation_ok": bool(validation.get("ok")),
            "entry_file": path,
            "permissions": ["read_sandbox", "write_workspace_files"],
            "denied_permissions": list(NAILDE_DENIED_ACTIONS),
            "filesystem_map_required": True,
            "github_push_pull_status": "planned_only_no_network_execution",
            "execution_authority": False,
            "live_apply_allowed": False,
        }
        addon_id = self._safe_addon_id(payload.get("addon_id") or app_name)
        addon_title = str(payload.get("display_name") or app_name).strip() or addon_id
        addon_manifest = {
            "id": addon_id,
            "addon_id": addon_id,
            "name": addon_title,
            "type": "nailde_dynamic_app",
            "version": str(payload.get("version") or "0.1.0"),
            "description": goal[:1000],
            "author": "NAILDE",
            "source": "NAILDE",
            "created_by": "SarahMemoryNAILDE",
            "created_from_workspace": workspace_id,
            "created_from_editor_path": path,
            "entry": {"mode": "ui_manifest", "ui": "ui.json", "runtime": "nailde_dynamic_runtime"},
            "entrypoint": {"module": "addon", "callable": "addon_init"},
            "permissions": ["sandbox_runtime", "ui_render", "read_addon_local_files"],
            "denied_permissions": list(NAILDE_DENIED_ACTIONS),
            "risk_tier": "LOW_SANDBOX_DYNAMIC_UI",
            "security": {"trusted": False, "requires_user_install": True, "requires_user_run": True, "auto_run_allowed": False},
            "governance": {
                "schema": "SarahMemory.nailde.generated_addon.v1",
                "sandbox_first": True,
                "source_workspace": workspace_id,
                "installed_by_user_only": True,
                "no_ui_rebuild_required": True,
                "runtime_icon_allowed": True,
                "live_core_write": False,
                "shell_execution": False,
                "device_write": False,
                "production_tensor_edit": False,
                "global_dlpanel_write": False,
                "execution_authority": False,
            },
            "ui": {"icon": "PackageCheck", "accent": "primary", "surface": "addons_runtime_panel"},
        }
        addon_ui = {
            "schema": "SarahMemory.addon.ui_manifest.v1",
            "title": addon_title,
            "icon": "PackageCheck",
            "runtime": "nailde_dynamic_runtime",
            "description": goal[:1000],
            "source_editor_file": path,
            "sections": [
                {"id": "overview", "title": "Overview", "kind": "markdown", "content": goal[:2000]},
                {"id": "code", "title": "Generated Code", "kind": "code", "file": "app/main.txt"},
                {"id": "governance", "title": "Governance", "kind": "facts", "facts": [
                    "Runs from Addons UI without rebuilding the React source tree.",
                    "Installed only after explicit user approval.",
                    "Does not grant live CORE/API/UI mutation authority.",
                    "Does not modify production tensors or global DLScreen values."
                ]},
            ],
            "actions": [
                {"id": "app.open", "label": "Open Runtime Panel", "mode": "ui_only"},
                {"id": "app.copy", "label": "Copy Application", "mode": "store_api"},
                {"id": "app.remove", "label": "Remove Application", "mode": "store_api"},
                {"id": "app.update", "label": "Update from NAILDE", "mode": "store_api"},
            ],
            "execution_authority": False,
        }
        addon_py = "\\n".join([
            "\"\"\"NAILDE-generated dynamic addon wrapper.",
            "",
            "This wrapper is intentionally inert by default. The Addons UI may render its",
            "manifest/ui.json dynamically without rebuilding the React source tree. Python",
            "execution remains behind explicit launcher/governance policy.",
            "\"\"\"",
            "from __future__ import annotations",
            "",
            "_SESSION = {\"status\": \"idle\", \"runs\": 0}",
            "",
            "def addon_info():",
            f"    return {{\"id\": {json.dumps(addon_id)}, \"name\": {json.dumps(addon_title)}, \"session\": dict(_SESSION), \"execution_authority\": False}}",
            "",
            "def addon_init(context=None, config=None):",
            "    _SESSION[\"status\"] = \"ready\"",
            "    return {\"ok\": True, \"info\": addon_info(), \"mode\": \"ui_manifest\", \"execution_authority\": False}",
            "",
            "def addon_status(context=None):",
            "    return {\"ok\": True, \"session\": dict(_SESSION), \"execution_authority\": False}",
            "",
            "def addon_shutdown(context=None):",
            "    _SESSION[\"status\"] = \"stopped\"",
            "    return True",
            "",
            "def addon_action(action_id, context=None, payload=None):",
            "    if action_id in (\"ping\", \"app.open\"):",
            "        _SESSION[\"runs\"] += 1",
            "        _SESSION[\"status\"] = \"running_ui_manifest\"",
            "        return {\"ok\": True, \"action\": action_id, \"session\": dict(_SESSION), \"execution_authority\": False}",
            "    return {\"ok\": False, \"error\": \"Action is not implemented by this dynamic UI addon.\", \"action\": str(action_id), \"execution_authority\": False}",
            "",
        ])
        saved_addon_manifest = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/addon_package/manifest.json", "content": json.dumps(addon_manifest, indent=2, sort_keys=True)})
        saved_addon_ui = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/addon_package/ui.json", "content": json.dumps(addon_ui, indent=2, sort_keys=True)})
        saved_addon_py = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/addon_package/addon.py", "content": addon_py})
        saved_addon_code = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/addon_package/app/main.txt", "content": content})
        editor_blueprint = {
            "name": app_name, "goal": goal, "project_kind": "editor_application",
            "files": [{"path": path, "purpose": "User-visible NAILDE editor application entry", "language": _language_from_path(path), "entrypoint": True}],
            "run": {"entrypoint": path, "arguments": []}, "requested_capabilities": [],
            "constraints": {"sandbox_only": True, "live_core_write": False, "self_approval": False},
        }
        universal_package = self._build_universal_addon_package(
            workspace_id,
            {"application_name": app_name, "addon_id": addon_id, "top_prompt": goal, "route_definition_owner": "SarahMemorySMLProtocol", "route_activation_owner": "SarahMemoryNeuron", "qsml_program_id": "editor-visible-source"},
            {"blueprint": editor_blueprint},
            {path: content},
        )
        if not universal_package.get("ok"):
            return {"ok": False, "error": "editor_addon_package_build_failed", "workspace_id": workspace_id, "addon_package": universal_package, "validation": validation, "execution_authority": False}
        addon_manifest = universal_package.get("manifest") or addon_manifest
        addon_ui = universal_package.get("ui") or addon_ui
        app_manifest["addon_package"] = {
            "addon_id": addon_id,
            "workspace_relative_path": "sandbox/addon_package",
            "absolute_path": os.path.join(self._workspace_root(workspace_id), "sandbox", "addon_package"),
            "manifest": addon_manifest,
            "ui": addon_ui,
            "runtime": universal_package.get("runtime") or {},
            "package_validation": universal_package.get("validation") or {},
            "install_requires_user_confirmation": True,
            "no_ui_rebuild_required": True,
            "runtime_icon_target": "AddonsScreen",
        }
        saved_manifest = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/application_manifest.json", "content": json.dumps(app_manifest, indent=2, sort_keys=True)})
        filesystem = self.filesystem_map({"workspace_id": workspace_id, "max_files": 120})
        receipt = self._record_receipt(
            "NAILDE_EDITOR_APPLICATION_DRAFT",
            workspace_id,
            f"Created sandbox application draft from editor: {app_name}",
            {"workspace_id": workspace_id, "path": path, "risk": "low", "verdict": "DRAFTED_SANDBOX", "validation_ok": bool(validation.get("ok"))},
        )
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.editor.application_draft.v1",
            "workspace_id": workspace_id,
            "app_manifest": app_manifest,
            "saved_main": saved_main,
            "saved_manifest": saved_manifest,
            "addon_package": app_manifest.get("addon_package"),
            "saved_addon_manifest": saved_addon_manifest,
            "saved_addon_ui": saved_addon_ui,
            "saved_addon_py": saved_addon_py,
            "saved_addon_code": saved_addon_code,
            "universal_addon_package": universal_package,
            "validation": validation,
            "filesystem": filesystem,
            "receipt": receipt,
            "sandbox_only": True,
            "execution_authority": False,
        }

    def addon_install_plan(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare a validated Addons installation plan from a NAILDE sandbox package.

        Planning never copies, launches, or grants authority. The package must pass
        the same deterministic ABI/integrity checks used after installation.
        """
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        source_path = str(payload.get("source_path") or "").strip()
        if not source_path and workspace_id:
            source_path = os.path.join(self._workspace_root(workspace_id), "sandbox", "addon_package")
        source_abs = os.path.abspath(source_path) if source_path else ""
        if not source_abs or not os.path.isdir(source_abs):
            return {"ok": False, "error": "addon_source_missing", "workspace_id": workspace_id, "source_path": source_path, "execution_authority": False}
        pending_root = os.path.abspath(os.path.join(self.data_dir, "addons", "pending"))
        if not (self._is_nailde_path(source_abs) or source_abs == pending_root or source_abs.startswith(pending_root + os.sep)):
            return {"ok": False, "error": "source_not_in_nailde_or_pending_zone", "source_path": source_abs, "execution_authority": False}
        integrity = self._validate_installable_addon_package(source_abs)
        manifest = integrity.get("manifest") if isinstance(integrity.get("manifest"), dict) else {}
        ui = integrity.get("ui") if isinstance(integrity.get("ui"), dict) else {}
        addon_id = self._safe_addon_id(payload.get("addon_id") or integrity.get("addon_id") or manifest.get("addon_id") or manifest.get("id") or os.path.basename(source_abs))
        target_root = self._addons_root()
        target = os.path.abspath(os.path.join(target_root, addon_id))
        try:
            if os.path.commonpath([os.path.abspath(target_root), target]) != os.path.abspath(target_root):
                return {"ok": False, "error": "addon_target_outside_addons_root", "execution_authority": False}
        except Exception:
            return {"ok": False, "error": "addon_target_validation_failed", "execution_authority": False}
        existing = os.path.isdir(target)
        plan = {
            "schema": "SarahMemory.nailde.addon_install_plan.v2",
            "workspace_id": workspace_id,
            "source_path": source_abs,
            "target_addons_dir": target_root,
            "target_path": target,
            "addon_id": addon_id,
            "package_integrity": integrity,
            "manifest_present": bool(manifest),
            "ui_present": bool(ui),
            "runnable": bool(integrity.get("runnable")),
            "existing_target": existing,
            "will_backup_existing": existing,
            "will_stage_atomic_copy": True,
            "will_create_runtime_icon": True,
            "no_ui_rebuild_required": True,
            "requires_confirm": True,
            "requires_confirmed": True,
            "copy_only_after_user_authorization": True,
            "auto_run_allowed": False,
            "execution_authority": False,
            "manifest": manifest,
            "ui": ui,
        }
        return {"ok": bool(integrity.get("ok")), "plan": plan, "execution_authority": False}

    def addon_install_authorized(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Atomically install a validated NAILDE package into canonical ADDONS_DIR.

        The call requires explicit user confirmation, creates rollback evidence for
        an existing installation, validates the staged copy, performs a same-root
        rename into place, validates again, and never auto-runs the application.
        """
        payload = payload if isinstance(payload, dict) else {}
        if not self._confirmed(payload):
            return {"ok": False, "error": "explicit_user_confirmation_required", "required": ["confirm", "confirmed"], "execution_authority": False}
        plan_packet = self.addon_install_plan(payload)
        plan = plan_packet.get("plan") if isinstance(plan_packet.get("plan"), dict) else {}
        if not plan_packet.get("ok") or not plan:
            return {"ok": False, "error": "install_plan_failed", "plan": plan_packet, "execution_authority": False}
        source = os.path.abspath(str(plan.get("source_path") or ""))
        target_root = os.path.abspath(str(plan.get("target_addons_dir") or self._addons_root()))
        target = os.path.abspath(str(plan.get("target_path") or ""))
        addon_id = self._safe_addon_id(plan.get("addon_id") or "nailde_app")
        try:
            if os.path.commonpath([target_root, target]) != target_root:
                return {"ok": False, "error": "addon_target_outside_addons_root", "execution_authority": False}
        except Exception:
            return {"ok": False, "error": "addon_target_validation_failed", "execution_authority": False}
        _ensure_dir(target_root)
        token = uuid.uuid4().hex[:10]
        staged = os.path.join(target_root, f"._installing_{addon_id}_{token}")
        rollback_dir = os.path.join(target_root, f"._rollback_{addon_id}_{token}")
        backup = None
        stats: Dict[str, Any] = {}
        old_moved = False
        new_installed = False
        try:
            if os.path.exists(staged):
                shutil.rmtree(staged)
            stats = self._copy_tree_bounded(source, staged, max_files=800, max_total_bytes=25 * 1024 * 1024)
            staged_validation = self._validate_installable_addon_package(staged)
            if not staged_validation.get("ok"):
                raise RuntimeError("staged_addon_integrity_failed:" + ",".join(staged_validation.get("errors") or []))
            if os.path.isdir(target):
                backup = self._zip_backup_dir(target, label=f"addon_{addon_id}")
                if os.path.exists(rollback_dir):
                    shutil.rmtree(rollback_dir)
                os.replace(target, rollback_dir)
                old_moved = True
            os.replace(staged, target)
            new_installed = True
            installed_validation = self._validate_installable_addon_package(target)
            if not installed_validation.get("ok"):
                raise RuntimeError("installed_addon_integrity_failed:" + ",".join(installed_validation.get("errors") or []))
            state = {
                "schema": "SarahMemory.addon.install_state.v2",
                "addon_id": addon_id,
                "installed_ts": _now_iso(),
                "source": source,
                "target": target,
                "backup": backup,
                "status": "installed_review_required",
                "activation_status": "installed_not_running",
                "created_by": "NAILDE",
                "package_validation": {"ok": True, "runnable": bool(installed_validation.get("runnable")), "runtime": installed_validation.get("runtime")},
                "no_ui_rebuild_required": True,
                "auto_run_allowed": False,
                "execution_authority": False,
            }
            self._write_json_file(os.path.join(target, "install_state.json"), state)
            if old_moved and os.path.isdir(rollback_dir):
                shutil.rmtree(rollback_dir)
            receipt = self._record_receipt(
                "NAILDE_ADDON_INSTALLED",
                addon_id,
                f"Installed validated NAILDE application into Addons folder: {addon_id}",
                {"addon_id": addon_id, "target": target, "backup": backup, "risk": "medium", "verdict": "USER_AUTHORIZED_INSTALL", "workspace_id": payload.get("workspace_id")},
            )
            return {
                "ok": True,
                "schema": "SarahMemory.nailde.addon_install_result.v2",
                "addon_id": addon_id,
                "source_path": source,
                "installed_path": target,
                "copy_stats": stats,
                "backup": backup,
                "state": state,
                "package_validation": installed_validation,
                "receipt": receipt,
                "runtime_icon_created": True,
                "no_ui_rebuild_required": True,
                "auto_run_performed": False,
                "execution_authority": False,
            }
        except Exception as exc:
            try:
                if os.path.isdir(staged):
                    shutil.rmtree(staged)
            except Exception:
                pass
            try:
                if new_installed and not old_moved and os.path.isdir(target):
                    shutil.rmtree(target)
            except Exception:
                pass
            try:
                if old_moved and os.path.isdir(rollback_dir):
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    os.replace(rollback_dir, target)
            except Exception as restore_exc:
                return {"ok": False, "error": "addon_install_failed_and_rollback_failed", "detail": str(exc), "rollback_error": str(restore_exc), "backup": backup, "execution_authority": False}
            return {"ok": False, "error": "addon_install_failed_rolled_back", "detail": str(exc), "backup": backup, "execution_authority": False}

    def settings_state(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load/save NAILDE settings. Secrets are never persisted here."""
        payload = payload if isinstance(payload, dict) else {}
        action = str(payload.get("action") or "load").strip().lower()
        path = self._settings_state_path()
        default_state = self._default_settings_state()
        if action in {"save", "update"}:
            incoming = payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
            state = self._sanitize_settings_state({**default_state, **(incoming if isinstance(incoming, dict) else {})})
            self._write_json_file(path, state)
            receipt = self._record_receipt("NAILDE_SETTINGS_SAVED", "nailde-settings", "NAILDE settings saved without secrets.", {"risk": "low", "verdict": "UI_STATE_ONLY"})
            return {"ok": True, "action": "save", "settings": state, "path": path, "receipt": receipt, "execution_authority": False}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    disk = json.load(fh)
                return {"ok": True, "action": "load", "settings": self._sanitize_settings_state({**default_state, **(disk if isinstance(disk, dict) else {})}), "path": path, "execution_authority": False}
            except Exception as exc:
                return {"ok": False, "action": "load", "error": str(exc), "settings": default_state, "execution_authority": False}
        return {"ok": True, "action": "default", "settings": default_state, "path": path, "execution_authority": False}

    def github_plan(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare a GitHub sandbox pull/push plan without running network/git."""
        payload = payload if isinstance(payload, dict) else {}
        operation = str(payload.get("operation") or "status").strip().lower()
        repo_url = str(payload.get("repo_url") or payload.get("repository_url") or "").strip()
        branch = _safe_name(payload.get("branch") or "main", "main")
        workspace_id = str(payload.get("workspace_id") or "").strip()
        allowed_ops = {"connect", "clone", "pull", "push", "fetch", "status", "disconnect"}
        if operation not in allowed_ops:
            return {"ok": False, "error": "unsupported_github_operation", "allowed": sorted(allowed_ops), "execution_authority": False}
        if not workspace_id and operation in {"clone", "pull", "push", "fetch"}:
            created = self.create_workspace({"goal": f"GitHub {operation} sandbox workspace", "mode": "GATHER"})
            workspace_id = str((created.get("workspace") or {}).get("workspace_id") or "")
        commands = []
        if operation == "clone":
            commands = [f"git clone --branch {branch} <repo_url> sandbox/github/repository"]
        elif operation == "pull":
            commands = ["git -C sandbox/github/repository fetch --all --prune", f"git -C sandbox/github/repository pull origin {branch}"]
        elif operation == "push":
            commands = ["git -C sandbox/github/repository status --short", f"git -C sandbox/github/repository push origin {branch}"]
        elif operation == "fetch":
            commands = ["git -C sandbox/github/repository fetch --all --prune"]
        elif operation == "status":
            commands = ["git -C sandbox/github/repository status --short"]
        plan = {
            "schema": "SarahMemory.nailde.github.plan.v1",
            "operation": operation,
            "workspace_id": workspace_id,
            "repo_url": repo_url,
            "branch": branch,
            "commands": commands,
            "network_execution": False,
            "credential_storage": False,
            "token_required_but_not_stored": operation in {"clone", "pull", "push", "fetch"},
            "requires_user_github_auth": operation in {"clone", "pull", "push", "fetch"},
            "requires_passported_terminal_or_devbridge": operation in {"clone", "pull", "push", "fetch"},
            "sandbox_target": "sandbox/github/repository",
            "execution_authority": False,
            "notes": [
                "NAILDE settings may remember repository metadata, not secrets.",
                "Push/pull execution must be performed by a governed Terminal/DevBridge path after explicit user approval.",
                "Use OAuth/GitHub App/PAT credential flows outside NAILDE storage; NAILDE must not persist tokens in settings.",
            ],
        }
        saved = None
        if workspace_id:
            saved = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/github/github_plan.json", "content": json.dumps(plan, indent=2, sort_keys=True)})
        receipt = self._record_receipt("NAILDE_GITHUB_PLAN", workspace_id or "nailde-github", f"Prepared GitHub {operation} plan; no network execution.", {"workspace_id": workspace_id, "risk": "medium" if operation in {"push", "pull", "clone", "fetch"} else "low", "verdict": "PLAN_ONLY"})
        return {"ok": True, "plan": plan, "saved": saved, "receipt": receipt, "execution_authority": False}


    # ------------------------------------------------------------------
    # Full IDE / Workbench environment
    # ------------------------------------------------------------------
    def environment_blueprint(self) -> Dict[str, Any]:
        """Return the NAILDE IDE environment contract for the UI.

        Design target:
        - VS Code-inspired menu bar, activity bar, sidebar/editor/bottom-panel layout.
        - Visual Studio / Visual Basic-style drag/drop Toolbox.
        - Microsoft Access-style data/form/report builder.
        - SarahMemory-governed sandbox boundaries remain hard-coded.
        """
        menus = [
            {
                "id": "file",
                "label": "File",
                "items": [
                    {"id": "file.new_workspace", "label": "New NAILDE Workspace", "shortcut": "Ctrl+Shift+N", "command": "create_workspace"},
                    {"id": "file.new_file", "label": "New Sandbox File", "shortcut": "Ctrl+N", "command": "new_sandbox_file"},
                    {"id": "file.new_form", "label": "New Form Designer Surface", "command": "new_form_surface"},
                    {"id": "file.new_database", "label": "New Addon-Local Database", "command": "new_addon_database"},
                    {"id": "file.open_workspace", "label": "Open Workspace...", "shortcut": "Ctrl+O", "command": "open_workspace"},
                    {"id": "file.open_recent", "label": "Open Recent", "command": "open_recent"},
                    {"id": "file.save", "label": "Save Sandbox Draft", "shortcut": "Ctrl+S", "command": "save_workspace_file"},
                    {"id": "file.save_as", "label": "Save Sandbox Draft As...", "shortcut": "Ctrl+Shift+S", "command": "save_workspace_file_as"},
                    {"id": "file.save_all", "label": "Save All Sandbox Drafts", "shortcut": "Ctrl+K S", "command": "save_all_sandbox_files"},
                    {"id": "file.export", "label": "Export Sandbox Application", "command": "package_workspace_export"},
                    {"id": "file.stage", "label": "Stage to DevBridge", "command": "stage_devbridge_proposal", "requires": ["validation", "compare", "ledger", "user_review"]},
                    {"id": "file.close", "label": "Close Workspace", "command": "close_workspace"},
                ],
            },
            {
                "id": "edit",
                "label": "Edit",
                "items": [
                    {"id": "edit.undo", "label": "Undo", "shortcut": "Ctrl+Z", "command": "editor_undo"},
                    {"id": "edit.redo", "label": "Redo", "shortcut": "Ctrl+Y", "command": "editor_redo"},
                    {"id": "edit.cut", "label": "Cut", "shortcut": "Ctrl+X", "command": "editor_cut"},
                    {"id": "edit.copy", "label": "Copy", "shortcut": "Ctrl+C", "command": "editor_copy"},
                    {"id": "edit.paste", "label": "Paste", "shortcut": "Ctrl+V", "command": "editor_paste"},
                    {"id": "edit.find", "label": "Find", "shortcut": "Ctrl+F", "command": "show_search"},
                    {"id": "edit.replace", "label": "Replace", "shortcut": "Ctrl+H", "command": "show_search_replace"},
                    {"id": "edit.reconcile", "label": "Reconcile Human Edits", "shortcut": "Ctrl+R", "command": "reconcile_edits"},
                    {"id": "edit.retry_failed", "label": "Retry Failed Sections Only", "command": "retry_failed_sections"},
                    {"id": "edit.make_safer", "label": "Make Safer", "command": "draft_safer_code"},
                ],
            },
            {
                "id": "selection",
                "label": "Selection",
                "items": [
                    {"id": "selection.select_all", "label": "Select All", "shortcut": "Ctrl+A", "command": "editor_select_all"},
                    {"id": "selection.expand", "label": "Expand Selection", "shortcut": "Shift+Alt+Right", "command": "expand_selection"},
                    {"id": "selection.shrink", "label": "Shrink Selection", "shortcut": "Shift+Alt+Left", "command": "shrink_selection"},
                    {"id": "selection.copy_line_up", "label": "Copy Line Up", "shortcut": "Shift+Alt+Up", "command": "copy_line_up"},
                    {"id": "selection.copy_line_down", "label": "Copy Line Down", "shortcut": "Shift+Alt+Down", "command": "copy_line_down"},
                    {"id": "selection.comment", "label": "Toggle Line Comment", "shortcut": "Ctrl+/", "command": "toggle_line_comment"},
                ],
            },
            {
                "id": "view",
                "label": "View",
                "items": [
                    {"id": "view.command_palette", "label": "Command Palette...", "shortcut": "Ctrl+Shift+P", "command": "show_command_palette"},
                    {"id": "view.explorer", "label": "Explorer", "shortcut": "Ctrl+Shift+E", "command": "show_explorer"},
                    {"id": "view.search", "label": "Search", "shortcut": "Ctrl+Shift+F", "command": "show_search"},
                    {"id": "view.source", "label": "Source / Diff", "shortcut": "Ctrl+Shift+G", "command": "show_diff"},
                    {"id": "view.run_debug", "label": "Run and Debug", "shortcut": "Ctrl+Shift+D", "command": "show_run_debug"},
                    {"id": "view.extensions", "label": "Extensions / SDK Library", "shortcut": "Ctrl+Shift+X", "command": "show_sdk"},
                    {"id": "view.problems", "label": "Problems / Validation", "shortcut": "Ctrl+Shift+M", "command": "show_validation"},
                    {"id": "view.output", "label": "Output", "shortcut": "Ctrl+Shift+U", "command": "show_output"},
                    {"id": "view.terminal", "label": "Terminal Output", "shortcut": "Ctrl+`", "command": "show_terminal"},
                    {"id": "view.graph", "label": "Flow Graph", "command": "show_flow_graph"},
                    {"id": "view.blocks", "label": "BlockForge", "command": "show_blockforge"},
                    {"id": "view.forms", "label": "Form Designer", "command": "show_form_designer"},
                    {"id": "view.toolbox", "label": "Toolbox", "command": "show_toolbox"},
                    {"id": "view.database", "label": "Access-Style Database Builder", "command": "show_database_builder"},
                    {"id": "view.device", "label": "Device Bay", "command": "show_device_bay"},
                    {"id": "view.holo", "label": "HoloForge / XR", "command": "show_holoforge"},
                    {"id": "view.reset_layout", "label": "Reset Workbench Layout", "command": "reset_layout"},
                ],
            },
            {
                "id": "go",
                "label": "Go",
                "items": [
                    {"id": "go.file", "label": "Go to File...", "shortcut": "Ctrl+P", "command": "go_to_file"},
                    {"id": "go.symbol", "label": "Go to Symbol in Workspace...", "shortcut": "Ctrl+T", "command": "go_to_symbol"},
                    {"id": "go.line", "label": "Go to Line/Column...", "shortcut": "Ctrl+G", "command": "go_to_line"},
                    {"id": "go.definition", "label": "Go to Definition", "shortcut": "F12", "command": "go_to_definition"},
                    {"id": "go.references", "label": "Go to References", "shortcut": "Shift+F12", "command": "go_to_references"},
                    {"id": "go.next_problem", "label": "Next Problem", "shortcut": "F8", "command": "next_problem"},
                    {"id": "go.previous_problem", "label": "Previous Problem", "shortcut": "Shift+F8", "command": "previous_problem"},
                ],
            },
            {
                "id": "run",
                "label": "Run",
                "items": [
                    {"id": "run.sandbox", "label": "Run Sandbox Validation", "shortcut": "F5", "command": "validate_sandbox"},
                    {"id": "run.text", "label": "Validate Current Text Artifact", "command": "validate_text_artifact"},
                    {"id": "run.thought", "label": "Think on This", "shortcut": "Ctrl+Alt+T", "command": "thought_loop"},
                    {"id": "run.weightlab", "label": "WeightLab Simulation", "command": "weightlab_simulate"},
                    {"id": "run.compare", "label": "Run Compare", "command": "compare_sandbox"},
                    {"id": "run.assurance", "label": "Run Assurance", "command": "assurance_review"},
                    {"id": "run.agent", "label": "Prepare Agent Mission", "command": "prepare_agent_mission"},
                ],
            },
            {
                "id": "terminal",
                "label": "Terminal",
                "items": [
                    {"id": "terminal.new", "label": "New Governed Terminal Output", "shortcut": "Ctrl+Shift+`", "command": "show_terminal"},
                    {"id": "terminal.clear", "label": "Clear Terminal Output", "command": "clear_terminal_output"},
                    {"id": "terminal.agent", "label": "Prepare /agent Mission Packet", "command": "prepare_agent_mission"},
                    {"id": "terminal.security_status", "label": "Agent Security Status", "command": "agent_security_status"},
                    {"id": "terminal.no_shell", "label": "Shell Execution Locked", "command": "show_terminal_boundary"},
                ],
            },
            {
                "id": "tools",
                "label": "Tools",
                "items": [
                    {"id": "tools.toolbox", "label": "Visual Object Toolbox", "command": "show_toolbox"},
                    {"id": "tools.form_designer", "label": "VB-Style Form Designer", "command": "show_form_designer"},
                    {"id": "tools.database_builder", "label": "Access-Style Table/Form/Report Builder", "command": "show_database_builder"},
                    {"id": "tools.blockforge", "label": "Scratch-Style BlockForge", "command": "show_blockforge"},
                    {"id": "tools.sdk", "label": "Internal SDK Library", "command": "show_sdk"},
                    {"id": "tools.model_bay", "label": "Model Bay", "command": "show_model_bay"},
                    {"id": "tools.device_bay", "label": "Device Bay", "command": "show_device_bay"},
                    {"id": "tools.simulation", "label": "Simulation Media", "command": "show_simulation"},
                ],
            },
            {
                "id": "governance",
                "label": "Governance",
                "items": [
                    {"id": "gov.status", "label": "Show Governance Gates", "command": "show_governance"},
                    {"id": "gov.backup", "label": "Create Backup ZIP", "command": "create_backup_zip", "requires": ["user_confirm_1", "user_confirm_2"]},
                    {"id": "gov.ledger", "label": "Record Ledger Receipt", "command": "record_receipt"},
                    {"id": "gov.rollback", "label": "Prepare Rollback", "command": "prepare_rollback"},
                    {"id": "gov.lockdown", "label": "Lock Workspace", "command": "lock_workspace"},
                    {"id": "gov.boundaries", "label": "Show Safety Boundaries", "command": "show_boundaries"},
                ],
            },
            {
                "id": "settings",
                "label": "Settings",
                "items": [
                    {"id": "settings.open", "label": "Open NAILDE Settings", "command": "show_settings"},
                    {"id": "settings.github", "label": "GitHub Sandbox Settings", "command": "show_github_settings"},
                    {"id": "settings.filesystem", "label": "Filesystem Map", "command": "show_filesystem"},
                    {"id": "settings.validate", "label": "Editor Diagnostics Settings", "command": "show_diagnostics"},
                ],
            },
            {
                "id": "help",
                "label": "Help",
                "items": [
                    {"id": "help.doctrine", "label": "NAILDE Doctrine", "command": "show_doctrine"},
                    {"id": "help.shortcuts", "label": "Keyboard Shortcuts", "command": "show_shortcuts"},
                    {"id": "help.boundaries", "label": "Safety Boundaries", "command": "show_boundaries"},
                    {"id": "help.research_notes", "label": "Workbench Research Notes", "command": "show_research_notes"},
                ],
            },
        ]
        activity_bar = [
            {"id": "explorer", "label": "Explorer", "icon": "files", "panel": "explorer"},
            {"id": "search", "label": "Search", "icon": "search", "panel": "search"},
            {"id": "source", "label": "Source / Diff", "icon": "git-compare", "panel": "diff"},
            {"id": "run_debug", "label": "Run and Debug", "icon": "play", "panel": "run_debug"},
            {"id": "extensions", "label": "SDK / Extensions", "icon": "package", "panel": "sdk"},
            {"id": "toolbox", "label": "Toolbox", "icon": "hammer", "panel": "toolbox"},
            {"id": "database", "label": "Database Builder", "icon": "database", "panel": "database_builder"},
            {"id": "forms", "label": "Form Designer", "icon": "layout", "panel": "form_designer"},
            {"id": "blocks", "label": "BlockForge", "icon": "blocks", "panel": "blockforge"},
            {"id": "agents", "label": "Agents", "icon": "bot", "panel": "agents"},
            {"id": "models", "label": "Model Bay", "icon": "cpu", "panel": "model_bay"},
            {"id": "devices", "label": "Device Bay", "icon": "usb", "panel": "device_bay"},
            {"id": "holoforge", "label": "HoloForge", "icon": "orbit", "panel": "holoforge"},
            {"id": "governance", "label": "Governance", "icon": "shield", "panel": "governance"},
        ]
        panels = [
            {"id": "battle_plan", "title": "Battle Plan", "zone": "primary_sidebar", "text_editable": True},
            {"id": "explorer", "title": "Sandbox Explorer", "zone": "primary_sidebar", "text_editable": False},
            {"id": "search", "title": "Search", "zone": "primary_sidebar", "text_editable": True},
            {"id": "prompt", "title": "Natural Language Prompt", "zone": "editor", "text_editable": True},
            {"id": "editor", "title": "Code Editor", "zone": "editor", "text_editable": True, "drop_target": True},
            {"id": "output", "title": "Output / Evidence", "zone": "bottom_panel", "text_editable": True},
            {"id": "terminal", "title": "Governed Terminal Output", "zone": "bottom_panel", "text_editable": True, "shell_authority": False},
            {"id": "validation", "title": "Problems / Validation", "zone": "bottom_panel", "text_editable": False},
            {"id": "toolbox", "title": "Toolbox", "zone": "secondary_sidebar", "drag_source": True},
            {"id": "database_builder", "title": "Database Builder", "zone": "secondary_sidebar", "drag_source": True},
            {"id": "form_designer", "title": "Form Designer", "zone": "editor", "drag_target": True},
            {"id": "blockforge", "title": "BlockForge", "zone": "editor", "drag_target": True},
            {"id": "run_debug", "title": "Run and Debug", "zone": "bottom_panel"},
            {"id": "sdk", "title": "SDK Library", "zone": "secondary_sidebar"},
            {"id": "diff", "title": "Diff Viewer", "zone": "editor"},
        ]
        commands = []
        for menu in menus:
            for item in menu.get("items", []):
                commands.append({
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "command": item.get("command"),
                    "shortcut": item.get("shortcut", ""),
                    "menu": menu.get("label"),
                    "execution_authority": False,
                    "requires": item.get("requires", []),
                })
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.environment.v2.vscode_vb_access_inspired",
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "ts": _now_iso(),
            "menus": menus,
            "activity_bar": activity_bar,
            "panels": panels,
            "commands": commands,
            "research_profile": {
                "vscode_inspired": ["menu_bar", "activity_bar", "primary_sidebar", "editor_group", "secondary_sidebar", "bottom_panel", "status_bar", "command_palette"],
                "visual_studio_inspired": ["toolbox", "designer_surface", "properties_window", "drag_drop_controls"],
                "access_inspired": ["tables", "forms", "queries", "reports", "bound_controls", "field_list"],
                "not_official_microsoft_product": True,
            },
            "status_bar": [
                "workspace", "sandbox", "selected_file", "model", "agent", "validation", "compare", "ledger", "read_only_live", "weight_isolated", "filesystem_mapped", "github_sandbox_only",
            ],
            "layout": {
                "activity_bar": [item["id"] for item in activity_bar],
                "primary_sidebar": ["explorer", "search", "battle_plan", "governance"],
                "editor": ["prompt", "editor", "diff", "graph", "blockforge", "form_designer", "holoforge"],
                "secondary_sidebar": ["toolbox", "database_builder", "sdk", "model_bay", "properties"],
                "bottom_panel": ["terminal", "output", "validation", "run_debug", "receipts"],
            },
            "hard_boundaries": {
                "execution_authority": False,
                "live_file_write": False,
                "shell_execution": False,
                "device_write": False,
                "production_tensor_edit": False,
                "global_dlpanel_write": False,
                "sandbox_only": True,
            },
        }

    def workspace_files(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        root = self._workspace_root(workspace_id)
        if not os.path.isdir(root):
            return {"ok": False, "error": "workspace_not_found", "workspace_id": workspace_id, "execution_authority": False}
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules"}]
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                try:
                    rel = os.path.relpath(path, root).replace("\\", "/")
                    size = os.path.getsize(path)
                    files.append({"path": rel, "size": size, "mtime": os.path.getmtime(path), "text_candidate": size <= 512000})
                except Exception:
                    continue
        files.sort(key=lambda row: row.get("path") or "")
        return {"ok": True, "workspace_id": workspace_id, "root": root, "files": files, "execution_authority": False}

    def read_workspace_file(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        rel_path = str(payload.get("path") or payload.get("file_path") or "").strip()
        if not workspace_id or not rel_path:
            return {"ok": False, "error": "workspace_id_and_path_required", "execution_authority": False}
        try:
            path = self._workspace_file_path(workspace_id, rel_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "execution_authority": False}
        if not os.path.isfile(path):
            return {"ok": False, "error": "file_not_found", "path": rel_path, "execution_authority": False}
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            return {"ok": False, "error": "file_too_large_for_text_read", "size": size, "execution_authority": False}
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return {"ok": True, "workspace_id": workspace_id, "path": rel_path, "content": content, "sha256": _sha256_text(content), "execution_authority": False}

    def save_workspace_file(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        rel_path = str(payload.get("path") or payload.get("file_path") or "sandbox/untitled.txt").strip()
        content = str(payload.get("content") if payload.get("content") is not None else "")
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        try:
            path = self._workspace_file_path(workspace_id, rel_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "execution_authority": False}
        _ensure_dir(os.path.dirname(path))
        before = ""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    before = fh.read()
            except Exception:
                before = ""
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        record = {
            "schema": "SarahMemory.nailde.human_edit_artifact.v1",
            "workspace_id": workspace_id,
            "path": rel_path,
            "saved_at": _now_iso(),
            "before_hash": _sha256_text(before),
            "after_hash": _sha256_text(content),
            "live_file_write": False,
            "execution_authority": False,
        }
        receipt = self._record_receipt("NAILDE_SANDBOX_FILE_SAVED", workspace_id, f"Saved sandbox file {rel_path}", {"workspace_id": workspace_id, "path": rel_path, "risk": "low", "verdict": "SANDBOX_WRITE"})
        return {"ok": True, "workspace_id": workspace_id, "path": rel_path, "absolute_path": path, "artifact": record, "receipt": receipt, "execution_authority": False}

    def package_workspace_export(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a bounded ZIP of a synthesized sandbox application.

        Export is an artifact operation inside NAILDE's allowed export zone. It
        does not install, launch, activate, or promote the generated application.
        """
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        root = self._workspace_root(workspace_id)
        sandbox_root = os.path.join(root, "sandbox")
        if not os.path.isdir(sandbox_root):
            return {"ok": False, "error": "workspace_sandbox_not_found", "workspace_id": workspace_id, "execution_authority": False}
        _ensure_dir(self.exports_dir)
        safe_id = _safe_name(workspace_id, "nailde_app")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_path = os.path.join(self.exports_dir, f"{safe_id}_{stamp}.zip")
        candidates: List[Tuple[str, str, int]] = []
        total_bytes = 0
        for dirpath, dirnames, filenames in os.walk(sandbox_root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
            for name in sorted(filenames):
                if name in {".env", "id_rsa", "id_dsa"} or name.lower().endswith((".pem", ".key")):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    continue
                rel = os.path.relpath(path, root).replace("\\", "/")
                if rel.startswith("../") or "/../" in rel:
                    continue
                size = os.path.getsize(path)
                total_bytes += size
                candidates.append((path, rel, size))
                if len(candidates) > _MAX_UNIVERSAL_EXPORT_FILES:
                    return {"ok": False, "error": "workspace_export_file_budget_exceeded", "max_files": _MAX_UNIVERSAL_EXPORT_FILES, "execution_authority": False}
                if total_bytes > _MAX_UNIVERSAL_PROJECT_BYTES:
                    return {"ok": False, "error": "workspace_export_size_budget_exceeded", "max_bytes": _MAX_UNIVERSAL_PROJECT_BYTES, "execution_authority": False}
        if not candidates:
            return {"ok": False, "error": "workspace_sandbox_has_no_exportable_files", "execution_authority": False}
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for path, rel, _size in candidates:
                zf.write(path, rel)
        digest = ""
        try:
            h = hashlib.sha256()
            with open(zip_path, "rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(block)
            digest = h.hexdigest()
        except Exception:
            pass
        receipt = self._record_receipt(
            "NAILDE_WORKSPACE_EXPORTED", workspace_id,
            f"Exported synthesized sandbox application package with {len(candidates)} files.",
            {"workspace_id": workspace_id, "risk": "low", "verdict": "SANDBOX_EXPORT", "sha256": digest},
        )
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.workspace_export.v2",
            "workspace_id": workspace_id,
            "zip_path": zip_path,
            "sha256": digest,
            "file_count": len(candidates),
            "total_uncompressed_bytes": total_bytes,
            "sandbox_only": True,
            "installed": False,
            "executed": False,
            "promoted": False,
            "receipt": receipt,
            "execution_authority": False,
        }

    def qsml_compile_request(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        '''Compile a NAILDE natural-language request through SarahMemorySMLProtocol.

        SML/QSML owns the language/route definition. NAILDE consumes the compiled
        program as a sandbox build specification. Neuron remains the route
        activation/weighting owner; NAILDE never self-authorizes execution.
        '''
        payload = payload if isinstance(payload, dict) else {}
        top = str(payload.get("top_prompt") or payload.get("goal") or payload.get("prompt") or payload.get("problem") or "").strip()
        details = str(payload.get("details_prompt") or payload.get("additional_instructions") or "").strip()
        source = "\n".join(x for x in (top, details) if x).strip()
        if not source:
            return {"ok": False, "error": "natural_language_source_required", "execution_authority": False}
        try:
            from SarahMemorySMLProtocol import sml_compile_natural_program  # type: ignore
            compiled = sml_compile_natural_program(
                source,
                context={
                    "target": "nailde",
                    "surface": "NAILDE",
                    "workspace_id": str(payload.get("workspace_id") or ""),
                    "sandbox_only": True,
                    "requested_target": str(payload.get("target") or ""),
                },
                target="nailde",
                collect_external_evidence=bool(payload.get("collect_language_evidence", True)),
            )
            if not isinstance(compiled, dict):
                return {"ok": False, "error": "qsml_compiler_returned_non_dict", "execution_authority": False}
            return {
                "ok": str(compiled.get("status") or "").upper() == "COMPILED",
                "schema": "SarahMemory.nailde.qsml_compile.v0_2",
                "source": source,
                "compiled": compiled,
                "execution_authority": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": "qsml_compiler_unavailable",
                "detail": str(exc)[:500],
                "source": source,
                "execution_authority": False,
            }

    def _expand_qsml_program(self, compiled: Dict[str, Any], top_prompt: str, details_prompt: str = "") -> Dict[str, Any]:
        """Convert a compiled QSML program into a NAILDE synthesis request.

        QSML/0.2 deliberately does not select a prompt-specific application
        template here. The local synthesis model expands the language-neutral
        contract into an application blueprint, then the universal file
        synthesizer generates the implementation.
        """
        result = compiled.get("compiled") if isinstance(compiled.get("compiled"), dict) else compiled
        program = result.get("program") if isinstance(result, dict) and isinstance(result.get("program"), dict) else {}
        build = program.get("build_spec") if isinstance(program.get("build_spec"), dict) else {}
        project_kind = str(build.get("project_kind") or "software_project")
        app_name = str(build.get("application_name") or self._title_from_prompt(top_prompt)).strip() or "NAILDE Application"
        constraints = build.get("constraints") if isinstance(build.get("constraints"), dict) else {}
        base_blueprint = program.get("synthesis_blueprint") if isinstance(program.get("synthesis_blueprint"), dict) else {}
        addon_package = True  # Every successful NAILDE application is packageable for Addons; install/run still require explicit user approval.
        return {
            "schema": "SarahMemory.nailde.qsml_build_spec.v0_2",
            "qsml_program_id": str(program.get("program_id") or ""),
            "qsml_language_version": str(program.get("language_version") or "QSML/0.2"),
            "project_type": project_kind,
            "project_kind": project_kind,
            "top_prompt": top_prompt,
            "details_prompt": details_prompt,
            "application_name": app_name,
            "addon_id": self._safe_addon_id(app_name),
            "addon_package": addon_package,
            "languages": list(build.get("languages") or []),
            "frameworks": list(build.get("frameworks") or []),
            "features": list(build.get("features") or []),
            "requested_capabilities": [dict(x) for x in list(build.get("requested_capabilities") or []) if isinstance(x, dict)][:64],
            "constraints": constraints,
            "candidate_routes": list(program.get("candidate_routes") or []),
            "route_definition_owner": str((program.get("metadata") or {}).get("route_definition_owner") or "SarahMemorySMLProtocol"),
            "route_activation_owner": str((program.get("metadata") or {}).get("route_activation_owner") or "SarahMemoryNeuron"),
            "required_authority": list(program.get("required_authority") or ["Read"]),
            "qsml_ast": program.get("ast") or {},
            "qsml_variables": program.get("variables") or {},
            "qsml_program": copy.deepcopy(program),
            "base_synthesis_blueprint": base_blueprint,
            "novice_summary": f"Synthesize a governed local {project_kind.replace('_', ' ')} named {app_name} from the natural-language requirements.",
            "required_files": [],
            "validation_plan": ["qsml_compile", "qsml_blueprint", "file_synthesis", "static_validation", "bounded_repair", "sandbox_containment", "compare_required", "ledger_required"],
            "synthesis_mode": "universal_local_model",
            "prompt_specific_template_selection": False,
            "local_model_required_for_arbitrary_synthesis": True,
            "denied": list(NAILDE_DENIED_ACTIONS),
            "sandbox_only": bool(constraints.get("sandbox_only", True)),
            "execution_authority": False,
        }

    def _qsml_manifest(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema": "SarahMemory.nailde.qsml_application_manifest.v0_1",
            "name": str(spec.get("application_name") or "NAILDE Application"),
            "project_type": str(spec.get("project_type") or "generic_sandbox_application"),
            "project_kind": str(spec.get("project_kind") or "software_project"),
            "languages": list(spec.get("languages") or []),
            "features": list(spec.get("features") or []),
            "constraints": dict(spec.get("constraints") or {}),
            "qsml_program_id": str(spec.get("qsml_program_id") or ""),
            "route_definition_owner": str(spec.get("route_definition_owner") or "SarahMemorySMLProtocol"),
            "route_activation_owner": str(spec.get("route_activation_owner") or "SarahMemoryNeuron"),
            "sandbox_only": True,
            "execution_authority": False,
            "live_core_write": False,
        }

    def _qsml_local_model_generate(
        self,
        prompt: str,
        *,
        purpose: str = "application_synthesis",
        max_tokens: int = 4096,
        temperature: float = 0.15,
        model: str = "",
    ) -> Dict[str, Any]:
        """Call only approved LOCAL generation backends for NAILDE synthesis.

        This method intentionally bypasses generic provider fallback so a failed
        local model cannot silently become a cloud dependency. It never executes
        generated code and returns model text as untrusted synthesis evidence.
        """
        api = _read_only_import("SarahMemoryAPI")
        if api is None:
            return {"ok": False, "error": "SarahMemoryAPI_unavailable", "purpose": purpose, "execution_authority": False}
        timeout = 90
        try:
            timeout = max(5, min(300, int(getattr(config, "NAILDE_LOCAL_MODEL_TIMEOUT_SECONDS", 90) if config else 90)))
        except Exception:
            timeout = 90
        max_tokens = max(128, min(int(max_tokens or 4096), 16384))
        attempts: List[Dict[str, Any]] = []

        local_fn = getattr(api, "_call_local_llm", None)
        if callable(local_fn):
            try:
                content, error = local_fn(
                    prompt=prompt,
                    model=(model or None),
                    timeout=timeout,
                    intent="programming",
                    user_text=prompt,
                    meta={"source": "SarahMemoryNAILDE", "purpose": purpose, "local_only": True, "sandbox_only": True},
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                )
                attempts.append({"backend": "local_llm", "ok": bool(content), "error": str(error or "")[:500]})
                if content and not error:
                    return {
                        "ok": True,
                        "backend": "local_llm",
                        "content": str(content),
                        "attempts": attempts,
                        "local_only": True,
                        "execution_authority": False,
                    }
            except Exception as exc:
                attempts.append({"backend": "local_llm", "ok": False, "error": str(exc)[:500]})

        ollama_fn = getattr(api, "_call_ollama", None)
        if callable(ollama_fn):
            try:
                default_models = getattr(api, "DEFAULT_MODELS", {}) or {}
                requested_model = model or str(default_models.get("ollama") or "")
                content, error = ollama_fn(prompt, requested_model, "")
                attempts.append({"backend": "ollama", "ok": bool(content), "error": str(error or "")[:500]})
                if content and not error:
                    return {
                        "ok": True,
                        "backend": "ollama",
                        "content": str(content),
                        "attempts": attempts,
                        "local_only": True,
                        "execution_authority": False,
                    }
            except Exception as exc:
                attempts.append({"backend": "ollama", "ok": False, "error": str(exc)[:500]})

        return {
            "ok": False,
            "error": "no_local_generation_backend_succeeded",
            "purpose": purpose,
            "attempts": attempts,
            "local_only": True,
            "execution_authority": False,
        }

    def _universal_architecture_prompt(self, spec: Dict[str, Any]) -> str:
        """Create the strict planner prompt for a language-neutral application blueprint."""
        prompt = str(spec.get("top_prompt") or "").strip()
        details = str(spec.get("details_prompt") or "").strip()
        constraints = dict(spec.get("constraints") or {})
        features = list(spec.get("features") or [])[:48]
        requested_capabilities = [dict(x) for x in list(spec.get("requested_capabilities") or []) if isinstance(x, dict)][:64]
        languages = list(spec.get("languages") or [])[:16]
        frameworks = list(spec.get("frameworks") or [])[:16]
        base_blueprint = spec.get("base_synthesis_blueprint") if isinstance(spec.get("base_synthesis_blueprint"), dict) else {}
        contract = {
            "name": "string",
            "goal": "string",
            "project_kind": "open-ended string; describe the requested application honestly",
            "languages": ["language"],
            "frameworks": ["framework"],
            "requirements": [{"requirement_id": "req_1", "text": "requirement", "kind": "functional|nonfunctional|constraint", "priority": "must|should|could", "acceptance": ["observable acceptance criterion"]}],
            "components": [{"id": "component_id", "name": "name", "responsibility": "single responsibility", "interfaces": ["contract/interface"]}],
            "dependencies": [{"name": "package_or_runtime", "kind": "stdlib|local_package|external_package", "required": True, "reason": "why"}],
            "requested_capabilities": [{"name": "capability", "reason": "why needed", "authority_required": ["Read|Write|Execute|Network|Filesystem"], "risk": "bounded|elevated|restricted", "granted": False}],
            "files": [{"path": "sandbox/<project-relative-path>", "purpose": "what this file implements", "language": "python|javascript|html|css|json|...", "artifact_role": "SOURCE|ENTRYPOINT|CONFIG|MANIFEST|TEST|DOCUMENTATION|ASSET_TEXT|DATA", "component_id": "component_id", "entrypoint": False, "depends_on": ["sandbox/other/file"], "acceptance": ["file-specific criterion"]}],
            "tests": [{"name": "test name", "type": "static|unit|integration", "target": "component or file", "description": "test objective"}],
            "acceptance_criteria": ["project-wide observable criterion"],
            "constraints": {"local_first": True, "sandbox_only": True, "live_core_write": False, "self_approval": False, "shell_allowed": False, "network_allowed": False},
            "run": {"entrypoint": "sandbox/path/to/entrypoint", "method": "language/runtime description", "arguments": []},
            "asset_requests": [{"kind": "image|audio|video|model|other", "description": "optional future asset", "required": False, "path": ""}],
            "phase": "ARCHITECT",
            "metadata": {"execution_authority": False},
        }
        return (
            "SARAHMEMORY QSML/0.2 UNIVERSAL APPLICATION ARCHITECT\n"
            "PHASE: ARCHITECTURE\n"
            "You are planning an arbitrary LOCAL application for the governed NAILDE sandbox.\n"
            "Do not choose a canned example or substitute a different application. Follow the user's exact goal.\n"
            "Hardcode rails, not thoughts: the architecture must be derived from the request, not from prompt-specific templates.\n"
            "Return ONE valid JSON object only. No markdown. No commentary. No code fences.\n"
            "The blueprint is a plan, not permission to execute. Do not request self-approval, live CORE writes, credentials, or hidden persistence.\n"
            "If the application needs network, filesystem, database, device, model, or process capabilities, declare them in requested_capabilities with granted=false; declaration is never permission.\n"
            "Prefer standard-library/local dependencies when they can satisfy the request. External packages may be DECLARED but are not installed here.\n"
            "If LANGUAGE HINTS is empty, choose the implementation language(s) from the application requirements; do not default every application to Python.\n"
            "Every generated source file must be necessary, have a clear purpose, and use a relative path beginning with sandbox/.\n"
            "The current universal file synthesizer produces text artifacts. Do not put binary PNG/JPG/audio/video/GLB/font/PDF files in files. Use SVG/CSS/procedural text assets, or declare unresolved media in asset_requests as required=false.\n"
            "Include enough files for a COMPLETE coherent first implementation, but keep the architecture bounded (maximum 96 files).\n"
            "Do not emit TODOs/placeholders as requirements; describe implementable behavior and acceptance criteria.\n\n"
            f"USER GOAL:\n{prompt}\n\n"
            f"ADDITIONAL REQUIREMENTS:\n{details or '(none)'}\n\n"
            f"QSML PROJECT HINT: {spec.get('project_kind') or 'software_project'}\n"
            f"LANGUAGE HINTS: {json.dumps(languages)}\n"
            f"FRAMEWORK HINTS: {json.dumps(frameworks)}\n"
            f"FEATURE HINTS: {json.dumps(features, ensure_ascii=False)}\n"
            f"QSML CAPABILITY REQUEST HINTS: {json.dumps(requested_capabilities, ensure_ascii=False, sort_keys=True)}\n"
            f"GOVERNED CONSTRAINTS: {json.dumps(constraints, ensure_ascii=False, sort_keys=True)}\n"
            f"BASE QSML BLUEPRINT: {json.dumps(base_blueprint, ensure_ascii=False, sort_keys=True)[:12000]}\n\n"
            f"REQUIRED JSON SHAPE:\n{json.dumps(contract, ensure_ascii=False, indent=2)}"
        )

    def _normalize_model_blueprint(self, raw: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        bp = dict(raw or {})
        bp["name"] = str(bp.get("name") or spec.get("application_name") or "NAILDE Application")[:160]
        bp["goal"] = str(bp.get("goal") or spec.get("top_prompt") or "")[:20000]
        bp["project_kind"] = str(bp.get("project_kind") or spec.get("project_kind") or "software_project")[:120]
        bp["languages"] = [str(x)[:80] for x in list(bp.get("languages") or spec.get("languages") or [])[:16] if str(x).strip()]
        bp["frameworks"] = [str(x)[:80] for x in list(bp.get("frameworks") or spec.get("frameworks") or [])[:16] if str(x).strip()]
        reqs = [dict(x) for x in list(bp.get("requirements") or []) if isinstance(x, dict)][:128]
        if not reqs:
            reqs.append({"requirement_id": "req_user_goal", "text": bp["goal"], "kind": "goal", "priority": "must", "source": "user", "acceptance": []})
        bp["requirements"] = reqs
        bp["components"] = [dict(x) for x in list(bp.get("components") or []) if isinstance(x, dict)][:64]
        bp["dependencies"] = [dict(x) if isinstance(x, dict) else {"name": str(x)} for x in list(bp.get("dependencies") or [])][:64]
        cap_index: Dict[str, Dict[str, Any]] = {}
        for cap in ([dict(x) for x in list(spec.get("requested_capabilities") or []) if isinstance(x, dict)][:64] + [dict(x) for x in list(bp.get("requested_capabilities") or []) if isinstance(x, dict)][:64]):
            name = str(cap.get("name") or cap.get("capability") or "").strip().lower().replace(" ", "_")[:120]
            if not name:
                continue
            row = dict(cap)
            row["name"] = name
            row["reason"] = str(row.get("reason") or "Application capability requested by QSML/application architecture.")[:1000]
            row["authority_required"] = [str(x)[:80] for x in list(row.get("authority_required") or row.get("authority") or [])[:16]]
            row["risk"] = str(row.get("risk") or "bounded")[:40]
            row["granted"] = False
            row["execution_authority"] = False
            cap_index[name] = row
        bp["requested_capabilities"] = list(cap_index.values())[:64]
        bp["tests"] = [dict(x) for x in list(bp.get("tests") or []) if isinstance(x, dict)][:96]
        bp["acceptance_criteria"] = [str(x)[:1000] for x in list(bp.get("acceptance_criteria") or [])[:128] if str(x).strip()]
        bp["asset_requests"] = [dict(x) for x in list(bp.get("asset_requests") or []) if isinstance(x, dict)][:64]
        model_constraints = dict(bp.get("constraints") or {})
        governed = dict(spec.get("constraints") or {})
        # Higher-authority compiler constraints win over model suggestions.
        for key, value in governed.items():
            model_constraints[key] = value
        model_constraints["sandbox_only"] = True
        model_constraints["live_core_write"] = False
        model_constraints["self_approval"] = False
        model_constraints.setdefault("shell_allowed", False)
        bp["constraints"] = model_constraints
        files = []
        for idx, item in enumerate(list(bp.get("files") or bp.get("file_plan") or [])[:96]):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            path = str(row.get("path") or "").replace("\\", "/").strip()
            path = re.sub(r"^\./+", "", path)
            if path and not path.startswith("sandbox/") and not path.startswith("/") and not re.match(r"^[A-Za-z]:/", path) and ".." not in path.split("/"):
                path = "sandbox/project/" + path.lstrip("/")
            row["path"] = path
            row["purpose"] = str(row.get("purpose") or f"Generated application artifact {idx + 1}")[:2000]
            row["language"] = str(row.get("language") or _language_from_path(path))[:80]
            row["artifact_role"] = str(row.get("artifact_role") or row.get("role") or ("ENTRYPOINT" if row.get("entrypoint") else "SOURCE"))[:80]
            row["component_id"] = str(row.get("component_id") or "")[:120]
            row["entrypoint"] = bool(row.get("entrypoint", False))
            row["depends_on"] = [str(x).replace("\\", "/")[:240] for x in list(row.get("depends_on") or [])[:32]]
            row["acceptance"] = [str(x)[:1000] for x in list(row.get("acceptance") or [])[:32]]
            files.append(row)
        bp["files"] = files
        run = dict(bp.get("run") or {})
        if not run.get("entrypoint"):
            entry = next((x.get("path") for x in files if x.get("entrypoint")), "")
            if entry:
                run["entrypoint"] = entry
        bp["run"] = run
        bp["phase"] = "ARCHITECT"
        bp["metadata"] = {
            **dict(bp.get("metadata") or {}),
            "schema": "SarahMemory.qsml.application_blueprint.v0_2",
            "planner": "NAILDE_local_synthesis_model",
            "prompt_specific_template_selection": False,
            "execution_authority": False,
        }
        return bp

    def _validate_qsml_synthesis_ownership(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce the SML/Neuron/NAILDE ownership boundary before synthesis.

        SMLProtocol defines the legal route. Neuron owns activation/weighting.
        NAILDE consumes the selected legal programming route as a sandbox
        synthesizer. This check prevents NAILDE from silently becoming a router.
        """
        route_definition_owner = str(spec.get("route_definition_owner") or "")
        route_activation_owner = str(spec.get("route_activation_owner") or "")
        routes = [list(r) for r in list(spec.get("candidate_routes") or []) if isinstance(r, (list, tuple))]
        compatible_route = None
        for route in routes:
            names = [str(x) for x in route]
            try:
                i_sml = names.index("SarahMemorySMLProtocol")
                i_neuron = names.index("SarahMemoryNeuron")
                i_nailde = names.index("SarahMemoryNAILDE")
            except ValueError:
                continue
            if i_sml < i_neuron < i_nailde:
                compatible_route = names
                break
        errors: List[str] = []
        if route_definition_owner != "SarahMemorySMLProtocol":
            errors.append("route_definition_owner_must_be_SarahMemorySMLProtocol")
        if route_activation_owner != "SarahMemoryNeuron":
            errors.append("route_activation_owner_must_be_SarahMemoryNeuron")
        if not compatible_route:
            errors.append("legal_programming_route_must_place_SMLProtocol_before_Neuron_before_NAILDE")
        if not bool(spec.get("sandbox_only", True)):
            errors.append("nailde_synthesis_must_be_sandbox_only")
        return {
            "ok": not errors,
            "schema": "SarahMemory.nailde.qsml_ownership_check.v0_2",
            "route_definition_owner": route_definition_owner,
            "route_activation_owner": route_activation_owner,
            "compatible_route": compatible_route,
            "errors": errors,
            "execution_authority": False,
        }

    @staticmethod
    def _ordered_universal_file_plan(blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """Topologically order declared file dependencies without inventing any.

        Unknown dependency labels are retained as warnings because a model may
        use component identifiers in depends_on. Cycles among actual file paths
        are rejected as an invalid architecture contract.
        """
        plans = [dict(x) for x in list(blueprint.get("files") or []) if isinstance(x, dict)][:96]
        by_path = {str(x.get("path") or ""): x for x in plans if str(x.get("path") or "")}
        original_order = {str(x.get("path") or ""): i for i, x in enumerate(plans)}
        incoming = {p: set() for p in by_path}
        dependents = {p: set() for p in by_path}
        warnings: List[Dict[str, Any]] = []
        for path, row in by_path.items():
            for dep in list(row.get("depends_on") or [])[:32]:
                dep_s = str(dep or "").replace("\\", "/").strip()
                if not dep_s:
                    continue
                if dep_s in by_path:
                    incoming[path].add(dep_s)
                    dependents[dep_s].add(path)
                else:
                    warnings.append({"path": path, "warning": f"depends_on_non_file_reference:{dep_s}"})
        ready = sorted([p for p, inc in incoming.items() if not inc], key=lambda x: original_order.get(x, 0))
        ordered_paths: List[str] = []
        while ready:
            node = ready.pop(0)
            ordered_paths.append(node)
            for child in sorted(dependents.get(node) or [], key=lambda x: original_order.get(x, 0)):
                incoming[child].discard(node)
                if not incoming[child] and child not in ordered_paths and child not in ready:
                    ready.append(child)
                    ready.sort(key=lambda x: original_order.get(x, 0))
        if len(ordered_paths) != len(by_path):
            cycle_paths = sorted(p for p, inc in incoming.items() if inc)
            return {
                "ok": False,
                "error": "file_dependency_cycle",
                "cycle_paths": cycle_paths,
                "warnings": warnings,
                "execution_authority": False,
            }
        ordered = [by_path[p] for p in ordered_paths]
        # Preserve any malformed/no-path plans at the end so later validation can report them.
        ordered.extend(x for x in plans if not str(x.get("path") or ""))
        return {"ok": True, "files": ordered, "warnings": warnings, "execution_authority": False}

    def _universal_blueprint_from_model(self, spec: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create and bounded-repair the arbitrary-application architecture.

        The local model proposes architecture; SMLProtocol validates the formal
        blueprint. Invalid JSON/contracts are repaired at most three times and
        never replaced with a canned application.
        """
        payload = payload if isinstance(payload, dict) else {}
        try:
            max_arch_repairs = max(0, min(3, int(payload.get("max_architecture_repair_rounds") if payload.get("max_architecture_repair_rounds") is not None else 2)))
        except Exception:
            max_arch_repairs = 2
        base_prompt = self._universal_architecture_prompt(spec)
        prompt = base_prompt
        attempts: List[Dict[str, Any]] = []
        last_validation: Dict[str, Any] = {}
        last_preview = ""
        for round_index in range(0, max_arch_repairs + 1):
            purpose = "arbitrary_application_architecture" if round_index == 0 else "arbitrary_application_architecture_repair"
            model_result = self._qsml_local_model_generate(
                prompt,
                purpose=purpose,
                max_tokens=int(payload.get("architecture_max_tokens") or 6000),
                temperature=float(payload.get("architecture_temperature") or (0.10 if round_index == 0 else 0.03)),
                model=str(payload.get("model") or ""),
            )
            model_meta = {k: v for k, v in model_result.items() if k != "content"}
            if not model_result.get("ok"):
                attempts.append({"round": round_index, "stage": "model", "model": model_meta})
                return {
                    "ok": False,
                    "error": "local_model_required_for_arbitrary_application_architecture",
                    "model": model_meta,
                    "attempts": attempts,
                    "execution_authority": False,
                }
            content = str(model_result.get("content") or "")
            last_preview = content[:1600]
            raw = _extract_first_json_object(content)
            if not isinstance(raw, dict):
                attempts.append({"round": round_index, "stage": "json_parse", "model": model_meta, "response_preview": last_preview})
                if round_index >= max_arch_repairs:
                    return {
                        "ok": False,
                        "error": "local_model_blueprint_not_valid_json_after_bounded_repairs",
                        "model": model_meta,
                        "attempts": attempts,
                        "response_preview": last_preview,
                        "execution_authority": False,
                    }
                prompt = (
                    "SARAHMEMORY QSML/0.2 APPLICATION ARCHITECTURE REPAIR\n"
                    "Return ONE valid JSON object only. Preserve the exact user goal. Do not substitute a template.\n"
                    "Your previous response was not parseable as the required application blueprint.\n"
                    f"USER GOAL: {spec.get('top_prompt','')}\n"
                    f"PREVIOUS RESPONSE PREVIEW: {last_preview}\n\n"
                    "Re-read and obey this original architecture contract:\n" + base_prompt
                )
                continue
            blueprint = self._normalize_model_blueprint(raw, spec)
            try:
                from SarahMemorySMLProtocol import sml_validate_application_blueprint  # type: ignore
                source_program = spec.get("qsml_program") if isinstance(spec.get("qsml_program"), dict) else None
                validation = sml_validate_application_blueprint(blueprint, require_files=True, source_program=source_program)
            except Exception as exc:
                validation = {"ok": False, "status": "ERROR", "issues": [{"code": "NAILDE-QSML-BP", "message": str(exc), "severity": "ERROR"}]}
            last_validation = validation
            attempts.append({
                "round": round_index,
                "stage": "validation",
                "model": model_meta,
                "status": validation.get("status"),
                "issues": list(validation.get("issues") or [])[:64],
            })
            if validation.get("ok"):
                return {
                    "ok": True,
                    "blueprint": validation.get("blueprint") if isinstance(validation.get("blueprint"), dict) else blueprint,
                    "validation": validation,
                    "model": model_meta,
                    "attempts": attempts,
                    "architecture_repairs": round_index,
                    "execution_authority": False,
                }
            if round_index >= max_arch_repairs:
                break
            prompt = (
                "SARAHMEMORY QSML/0.2 APPLICATION ARCHITECTURE REPAIR\n"
                "Repair the blueprint contract. Return ONE complete valid JSON object only; no markdown.\n"
                "Keep the same requested application and functionality. Do not remove requirements merely to pass validation.\n"
                "No live-core writes, no self-approval, and every file path must begin with sandbox/.\n"
                f"USER GOAL: {spec.get('top_prompt','')}\n"
                f"VALIDATION ISSUES: {json.dumps(validation.get('issues') or [], ensure_ascii=False, sort_keys=True)[:12000]}\n"
                f"INVALID BLUEPRINT: {json.dumps(blueprint, ensure_ascii=False, sort_keys=True)[:24000]}\n\n"
                "Original contract follows:\n" + base_prompt
            )
        return {
            "ok": False,
            "error": "local_model_blueprint_failed_validation_after_bounded_repairs",
            "validation": last_validation,
            "attempts": attempts,
            "response_preview": last_preview,
            "execution_authority": False,
        }

    @staticmethod
    def _file_contract_summary(blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for row in list(blueprint.get("files") or [])[:96]:
            if not isinstance(row, dict):
                continue
            out.append({
                "path": row.get("path"),
                "purpose": row.get("purpose"),
                "language": row.get("language"),
                "component_id": row.get("component_id"),
                "entrypoint": bool(row.get("entrypoint")),
                "depends_on": list(row.get("depends_on") or [])[:16],
            })
        return out

    @staticmethod
    def _generated_interface_summary(path: str, content: str) -> Dict[str, Any]:
        """Extract bounded cross-file interface evidence without executing code."""
        language = _language_from_path(path)
        summary: Dict[str, Any] = {"path": path, "language": language, "sha256": _sha256_text(content)}
        raw = str(content or "")
        if language == "python":
            try:
                tree = ast.parse(raw, filename=path)
                summary["classes"] = [n.name for n in tree.body if isinstance(n, ast.ClassDef)][:64]
                summary["functions"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][:96]
                summary["imports"] = [
                    (n.module or "") if isinstance(n, ast.ImportFrom) else ",".join(a.name for a in n.names)
                    for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                ][:96]
            except Exception:
                pass
        elif language in {"javascript", "typescript"}:
            summary["exports"] = re.findall(r"(?m)^\s*export\s+(?:default\s+)?(?:class|function|const|let|var)?\s*([A-Za-z_$][\w$]*)", raw)[:96]
            summary["functions"] = re.findall(r"(?m)^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", raw)[:96]
        elif language == "html":
            summary["ids"] = re.findall(r"\bid=[\"']([^\"']+)[\"']", raw, flags=re.I)[:96]
            summary["scripts"] = re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", raw, flags=re.I)[:48]
            summary["stylesheets"] = re.findall(r"<link[^>]+href=[\"']([^\"']+)[\"']", raw, flags=re.I)[:48]
        elif language == "json":
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    summary["top_level_keys"] = list(obj.keys())[:96]
            except Exception:
                pass
        return summary

    def _universal_file_prompt(self, spec: Dict[str, Any], blueprint: Dict[str, Any], file_plan: Dict[str, Any], generated_index: List[Dict[str, Any]]) -> str:
        language = str(file_plan.get("language") or _language_from_path(str(file_plan.get("path") or "")))
        return (
            "SARAHMEMORY QSML/0.2 UNIVERSAL APPLICATION FILE SYNTHESIZER\n"
            "PHASE: GENERATE\n"
            "Generate exactly ONE complete file for the governed NAILDE sandbox application.\n"
            "Return raw file content only. Do not use markdown fences. Do not add explanations before or after the file.\n"
            "Implement the requested behavior; do not replace it with a demo from another application.\n"
            "No TODO, FIXME, 'implement later', pseudo-code, or placeholder business logic.\n"
            "Do not write or execute outside the generated application. Do not embed credentials or secrets.\n"
            "Respect interfaces/dependencies from the application blueprint. Prefer deterministic/local behavior.\n"
            "If an optional external package is unavailable, fail clearly or provide a standard-library fallback when feasible.\n"
            "Generated code is sandbox evidence only and has no execution authority.\n\n"
            f"USER GOAL:\n{spec.get('top_prompt','')}\n\n"
            f"ADDITIONAL REQUIREMENTS:\n{spec.get('details_prompt','')}\n\n"
            f"APPLICATION BLUEPRINT:\n{json.dumps(blueprint, ensure_ascii=False, sort_keys=True)[:30000]}\n\n"
            f"TARGET FILE CONTRACT:\n{json.dumps(file_plan, ensure_ascii=False, sort_keys=True)}\n"
            f"LANGUAGE: {language}\n"
            f"FILES ALREADY GENERATED (interfaces only):\n{json.dumps(generated_index[-48:], ensure_ascii=False, sort_keys=True)}\n"
        )

    def _static_validate_generated_content(self, path: str, content: str) -> Dict[str, Any]:
        language = _language_from_path(path)
        errors: List[str] = []
        warnings: List[str] = []
        raw = str(content or "")
        raw_bytes = len(raw.encode("utf-8", "replace"))
        if raw_bytes > _MAX_UNIVERSAL_TEXT_FILE_BYTES:
            errors.append(f"generated_file_exceeds_text_budget:{raw_bytes}>{_MAX_UNIVERSAL_TEXT_FILE_BYTES}")
        if not raw.strip():
            errors.append("empty_file")
        low = raw.lower()
        if any(tok in low for tok in ("todo:", "todo ", "fixme", "implement later", "placeholder business logic")):
            errors.append("unfinished_placeholder_detected")
        if language == "python" and raw.strip():
            try:
                ast.parse(raw, filename=path)
            except SyntaxError as exc:
                errors.append(f"python_syntax:{exc.msg}:line_{exc.lineno}")
        elif language == "json" and raw.strip():
            try:
                json.loads(raw)
            except Exception as exc:
                errors.append(f"json_syntax:{exc}")
        elif language == "html" and raw.strip():
            if "<html" not in low and "<!doctype" not in low:
                warnings.append("html_document_root_not_detected")
        elif language in {"javascript", "typescript", "css"} and raw.strip():
            pairs = [("{", "}"), ("(", ")"), ("[", "]")]
            for left, right in pairs:
                if raw.count(left) != raw.count(right):
                    warnings.append(f"unbalanced_{left}{right}")
        if re.search(r"(?i)(?:[A-Za-z]:[\\/].*SarahMemory[\\/](?:core|api)|/SarahMemory/(?:core|api))", raw):
            warnings.append("generated_code_references_live_sarahmemory_core_or_api")
        risk_patterns = {
            "shell_process_usage": r"(?i)\b(?:os\.system|subprocess\.(?:run|Popen|call)|powershell|cmd\.exe)\b",
            "dynamic_code_execution": r"(?i)\b(?:eval|exec)\s*\(",
            "credential_access": r"(?i)\b(?:api[_-]?key|password|private[_-]?key|credential)\b",
            "network_usage": r"(?i)\b(?:requests\.|urllib\.|http\.client|socket\.|aiohttp|websockets?|fetch\s*\(|XMLHttpRequest|WebSocket\s*\()",
            "device_or_sensor_usage": r"(?i)\b(?:cv2\.VideoCapture|pyaudio|sounddevice|serial\.Serial|hid\.|pygame\.joystick)\b",
        }
        risk_hits = [name for name, pattern in risk_patterns.items() if re.search(pattern, raw)]
        return {
            "ok": not errors,
            "path": path,
            "language": language,
            "errors": errors,
            "warnings": warnings,
            "risk_hits": risk_hits,
            "sha256": _sha256_text(raw),
            "size_bytes": raw_bytes,
            "max_size_bytes": _MAX_UNIVERSAL_TEXT_FILE_BYTES,
            "execution_authority": False,
        }

    def _compare_generated_artifact(self, user_goal: str, path: str, content: str) -> Dict[str, Any]:
        """Run Compare GuardDog A when available; never grants execution."""
        mod = _read_only_import("SarahMemoryCompare")
        fn = getattr(mod, "validate_response_quality", None) if mod is not None else None
        if not callable(fn):
            return {"ok": False, "status": "NOT_AVAILABLE", "accepted": True, "execution_authority": False}
        try:
            out = fn(str(user_goal or path), str(content or ""), intent="code")
            return {"ok": True, "status": "PASS" if bool(out.get("accepted")) else "REJECT", **dict(out), "execution_authority": False}
        except Exception as exc:
            return {"ok": False, "status": "ERROR", "accepted": True, "error": str(exc)[:500], "execution_authority": False}

    def _repair_generated_file(
        self,
        *,
        spec: Dict[str, Any],
        blueprint: Dict[str, Any],
        file_plan: Dict[str, Any],
        content: str,
        validation: Dict[str, Any],
        payload: Dict[str, Any],
        repair_round: int,
    ) -> Dict[str, Any]:
        repair_prompt = (
            "SARAHMEMORY QSML/0.2 FILE REPAIR\n"
            "PHASE: REPAIR\n"
            "Repair this generated file so it satisfies its contract and static validation.\n"
            "Return the COMPLETE corrected raw file only. No markdown fences or explanation.\n"
            "Do not remove requested functionality merely to make validation pass.\n"
            "Do not add TODOs/placeholders. Preserve sandbox-only/local-first constraints.\n\n"
            f"USER GOAL: {spec.get('top_prompt','')}\n"
            f"FILE CONTRACT: {json.dumps(file_plan, ensure_ascii=False, sort_keys=True)}\n"
            f"VALIDATION: {json.dumps(validation, ensure_ascii=False, sort_keys=True)}\n"
            f"REPAIR ROUND: {repair_round}\n\n"
            f"CURRENT FILE:\n{content}\n"
        )
        return self._qsml_local_model_generate(
            repair_prompt,
            purpose="arbitrary_application_file_repair",
            max_tokens=int(payload.get("file_max_tokens") or 6000),
            temperature=float(payload.get("repair_temperature") or 0.05),
            model=str(payload.get("model") or ""),
        )

    def _project_static_validation(self, files: Dict[str, str], blueprint: Dict[str, Any]) -> Dict[str, Any]:
        results = [self._static_validate_generated_content(path, content) for path, content in sorted(files.items())]
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        risk_hits: List[Dict[str, Any]] = []
        for row in results:
            for err in row.get("errors") or []:
                errors.append({"path": row.get("path"), "error": err})
            for warn in row.get("warnings") or []:
                warnings.append({"path": row.get("path"), "warning": warn})
            for risk in row.get("risk_hits") or []:
                risk_hits.append({"path": row.get("path"), "risk": risk})
        planned = {str(x.get("path") or "") for x in list(blueprint.get("files") or []) if isinstance(x, dict)}
        actual = set(files.keys())
        total_bytes = sum(len(str(content or "").encode("utf-8", "replace")) for content in files.values())
        if total_bytes > _MAX_UNIVERSAL_PROJECT_BYTES:
            errors.append({"path": "<project>", "error": f"generated_project_exceeds_text_budget:{total_bytes}>{_MAX_UNIVERSAL_PROJECT_BYTES}"})
        missing = sorted(p for p in planned if p and p not in actual)
        for path in missing:
            errors.append({"path": path, "error": "planned_file_missing"})
        entrypoint = str((blueprint.get("run") or {}).get("entrypoint") or "") if isinstance(blueprint.get("run"), dict) else ""
        if entrypoint and entrypoint not in actual:
            errors.append({"path": entrypoint, "error": "declared_entrypoint_missing"})

        # External dependency declarations require a reproducible manifest, but
        # NAILDE never invents versions or auto-installs packages.
        external_deps = []
        for dep in list(blueprint.get("dependencies") or [])[:64]:
            if not isinstance(dep, dict):
                continue
            kind = str(dep.get("kind") or dep.get("type") or "").strip().lower()
            if kind in {"external", "external_package", "package", "pip", "npm", "library"}:
                external_deps.append(str(dep.get("name") or "").strip())
        if external_deps:
            manifest_names = {
                "requirements.txt", "pyproject.toml", "setup.cfg", "setup.py",
                "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
                "pom.xml", "build.gradle", "build.gradle.kts", "cargo.toml",
                "go.mod", "composer.json", "gemfile",
            }
            has_dep_manifest = any(os.path.basename(p).lower() in manifest_names for p in actual)
            if not has_dep_manifest:
                warnings.append({
                    "path": "<project>",
                    "warning": "external_dependencies_declared_without_dependency_manifest:" + ",".join(x for x in external_deps if x)[:500],
                })
        # Check simple Python relative imports without importing or executing code.
        py_paths = {p for p in actual if p.lower().endswith(".py")}
        for path in sorted(py_paths):
            content = files.get(path, "")
            try:
                tree = ast.parse(content, filename=path)
            except Exception:
                continue
            base_dir = os.path.dirname(path).replace("\\", "/")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and int(getattr(node, "level", 0) or 0) > 0:
                    module = str(node.module or "").replace(".", "/")
                    up = max(0, int(node.level or 1) - 1)
                    parent = base_dir
                    for _ in range(up):
                        parent = os.path.dirname(parent).replace("\\", "/")
                    candidate = (parent.rstrip("/") + "/" + module + ".py").replace("//", "/") if module else ""
                    pkg = (parent.rstrip("/") + "/" + module + "/__init__.py").replace("//", "/") if module else parent.rstrip("/") + "/__init__.py"
                    if candidate and candidate not in actual and pkg not in actual:
                        warnings.append({"path": path, "warning": f"relative_import_target_not_found:{module}"})
        return {
            "ok": not errors,
            "schema": "SarahMemory.nailde.universal_synthesis_validation.v0_2",
            "results": results,
            "errors": errors,
            "warnings": warnings,
            "risk_hits": risk_hits,
            "risk_review_required": bool(risk_hits),
            "total_generated_bytes": total_bytes,
            "max_project_bytes": _MAX_UNIVERSAL_PROJECT_BYTES,
            "execution_authority": False,
            "live_runtime_verified": False,
        }

    def _universal_synthesize_spec(self, spec: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Architect, generate, validate, and bounded-repair an arbitrary app."""
        payload = payload if isinstance(payload, dict) else {}
        ownership = self._validate_qsml_synthesis_ownership(spec)
        if not ownership.get("ok"):
            return {
                "ok": False,
                "phase": "COMPILE",
                "error": "qsml_synthesis_ownership_contract_failed",
                "ownership": ownership,
                "message": "NAILDE refused synthesis because SMLProtocol/Neuron/NAILDE ownership ordering was not preserved.",
                "execution_authority": False,
            }
        architecture = self._universal_blueprint_from_model(spec, payload)
        if not architecture.get("ok"):
            return {
                "ok": False,
                "phase": "ARCHITECT",
                "error": architecture.get("error") or "blueprint_validation_failed",
                "architecture": architecture,
                "message": "Arbitrary application synthesis stopped rather than substituting an unrelated template.",
                "execution_authority": False,
            }
        blueprint = dict(architecture.get("blueprint") or {})
        ordering = self._ordered_universal_file_plan(blueprint)
        if not ordering.get("ok"):
            return {
                "ok": False,
                "phase": "ARCHITECT",
                "error": ordering.get("error") or "invalid_file_dependency_graph",
                "ordering": ordering,
                "blueprint": blueprint,
                "execution_authority": False,
            }
        file_plan = [dict(x) for x in list(ordering.get("files") or []) if isinstance(x, dict)][:96]
        if ordering.get("warnings"):
            architecture["ordering_warnings"] = list(ordering.get("warnings") or [])
        if not file_plan:
            return {"ok": False, "phase": "ARCHITECT", "error": "blueprint_has_no_files", "architecture": architecture, "execution_authority": False}
        max_repairs = 3
        try:
            max_repairs = max(0, min(3, int(payload.get("max_repair_rounds") if payload.get("max_repair_rounds") is not None else 3)))
        except Exception:
            max_repairs = 3
        files: Dict[str, str] = {}
        generation_trace: List[Dict[str, Any]] = []
        generated_index: List[Dict[str, Any]] = []
        for fp in file_plan:
            path = str(fp.get("path") or "")
            generation = self._qsml_local_model_generate(
                self._universal_file_prompt(spec, blueprint, fp, generated_index),
                purpose="arbitrary_application_file_generation",
                max_tokens=int(payload.get("file_max_tokens") or 6000),
                temperature=float(payload.get("file_temperature") or 0.12),
                model=str(payload.get("model") or ""),
            )
            if not generation.get("ok"):
                return {
                    "ok": False,
                    "phase": "GENERATE",
                    "error": "local_model_failed_while_generating_file",
                    "failed_path": path,
                    "model": generation,
                    "blueprint": blueprint,
                    "generated_paths": list(files.keys()),
                    "execution_authority": False,
                }
            content = _strip_code_fence(generation.get("content"))
            validation = self._static_validate_generated_content(path, content)
            compare_check = self._compare_generated_artifact(str(spec.get("top_prompt") or ""), path, content)
            validation["compare"] = compare_check
            if compare_check.get("ok") and not bool(compare_check.get("accepted", True)):
                validation.setdefault("errors", []).append("compare_output_validity_reject")
                validation["ok"] = False
            repairs = []
            for repair_round in range(1, max_repairs + 1):
                if validation.get("ok"):
                    break
                repair = self._repair_generated_file(
                    spec=spec,
                    blueprint=blueprint,
                    file_plan=fp,
                    content=content,
                    validation=validation,
                    payload=payload,
                    repair_round=repair_round,
                )
                repairs.append({k: v for k, v in repair.items() if k != "content"})
                if not repair.get("ok"):
                    break
                content = _strip_code_fence(repair.get("content"))
                validation = self._static_validate_generated_content(path, content)
                compare_check = self._compare_generated_artifact(str(spec.get("top_prompt") or ""), path, content)
                validation["compare"] = compare_check
                if compare_check.get("ok") and not bool(compare_check.get("accepted", True)):
                    validation.setdefault("errors", []).append("compare_output_validity_reject")
                    validation["ok"] = False
            generation_trace.append({
                "path": path,
                "model": {k: v for k, v in generation.items() if k != "content"},
                "validation": validation,
                "repairs": repairs,
            })
            if not validation.get("ok"):
                return {
                    "ok": False,
                    "phase": "VALIDATE",
                    "error": "generated_file_failed_static_validation_after_bounded_repairs",
                    "failed_path": path,
                    "validation": validation,
                    "blueprint": blueprint,
                    "generation_trace": generation_trace,
                    "execution_authority": False,
                }
            files[path] = content
            interface_summary = self._generated_interface_summary(path, content)
            interface_summary["purpose"] = fp.get("purpose")
            generated_index.append(interface_summary)

        # Deterministic governance metadata is part of the rails, not app logic.
        synthesis_manifest = {
            "schema": "SarahMemory.nailde.universal_synthesis_manifest.v0_2",
            "application_name": blueprint.get("name"),
            "goal": blueprint.get("goal"),
            "qsml_program_id": spec.get("qsml_program_id"),
            "qsml_language_version": spec.get("qsml_language_version"),
            "blueprint_id": blueprint.get("blueprint_id"),
            "project_kind": blueprint.get("project_kind"),
            "languages": blueprint.get("languages"),
            "frameworks": blueprint.get("frameworks"),
            "dependencies": blueprint.get("dependencies"),
            "requested_capabilities": blueprint.get("requested_capabilities"),
            "run": blueprint.get("run"),
            "acceptance_criteria": blueprint.get("acceptance_criteria"),
            "route_definition_owner": "SarahMemorySMLProtocol",
            "route_activation_owner": "SarahMemoryNeuron",
            "synthesis_owner": "SarahMemoryNAILDE",
            "sandbox_only": True,
            "live_core_write": False,
            "execution_authority": False,
        }
        manifest_path = "sandbox/QSML_SYNTHESIS_MANIFEST.json"
        if manifest_path not in files:
            files[manifest_path] = json.dumps(synthesis_manifest, indent=2, sort_keys=True, ensure_ascii=False)
        blueprint_path = "sandbox/QSML_APPLICATION_BLUEPRINT.json"
        if blueprint_path not in files:
            files[blueprint_path] = json.dumps(blueprint, indent=2, sort_keys=True, ensure_ascii=False)

        project_validation = self._project_static_validation(files, blueprint)
        return {
            "ok": bool(project_validation.get("ok")),
            "phase": "READY" if project_validation.get("ok") else "VALIDATE",
            "schema": "SarahMemory.nailde.universal_arbitrary_application_synthesis.v0_2",
            "blueprint": blueprint,
            "architecture": architecture,
            "ownership": ownership,
            "files": files,
            "generation_trace": generation_trace,
            "validation": project_validation,
            "risk_review_required": bool(project_validation.get("risk_review_required")),
            "sandbox_only": True,
            "live_file_write": False,
            "execution_authority": False,
        }

    def synthesize_arbitrary_application(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Public NAILDE entrypoint for the QSML universal application synthesizer."""
        payload = payload if isinstance(payload, dict) else {}
        normalized = dict(payload)
        if not normalized.get("top_prompt"):
            normalized["top_prompt"] = normalized.get("goal") or normalized.get("prompt") or normalized.get("problem") or ""
        normalized["universal_synthesis"] = True
        return self.auto_build_from_prompt(normalized)

    def handle_gcop_capability_gap(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Translate a GCOP capability gap into a sandbox-only NAILDE build mission.

        This method is deliberately proposal-only.  GCOP/SML may identify that a
        capability is missing, but NAILDE is not allowed to grant the capability,
        install the generated package, write live Core/API files, or execute it.
        The result is a staged application proposal that must continue through the
        existing validation/Compare/AppStore/DevBridge/human-approval path.
        """
        payload = payload if isinstance(payload, dict) else {}
        gap = payload.get("capability_gap") if isinstance(payload.get("capability_gap"), dict) else {}
        event_type = str(payload.get("event_type") or payload.get("type") or gap.get("event_type") or "CAPABILITY_GAP").strip().upper()
        capability = str(
            gap.get("capability")
            or gap.get("required_capability")
            or payload.get("capability")
            or payload.get("required_capability")
            or ""
        ).strip()[:240]
        goal = str(
            gap.get("goal")
            or gap.get("reason")
            or payload.get("goal")
            or payload.get("reason")
            or payload.get("prompt")
            or ""
        ).strip()[:4000]
        if event_type not in {"CAPABILITY_GAP", "GCOP_CAPABILITY_GAP"}:
            return {
                "ok": False,
                "blocked": True,
                "reason": "unsupported_gcop_event",
                "event_type": event_type,
                "sandbox_only": True,
                "execution_authority": False,
            }
        if not capability and not goal:
            return {
                "ok": False,
                "blocked": True,
                "reason": "capability_gap_requires_capability_or_goal",
                "sandbox_only": True,
                "execution_authority": False,
            }

        top_prompt = goal or f"Create a governed local capability for: {capability}"
        normalized = dict(payload)
        normalized.update({
            "top_prompt": top_prompt,
            "goal": top_prompt,
            "universal_synthesis": True,
            "gcop_event_type": "CAPABILITY_GAP",
            "gcop_capability": capability,
            "sandbox_only": True,
            "live_file_write": False,
            "execution_authority": False,
            "auto_install": False,
            "auto_run": False,
            "promotion_required": True,
        })
        result = self.synthesize_arbitrary_application(normalized)
        return {
            "ok": bool(isinstance(result, dict) and result.get("ok")),
            "schema": "SarahMemory.nailde.gcop_capability_gap.v1",
            "event_type": "CAPABILITY_GAP",
            "capability": capability,
            "goal": top_prompt,
            "result": result,
            "next_stage": "validation_compare_then_explicit_governed_install",
            "sandbox_only": True,
            "live_file_write": False,
            "auto_install": False,
            "auto_run": False,
            "execution_authority": False,
        }

    def _python_qsml_desktop_files(self, spec: Dict[str, Any]) -> Dict[str, str]:
        app_name = str(spec.get("application_name") or "NAILDE Application")
        features = [str(x) for x in list(spec.get("features") or [])[:24]]
        manifest = self._qsml_manifest(spec)
        code = f'''"""QSML-compiled local desktop application scaffold.

Natural-language source:
{str(spec.get('top_prompt') or '')[:1200]}

Generated in NAILDE sandbox. No live SarahMemory authority is granted.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

APP_NAME = {app_name!r}
FEATURES = {features!r}

class Application:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("820x560")
        self.data_path = Path(__file__).with_name("app_data.json")
        self.status = tk.StringVar(value="Ready")
        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=APP_NAME, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="QSML / NAILDE local sandbox application").pack(anchor="w", pady=(0, 10))
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=3)
        ttk.Label(left, text="Requested Features").pack(anchor="w")
        self.feature_list = tk.Listbox(left, height=18)
        self.feature_list.pack(fill="both", expand=True, pady=6)
        for feature in FEATURES or ["application workspace"]:
            self.feature_list.insert("end", feature)
        ttk.Label(right, text="Workspace").pack(anchor="w")
        self.editor = tk.Text(right, wrap="word")
        self.editor.pack(fill="both", expand=True, pady=6)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Save Local Data", command=self.save).pack(side="left")
        ttk.Button(buttons, text="Load Local Data", command=self.load).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear", command=lambda: self.editor.delete("1.0", "end")).pack(side="left")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))

    def save(self) -> None:
        payload = {{"text": self.editor.get("1.0", "end-1c"), "features": FEATURES}}
        self.data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status.set(f"Saved {{self.data_path.name}}")

    def load(self) -> None:
        if not self.data_path.exists():
            self.status.set("No local data saved yet")
            return
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", str(payload.get("text") or ""))
        self.status.set(f"Loaded {{self.data_path.name}}")


def main() -> None:
    root = tk.Tk()
    Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
'''
        readme = f'''# {app_name}

QSML-compiled local desktop application scaffold.

## Natural Language Goal

{spec.get("top_prompt", "")}

## Features

''' + "\n".join(f"- {x}" for x in features) + '''

## Governance

- Sandbox-only generation
- No live CORE/API/UI modification
- Neuron remains route activation owner
- User approval required before installation/execution outside sandbox
'''
        return {
            "sandbox/app/main.py": code,
            "sandbox/app/app_manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/README.md": readme,
        }

    def _python_qsml_cli_files(self, spec: Dict[str, Any]) -> Dict[str, str]:
        app_name = str(spec.get("application_name") or "NAILDE CLI")
        features = [str(x) for x in list(spec.get("features") or [])[:24]]
        manifest = self._qsml_manifest(spec)
        code = f'''"""QSML-compiled local command-line application."""
from __future__ import annotations
import argparse
import json

APP_NAME = {app_name!r}
FEATURES = {features!r}

def main() -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--info", action="store_true", help="Show compiled application information")
    parser.add_argument("text", nargs="*", help="Input to the local sandbox application")
    args = parser.parse_args()
    if args.info:
        print(json.dumps({{"name": APP_NAME, "features": FEATURES, "sandbox_only": True, "execution_authority": False}}, indent=2))
        return 0
    print(" ".join(args.text) if args.text else f"{{APP_NAME}} ready")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''
        return {
            "sandbox/app/main.py": code,
            "sandbox/app/app_manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/README.md": f"# {app_name}\n\n{spec.get('top_prompt','')}\n",
        }

    def _web_qsml_application_files(self, spec: Dict[str, Any]) -> Dict[str, str]:
        app_name = str(spec.get("application_name") or "NAILDE Web App")
        features = [str(x) for x in list(spec.get("features") or [])[:24]]
        cards = "".join(f"<li>{x}</li>" for x in features) or "<li>Application workspace</li>"
        html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{app_name}</title><link rel="stylesheet" href="styles.css"></head>
<body><main><h1>{app_name}</h1><p>QSML / NAILDE local sandbox web application.</p>
<section><h2>Requested Features</h2><ul id="features">{cards}</ul></section>
<section><h2>Workspace</h2><textarea id="workspace" placeholder="Local sandbox workspace"></textarea>
<div><button id="clear">Clear</button><button id="download">Export JSON</button></div></section>
<pre id="status"></pre></main><script src="app.js"></script></body></html>'''
        css = '''body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:900px;margin:auto;padding:2rem}section{background:#1f2937;padding:1rem;border-radius:12px;margin:1rem 0}textarea{width:100%;min-height:220px;background:#0f172a;color:#e5e7eb;border:1px solid #475569;border-radius:8px;padding:.8rem;box-sizing:border-box}button{margin:.6rem .4rem 0 0;padding:.6rem 1rem}'''
        js = '''const workspace=document.getElementById("workspace");const status=document.getElementById("status");document.getElementById("clear").onclick=()=>{workspace.value="";status.textContent="Cleared"};document.getElementById("download").onclick=()=>{const blob=new Blob([JSON.stringify({text:workspace.value,sandboxOnly:true,executionAuthority:false},null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="app_data.json";a.click();URL.revokeObjectURL(a.href);status.textContent="Exported local JSON"};'''
        manifest = self._qsml_manifest(spec)
        return {
            "sandbox/web/index.html": html,
            "sandbox/web/styles.css": css,
            "sandbox/web/app.js": js,
            "sandbox/app_manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/README.md": f"# {app_name}\n\n{spec.get('top_prompt','')}\n",
        }

    def _python_qsml_game_files(self, spec: Dict[str, Any]) -> Dict[str, str]:
        '''Generate a genre-aware local game scaffold from QSML.

        This deliberately avoids the old behavior where unrelated game prompts
        collapsed into a PACMAN-style template.
        '''
        app_name = str(spec.get("application_name") or "NAILDE Game")
        genre = str(spec.get("game_style") or "generic").lower()
        if genre == "maze_chase":
            return self._python_game_addon_files(spec)
        addon_id = str(spec.get("addon_id") or self._safe_addon_id(app_name))
        manifest = {
            "addon_id": addon_id, "id": addon_id, "name": app_name, "type": "application", "version": "0.1.0",
            "description": f"QSML-compiled {genre} game generated in NAILDE.",
            "entrypoint": {"module": "addon", "callable": "run"}, "entry": {"module": "addon", "callable": "run"},
            "execution": {"mode": "subprocess"}, "risk_tier": "low",
            "permissions": ["keyboard_input", "addon_local_file_write"], "denied": list(spec.get("denied") or NAILDE_DENIED_ACTIONS),
            "security": {"trusted": False, "requires_user_run": True, "auto_run_allowed": False},
            "governance": {"created_by": "NAILDE_QSML", "sandbox_first": True, "execution_authority": False, "route_activation_owner": "SarahMemoryNeuron"},
        }
        ui = {"schema":"SarahMemory.addon.ui_manifest.v1","title":app_name,"icon":"PackageCheck","action":"run","runtime":"python_subprocess_manifest","description":manifest["description"],"execution_authority":False}
        addon_py='''"""QSML-generated game addon wrapper."""\nfrom __future__ import annotations\n\ndef run(context=None):\n    from game.sm_game import run_game\n    return run_game(context=context or {})\n\ndef addon_info():\n    return {"ok": True, "type": "qsml_game_addon", "execution_authority": False}\n'''
        if genre == "racing":
            game_code = f'''"""QSML-compiled local racing game."""
from __future__ import annotations
import random
import tkinter as tk

WIDTH, HEIGHT = 640, 520
LANES = [120, 240, 360, 480]
TICK = 35

class RacingGame:
    def __init__(self, root):
        self.root=root; self.root.title({app_name!r})
        self.canvas=tk.Canvas(root,width=WIDTH,height=HEIGHT,bg="#101820",highlightthickness=0); self.canvas.pack()
        self.lane=1; self.score=0; self.speed=6; self.running=True; self.obstacles=[]
        root.bind("<Left>",lambda e:self.move(-1)); root.bind("<Right>",lambda e:self.move(1)); root.bind("<a>",lambda e:self.move(-1)); root.bind("<d>",lambda e:self.move(1)); root.bind("<r>",lambda e:self.restart())
        self.spawn_counter=0; self.loop()
    def move(self,d): self.lane=max(0,min(len(LANES)-1,self.lane+d))
    def restart(self): self.score=0; self.speed=6; self.running=True; self.obstacles=[]
    def loop(self):
        if self.running:
            self.spawn_counter+=1
            if self.spawn_counter%28==0: self.obstacles.append([random.randrange(len(LANES)),-40])
            for o in self.obstacles: o[1]+=self.speed
            self.obstacles=[o for o in self.obstacles if o[1]<HEIGHT+50]
            self.score+=1
            if self.score%350==0: self.speed=min(14,self.speed+1)
            px=LANES[self.lane]
            if any(abs(LANES[o[0]]-px)<30 and HEIGHT-105<o[1]<HEIGHT-35 for o in self.obstacles): self.running=False
        self.render(); self.root.after(TICK,self.loop)
    def render(self):
        c=self.canvas; c.delete("all")
        for x in LANES: c.create_line(x,0,x,HEIGHT,fill="#405060",dash=(14,14))
        px=LANES[self.lane]; c.create_rectangle(px-24,HEIGHT-90,px+24,HEIGHT-35,fill="#00d4ff",outline="white")
        for lane,y in self.obstacles:
            x=LANES[lane]; c.create_rectangle(x-24,y-45,x+24,y+10,fill="#ff5c5c",outline="white")
        c.create_text(12,12,anchor="nw",fill="white",font=("Consolas",14),text=f"Score: {{self.score}}  Speed: {{self.speed}}")
        if not self.running: c.create_text(WIDTH/2,HEIGHT/2,fill="white",font=("Segoe UI",24,"bold"),text="CRASH - press R to restart")

def run_game(context=None):
    root=tk.Tk(); RacingGame(root); root.mainloop(); return {{"ok":True,"game":{app_name!r},"genre":"racing","execution_authority":False}}
'''
        elif genre == "snake":
            game_code = f'''"""QSML-compiled local snake game."""
from __future__ import annotations
import random, tkinter as tk
CELL=20; W=30; H=22; TICK=110
class SnakeGame:
    def __init__(self,root):
        self.root=root; root.title({app_name!r}); self.c=tk.Canvas(root,width=W*CELL,height=H*CELL,bg="#101820"); self.c.pack()
        self.snake=[(10,10),(9,10),(8,10)]; self.direction=(1,0); self.food=(18,10); self.running=True; self.score=0
        for key,d in [("<Up>",(0,-1)),("<Down>",(0,1)),("<Left>",(-1,0)),("<Right>",(1,0))]: root.bind(key,lambda e,d=d:self.set_dir(d))
        self.loop()
    def set_dir(self,d):
        if (d[0]+self.direction[0],d[1]+self.direction[1])!=(0,0): self.direction=d
    def loop(self):
        if self.running:
            x,y=self.snake[0]; dx,dy=self.direction; head=((x+dx)%W,(y+dy)%H)
            if head in self.snake: self.running=False
            else:
                self.snake.insert(0,head)
                if head==self.food: self.score+=10; self.food=(random.randrange(W),random.randrange(H))
                else: self.snake.pop()
        self.render(); self.root.after(TICK,self.loop)
    def render(self):
        self.c.delete("all")
        for x,y in self.snake: self.c.create_rectangle(x*CELL,y*CELL,(x+1)*CELL,(y+1)*CELL,fill="#60e080",outline="#204030")
        x,y=self.food; self.c.create_oval(x*CELL+3,y*CELL+3,(x+1)*CELL-3,(y+1)*CELL-3,fill="#ff5c5c",outline="")
        self.c.create_text(8,8,anchor="nw",fill="white",text=f"Score: {{self.score}}")
        if not self.running:self.c.create_text(W*CELL/2,H*CELL/2,fill="white",font=("Segoe UI",22,"bold"),text="GAME OVER")
def run_game(context=None):
    root=tk.Tk(); SnakeGame(root); root.mainloop(); return {{"ok":True,"game":{app_name!r},"genre":"snake","execution_authority":False}}
'''
        else:
            game_code = f'''"""QSML-compiled generic local game scaffold."""
from __future__ import annotations
import tkinter as tk
WIDTH,HEIGHT=640,480
class Game:
    def __init__(self,root):
        self.root=root; root.title({app_name!r}); self.c=tk.Canvas(root,width=WIDTH,height=HEIGHT,bg="#101820"); self.c.pack()
        self.x,self.y=WIDTH//2,HEIGHT//2; self.score=0
        root.bind("<KeyPress>",self.key); self.render()
    def key(self,e):
        d={{"Left":(-12,0),"Right":(12,0),"Up":(0,-12),"Down":(0,12),"a":(-12,0),"d":(12,0),"w":(0,-12),"s":(0,12)}}.get(e.keysym)
        if d:self.x=max(15,min(WIDTH-15,self.x+d[0]));self.y=max(15,min(HEIGHT-15,self.y+d[1]));self.score+=1;self.render()
    def render(self):
        self.c.delete("all");self.c.create_oval(self.x-14,self.y-14,self.x+14,self.y+14,fill="#00d4ff");self.c.create_text(10,10,anchor="nw",fill="white",text=f"QSML {genre} scaffold | score {{self.score}}")
def run_game(context=None):
    root=tk.Tk();Game(root);root.mainloop();return {{"ok":True,"game":{app_name!r},"genre":{genre!r},"execution_authority":False}}
'''
        readme = f"# {app_name}\n\nQSML-compiled {genre} game.\n\nNatural language source: {spec.get('top_prompt','')}\n"
        return {
            "sandbox/addon_package/manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/addon_package/ui.json": json.dumps(ui, indent=2, sort_keys=True),
            "sandbox/addon_package/addon.py": addon_py,
            "sandbox/addon_package/game/__init__.py": "# QSML generated game package\n",
            "sandbox/addon_package/game/sm_game.py": game_code,
            "sandbox/addon_package/README.md": readme,
        }

    def _files_from_qsml_spec(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """Compatibility bridge into the universal synthesizer.

        This intentionally no longer dispatches prompt-specific templates.
        Callers receive a complete model-synthesized file set or a clear error.
        """
        result = self._universal_synthesize_spec(spec, {})
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "universal_synthesis_failed"))
        return dict(result.get("files") or {})

    def natural_language_code_draft(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        prompt = str(payload.get("prompt") or payload.get("problem") or payload.get("goal") or "Build a NAILDE sandbox addon.").strip()
        workspace_id = str(payload.get("workspace_id") or "").strip()
        if not workspace_id:
            created = self.create_workspace({"goal": prompt, "mode": "BUILD_SANDBOX"})
            workspace_id = str((created.get("workspace") or {}).get("workspace_id") or "")
        target = str(payload.get("target") or "nailde_local_application").lower()
        compiled = self.qsml_compile_request({**payload, "prompt": prompt, "target": target, "workspace_id": workspace_id})
        if not compiled.get("ok"):
            return {
                "ok": False, "schema": "SarahMemory.nailde.code_draft.v2", "workspace_id": workspace_id,
                "error": "qsml_compile_failed", "qsml": compiled,
                "message": "NAILDE did not substitute a canned draft.", "execution_authority": False,
            }
        spec = self._expand_qsml_program(compiled, prompt, "")
        if payload.get("app_name"):
            spec["application_name"] = _safe_name(payload.get("app_name"), "NaildeGeneratedTool")
        app_name = _safe_name(spec.get("application_name") or self._title_from_prompt(prompt), "NaildeGeneratedTool")
        synthesis = self._universal_synthesize_spec(spec, payload)
        if not synthesis.get("ok"):
            return {
                "ok": False, "schema": "SarahMemory.nailde.code_draft.v2", "workspace_id": workspace_id,
                "prompt": prompt, "app_name": app_name, "target": target, "qsml": compiled,
                "spec": spec, "synthesis": synthesis,
                "message": "Universal application synthesis stopped safely rather than returning an unrelated draft.",
                "sandbox_only": True, "execution_authority": False,
            }
        files = dict(synthesis.get("files") or {})
        spec["universal_blueprint"] = synthesis.get("blueprint") or {}
        saved = []
        for rel_path, content in files.items():
            saved.append(self.save_workspace_file({"workspace_id": workspace_id, "path": rel_path, "content": content}))
        addon_package = self._build_universal_addon_package(workspace_id, spec, synthesis, files)
        if not addon_package.get("ok"):
            return {
                "ok": False, "schema": "SarahMemory.nailde.code_draft.v3", "workspace_id": workspace_id,
                "error": "addon_package_build_failed", "addon_package": addon_package, "qsml": compiled, "spec": spec, "synthesis": synthesis,
                "sandbox_only": True, "execution_authority": False,
            }
        spec["addon_package"] = True
        spec["addon_id"] = addon_package.get("addon_id") or spec.get("addon_id")
        spec["addon_runtime"] = addon_package.get("runtime") or {}
        package_file_paths = [str(row.get("path") or "") for row in list((self.workspace_files({"workspace_id": workspace_id}).get("files") or [])) if str(row.get("path") or "").startswith("sandbox/addon_package/")]
        mission = self.agent_mission({"workspace_id": workspace_id, "goal": prompt, "mission_type": "code_generation_review", "target_files": list(files.keys())})
        validation = self.validate_text_artifacts({"workspace_id": workspace_id, "paths": sorted(set(list(files.keys()) + package_file_paths))})
        install_plan = self.addon_install_plan({"workspace_id": workspace_id})
        packet = {
            "ok": True,
            "schema": "SarahMemory.nailde.code_draft.v2",
            "workspace_id": workspace_id,
            "prompt": prompt,
            "app_name": app_name,
            "target": target,
            "qsml": compiled,
            "spec": spec,
            "synthesis": synthesis,
            "addon_package": addon_package,
            "install_plan": install_plan,
            "files": [{"path": path, "content": content, "sha256": _sha256_text(content)} for path, content in files.items()],
            "saved": saved,
            "agent_mission": mission.get("mission"),
            "validation": validation,
            "sandbox_only": True,
            "live_file_write": False,
            "execution_authority": False,
        }
        receipt = self._record_receipt("NAILDE_CODE_DRAFT_CREATED", workspace_id, f"Drafted {len(files)} sandbox code files from natural language.", {"workspace_id": workspace_id, "file_count": len(files), "risk": "low", "verdict": "DRAFTED_SANDBOX"})
        packet["receipt"] = receipt
        return packet

    def agent_mission(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "")
        goal = str(payload.get("goal") or payload.get("task") or "Review NAILDE sandbox project.")[:1200]
        mission_type = str(payload.get("mission_type") or "sandbox_code_assist")
        target_files = payload.get("target_files") if isinstance(payload.get("target_files"), list) else []
        mission = {
            "schema": "SarahMemory.nailde.agent_mission.v1",
            "mission_id": "nailde-mission-" + uuid.uuid4().hex[:12],
            "workspace_id": workspace_id,
            "mission_type": mission_type,
            "goal": goal,
            "target_files": target_files,
            "agent_lane": "terminal_agent_proposal_only",
            "passport_required": True,
            "roachmotel_required": True,
            "compare_required": True,
            "ledger_required": True,
            "allowed_capabilities": ["read_sandbox", "generate_code", "explain_code", "suggest_tests", "summarize_evidence"],
            "denied_capabilities": ["shell", "live_file_write", "credential_access", "device_control", "production_tensor_edit", "self_approval"],
            "ttl_seconds": 600,
            "execution_authority": False,
            "launch_status": "prepared_not_launched",
        }
        return {"ok": True, "mission": mission, "execution_authority": False}

    def validate_text_artifacts(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        paths = payload.get("paths") if isinstance(payload.get("paths"), list) else []
        direct_content = payload.get("content")
        results = []
        if direct_content is not None:
            results.append(self._validate_text(str(direct_content), str(payload.get("path") or "inline.txt")))
        elif workspace_id:
            if not paths:
                listed = self.workspace_files({"workspace_id": workspace_id})
                paths = [row.get("path") for row in listed.get("files", []) if row.get("text_candidate")]
            for rel_path in paths[:80]:
                read = self.read_workspace_file({"workspace_id": workspace_id, "path": rel_path})
                if read.get("ok"):
                    results.append(self._validate_text(str(read.get("content") or ""), str(rel_path)))
                else:
                    results.append({"path": rel_path, "ok": False, "errors": [read.get("error") or "read_failed"]})
        overall = all(bool(r.get("ok")) for r in results) if results else False
        problems: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = []
        for item in results:
            if isinstance(item, dict):
                problems.extend(item.get("problems") if isinstance(item.get("problems"), list) else [])
                tasks.extend(self._tasks_from_validation(item, workspace_id=workspace_id))
        return {"ok": overall, "schema": "SarahMemory.nailde.validation.text.v1", "workspace_id": workspace_id, "results": results, "problems": problems, "tasks": tasks, "execution_authority": False, "live_runtime_verified": False}

    def reconcile_edits(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        original = str(payload.get("original") if payload.get("original") is not None else "")
        edited = str(payload.get("edited") if payload.get("edited") is not None else "")
        path = str(payload.get("path") or "sandbox/edited.txt")
        import difflib
        diff = "\n".join(difflib.unified_diff(original.splitlines(), edited.splitlines(), fromfile="original", tofile="edited", lineterm=""))
        artifact = {
            "schema": "SarahMemory.nailde.reconcile.v1",
            "path": path,
            "changed": original != edited,
            "original_hash": _sha256_text(original),
            "edited_hash": _sha256_text(edited),
            "diff": diff,
            "requires_validation": True,
            "execution_authority": False,
        }
        return {"ok": True, "artifact": artifact, "execution_authority": False}


    # ------------------------------------------------------------------
    # Extreme IDE / floating workbench contracts
    # ------------------------------------------------------------------
    def workbench_layout(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load/save/reset the NAILDE floating window layout.

        Layout persistence is sandbox UI state only. It has no execution authority
        and cannot change live files, devices, or model weights.
        """
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "__global__").strip() or "__global__"
        action = str(payload.get("action") or payload.get("mode") or "load").strip().lower()
        path = self._layout_state_path(workspace_id)
        if action == "reset":
            layout = self._default_floating_layout()
            self._write_json_file(path, layout)
            return {"ok": True, "action": "reset", "workspace_id": workspace_id, "layout": layout, "execution_authority": False}
        if action == "save":
            layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
            if not layout:
                return {"ok": False, "error": "layout_required", "execution_authority": False}
            normalized = self._normalize_floating_layout(layout)
            self._write_json_file(path, normalized)
            receipt = self._record_receipt(
                "NAILDE_LAYOUT_SAVED",
                workspace_id,
                "NAILDE floating window layout saved.",
                {"workspace_id": workspace_id, "risk": "low", "verdict": "UI_STATE_ONLY"},
            )
            return {"ok": True, "action": "save", "workspace_id": workspace_id, "layout": normalized, "receipt": receipt, "execution_authority": False}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return {"ok": True, "action": "load", "workspace_id": workspace_id, "layout": self._normalize_floating_layout(data), "execution_authority": False}
            except Exception as exc:
                return {"ok": False, "error": str(exc), "workspace_id": workspace_id, "layout": self._default_floating_layout(), "execution_authority": False}
        return {"ok": True, "action": "default", "workspace_id": workspace_id, "layout": self._default_floating_layout(), "execution_authority": False}

    def toolbox_catalog(self) -> Dict[str, Any]:
        """Return a drag/drop toolbox catalog for NAILDE.

        The catalog intentionally resembles the role of Visual Studio / VB form
        designers and Access form/report builders, but every item inserts sandbox
        snippets or sandbox model objects only.
        """
        def item(category: str, id_: str, label: str, kind: str, snippet: str, description: str, *, icon: str = "box", form: Optional[Dict[str, Any]] = None, block: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "id": id_,
                "category": category,
                "label": label,
                "kind": kind,
                "icon": icon,
                "description": description,
                "insert_targets": ["code_editor", "form_designer", "blockforge", "sandbox_model"],
                "tsx_snippet": snippet,
                "code_snippet": snippet,
                "form_object": form or {"object_id": id_, "type": label, "label": label, "required": False, "execution_authority": False},
                "block_object": block or {"id": id_, "type": kind, "label": label, "execution_authority": False},
                "sandbox_only": True,
                "execution_authority": False,
                "denied_actions": ["live_file_write", "self_approval", "device_write", "production_tensor_edit", "global_dlpanel_write"],
            }

        categories = [
            {
                "id": "basic_ui",
                "label": "Basic UI Controls",
                "description": "VB-style controls for sandbox React/TSX forms.",
                "items": [
                    item("basic_ui", "label", "Label", "ui_control", "<label className=\"text-sm font-medium\">Label Text</label>", "Static text label."),
                    item("basic_ui", "textbox", "TextBox", "ui_control", "<Input value={value} onChange={(event) => setValue(event.target.value)} placeholder=\"Enter text\" />", "Single-line text input."),
                    item("basic_ui", "textarea", "TextArea", "ui_control", "<Textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder=\"Enter notes\" />", "Multi-line input."),
                    item("basic_ui", "button", "Button", "ui_control", "<Button onClick={() => handleAction()} type=\"button\">Run Action</Button>", "Clickable command button."),
                    item("basic_ui", "checkbox", "Checkbox", "ui_control", "<label className=\"flex items-center gap-2\"><input type=\"checkbox\" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enabled</label>", "Boolean option control."),
                    item("basic_ui", "dropdown", "Dropdown", "ui_control", "<select value={mode} onChange={(event) => setMode(event.target.value)}><option value=\"plan\">Plan</option><option value=\"build\">Build</option></select>", "Select list."),
                    item("basic_ui", "image", "Image", "ui_control", "<img src={previewUrl} alt=\"Sandbox preview\" className=\"rounded border border-border\" />", "Image preview control."),
                ],
            },
            {
                "id": "layout",
                "label": "Layout / Window Controls",
                "description": "Panels, tabs, modals, toolbars, splitters, and floating windows.",
                "items": [
                    item("layout", "panel", "Panel", "layout_control", "<section className=\"rounded-xl border border-border bg-card p-4\">Panel content</section>", "Container panel."),
                    item("layout", "tabview", "TabView", "layout_control", "<div role=\"tablist\" className=\"flex gap-2\"><Button variant=\"outline\">Tab 1</Button><Button variant=\"ghost\">Tab 2</Button></div>", "Simple tabs."),
                    item("layout", "toolbar", "Toolbar", "layout_control", "<div className=\"flex items-center gap-2 border-b border-border p-2\"><Button size=\"sm\">Action</Button></div>", "Toolbar strip."),
                    item("layout", "floating_window", "FloatingWindow", "layout_control", "<WindowFrame title=\"Sandbox Window\">Window content</WindowFrame>", "Floating workbench window object."),
                ],
            },
            {
                "id": "data_access",
                "label": "Data / Microsoft Access Style",
                "description": "Addon-local SQLite tables, bound forms, queries, reports, and grids.",
                "items": [
                    item("data_access", "table_grid", "Table/Grid", "data_control", "<table className=\"w-full text-sm\"><thead><tr><th>Name</th><th>Status</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id}><td>{row.name}</td><td>{row.status}</td></tr>)}</tbody></table>", "Data grid."),
                    item("data_access", "bound_form", "Bound Form", "data_control", "// Bound form model\nconst formBinding = { table: \"items\", fields: [\"name\", \"category\", \"notes\"], sandboxOnly: true };", "Access-style bound form definition."),
                    item("data_access", "sqlite_table", "SQLite Local Table", "sql_schema", "CREATE TABLE IF NOT EXISTS items (\n  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  name TEXT NOT NULL,\n  category TEXT,\n  notes TEXT,\n  created_ts TEXT,\n  updated_ts TEXT\n);", "Addon-local SQLite table schema."),
                    item("data_access", "query_builder", "Query Builder", "data_control", "// Query Builder\nconst query = { table: \"items\", where: [], orderBy: \"created_ts DESC\", sandboxOnly: true };", "Visual query model."),
                    item("data_access", "report", "Report", "data_control", "// Report definition\nconst report = { title: \"Sandbox Report\", source: \"items\", groupBy: \"category\", sandboxOnly: true };", "Access-style report model."),
                    item("data_access", "field_list", "Field List", "data_control", "const fields = [{ name: \"name\", type: \"text\", required: true }, { name: \"notes\", type: \"textarea\" }];", "Fields that can bind to controls."),
                ],
            },
            {
                "id": "ai_agents",
                "label": "AI / Agents",
                "description": "Passported mission, local model, evidence, and return-capture controls.",
                "items": [
                    item("ai_agents", "ask_local_model", "Ask Local Model", "ai_block", "// Ask Local Model\nconst modelRequest = { role: \"code_drafting\", target: \"sandbox_only\", execution_authority: false };", "Local model request block."),
                    item("ai_agents", "agent_mission", "Send Passported Agent", "agent_block", "// Passported Agent Mission\nconst mission = { passport_required: true, roachmotel_required: true, scope: [\"read_sandbox\"], execution_authority: false };", "Agent mission proposal only."),
                    item("ai_agents", "capture_return", "Capture Agent Return", "agent_block", "// Capture Agent Return\nconst capture = { compare_required: true, ledger_required: true, execution_authority: false };", "RoachMotel return capture."),
                    item("ai_agents", "model_selector", "Model Selector", "ai_control", "<select value={modelId} onChange={(event) => setModelId(event.target.value)}><option>Qwen2.5-Coder-3B-Instruct</option></select>", "Model Bay selector."),
                ],
            },
            {
                "id": "governance",
                "label": "Governance Gates",
                "description": "Required gates for safe sandbox promotion.",
                "items": [
                    item("governance", "require_approval", "Require Approval", "governance_block", "const approvalGate = { required: true, confirmations: 2, final_authority: \"USER\", execution_authority: false };", "User approval gate."),
                    item("governance", "run_compare", "Run Compare", "governance_block", "const compareGate = { compare_required: true, status: \"pending\", execution_authority: false };", "Compare gate."),
                    item("governance", "record_ledger", "Record Ledger", "governance_block", "const ledgerReceipt = { domain: \"nailde\", action: \"sandbox_event\", immutable: true, execution_authority: false };", "Ledger receipt model."),
                    item("governance", "backup_zip", "Create Backup ZIP", "governance_block", "const backupGate = { backup_zip_required: true, verify_hash: true, double_confirm: true, execution_authority: false };", "Backup gate."),
                    item("governance", "rollback", "Rollback Viewer", "governance_block", "const rollbackPlan = { source: \"verified_backup_zip\", user_approval_required: true, execution_authority: false };", "Rollback model."),
                ],
            },
            {
                "id": "device_xr",
                "label": "Device / XR / HoloForge",
                "description": "Read-only device nodes and XR spatial design objects.",
                "items": [
                    item("device_xr", "com_port", "COM Port Node", "device_node", "const comNode = { mode: \"READ_ONLY_DISCOVERY\", write_locked: true, execution_authority: false };", "COM/USB read-only witness."),
                    item("device_xr", "arduino", "Arduino Node", "device_node", "const arduinoNode = { board: \"unknown\", port: \"unselected\", upload_allowed: false, execution_authority: false };", "Arduino design node."),
                    item("device_xr", "plc", "PLC ReadOnly Node", "device_node", "const plcNode = { mode: \"READ_ONLY\", write_plc_logic: false, high_risk_gate_required: true };", "PLC witness node."),
                    item("device_xr", "physics_body", "Physics Body", "xr_object", "const physicsBody = { massKg: 1, collider: \"box\", simulation_only: true, execution_authority: false };", "Simulation-only physical object."),
                    item("device_xr", "xr_panel", "XR Floating Panel", "xr_object", "const xrPanel = { surface: \"HoloForge\", position: [0, 1.4, -2], sandbox_only: true };", "Spatial coding panel."),
                ],
            },
            {
                "id": "debug_validation",
                "label": "Debug / Validation",
                "description": "Problems, output, logs, sandbox status, and validation controls.",
                "items": [
                    item("debug_validation", "log_console", "LogConsole", "debug_control", "<pre className=\"rounded border border-border bg-muted p-2 text-xs\">{logs.join(\"\\n\")}</pre>", "Log output."),
                    item("debug_validation", "test_runner", "TestRunner", "debug_control", "const testPlan = { syntax: \"pending\", imports: \"pending\", ui: \"pending\", runtime: \"not_verified\" };", "Test plan object."),
                    item("debug_validation", "error_panel", "ErrorPanel", "debug_control", "<div className=\"rounded border border-destructive p-2 text-sm\">{errorMessage}</div>", "Error display."),
                    item("debug_validation", "sandbox_status", "SandboxStatus", "debug_control", "const sandboxStatus = { live_file_write: false, shell: false, device_write: false, production_tensor_edit: false };", "Boundary status."),
                ],
            },
        ]
        flat_items: List[Dict[str, Any]] = []
        for category in categories:
            for entry in category.get("items", []):
                flat_items.append(entry)
        block_types = [
            "WHEN screen loads", "WHEN button clicked", "VALIDATE field", "READ addon database", "WRITE addon database",
            "CALL local API", "ASK local LLM", "SEND passported agent", "CAPTURE agent return", "RUN sandbox test",
            "RUN Compare", "RECORD Ledger receipt", "REQUIRE user approval", "IF condition", "LOOP over records", "HANDLE error", "RETURN value",
        ]
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.toolbox.v2.vb_access_dragdrop",
            "module": MODULE_NAME,
            "ts": _now_iso(),
            "categories": categories,
            "items": flat_items,
            "block_types": block_types,
            "typed_sockets": ["event_output", "data_input", "data_output", "approval_required", "sandbox_file", "agent_mission", "validation_result", "device_witness", "xr_object"],
            "drop_contract": {
                "drag_mime": "application/x-nailde-toolbox-item",
                "allowed_drop_targets": ["code_editor", "form_designer", "blockforge", "sandbox_model"],
                "live_file_write": False,
                "execution_authority": False,
            },
            "doctrine": {
                "toolbox_items_are_templates": True,
                "drag_drop_creates_sandbox_snippet_or_model_object": True,
                "no_live_apply_from_drag_drop": True,
                "user_validation_required": True,
            },
            "execution_authority": False,
            "sandbox_only": True,
        }

    def command_dispatch(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch a NAILDE UI command through a safe command router.

        This router only calls sandbox/read-only/advisory handlers. Unknown or
        dangerous commands return a blocked/proposal response instead of executing.
        """
        payload = payload if isinstance(payload, dict) else {}
        command = str(payload.get("command") or payload.get("id") or "").strip()
        args = payload.get("args") if isinstance(payload.get("args"), dict) else payload
        safe_view_commands = {"show_explorer", "show_search", "show_flow_graph", "show_blockforge", "show_form_designer", "show_device_bay", "show_holoforge", "show_output", "show_doctrine", "show_shortcuts", "show_boundaries", "show_settings", "show_github_settings", "show_filesystem", "show_diagnostics"}
        blocked_apply_commands = {"stage_devbridge_proposal", "create_backup_zip", "prepare_rollback", "package_addon_pending_review", "validate_sandbox", "compare_sandbox", "assurance_review"}
        try:
            if command in {"create_workspace", "workspace.new"}:
                return {"ok": True, "command": command, "result": self.create_workspace(args), "execution_authority": False}
            if command in {"natural_language_code_draft", "nl.draft"}:
                return {"ok": True, "command": command, "result": self.natural_language_code_draft(args), "execution_authority": False}
            if command in {"qsml_compile", "qsml.compile", "compile_natural_program"}:
                return {"ok": True, "command": command, "result": self.qsml_compile_request(args), "execution_authority": False}
            if command in {"auto_build_from_prompt", "nl.build", "qsml.build"}:
                return {"ok": True, "command": command, "result": self.auto_build_from_prompt(args), "execution_authority": False}
            if command in {"synthesize_arbitrary_application", "qsml.synthesize", "app.build", "universal.build"}:
                return {"ok": True, "command": command, "result": self.synthesize_arbitrary_application(args), "execution_authority": False}
            if command in {"gcop.capability_gap", "capability_gap", "gcop_capability_gap"}:
                return {"ok": True, "command": command, "result": self.handle_gcop_capability_gap(args), "execution_authority": False}
            if command in {"thought_loop", "run.thought"}:
                return {"ok": True, "command": command, "result": self.thought_loop(args), "execution_authority": False}
            if command in {"weightlab_simulate", "run.weightlab"}:
                return {"ok": True, "command": command, "result": self.weightlab_simulate(args), "execution_authority": False}
            if command in {"save_workspace_file", "file.save_sandbox"}:
                return {"ok": True, "command": command, "result": self.save_workspace_file(args), "execution_authority": False}
            if command in {"package_workspace_export", "workspace.export", "file.export", "app.export"}:
                return {"ok": True, "command": command, "result": self.package_workspace_export(args), "execution_authority": False}
            if command in {"reconcile_edits", "edit.reconcile"}:
                return {"ok": True, "command": command, "result": self.reconcile_edits(args), "execution_authority": False}
            if command in {"editor_validate", "validate_editor"}:
                return {"ok": True, "command": command, "result": self.editor_validate(args), "execution_authority": False}
            if command in {"create_application_from_editor", "editor.create_application"}:
                return {"ok": True, "command": command, "result": self.create_application_from_editor(args), "execution_authority": False}
            if command in {"filesystem_map", "show_filesystem"}:
                return {"ok": True, "command": command, "result": self.filesystem_map(args), "execution_authority": False}
            if command in {"settings_state", "settings.open", "show_settings"}:
                return {"ok": True, "command": command, "result": self.settings_state(args), "execution_authority": False}
            if command in {"github_plan", "settings.github", "show_github_settings"}:
                return {"ok": True, "command": command, "result": self.github_plan(args), "execution_authority": False}
            if command in {"prepare_agent_mission", "agent.plan", "prepare_internal_agent", "prepare_api_agent", "prepare_web_agent", "prepare_swarm_group"}:
                return {"ok": True, "command": command, "result": self.agent_mission(args), "execution_authority": False}
            if command in safe_view_commands:
                return {"ok": True, "command": command, "ui_only": True, "message": "Open or focus the requested NAILDE panel.", "execution_authority": False}
            if command in blocked_apply_commands:
                return {
                    "ok": False,
                    "blocked": True,
                    "command": command,
                    "reason": "This command requires the existing DevBridge/Compare/Ledger/backup/user-confirmation path and is not executed by NAILDE command_dispatch.",
                    "execution_authority": False,
                }
            return {"ok": False, "blocked": True, "command": command, "reason": "unknown_or_unimplemented_command", "execution_authority": False}
        except Exception as exc:
            return {"ok": False, "command": command, "error": str(exc), "execution_authority": False}

    def search_workspace(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        query = str(payload.get("query") or payload.get("q") or "").strip()
        max_files = max(1, min(200, int(payload.get("max_files") or 80)))
        max_matches = max(1, min(1000, int(payload.get("max_matches") or 200)))
        if not workspace_id or not query:
            return {"ok": False, "error": "workspace_id_and_query_required", "execution_authority": False}
        listed = self.workspace_files({"workspace_id": workspace_id})
        if not listed.get("ok"):
            return listed
        matches: List[Dict[str, Any]] = []
        q_lower = query.lower()
        for row in (listed.get("files") or [])[:max_files]:
            if not row.get("text_candidate"):
                continue
            rel_path = str(row.get("path") or "")
            read = self.read_workspace_file({"workspace_id": workspace_id, "path": rel_path})
            if not read.get("ok"):
                continue
            lines = str(read.get("content") or "").splitlines()
            for idx, line in enumerate(lines, start=1):
                if q_lower in line.lower():
                    matches.append({"path": rel_path, "line": idx, "preview": line[:300]})
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break
        return {"ok": True, "workspace_id": workspace_id, "query": query, "matches": matches, "execution_authority": False}

    def scaffold_extreme_project(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a richer multi-surface sandbox project scaffold."""
        payload = payload if isinstance(payload, dict) else {}
        goal = str(payload.get("goal") or payload.get("prompt") or "Build an extreme NAILDE sandbox project.").strip()
        created = self.create_workspace({"goal": goal, "mode": "BUILD_SANDBOX", "workspace_id": payload.get("workspace_id")})
        if not created.get("ok"):
            return created
        workspace_id = str((created.get("workspace") or {}).get("workspace_id") or "")
        draft = self.natural_language_code_draft({"workspace_id": workspace_id, "prompt": goal, "target": payload.get("target") or "extreme_workbench_addon"})
        graph = {
            "schema": "SarahMemory.nailde.graph.v1",
            "nodes": [
                {"id": "user_request", "label": "User Request", "type": "input"},
                {"id": "nailde_planner", "label": "NAILDE Planner", "type": "governed_planner"},
                {"id": "agent_mission", "label": "Agent Mission", "type": "passport_required"},
                {"id": "sandbox_code", "label": "Sandbox Code", "type": "sandbox_file"},
                {"id": "validation", "label": "Validation Gate", "type": "governance_gate"},
                {"id": "pending_addon", "label": "Pending Addon", "type": "package"},
            ],
            "edges": [
                {"from": "user_request", "to": "nailde_planner", "type": "requests"},
                {"from": "nailde_planner", "to": "agent_mission", "type": "uses_agent"},
                {"from": "agent_mission", "to": "sandbox_code", "type": "proposes_patch"},
                {"from": "sandbox_code", "to": "validation", "type": "validates_with_compare"},
                {"from": "validation", "to": "pending_addon", "type": "stages_devbridge"},
            ],
            "execution_authority": False,
        }
        blockgraph = {
            "schema": "SarahMemory.nailde.blockgraph.v1",
            "blocks": [
                {"id": "when_build_clicked", "type": "event", "label": "WHEN Build Sandbox clicked"},
                {"id": "require_approval", "type": "governance", "label": "REQUIRE user approval before live apply"},
                {"id": "run_validation", "type": "validation", "label": "RUN sandbox validation"},
                {"id": "record_ledger", "type": "governance", "label": "RECORD Ledger receipt"},
            ],
            "connections": [["when_build_clicked", "run_validation"], ["run_validation", "record_ledger"], ["record_ledger", "require_approval"]],
            "execution_authority": False,
        }
        form = {
            "schema": "SarahMemory.nailde.form_designer.v1",
            "objects": [
                {"object_id": "txtGoal", "type": "TextArea", "label": "Goal", "bound_field": "project.goal", "required": True},
                {"object_id": "btnBuild", "type": "Button", "label": "Build Sandbox", "event": "onClick"},
                {"object_id": "gridFiles", "type": "Table/Grid", "label": "Sandbox Files", "source": "workspace_files"},
            ],
            "execution_authority": False,
        }
        device = {
            "schema": "SarahMemory.nailde.device_bay.v1",
            "mode": "READ_ONLY_DISCOVERY",
            "blocked_actions": ["upload_firmware", "write_plc_logic", "toggle_outputs", "start_motors"],
            "execution_authority": False,
        }
        saved_extra = [
            self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/graph/nailde_graph.json", "content": json.dumps(graph, indent=2, sort_keys=True)}),
            self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/blocks/blockgraph.json", "content": json.dumps(blockgraph, indent=2, sort_keys=True)}),
            self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/forms/form_designer.json", "content": json.dumps(form, indent=2, sort_keys=True)}),
            self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/device_bay/device_contract.json", "content": json.dumps(device, indent=2, sort_keys=True)}),
        ]
        validation = self.validate_text_artifacts({"workspace_id": workspace_id})
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "workspace": created.get("workspace"),
            "draft": draft,
            "extra_saved": saved_extra,
            "validation": validation,
            "sandbox_only": True,
            "execution_authority": False,
        }

    # ------------------------------------------------------------------
    # Extreme IDE helper methods
    # ------------------------------------------------------------------
    def _layout_state_path(self, workspace_id: str) -> str:
        safe = _safe_name(workspace_id or "__global__", "__global__")
        if safe == "__global__":
            root = _ensure_dir(os.path.join(self.nailde_dir, "ui_state"))
        else:
            root = _ensure_dir(os.path.join(self._workspace_root(safe), "ui_state"))
        return os.path.join(root, "floating_layout.json")

    @staticmethod
    def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
        _ensure_dir(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    def _default_floating_layout(self) -> Dict[str, Any]:
        windows = [
            ("battle_plan", "Battle Plan", 72, 86, 320, 470),
            ("explorer", "Sandbox Explorer", 96, 188, 340, 440),
            ("prompt", "Natural Language Prompt", 456, 86, 560, 230),
            ("editor", "Code Editor", 1038, 86, 620, 560),
            ("output", "Output / Evidence", 456, 338, 560, 310),
            ("sdk", "SDK Library", 72, 580, 400, 310),
            ("agents", "Agent Mission Bay", 496, 670, 430, 260),
            ("weightlab", "WeightLab", 948, 670, 430, 260),
            ("toolbox", "Visual Toolbox", 1398, 670, 360, 260),
            ("device_bay", "Device Bay", 1398, 356, 360, 280),
            ("holoforge", "HoloForge / XR", 948, 356, 430, 280),
            ("validation", "Validation", 72, 914, 860, 260),
            ("terminal", "Governed Terminal Output", 948, 954, 810, 220),
        ]
        return {
            "schema": "SarahMemory.nailde.floating_layout.v1",
            "updated_at": _now_iso(),
            "windows": {
                wid: {"id": wid, "title": title, "x": x, "y": y, "w": w, "h": h, "z": index + 1, "open": True, "minimized": False, "maximized": False, "dock": "float"}
                for index, (wid, title, x, y, w, h) in enumerate(windows)
            },
            "execution_authority": False,
        }

    @staticmethod
    def _normalize_floating_layout(layout: Dict[str, Any]) -> Dict[str, Any]:
        raw_windows = layout.get("windows") if isinstance(layout.get("windows"), dict) else {}
        normalized: Dict[str, Any] = {}
        for key, value in raw_windows.items():
            if not isinstance(value, dict):
                continue
            wid = _safe_name(value.get("id") or key, "panel")
            normalized[wid] = {
                "id": wid,
                "title": str(value.get("title") or wid.replace("_", " ").title())[:80],
                "x": max(0, min(3200, int(float(value.get("x", 80) or 80)))),
                "y": max(0, min(2400, int(float(value.get("y", 80) or 80)))),
                "w": max(220, min(2400, int(float(value.get("w", 420) or 420)))),
                "h": max(160, min(1800, int(float(value.get("h", 320) or 320)))),
                "z": max(1, min(9999, int(float(value.get("z", 1) or 1)))),
                "open": bool(value.get("open", True)),
                "minimized": bool(value.get("minimized", False)),
                "maximized": bool(value.get("maximized", False)),
                "dock": str(value.get("dock") or "float"),
            }
        if not normalized:
            # Avoid recursion into default; keep minimal valid layout.
            normalized = {"battle_plan": {"id": "battle_plan", "title": "Battle Plan", "x": 80, "y": 80, "w": 360, "h": 420, "z": 1, "open": True, "minimized": False, "maximized": False, "dock": "float"}}
        return {"schema": "SarahMemory.nailde.floating_layout.v1", "updated_at": _now_iso(), "windows": normalized, "execution_authority": False}

    # ------------------------------------------------------------------
    # Settings / diagnostics helpers
    # ------------------------------------------------------------------
    def _is_nailde_path(self, path: str) -> bool:
        try:
            p = os.path.abspath(path)
            roots = [os.path.abspath(self.nailde_dir), os.path.abspath(os.path.join(self.data_dir, "addons", "pending")), os.path.abspath(os.path.join(self.data_dir, "devbridge", "staged"))]
            return any(p == root or p.startswith(root + os.sep) for root in roots)
        except Exception:
            return False

    def _addons_root(self) -> str:
        try:
            value = getattr(config, "ADDONS_DIR", None) if config else None
            if value:
                return _ensure_dir(os.path.abspath(os.path.expanduser(str(value))))
        except Exception:
            pass
        # Canonical fallback is data/addons so NAILDE-created local apps,
        # the Addons UI registry, and future PowerStore staging agree by default.
        return _ensure_dir(os.path.join(self.data_dir, "addons"))

    @staticmethod
    def _safe_addon_id(value: Any) -> str:
        raw = str(value or "nailde_app").strip().replace("\\", "/").split("/")[-1]
        raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).replace("..", "_").strip("._-")
        if not raw:
            raw = f"nailde_app_{uuid.uuid4().hex[:8]}"
        return raw[:96]

    @staticmethod
    def _confirmed(payload: Dict[str, Any]) -> bool:
        for key in ("confirm", "confirmed", "user_confirmed", "user_authorized", "approved", "explicit_user_approval"):
            value = payload.get(key)
            if value is True:
                return True
            if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "approved", "confirm", "confirmed", "user_approved"}:
                return True
        phrase = str(payload.get("confirm_phrase") or "").strip().upper()
        return phrase in {"I APPROVE", "USER APPROVED", "INSTALL ADDON", "APPROVE ADDON INSTALL"}

    @staticmethod
    def _read_json_file(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _zip_backup_dir(self, folder: str, label: str = "backup") -> Dict[str, Any]:
        backup_root = _ensure_dir(os.path.join(self.data_dir, "backup", "addons"))
        safe = self._safe_addon_id(label)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_path = os.path.join(backup_root, f"{safe}_{stamp}.zip")
        count = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
            for dirpath, dirnames, filenames in os.walk(folder):
                dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    if os.path.islink(path):
                        continue
                    rel = os.path.relpath(path, folder).replace("\\", "/")
                    zf.write(path, rel)
                    count += 1
        digest = ""
        try:
            h = hashlib.sha256()
            with open(zip_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    if not chunk:
                        break
                    h.update(chunk)
            digest = h.hexdigest()
        except Exception:
            pass
        return {"path": zip_path, "sha256": digest, "file_count": count}

    def _copy_tree_bounded(self, source: str, target: str, *, max_files: int = 800, max_total_bytes: int = 25 * 1024 * 1024) -> Dict[str, Any]:
        source_abs = os.path.abspath(source)
        target_abs = os.path.abspath(target)
        if not os.path.isdir(source_abs):
            raise FileNotFoundError(source_abs)
        files = []
        total = 0
        for dirpath, dirnames, filenames in os.walk(source_abs):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
            for name in filenames:
                if name in {".env", "id_rsa", "id_dsa"} or name.lower().endswith((".pem", ".key")):
                    continue
                src = os.path.join(dirpath, name)
                if os.path.islink(src):
                    continue
                size = os.path.getsize(src)
                total += size
                if len(files) >= max_files or total > max_total_bytes:
                    raise RuntimeError("addon_copy_budget_exceeded")
                rel = os.path.relpath(src, source_abs).replace("\\", "/")
                if rel.startswith("../") or "/../" in rel:
                    continue
                files.append((src, rel, size))
        os.makedirs(target_abs, exist_ok=True)
        copied = 0
        for src, rel, _size in files:
            dst = os.path.join(target_abs, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        return {"files_copied": copied, "bytes_copied": total, "source": source_abs, "target": target_abs}


    @staticmethod
    def _addon_source_relative_path(path: str) -> str:
        raw = str(path or "").replace("\\", "/").strip()
        raw = re.sub(r"^\./+", "", raw)
        if raw.startswith("sandbox/"):
            raw = raw[len("sandbox/"):]
        raw = raw.lstrip("/")
        parts = [part for part in raw.split("/") if part not in {"", "."}]
        if not parts or ".." in parts:
            raise ValueError("invalid_generated_application_path")
        return "/".join(parts)

    def _infer_universal_addon_runtime(self, blueprint: Dict[str, Any], source_files: Dict[str, str]) -> Dict[str, Any]:
        """Infer a contained Addons runtime from the QSML application blueprint.

        This is packaging/runtime adaptation only. It does not select application
        behavior, grant authority, install dependencies, or execute the program.
        """
        files = [str(path).replace("\\", "/") for path in source_files.keys()]
        run = blueprint.get("run") if isinstance(blueprint.get("run"), dict) else {}
        declared = str(run.get("entrypoint") or "").replace("\\", "/").strip()
        args = [str(x)[:500] for x in list(run.get("arguments") or [])[:32]]

        def _find(*suffixes: str) -> str:
            lowered = [(path, path.lower()) for path in files]
            for suffix in suffixes:
                for path, low in lowered:
                    if low.endswith(suffix.lower()):
                        return path
            return ""

        entry = declared if declared in files else ""
        if not entry and declared:
            declared_tail = declared.split("sandbox/", 1)[-1]
            matches = [path for path in files if path.split("sandbox/", 1)[-1] == declared_tail]
            entry = matches[0] if len(matches) == 1 else ""
        if not entry:
            entry = _find("/main.py", "main.py", "/app.py", "app.py", "/index.html", "index.html", "/main.js", "main.js", "/index.js", "index.js")

        mode = "source_only"
        runnable = False
        reason = "No directly runnable text entrypoint was declared by the QSML blueprint."
        low = entry.lower()
        if low.endswith(".py"):
            mode, runnable, reason = "python", True, "Python entrypoint packaged for the current SarahMemory interpreter."
        elif low.endswith((".html", ".htm")):
            mode, runnable, reason = "static_web", True, "Static web entrypoint packaged for a localhost Addons server."
        elif low.endswith((".js", ".mjs", ".cjs")):
            mode, runnable, reason = "node", True, "JavaScript entrypoint requires a locally installed Node runtime."
        elif low.endswith(".jar"):
            mode, runnable, reason = "java_jar", True, "JAR entrypoint requires a locally installed Java runtime."
        elif low.endswith(".exe"):
            mode, runnable, reason = "windows_executable", True, "Contained Windows executable entrypoint."

        packaged_entry = ""
        if entry:
            try:
                packaged_entry = "app/" + self._addon_source_relative_path(entry)
            except Exception:
                packaged_entry = ""
                mode, runnable, reason = "source_only", False, "Declared entrypoint was not safely containable."
        return {
            "schema": "SarahMemory.nailde.addon_runtime.v1",
            "mode": mode,
            "runnable": bool(runnable and packaged_entry),
            "source_entrypoint": entry,
            "entrypoint": packaged_entry,
            "arguments": args,
            "reason": reason,
            "local_only": True,
            "auto_run_allowed": False,
            "execution_authority": False,
        }

    @staticmethod
    def _universal_addon_wrapper_source() -> str:
        return '"""NAILDE generated universal Addons runtime wrapper.\n\nThe wrapper launches only the application packaged inside this Addon directory.\nIt never writes SarahMemory CORE/API/UI files and never auto-runs.\n"""\nfrom __future__ import annotations\n\nimport functools\nimport json\nimport os\nimport shutil\nimport subprocess\nimport sys\nimport webbrowser\nfrom http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer\nfrom pathlib import Path\nfrom typing import Any, Dict\n\n_ROOT = Path(__file__).resolve().parent\n_APP_ROOT = (_ROOT / "app").resolve()\n_RUNTIME_FILE = _ROOT / "app_runtime.json"\n_SESSION: Dict[str, Any] = {"status": "idle", "runs": 0, "last_result": None}\n\n\ndef _runtime() -> Dict[str, Any]:\n    try:\n        value = json.loads(_RUNTIME_FILE.read_text(encoding="utf-8-sig"))\n        return value if isinstance(value, dict) else {}\n    except Exception as exc:\n        return {"mode": "source_only", "runnable": False, "error": str(exc)}\n\n\ndef _contained_entry(rel: str) -> Path:\n    raw = str(rel or "").replace("\\\\", "/").lstrip("/")\n    if not raw or ".." in raw.split("/"):\n        raise RuntimeError("invalid_addon_runtime_entrypoint")\n    target = (_ROOT / raw).resolve()\n    root = _ROOT.resolve()\n    try:\n        if os.path.commonpath([str(root), str(target)]) != str(root):\n            raise RuntimeError("runtime_entrypoint_outside_addon")\n    except ValueError:\n        raise RuntimeError("runtime_entrypoint_outside_addon")\n    if not target.exists():\n        raise FileNotFoundError(str(target))\n    return target\n\n\ndef _subprocess_result(proc: subprocess.CompletedProcess, mode: str) -> Dict[str, Any]:\n    return {"ok": proc.returncode == 0, "mode": mode, "returncode": int(proc.returncode), "execution_authority": False}\n\n\ndef _child_env() -> Dict[str, str]:\n    env = dict(os.environ)\n    prior = str(env.get("PYTHONPATH") or "")\n    env["PYTHONPATH"] = str(_APP_ROOT) + (os.pathsep + prior if prior else "")\n    env["SARAHMEMORY_ADDON_ROOT"] = str(_ROOT)\n    env["SARAHMEMORY_ADDON_APP_ROOT"] = str(_APP_ROOT)\n    return env\n\n\ndef run(context=None, payload=None):\n    cfg = _runtime()\n    mode = str(cfg.get("mode") or "source_only").strip().lower()\n    if not bool(cfg.get("runnable")):\n        result = {"ok": False, "blocked": True, "mode": mode, "error": "application_is_source_only", "execution_authority": False}\n        _SESSION.update({"status": "source_only", "last_result": result})\n        return result\n    entry = _contained_entry(str(cfg.get("entrypoint") or ""))\n    args = [str(x) for x in list(cfg.get("arguments") or [])[:32]]\n    _SESSION["runs"] = int(_SESSION.get("runs") or 0) + 1\n    _SESSION["status"] = "running"\n    if mode == "python":\n        proc = subprocess.run([sys.executable, str(entry), *args], cwd=str(_APP_ROOT), env=_child_env(), shell=False, check=False)\n        result = _subprocess_result(proc, mode)\n    elif mode == "node":\n        node = shutil.which("node")\n        if not node:\n            result = {"ok": False, "blocked": True, "mode": mode, "error": "node_runtime_not_found", "execution_authority": False}\n        else:\n            proc = subprocess.run([node, str(entry), *args], cwd=str(_APP_ROOT), shell=False, check=False)\n            result = _subprocess_result(proc, mode)\n    elif mode == "java_jar":\n        java = shutil.which("java")\n        if not java:\n            result = {"ok": False, "blocked": True, "mode": mode, "error": "java_runtime_not_found", "execution_authority": False}\n        else:\n            proc = subprocess.run([java, "-jar", str(entry), *args], cwd=str(_APP_ROOT), shell=False, check=False)\n            result = _subprocess_result(proc, mode)\n    elif mode == "windows_executable":\n        if os.name != "nt" or entry.suffix.lower() != ".exe":\n            result = {"ok": False, "blocked": True, "mode": mode, "error": "windows_executable_not_supported_on_this_host", "execution_authority": False}\n        else:\n            proc = subprocess.run([str(entry), *args], cwd=str(entry.parent), shell=False, check=False)\n            result = _subprocess_result(proc, mode)\n    elif mode == "static_web":\n        serve_dir = entry.parent\n        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(serve_dir))\n        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)\n        host, port = server.server_address[:2]\n        url = f"http://{host}:{port}/{entry.name}"\n        try:\n            webbrowser.open(url, new=1, autoraise=True)\n        except Exception:\n            pass\n        _SESSION.update({"status": "serving", "url": url})\n        try:\n            server.serve_forever(poll_interval=0.35)\n        finally:\n            server.server_close()\n        result = {"ok": True, "mode": mode, "url": url, "execution_authority": False}\n    else:\n        result = {"ok": False, "blocked": True, "mode": mode, "error": "unsupported_addon_runtime_mode", "execution_authority": False}\n    _SESSION["last_result"] = result\n    _SESSION["status"] = "completed" if bool(result.get("ok")) else "failed"\n    return result\n\n\ndef addon_init(context=None, config=None):\n    return run(context=context or {})\n\n\ndef addon_info():\n    cfg = _runtime()\n    return {"ok": True, "id": cfg.get("addon_id"), "name": cfg.get("application_name"), "runtime": cfg, "session": dict(_SESSION), "execution_authority": False}\n\n\ndef addon_status(context=None):\n    return {"ok": True, "session": dict(_SESSION), "runtime": _runtime(), "execution_authority": False}\n\n\ndef addon_shutdown(context=None):\n    _SESSION["status"] = "stopped"\n    return True\n'

    def _validate_installable_addon_package(self, source_path: str) -> Dict[str, Any]:
        """Validate a NAILDE-generated Addons package without importing/executing it."""
        source_abs = os.path.abspath(str(source_path or ""))
        errors: List[str] = []
        warnings: List[str] = []
        if not source_abs or not os.path.isdir(source_abs):
            return {"ok": False, "error": "addon_source_missing", "source_path": source_abs, "errors": ["addon_source_missing"], "execution_authority": False}
        manifest = self._read_json_file(os.path.join(source_abs, "manifest.json"))
        ui = self._read_json_file(os.path.join(source_abs, "ui.json"))
        runtime = self._read_json_file(os.path.join(source_abs, "app_runtime.json"))
        addon_py_path = os.path.join(source_abs, "addon.py")
        addon_py = self._read_text_file(addon_py_path)
        if not manifest:
            errors.append("manifest_json_missing_or_invalid")
        if not ui:
            errors.append("ui_json_missing_or_invalid")
        if not runtime:
            errors.append("app_runtime_json_missing_or_invalid")
        if not addon_py:
            errors.append("addon_py_missing")
        addon_id = self._safe_addon_id(manifest.get("addon_id") or manifest.get("id") or os.path.basename(source_abs)) if manifest else ""
        if manifest:
            declared_id = str(manifest.get("addon_id") or manifest.get("id") or "")
            if not declared_id or addon_id != declared_id:
                errors.append("manifest_addon_id_not_safe_or_missing")
            entry = manifest.get("entrypoint") if isinstance(manifest.get("entrypoint"), dict) else manifest.get("entry") if isinstance(manifest.get("entry"), dict) else {}
            if str(entry.get("module") or "") != "addon" or str(entry.get("callable") or "") != "addon_init":
                errors.append("manifest_entrypoint_must_be_addon_addon_init")
            execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
            if str(execution.get("mode") or "").strip().lower() != "subprocess":
                errors.append("manifest_execution_mode_must_be_subprocess")
            security = manifest.get("security") if isinstance(manifest.get("security"), dict) else {}
            if bool(security.get("auto_run_allowed")):
                errors.append("addon_auto_run_must_be_false")
        if addon_py:
            try:
                tree = ast.parse(addon_py, filename=addon_py_path)
                functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for required in ("addon_init", "run", "addon_info"):
                    if required not in functions:
                        errors.append(f"addon_wrapper_missing_{required}")
            except SyntaxError as exc:
                errors.append(f"addon_wrapper_python_syntax:{exc.msg}:line_{exc.lineno}")
        runtime_mode = str(runtime.get("mode") or "source_only").strip().lower() if runtime else "source_only"
        runnable = bool(runtime.get("runnable")) if runtime else False
        runtime_entry = str(runtime.get("entrypoint") or "").replace("\\", "/").strip() if runtime else ""
        if runnable:
            if runtime_mode not in {"python", "static_web", "node", "java_jar", "windows_executable"}:
                errors.append("unsupported_runnable_runtime_mode")
            if not runtime_entry or runtime_entry.startswith("/") or ".." in runtime_entry.split("/"):
                errors.append("invalid_runtime_entrypoint")
            else:
                entry_abs = os.path.abspath(os.path.join(source_abs, *runtime_entry.split("/")))
                try:
                    if os.path.commonpath([source_abs, entry_abs]) != source_abs:
                        errors.append("runtime_entrypoint_outside_addon")
                    elif not os.path.isfile(entry_abs):
                        errors.append("runtime_entrypoint_missing")
                except Exception:
                    errors.append("runtime_entrypoint_outside_addon")
        else:
            warnings.append("application_is_source_only_until_a_supported_runtime_entrypoint_exists")
        app_root = os.path.join(source_abs, "app")
        if not os.path.isdir(app_root):
            errors.append("packaged_application_directory_missing")
        integrity_path = os.path.join(source_abs, "PACKAGE_INTEGRITY.json")
        package_integrity = self._read_json_file(integrity_path) if os.path.isfile(integrity_path) else {}
        new_nailde_package = str(manifest.get("type") or "").strip().lower() == "nailde_application" if manifest else False
        if new_nailde_package and not package_integrity:
            errors.append("package_integrity_manifest_missing")
        elif package_integrity:
            expected_files = package_integrity.get("files") if isinstance(package_integrity.get("files"), dict) else {}
            for rel, expected_hash in list(expected_files.items())[:1000]:
                rel_s = str(rel or "").replace("\\", "/").strip()
                if not rel_s or rel_s.startswith("/") or ".." in rel_s.split("/"):
                    errors.append("package_integrity_invalid_path")
                    continue
                file_path = os.path.abspath(os.path.join(source_abs, *rel_s.split("/")))
                try:
                    if os.path.commonpath([source_abs, file_path]) != source_abs or not os.path.isfile(file_path):
                        errors.append(f"package_integrity_file_missing:{rel_s}")
                        continue
                except Exception:
                    errors.append(f"package_integrity_file_missing:{rel_s}")
                    continue
                h = hashlib.sha256()
                try:
                    with open(file_path, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest() != str(expected_hash):
                        errors.append(f"package_integrity_hash_mismatch:{rel_s}")
                except Exception:
                    errors.append(f"package_integrity_hash_read_failed:{rel_s}")
        return {
            "ok": not errors,
            "schema": "SarahMemory.nailde.addon_package_validation.v1",
            "source_path": source_abs,
            "addon_id": addon_id,
            "manifest": manifest,
            "ui": ui,
            "runtime": runtime,
            "package_integrity": package_integrity,
            "runnable": runnable,
            "errors": errors,
            "warnings": warnings,
            "execution_authority": False,
        }

    def _build_universal_addon_package(self, workspace_id: str, spec: Dict[str, Any], synthesis: Dict[str, Any], source_files: Dict[str, str]) -> Dict[str, Any]:
        """Package any successful QSML/NAILDE application for the existing Addons runtime.

        Application logic remains model-synthesized. This method deterministically
        supplies only the SarahMemory Addons ABI: manifest, UI metadata, runtime
        adapter, local icon, source copy, and governance metadata.
        """
        workspace_id = str(workspace_id or "").strip()
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        root = self._workspace_root(workspace_id)
        package_root = os.path.join(root, "sandbox", "addon_package")
        try:
            expected = os.path.abspath(os.path.join(root, "sandbox"))
            package_abs = os.path.abspath(package_root)
            if os.path.commonpath([expected, package_abs]) != expected:
                return {"ok": False, "error": "addon_package_outside_workspace_sandbox", "execution_authority": False}
        except Exception:
            return {"ok": False, "error": "addon_package_path_validation_failed", "execution_authority": False}
        if os.path.isdir(package_root):
            shutil.rmtree(package_root)
        _ensure_dir(os.path.join(package_root, "app"))

        copied: List[Dict[str, Any]] = []
        total = 0
        for source_path, content in sorted(source_files.items()):
            source_key = str(source_path or "").replace("\\", "/")
            if source_key.startswith("sandbox/addon_package/"):
                continue
            try:
                rel = self._addon_source_relative_path(source_key)
            except Exception as exc:
                return {"ok": False, "error": "generated_source_path_not_packageable", "path": source_key, "detail": str(exc), "execution_authority": False}
            data = str(content if content is not None else "")
            size = len(data.encode("utf-8", "replace"))
            total += size
            if len(copied) >= _MAX_UNIVERSAL_EXPORT_FILES or total > _MAX_UNIVERSAL_PROJECT_BYTES:
                return {"ok": False, "error": "addon_package_budget_exceeded", "execution_authority": False}
            target_rel = f"sandbox/addon_package/app/{rel}"
            saved = self.save_workspace_file({"workspace_id": workspace_id, "path": target_rel, "content": data})
            if not saved.get("ok"):
                return {"ok": False, "error": "addon_application_source_copy_failed", "path": source_key, "saved": saved, "execution_authority": False}
            copied.append({"source": source_key, "packaged": f"app/{rel}", "size_bytes": size, "sha256": _sha256_text(data)})

        blueprint = synthesis.get("blueprint") if isinstance(synthesis.get("blueprint"), dict) else spec.get("universal_blueprint") if isinstance(spec.get("universal_blueprint"), dict) else {}
        runtime = self._infer_universal_addon_runtime(blueprint, source_files)
        app_name = str(spec.get("application_name") or blueprint.get("name") or "NAILDE Application").strip()[:160] or "NAILDE Application"
        addon_id = self._safe_addon_id(spec.get("addon_id") or app_name)
        requested_caps = [dict(x) for x in list(blueprint.get("requested_capabilities") or spec.get("requested_capabilities") or []) if isinstance(x, dict)][:64]
        for cap in requested_caps:
            cap["granted"] = False
            cap["execution_authority"] = False
        runtime.update({
            "addon_id": addon_id,
            "application_name": app_name,
            "requested_capabilities": requested_caps,
            "generated_by": "SarahMemoryNAILDE",
            "qsml_program_id": str(spec.get("qsml_program_id") or ""),
            "execution_authority": False,
        })
        buttons = ["RUN", "COPY", "REMOVE", "UPDATE"] if runtime.get("runnable") else ["COPY", "REMOVE", "UPDATE"]
        manifest = {
            "id": addon_id,
            "addon_id": addon_id,
            "name": app_name,
            "type": "nailde_application",
            "version": str(spec.get("version") or "0.1.0"),
            "description": str(blueprint.get("goal") or spec.get("top_prompt") or "NAILDE generated local application")[:1200],
            "author": "NAILDE",
            "source": "NAILDE",
            "created_by": "SarahMemoryNAILDE",
            "created_from_workspace": workspace_id,
            "qsml_program_id": str(spec.get("qsml_program_id") or ""),
            "entrypoint": {"module": "addon", "callable": "addon_init"},
            "entry": {"module": "addon", "callable": "addon_init"},
            "execution": {"mode": "subprocess", "runtime_adapter": runtime.get("mode"), "auto_run_allowed": False},
            "permissions": ["addon_local_runtime", "addon_local_file_read", "addon_local_file_write"],
            "requested_capabilities": requested_caps,
            "denied_permissions": list(NAILDE_DENIED_ACTIONS),
            "risk_tier": "REVIEW_GENERATED_APPLICATION",
            "security": {"trusted": False, "requires_user_install": True, "requires_user_run": True, "auto_run_allowed": False},
            "governance": {
                "schema": "SarahMemory.nailde.generated_application.v1",
                "sandbox_first": True,
                "source_workspace": workspace_id,
                "installed_by_user_only": True,
                "run_by_user_only": True,
                "no_ui_rebuild_required": True,
                "live_core_write": False,
                "self_approval": False,
                "execution_authority": False,
                "route_definition_owner": str(spec.get("route_definition_owner") or "SarahMemorySMLProtocol"),
                "route_activation_owner": str(spec.get("route_activation_owner") or "SarahMemoryNeuron"),
            },
            "ui": {"icon": "PackageCheck", "surface": "addons_runtime_panel"},
        }
        ui = {
            "schema": "SarahMemory.addon.ui_manifest.v1",
            "title": app_name,
            "icon": "PackageCheck",
            "runtime": "python_subprocess_manifest",
            "description": manifest["description"],
            "buttons": buttons,
            "runtime_mode": runtime.get("mode"),
            "runnable": bool(runtime.get("runnable")),
            "execution_authority": False,
        }
        icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#111827"/><path d="M18 20h28v24H18z" fill="#0ea5e9"/><path d="M24 27l7 5-7 5M34 38h7" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>\n'
        readme = "\n".join([
            f"# {app_name}",
            "",
            "Generated by SarahMemory NAILDE through QSML universal application synthesis.",
            "",
            f"Workspace: `{workspace_id}`",
            f"Runtime adapter: `{runtime.get('mode')}`",
            f"Runnable: `{bool(runtime.get('runnable'))}`",
            "",
            "Installation requires explicit user approval. Running the Addon is a separate explicit user action.",
            "The package does not modify SarahMemory CORE/API/UI files and does not auto-run.",
            "",
        ])
        package_payloads = {
            "sandbox/addon_package/manifest.json": json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
            "sandbox/addon_package/ui.json": json.dumps(ui, indent=2, sort_keys=True, ensure_ascii=False),
            "sandbox/addon_package/app_runtime.json": json.dumps(runtime, indent=2, sort_keys=True, ensure_ascii=False),
            "sandbox/addon_package/addon.py": self._universal_addon_wrapper_source(),
            "sandbox/addon_package/assets/icon.svg": icon_svg,
            "sandbox/addon_package/README.md": readme,
        }
        saved_meta = []
        for rel_path, content in package_payloads.items():
            result = self.save_workspace_file({"workspace_id": workspace_id, "path": rel_path, "content": content})
            saved_meta.append(result)
            if not result.get("ok"):
                return {"ok": False, "error": "addon_metadata_write_failed", "path": rel_path, "saved": result, "execution_authority": False}
        integrity_files: Dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(package_root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
            for name in sorted(filenames):
                if name in {"PACKAGE_INTEGRITY.json", "install_state.json", "manifest_launch.log"}:
                    continue
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    continue
                rel = os.path.relpath(full, package_root).replace("\\", "/")
                h = hashlib.sha256()
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                integrity_files[rel] = h.hexdigest()
        integrity_doc = {
            "schema": "SarahMemory.nailde.addon_package_integrity.v1",
            "addon_id": addon_id,
            "generated_at": _now_iso(),
            "file_count": len(integrity_files),
            "files": integrity_files,
            "execution_authority": False,
        }
        integrity_saved = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/addon_package/PACKAGE_INTEGRITY.json", "content": json.dumps(integrity_doc, indent=2, sort_keys=True)})
        if not integrity_saved.get("ok"):
            return {"ok": False, "error": "addon_integrity_manifest_write_failed", "saved": integrity_saved, "execution_authority": False}
        saved_meta.append(integrity_saved)
        validation = self._validate_installable_addon_package(package_root)
        return {
            "ok": bool(validation.get("ok")),
            "schema": "SarahMemory.nailde.universal_addon_package.v1",
            "workspace_id": workspace_id,
            "addon_id": addon_id,
            "application_name": app_name,
            "source_path": package_root,
            "runtime": runtime,
            "manifest": manifest,
            "ui": ui,
            "copied_sources": copied,
            "saved_metadata": saved_meta,
            "validation": validation,
            "install_requires_user_confirmation": True,
            "run_requires_user_confirmation": True,
            "no_ui_rebuild_required": True,
            "auto_run_performed": False,
            "execution_authority": False,
        }

    def _settings_state_path(self) -> str:
        return os.path.join(_ensure_dir(os.path.join(self.nailde_dir, "settings")), "nailde_settings.json")

    def _default_settings_state(self) -> Dict[str, Any]:
        return {
            "schema": "SarahMemory.nailde.settings.v1",
            "editor": {"line_numbers": True, "indent_grid": True, "indent_size": 4, "validate_on_change": False, "show_problems": True},
            "filesystem": {"show_sandbox_map": True, "checksum_on_demand": True, "live_files_read_only": True},
            "github": {"enabled": False, "auth_mode": "user_managed", "username": "", "repo_url": "", "branch": "main", "remember_repo_metadata": True, "store_tokens_here": False},
            "security": {"secrets_never_persisted": True, "push_pull_plan_only": True, "requires_user_approval": True},
            "execution_authority": False,
        }

    def _sanitize_settings_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        clean = self._default_settings_state()
        if isinstance(state.get("editor"), dict):
            clean["editor"].update({k: state["editor"].get(k) for k in clean["editor"].keys() if k in state["editor"]})
        if isinstance(state.get("filesystem"), dict):
            clean["filesystem"].update({k: state["filesystem"].get(k) for k in clean["filesystem"].keys() if k in state["filesystem"]})
        if isinstance(state.get("github"), dict):
            allowed = {k: state["github"].get(k) for k in clean["github"].keys() if k in state["github"]}
            clean["github"].update(allowed)
        # Strip common secret fields even if the UI accidentally sends them.
        for secret_key in ("token", "access_token", "password", "client_secret", "private_key", "pat"):
            clean.pop(secret_key, None)
            if isinstance(clean.get("github"), dict):
                clean["github"].pop(secret_key, None)
        clean["github"]["store_tokens_here"] = False
        clean["security"]["secrets_never_persisted"] = True
        clean["execution_authority"] = False
        return clean

    @staticmethod
    def _tasks_from_validation(result: Dict[str, Any], *, workspace_id: str = "") -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for idx, problem in enumerate(result.get("problems") if isinstance(result.get("problems"), list) else [], start=1):
            tasks.append({
                "task_id": f"nailde-task-{idx}-{_safe_name(problem.get('code'), 'problem')}",
                "workspace_id": workspace_id,
                "path": result.get("path", problem.get("path", "")),
                "title": str(problem.get("message") or problem.get("code") or "Editor issue")[:160],
                "severity": problem.get("severity", "warning"),
                "line": problem.get("line", 1),
                "column": problem.get("column", 1),
                "recommended_action": problem.get("task", "review"),
                "sandbox_only": True,
                "execution_authority": False,
            })
        return tasks

    @staticmethod
    def _indent_grid(content: str) -> Dict[str, Any]:
        lines = (content or "").splitlines()
        return {
            "indent_size": 4,
            "tabs_allowed": False,
            "line_count": len(lines),
            "max_line_length": max([len(line) for line in lines] or [0]),
            "indents": [len(line) - len(line.lstrip(" ")) for line in lines[:1000]],
        }

    # ------------------------------------------------------------------
    # Workspace filesystem helpers - sandbox only
    # ------------------------------------------------------------------
    def _workspace_root(self, workspace_id: str) -> str:
        safe = _safe_name(workspace_id, "workspace")
        root = os.path.abspath(os.path.join(self.workspaces_dir, safe))
        base = os.path.abspath(self.workspaces_dir)
        if not (root == base or root.startswith(base + os.sep)):
            raise ValueError("workspace_path_escape_blocked")
        return root

    def _workspace_file_path(self, workspace_id: str, rel_path: str) -> str:
        rel = str(rel_path or "").replace("\\", "/").lstrip("/")
        if not rel or ".." in rel.split("/"):
            raise ValueError("relative_path_required_inside_workspace")
        root = self._workspace_root(workspace_id)
        path = os.path.abspath(os.path.join(root, rel))
        if not path.startswith(root + os.sep) and path != root:
            raise ValueError("workspace_file_escape_blocked")
        return path

    @staticmethod
    def _title_from_prompt(prompt: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", prompt or "")[:5]
        return "".join(w.capitalize() for w in words) or "NaildeGeneratedTool"

    def _draft_files_for_prompt(self, app_name: str, prompt: str, target: str) -> Dict[str, str]:
        component_name = re.sub(r"[^A-Za-z0-9]", "", app_name) or "NaildeGeneratedTool"
        if not component_name[0].isalpha():
            component_name = "Nailde" + component_name
        lower_slug = re.sub(r"[^a-z0-9]+", "_", app_name.lower()).strip("_") or "nailde_tool"
        tsx = f'''import {{ useState }} from "react";\n\nexport default function {component_name}Panel() {{\n  const [notes, setNotes] = useState("");\n\n  return (\n    <section className="rounded-xl border border-border bg-card p-4 space-y-3">\n      <div>\n        <h2 className="text-lg font-semibold">{component_name}</h2>\n        <p className="text-sm text-muted-foreground">Sandbox-generated by NAILDE from natural language. Live apply is not authorized.</p>\n      </div>\n      <textarea\n        className="w-full min-h-32 rounded-md border border-border bg-background p-3 text-sm"\n        value={{notes}}\n        onChange={{(event) => setNotes(event.target.value)}}\n        placeholder="Edit this sandbox panel before validation."\n      />\n      <pre className="rounded-md bg-muted p-3 text-xs whitespace-pre-wrap">{{JSON.stringify({{ notes, sandboxOnly: true, executionAuthority: false }}, null, 2)}}</pre>\n    </section>\n  );\n}}\n'''
        py = f'''"""Sandbox backend draft generated by NAILDE.\n\nGoal:\n{prompt[:1000]}\n\nThis file is a sandbox artifact only. It has no live route authority until\nDevBridge/Compare/Ledger/user approval gates are completed.\n"""\n\nfrom __future__ import annotations\n\nfrom datetime import datetime\nfrom typing import Any, Dict\n\n\ndef handle_{lower_slug}_request(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:\n    payload = payload if isinstance(payload, dict) else {{}}\n    return {{\n        "ok": True,\n        "tool": "{component_name}",\n        "ts": datetime.now().isoformat(timespec="seconds"),\n        "input": payload,\n        "sandbox_only": True,\n        "execution_authority": False,\n        "live_apply_required_gates": ["sandbox_validation", "Compare", "Ledger", "DevBridge", "user_double_confirmation"],\n    }}\n'''
        manifest = {
            "schema": "SarahMemory.nailde.addon_manifest.v1",
            "name": component_name,
            "slug": lower_slug,
            "goal": prompt,
            "status": "draft",
            "sandbox_only": True,
            "execution_authority": False,
            "permissions": ["read_sandbox", "write_addon_local_db_if_approved"],
            "denied_permissions": NAILDE_DENIED_ACTIONS,
        }
        readme = f'''# {component_name}\n\nGenerated by NAILDE from natural language.\n\n## Goal\n\n{prompt}\n\n## Status\n\n- Sandbox artifact: YES\n- Live apply: NO\n- Production tensor edit: NO\n- Device write: NO\n- Requires validation: YES\n- Requires DevBridge/Compare/Ledger/user approval before live use: YES\n'''
        return {
            f"sandbox/ui/{component_name}Panel.tsx": tsx,
            f"sandbox/backend/{lower_slug}_backend.py": py,
            "sandbox/addon_manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/README.md": readme,
        }

    @staticmethod
    def _validate_text(content: str, path: str) -> Dict[str, Any]:
        """Static editor validation with line/column diagnostics.

        This is syntax/lint inspection only. It does not execute code.
        """
        errors: List[str] = []
        warnings: List[str] = []
        problems: List[Dict[str, Any]] = []
        lower = path.lower()
        lines = (content or "").splitlines()

        def problem(severity: str, code: str, message: str, line: int = 1, column: int = 1, task: str = "review") -> None:
            entry = {
                "severity": severity,
                "code": code,
                "message": message,
                "line": max(1, int(line or 1)),
                "column": max(1, int(column or 1)),
                "task": task,
                "path": path,
                "execution_authority": False,
            }
            problems.append(entry)
            if severity == "error":
                errors.append(f"{code}:{line}:{column}:{message}")
            else:
                warnings.append(f"{code}:{line}:{column}:{message}")

        # Universal indentation / formatting inspection.
        indent_stack = [0]
        for idx, line in enumerate(lines, start=1):
            if "\t" in line:
                problem("warning", "tab_indent_detected", "Tab indentation detected; use spaces for deterministic alignment.", idx, line.index("\t") + 1, "normalize_indent")
            if line.rstrip(" \t") != line:
                problem("warning", "trailing_whitespace", "Trailing whitespace detected.", idx, len(line.rstrip(" \t")) + 1, "trim_line")
            stripped = line.lstrip(" ")
            leading = len(line) - len(stripped)
            if stripped and leading % 4 != 0 and lower.endswith(('.py', '.tsx', '.ts', '.jsx', '.js', '.json', '.md')):
                problem("warning", "indent_grid_misaligned", "Indent is not aligned to the 4-space NAILDE grid.", idx, leading + 1, "align_to_grid")
            if lower.endswith(".py") and stripped:
                if leading > indent_stack[-1]:
                    indent_stack.append(leading)
                while leading < indent_stack[-1] and len(indent_stack) > 1:
                    indent_stack.pop()
                if leading not in indent_stack:
                    problem("error", "python_indent_level_mismatch", "Indentation does not match any previous Python block level.", idx, leading + 1, "fix_indent")

        if lower.endswith(".py"):
            try:
                import ast
                ast.parse(content or "\n")
            except IndentationError as exc:
                problem("error", "python_indentation_error", str(exc), getattr(exc, "lineno", 1) or 1, getattr(exc, "offset", 1) or 1, "fix_indent")
            except SyntaxError as exc:
                problem("error", "python_syntax_error", str(exc), getattr(exc, "lineno", 1) or 1, getattr(exc, "offset", 1) or 1, "fix_syntax")
            except Exception as exc:
                problem("error", "python_parse_error", str(exc), 1, 1, "fix_syntax")
        if lower.endswith((".json", ".jsonc")):
            try:
                json.loads(content or "{}")
            except Exception as exc:
                problem("error", "json_parse_error", str(exc), 1, 1, "fix_json")
        if lower.endswith((".tsx", ".ts", ".jsx", ".js")):
            pairs = [("(", ")"), ("[", "]"), ("{", "}")]
            for left, right in pairs:
                if content.count(left) != content.count(right):
                    problem("warning", f"unbalanced_{left}_{right}_count_static_check", f"Unbalanced {left}{right} count in static editor check.", 1, 1, "check_brackets")
            if "executionAuthority" not in content and "execution_authority" not in content:
                problem("warning", "missing_visible_execution_authority_marker", "Add an execution_authority / executionAuthority marker for governed generated code.", 1, 1, "add_governance_marker")
        unsafe_terms = ["os.system(", "subprocess.", "eval(", "exec(", "torch.save(", "requests.post(", "child_process", "localStorage.setItem('token", 'localStorage.setItem("token']
        for term in unsafe_terms:
            pos = content.find(term)
            if pos >= 0:
                before = content[:pos]
                line = before.count("\n") + 1
                col = pos - before.rfind("\n")
                problem("warning", "manual_review_term", f"Manual review term detected: {term}", line, col, "security_review")
        return {
            "path": path,
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "problems": problems,
            "line_count": len(lines),
            "sha256": _sha256_text(content),
            "indent_grid": {"spaces": 4, "tabs_allowed": False, "line_count": len(lines)},
            "execution_authority": False,
        }

    # ------------------------------------------------------------------
    # Novice-to-enterprise Natural Language Auto-Build Pipeline
    # ------------------------------------------------------------------
    def auto_build_from_prompt(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Expand a novice natural-language prompt into a governed sandbox build.

        The top prompt is the project seed. Changing that seed creates a new
        workspace only after the user decides what to do with the previous one.
        The lower/detail prompt changes requirements inside the active workspace.
        """
        payload = payload if isinstance(payload, dict) else {}
        top_prompt = str(
            payload.get("top_prompt")
            or payload.get("project_seed")
            or payload.get("goal")
            or ""
        ).strip()
        details_prompt = str(
            payload.get("details_prompt")
            or payload.get("additional_instructions")
            or payload.get("prompt")
            or ""
        ).strip()
        if not top_prompt:
            return {
                "ok": False,
                "error": "top_natural_language_prompt_required",
                "message": "Type the main project idea first, for example: Build a Pacman style game.",
                "execution_authority": False,
            }

        current_workspace_id = str(payload.get("current_workspace_id") or payload.get("workspace_id") or "").strip()
        current_seed_hash = str(payload.get("current_project_seed_hash") or payload.get("project_seed_hash") or "").strip()
        prompt_decision = str(payload.get("prompt_change_decision") or payload.get("decision") or "").strip().lower()
        seed_hash = self._prompt_seed_hash(top_prompt)
        seed_changed = bool(current_workspace_id and current_seed_hash and current_seed_hash != seed_hash)

        if seed_changed and prompt_decision not in {"keep_save_current", "save_as_current", "discard_current", "cancel_prompt_change"}:
            return {
                "ok": False,
                "requires_workspace_decision": True,
                "schema": "SarahMemory.nailde.workspace_prompt_change.v1",
                "message": "The top Natural Language Prompt changed. Choose what to do with the current workspace before NAILDE creates a new one.",
                "current_workspace_id": current_workspace_id,
                "previous_seed_hash": current_seed_hash,
                "next_seed_hash": seed_hash,
                "previous_prompt": str(payload.get("previous_top_prompt") or payload.get("last_top_prompt") or ""),
                "next_prompt": top_prompt,
                "options": [
                    {"id": "keep_save_current", "label": "Keep and Save Current"},
                    {"id": "save_as_current", "label": "Save Current As..."},
                    {"id": "discard_current", "label": "Discard Current"},
                    {"id": "cancel_prompt_change", "label": "Cancel Prompt Change"},
                ],
                "execution_authority": False,
            }

        if seed_changed and prompt_decision == "cancel_prompt_change":
            return {
                "ok": False,
                "cancelled": True,
                "message": "Prompt change cancelled. Current workspace remains active.",
                "workspace_id": current_workspace_id,
                "execution_authority": False,
            }

        if current_workspace_id and prompt_decision in {"keep_save_current", "save_as_current"}:
            self.workspace_autosave({
                "workspace_id": current_workspace_id,
                "top_prompt": payload.get("previous_top_prompt") or payload.get("last_top_prompt") or "",
                "details_prompt": payload.get("previous_details_prompt") or "",
                "editor_text": payload.get("editor_text") or "",
                "file_path": payload.get("file_path") or "",
                "status": "saved_before_prompt_change",
            })
            if prompt_decision == "save_as_current":
                self.workspace_decision({
                    "action": "save_as",
                    "workspace_id": current_workspace_id,
                    "new_workspace_id": payload.get("save_as_workspace_id") or f"{current_workspace_id}_saved_{int(time.time())}",
                })

        if current_workspace_id and prompt_decision == "discard_current":
            # Do not delete automatically. Mark as discarded/recoverable to protect work.
            self.workspace_autosave({
                "workspace_id": current_workspace_id,
                "status": "discarded_by_user_before_new_prompt_but_recoverable",
                "top_prompt": payload.get("previous_top_prompt") or "",
                "details_prompt": payload.get("previous_details_prompt") or "",
            })

        reuse_current = bool(current_workspace_id and not seed_changed)
        if reuse_current:
            workspace_id = current_workspace_id
            root = self._workspace_root(workspace_id)
            _ensure_dir(root)
        else:
            workspace_id = self._workspace_id_from_prompt(top_prompt)
            root = self._workspace_root(workspace_id)
            if not os.path.isdir(root):
                created = self.create_workspace({"workspace_id": workspace_id, "goal": top_prompt, "mode": "BUILD_SANDBOX"})
                if not created.get("ok"):
                    return created
            else:
                _ensure_dir(root)

        compiled = self.qsml_compile_request({**payload, "top_prompt": top_prompt, "details_prompt": details_prompt, "workspace_id": workspace_id, "target": "nailde"})
        if not compiled.get("ok"):
            return {
                "ok": False,
                "schema": "SarahMemory.nailde.auto_build.v2",
                "workspace_id": workspace_id,
                "error": "qsml_compile_failed",
                "qsml": compiled,
                "message": "NAILDE stopped instead of substituting a canned or unrelated project.",
                "sandbox_only": True,
                "execution_authority": False,
            }
        spec = self._expand_qsml_program(compiled, top_prompt, details_prompt)
        synthesis = self._universal_synthesize_spec(spec, payload)
        if not synthesis.get("ok"):
            receipt = self._record_receipt(
                "NAILDE_UNIVERSAL_SYNTHESIS_BLOCKED",
                workspace_id,
                f"Universal application synthesis stopped safely: {str(synthesis.get('error') or 'unknown')[:300]}",
                {"workspace_id": workspace_id, "risk": "low", "verdict": "SYNTHESIS_BLOCKED", "phase": synthesis.get("phase")},
            )
            return {
                "ok": False,
                "schema": "SarahMemory.nailde.auto_build.v2",
                "workspace_id": workspace_id,
                "workspace_root": root,
                "top_prompt": top_prompt,
                "details_prompt": details_prompt,
                "project_seed_hash": seed_hash,
                "spec": spec,
                "qsml": compiled,
                "synthesis": synthesis,
                "receipt": receipt,
                "message": "NAILDE did not fall back to a different application. Resolve the reported local synthesis/model issue and retry.",
                "sandbox_only": True,
                "execution_authority": False,
                "live_file_write": False,
            }
        files = dict(synthesis.get("files") or {})
        spec["universal_blueprint"] = synthesis.get("blueprint") or {}
        spec["synthesis_validation"] = synthesis.get("validation") or {}
        spec["synthesis_mode"] = "universal_local_model"
        spec["required_files"] = sorted(files.keys())
        saved: List[Dict[str, Any]] = []
        for rel_path, content in files.items():
            saved.append(self.save_workspace_file({"workspace_id": workspace_id, "path": rel_path, "content": content}))

        addon_package = self._build_universal_addon_package(workspace_id, spec, synthesis, files)
        if not addon_package.get("ok"):
            return {
                "ok": False, "schema": "SarahMemory.nailde.auto_build.v3", "workspace_id": workspace_id,
                "error": "addon_package_build_failed", "addon_package": addon_package, "spec": spec, "synthesis": synthesis,
                "message": "The application source was synthesized, but NAILDE refused promotion because the Addons package ABI/integrity validation failed.",
                "sandbox_only": True, "execution_authority": False, "live_file_write": False,
            }
        spec["addon_package"] = True
        spec["addon_id"] = addon_package.get("addon_id") or spec.get("addon_id")
        spec["addon_runtime"] = addon_package.get("runtime") or {}
        package_file_paths = [str(row.get("path") or "") for row in list((self.workspace_files({"workspace_id": workspace_id}).get("files") or [])) if str(row.get("path") or "").startswith("sandbox/addon_package/")]

        plan = self._build_auto_battle_plan(spec)
        plan_saved = self.save_workspace_file({"workspace_id": workspace_id, "path": "sandbox/BATTLE_PLAN.md", "content": plan})
        state = {
            "schema": "SarahMemory.nailde.auto_build_state.v2",
            "workspace_id": workspace_id,
            "top_prompt": top_prompt,
            "details_prompt": details_prompt,
            "project_seed_hash": seed_hash,
            "project_type": spec.get("project_type"),
            "qsml_program_id": spec.get("qsml_program_id"),
            "qsml_language_version": spec.get("qsml_language_version"),
            "application_name": spec.get("application_name"),
            "addon_id": spec.get("addon_id"),
            "created_at": _now_iso(),
            "status": "universal_application_synthesized",
            "novice_mode": bool(payload.get("novice_mode", True)),
            "enterprise_mode_available": True,
            "sandbox_only": True,
            "execution_authority": False,
        }
        self._write_workspace_state(workspace_id, state)
        autosave = self.workspace_autosave({
            "workspace_id": workspace_id,
            "top_prompt": top_prompt,
            "details_prompt": details_prompt,
            "project_seed_hash": seed_hash,
            "editor_text": payload.get("editor_text") or "",
            "file_path": payload.get("file_path") or "",
            "status": "auto_build_complete",
            "generated_files": list(files.keys()),
            "addon_package_files": package_file_paths,
        })
        validation_paths = sorted(set(list(files.keys()) + package_file_paths))
        validation = self.validate_text_artifacts({"workspace_id": workspace_id, "paths": validation_paths})
        test_run = self._sandbox_test_run_readiness(workspace_id, spec, validation)
        export_package = self.package_workspace_export({"workspace_id": workspace_id}) if validation.get("ok") else {"ok": False, "reason": "validation_required_before_export", "execution_authority": False}
        install_plan = self.addon_install_plan({"workspace_id": workspace_id})
        post_popup = self._post_test_popup(spec, validation, test_run, install_plan)
        receipt = self._record_receipt(
            "NAILDE_AUTO_BUILD_FROM_PROMPT",
            workspace_id,
            f"NAILDE auto-built sandbox project from prompt: {top_prompt[:160]}",
            {"workspace_id": workspace_id, "risk": "low", "verdict": "SANDBOX_GENERATED", "project_seed_hash": seed_hash},
        )
        return {
            "ok": True,
            "schema": "SarahMemory.nailde.auto_build.v2",
            "workspace_id": workspace_id,
            "workspace_root": root,
            "top_prompt": top_prompt,
            "details_prompt": details_prompt,
            "project_seed_hash": seed_hash,
            "spec": spec,
            "qsml": compiled,
            "synthesis": synthesis,
            "addon_package": addon_package,
            "battle_plan": plan,
            "battle_plan_saved": plan_saved,
            "files": [{"path": path, "sha256": _sha256_text(content)} for path, content in files.items()],
            "saved": saved,
            "validation": validation,
            "test_run": test_run,
            "export_package": export_package,
            "install_plan": install_plan,
            "post_test_popup": post_popup,
            "autosave": autosave,
            "receipt": receipt,
            "sandbox_only": True,
            "execution_authority": False,
            "live_file_write": False,
        }

    def workspace_autosave(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Persist recoverable NAILDE state for power-loss/session recovery."""
        payload = payload if isinstance(payload, dict) else {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        root = self._workspace_root(workspace_id)
        _ensure_dir(root)
        autosave_dir = _ensure_dir(os.path.join(root, "autosave"))
        recovery_dir = _ensure_dir(os.path.join(root, "recovery"))
        snapshot = {
            "schema": "SarahMemory.nailde.autosave.snapshot.v1",
            "workspace_id": workspace_id,
            "saved_at": _now_iso(),
            "top_prompt": str(payload.get("top_prompt") or ""),
            "details_prompt": str(payload.get("details_prompt") or ""),
            "project_seed_hash": str(payload.get("project_seed_hash") or ""),
            "file_path": str(payload.get("file_path") or ""),
            "editor_text": str(payload.get("editor_text") if payload.get("editor_text") is not None else ""),
            "battle_plan": str(payload.get("battle_plan") or ""),
            "status": str(payload.get("status") or "autosaved"),
            "generated_files": payload.get("generated_files") if isinstance(payload.get("generated_files"), list) else [],
            "dirty": bool(payload.get("dirty", False)),
            "sandbox_only": True,
            "execution_authority": False,
        }
        self._write_json_file(os.path.join(autosave_dir, "latest_snapshot.json"), snapshot)
        self._write_json_file(os.path.join(autosave_dir, "prompt_state.json"), {
            "top_prompt": snapshot["top_prompt"],
            "details_prompt": snapshot["details_prompt"],
            "project_seed_hash": snapshot["project_seed_hash"],
            "saved_at": snapshot["saved_at"],
            "execution_authority": False,
        })
        self._write_json_file(os.path.join(autosave_dir, "editor_buffer.json"), {
            "file_path": snapshot["file_path"],
            "editor_text": snapshot["editor_text"],
            "sha256": _sha256_text(snapshot["editor_text"]),
            "saved_at": snapshot["saved_at"],
            "execution_authority": False,
        })
        recovery = {
            "schema": "SarahMemory.nailde.recovery_manifest.v1",
            "workspace_id": workspace_id,
            "latest_snapshot": os.path.join(autosave_dir, "latest_snapshot.json"),
            "last_saved_at": snapshot["saved_at"],
            "status": snapshot["status"],
            "restore_available": True,
            "execution_authority": False,
        }
        self._write_json_file(os.path.join(recovery_dir, "recovery_manifest.json"), recovery)
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "snapshot": snapshot,
            "recovery": recovery,
            "execution_authority": False,
        }

    def workspace_recovery(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Locate or restore the latest NAILDE autosave snapshot."""
        payload = payload if isinstance(payload, dict) else {}
        action = str(payload.get("action") or "latest").strip().lower()
        candidates = self._recovery_candidates()
        latest = candidates[0] if candidates else None
        if action == "list":
            return {"ok": True, "candidates": candidates, "count": len(candidates), "execution_authority": False}
        if not latest:
            return {"ok": True, "restore_available": False, "candidates": [], "execution_authority": False}
        if action == "restore":
            return {
                "ok": True,
                "restore_available": True,
                "restored": True,
                "workspace_id": latest.get("workspace_id"),
                "snapshot": latest.get("snapshot"),
                "message": "Recovered NAILDE workspace snapshot loaded. No live files were modified.",
                "execution_authority": False,
            }
        return {
            "ok": True,
            "restore_available": True,
            "latest": latest,
            "candidates": candidates[:10],
            "message": "Previous unfinished NAILDE workspace found.",
            "execution_authority": False,
        }

    def workspace_decision(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle Save / Save As / Cancel decisions after sandbox test or prompt change."""
        payload = payload if isinstance(payload, dict) else {}
        action = str(payload.get("action") or "save").strip().lower()
        workspace_id = str(payload.get("workspace_id") or "").strip()
        if not workspace_id:
            return {"ok": False, "error": "workspace_id_required", "execution_authority": False}
        if action == "cancel":
            return {"ok": True, "action": "cancel", "message": "Decision cancelled. Continue editing in the current NAILDE session.", "workspace_id": workspace_id, "execution_authority": False}
        if action == "save":
            saved = self.workspace_autosave({**payload, "status": "user_saved_workspace"})
            return {"ok": True, "action": "save", "workspace_id": workspace_id, "saved": saved, "execution_authority": False}
        if action == "save_as":
            new_workspace_id = _safe_name(payload.get("new_workspace_id") or f"{workspace_id}_copy_{int(time.time())}", "workspace_copy")
            src = self._workspace_root(workspace_id)
            dst = self._workspace_root(new_workspace_id)
            if os.path.exists(dst):
                return {"ok": False, "error": "target_workspace_exists", "target": new_workspace_id, "execution_authority": False}
            stats = self._copy_tree_bounded(src, dst, max_files=1200, max_total_bytes=40 * 1024 * 1024)
            saved = self.workspace_autosave({**payload, "workspace_id": new_workspace_id, "status": "user_save_as_workspace"})
            return {"ok": True, "action": "save_as", "workspace_id": new_workspace_id, "copy_stats": stats, "saved": saved, "execution_authority": False}
        if action == "add_to_addons":
            # This still requires explicit install confirmation; do not silently install here.
            plan = self.addon_install_plan({"workspace_id": workspace_id})
            return {"ok": bool(plan.get("ok")), "action": "add_to_addons", "install_plan": plan, "requires_explicit_install_confirmation": True, "execution_authority": False}
        return {"ok": False, "error": "unsupported_workspace_decision", "allowed": ["save", "save_as", "cancel", "add_to_addons"], "execution_authority": False}

    def _prompt_seed_hash(self, prompt: str) -> str:
        return hashlib.sha256(self._normalize_prompt_seed(prompt).encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _normalize_prompt_seed(prompt: str) -> str:
        text = str(prompt or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text[:500]

    def _workspace_id_from_prompt(self, prompt: str) -> str:
        title = self._title_from_prompt(prompt)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-") or "NAILDEProject"
        return _safe_name(f"SM_{slug}_{self._prompt_seed_hash(prompt)[:8]}", "SM_NAILDE_Project")

    def _expand_novice_prompt(self, top_prompt: str, details_prompt: str = "") -> Dict[str, Any]:
        text = f"{top_prompt}\n{details_prompt}".lower()
        wants_game = any(word in text for word in ("game", "pacman", "snake", "tetris", "pong", "maze"))
        wants_python = "python" in text or wants_game
        controller = any(word in text for word in ("controller", "gamepad", "joystick", "drivers"))
        pacman = "pacman" in text or "pac-man" in text or "maze chase" in text
        snake = "snake" in text
        if wants_game and wants_python:
            style = "snake" if snake else "pacman_style" if pacman else "arcade_game"
            app_title = "SarahMemory Snake Style Game" if snake else "SarahMemory PACMAN Style Game" if pacman else self._title_from_prompt(top_prompt)
            addon_id = self._safe_addon_id("sm_snake_style_game" if snake else "sm_pacman_style_game" if pacman else self._title_from_prompt(top_prompt))
            return {
                "schema": "SarahMemory.nailde.prompt_expanded.v1",
                "project_type": "python_game_addon",
                "game_style": style,
                "top_prompt": top_prompt,
                "details_prompt": details_prompt,
                "application_name": app_title,
                "addon_id": addon_id,
                "addon_package": True,
                "novice_summary": f"Build a Python game addon named {app_title}.",
                "required_files": [
                    "sandbox/addon_package/manifest.json",
                    "sandbox/addon_package/ui.json",
                    "sandbox/addon_package/addon.py",
                    "sandbox/addon_package/game/__init__.py",
                    "sandbox/addon_package/game/sm_game.py",
                    "sandbox/addon_package/game/input_adapter.py",
                    "sandbox/addon_package/game/driver_gamepad_adapter.py",
                    "sandbox/addon_package/assets/icon.svg",
                    "sandbox/addon_package/data/highscores.json",
                    "sandbox/addon_package/README.md",
                ],
                "validation_plan": ["python_syntax", "python_indentation", "json_manifest", "ui_json", "entrypoint", "sandbox_containment", "controller_fail_soft"],
                "input": {"keyboard": True, "game_controller_optional": bool(controller or wants_game), "drivers_read_only": True},
                "denied": ["live_core_write", "shell_command_generation", "network_access", "device_write", "driver_mutation", "production_tensor_edit", "global_dlpanel_write", "self_approval"],
                "execution_authority": False,
            }
        app_title = self._title_from_prompt(top_prompt)
        return {
            "schema": "SarahMemory.nailde.prompt_expanded.v1",
            "project_type": "generic_sandbox_addon",
            "top_prompt": top_prompt,
            "details_prompt": details_prompt,
            "application_name": app_title,
            "addon_id": self._safe_addon_id(app_title),
            "addon_package": False,
            "novice_summary": f"Build a sandbox project named {app_title}.",
            "required_files": ["sandbox/README.md", "sandbox/addon_manifest.json"],
            "validation_plan": ["syntax", "manifest", "sandbox_containment"],
            "denied": list(NAILDE_DENIED_ACTIONS),
            "execution_authority": False,
        }

    def _files_for_auto_spec(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """Legacy compatibility entrypoint; all new builds use universal synthesis."""
        result = self._universal_synthesize_spec(spec, {})
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "universal_synthesis_failed"))
        return dict(result.get("files") or {})

    def _python_game_addon_files(self, spec: Dict[str, Any]) -> Dict[str, str]:
        app_name = str(spec.get("application_name") or "SarahMemory Game")
        addon_id = str(spec.get("addon_id") or self._safe_addon_id(app_name))
        game_style = str(spec.get("game_style") or "arcade_game")
        top_prompt = str(spec.get("top_prompt") or "")
        details_prompt = str(spec.get("details_prompt") or "")
        manifest = {
            "addon_id": addon_id,
            "id": addon_id,
            "name": app_name,
            "type": "application",
            "version": "0.1.0",
            "description": f"Original Python {game_style.replace('_', ' ')} built in NAILDE from a natural language prompt.",
            "entrypoint": {"module": "addon", "callable": "run"},
            "entry": {"module": "addon", "callable": "run"},
            "execution": {"mode": "subprocess"},
            "risk_tier": "low",
            "permissions": ["keyboard_input", "optional_game_controller_readonly", "addon_local_file_write"],
            "denied": spec.get("denied") or list(NAILDE_DENIED_ACTIONS),
            "security": {"trusted": False, "requires_user_run": True, "auto_run_allowed": False},
            "governance": {"created_by": "NAILDE", "sandbox_first": True, "no_ui_rebuild_required": True, "execution_authority": False},
        }
        ui = {
            "schema": "SarahMemory.addon.ui_manifest.v1",
            "title": app_name.replace("Style Game", ""),
            "icon": "assets/icon.svg",
            "action": "run",
            "runtime": "python_subprocess_manifest",
            "description": f"Original Python {game_style.replace('_', ' ')} built in NAILDE.",
            "buttons": ["RUN", "COPY", "REMOVE", "UPDATE"],
            "sections": [
                {"id": "overview", "title": "Overview", "kind": "markdown", "content": f"# {app_name}\n\nGenerated from: {top_prompt}"},
                {"id": "controls", "title": "Controls", "kind": "facts", "facts": ["Arrow keys / WASD", "P or Escape for pause", "Optional controller if discovered read-only"]},
            ],
            "execution_authority": False,
        }
        addon_py = '''"""NAILDE-generated Python game addon wrapper."""
from __future__ import annotations


def run(context=None):
    from game.sm_game import run_game
    return run_game(context=context or {})


def addon_info():
    return {"ok": True, "type": "python_game_addon", "execution_authority": False}
'''
        game_py = f'''"""Original SarahMemory-style Python arcade game.

Generated by NAILDE from a natural-language prompt.
No copyrighted assets, logos, sounds, or maze layouts are included.
"""
from __future__ import annotations

import json
import random
import tkinter as tk
from pathlib import Path

try:
    from .input_adapter import KeyboardInputAdapter
    from .driver_gamepad_adapter import ReadOnlyGamepadAdapter
except Exception:
    from input_adapter import KeyboardInputAdapter
    from driver_gamepad_adapter import ReadOnlyGamepadAdapter

CELL = 24
ROWS = 21
COLS = 23
TICK_MS = 115

MAZE = []
for r in range(ROWS):
    row = []
    for c in range(COLS):
        border = r in (0, ROWS - 1) or c in (0, COLS - 1)
        pillar = (r % 4 == 0 and c % 5 in (0, 1) and 2 < r < ROWS - 3 and 2 < c < COLS - 3)
        row.append(1 if border or pillar else 0)
    MAZE.append(row)
MAZE[1][1] = 0

class SMArcadeGame:
    def __init__(self, root, addon_root: Path):
        self.root = root
        self.addon_root = addon_root
        self.root.title({app_name!r})
        self.canvas = tk.Canvas(root, width=COLS * CELL, height=ROWS * CELL + 44, bg="#05060a", highlightthickness=0)
        self.canvas.pack()
        self.input = KeyboardInputAdapter(root)
        self.gamepad = ReadOnlyGamepadAdapter()
        self.gamepad_status = self.gamepad.detect()
        self.player = [1, 1]
        self.direction = [1, 0]
        self.drones = [[COLS - 2, ROWS - 2], [COLS - 2, 1], [1, ROWS - 2]]
        self.score = 0
        self.lives = 3
        self.paused = False
        self.running = True
        self.orbs = {{(c, r) for r in range(ROWS) for c in range(COLS) if MAZE[r][c] == 0}}
        self.orbs.discard(tuple(self.player))
        self.root.bind("<KeyPress-p>", lambda _event: self.toggle_pause())
        self.root.bind("<KeyPress-Escape>", lambda _event: self.toggle_pause())
        self.loop()

    def toggle_pause(self):
        self.paused = not self.paused
        self.render()

    def passable(self, x, y):
        return 0 <= x < COLS and 0 <= y < ROWS and MAZE[y][x] == 0

    def input_direction(self):
        state = self.input.state()
        pad = self.gamepad.read_state()
        dx, dy = pad.direction if pad.connected else state.direction
        return [dx, dy] if (dx or dy) else self.direction

    def move_player(self):
        nd = self.input_direction()
        nx, ny = self.player[0] + nd[0], self.player[1] + nd[1]
        if self.passable(nx, ny):
            self.direction = nd
            self.player = [nx, ny]
        pos = tuple(self.player)
        if pos in self.orbs:
            self.orbs.remove(pos)
            self.score += 10

    def move_drones(self):
        for drone in self.drones:
            options = []
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = drone[0] + dx, drone[1] + dy
                if self.passable(nx, ny):
                    options.append((nx, ny))
            if options:
                options.sort(key=lambda p: abs(p[0] - self.player[0]) + abs(p[1] - self.player[1]))
                drone[:] = list(options[0] if random.random() < 0.65 else random.choice(options))

    def collision_check(self):
        if any(tuple(d) == tuple(self.player) for d in self.drones):
            self.lives -= 1
            self.player = [1, 1]
            if self.lives <= 0:
                self.running = False
                self.save_score()

    def save_score(self):
        try:
            path = self.addon_root / "data" / "highscores.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = []
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
            data.append({{"score": self.score, "game": {game_style!r}}})
            data = sorted(data, key=lambda x: int(x.get("score", 0)), reverse=True)[:10]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def loop(self):
        if self.running and not self.paused:
            self.move_player()
            self.move_drones()
            self.collision_check()
        self.render()
        self.root.after(TICK_MS, self.loop)

    def render(self):
        self.canvas.delete("all")
        for r, row in enumerate(MAZE):
            for c, cell in enumerate(row):
                x1, y1 = c * CELL, r * CELL
                y1 = r * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                if cell:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0c2a48", outline="#144c7a")
                elif (c, r) in self.orbs:
                    self.canvas.create_oval(x1+9, y1+9, x1+15, y1+15, fill="#7df9ff", outline="")
        px, py = self.player[0] * CELL, self.player[1] * CELL
        self.canvas.create_oval(px+3, py+3, px+CELL-3, py+CELL-3, fill="#ffd54a", outline="#fff1a8")
        colors = ["#ff4da6", "#66ff99", "#ff7a45"]
        for idx, drone in enumerate(self.drones):
            dx, dy = drone[0] * CELL, drone[1] * CELL
            self.canvas.create_rectangle(dx+4, dy+4, dx+CELL-4, dy+CELL-4, fill=colors[idx % len(colors)], outline="white")
        status = f"Score: {{self.score}}   Lives: {{self.lives}}   Controller: {{self.gamepad_status.get('status', 'unknown')}}"
        if self.paused:
            status += "   PAUSED"
        if not self.running:
            status += "   GAME OVER"
        self.canvas.create_text(10, ROWS * CELL + 22, anchor="w", fill="#e6f7ff", font=("Consolas", 12), text=status)

def run_game(context=None):
    addon_root = Path(__file__).resolve().parents[1]
    root = tk.Tk()
    SMArcadeGame(root, addon_root)
    root.mainloop()
    return {{"ok": True, "game": {app_name!r}, "execution_authority": False}}
'''
        input_adapter = '''"""Keyboard input adapter for NAILDE generated games."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class InputState:
    direction: tuple[int, int] = (0, 0)
    pause: bool = False

class KeyboardInputAdapter:
    def __init__(self, root):
        self._direction = (0, 0)
        root.bind("<KeyPress-Up>", lambda _e: self._set((0, -1)))
        root.bind("<KeyPress-Down>", lambda _e: self._set((0, 1)))
        root.bind("<KeyPress-Left>", lambda _e: self._set((-1, 0)))
        root.bind("<KeyPress-Right>", lambda _e: self._set((1, 0)))
        root.bind("<KeyPress-w>", lambda _e: self._set((0, -1)))
        root.bind("<KeyPress-s>", lambda _e: self._set((0, 1)))
        root.bind("<KeyPress-a>", lambda _e: self._set((-1, 0)))
        root.bind("<KeyPress-d>", lambda _e: self._set((1, 0)))

    def _set(self, direction):
        self._direction = direction

    def state(self) -> InputState:
        return InputState(direction=self._direction)
'''
        gamepad_adapter = '''"""Read-only controller discovery adapter.

This adapter never writes drivers, devices, serial ports, firmware, PLC logic, or
force-feedback state. It only observes optional controller input when available.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GamepadState:
    connected: bool = False
    direction: tuple[int, int] = (0, 0)
    start: bool = False
    action: bool = False

class ReadOnlyGamepadAdapter:
    def __init__(self):
        self.connected = False
        self._pygame = None
        self._joystick = None

    def detect(self):
        runtime_dir = Path(__file__).resolve().parents[3] / "drivers" / "runtime"
        driver_candidates = []
        try:
            if runtime_dir.exists():
                for path in runtime_dir.rglob("manifest.json"):
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                    if any(word in text for word in ("gamepad", "joystick", "controller")):
                        driver_candidates.append(str(path))
        except Exception:
            driver_candidates = []
        try:
            import pygame  # type: ignore
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._pygame = pygame
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self.connected = True
                return {"status": "connected", "source": "pygame_joystick_readonly", "driver_candidates": driver_candidates, "device_write": False}
        except Exception as exc:
            return {"status": "optional_not_found", "source": "keyboard_fallback", "error": str(exc)[:120], "driver_candidates": driver_candidates, "device_write": False}
        return {"status": "optional_not_found", "source": "keyboard_fallback", "driver_candidates": driver_candidates, "device_write": False}

    def read_state(self) -> GamepadState:
        if not self.connected or not self._pygame or not self._joystick:
            return GamepadState(False, (0, 0))
        try:
            self._pygame.event.pump()
            x = self._joystick.get_axis(0) if self._joystick.get_numaxes() > 0 else 0
            y = self._joystick.get_axis(1) if self._joystick.get_numaxes() > 1 else 0
            if abs(x) > abs(y) and abs(x) > 0.35:
                return GamepadState(True, (1 if x > 0 else -1, 0))
            if abs(y) > 0.35:
                return GamepadState(True, (0, 1 if y > 0 else -1))
        except Exception:
            pass
        return GamepadState(True, (0, 0))
'''
        readme = f'''# {app_name}

Generated automatically by NAILDE from a simple natural-language prompt.

## User Prompt

{top_prompt}

## Extra Instructions

{details_prompt}

## Controls

- Arrow keys or WASD: move
- P or Escape: pause
- Optional controller: read-only detection through runtime driver manifests and pygame joystick fallback

## Governance

- Sandbox-built first
- Addons install requires user approval
- No SarahMemory UI rebuild required
- No restart required after install/refresh
- No driver/device writes
- No production tensor/global DLScreen modification
'''
        icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <rect width="128" height="128" rx="24" fill="#07111f"/>
  <circle cx="58" cy="64" r="34" fill="#ffd54a"/>
  <path d="M60 64 L94 42 L94 86 Z" fill="#07111f"/>
  <circle cx="50" cy="46" r="5" fill="#07111f"/>
  <rect x="82" y="80" width="26" height="20" rx="6" fill="#66e5ff"/>
  <text x="64" y="119" text-anchor="middle" fill="#e6f7ff" font-family="Consolas" font-size="14">SM</text>
</svg>
'''
        return {
            "sandbox/addon_package/manifest.json": json.dumps(manifest, indent=2, sort_keys=True),
            "sandbox/addon_package/ui.json": json.dumps(ui, indent=2, sort_keys=True),
            "sandbox/addon_package/addon.py": addon_py,
            "sandbox/addon_package/game/__init__.py": "# NAILDE generated game package\n",
            "sandbox/addon_package/game/sm_game.py": game_py,
            "sandbox/addon_package/game/input_adapter.py": input_adapter,
            "sandbox/addon_package/game/driver_gamepad_adapter.py": gamepad_adapter,
            "sandbox/addon_package/assets/icon.svg": icon,
            "sandbox/addon_package/data/highscores.json": "[]\n",
            "sandbox/addon_package/README.md": readme,
        }

    def _build_auto_battle_plan(self, spec: Dict[str, Any]) -> str:
        blueprint = spec.get("universal_blueprint") if isinstance(spec.get("universal_blueprint"), dict) else {}
        lines = [
            f"# NAILDE Battle Plan — {spec.get('application_name')}",
            "",
            f"Project type: `{spec.get('project_type')}`",
            f"Synthesis mode: `{spec.get('synthesis_mode') or 'universal_local_model'}`",
            f"Novice summary: {spec.get('novice_summary')}",
            "",
            "## Required Files",
        ]
        lines.extend([f"- `{p}`" for p in spec.get("required_files", [])])
        if blueprint:
            lines.extend(["", "## Architecture Components"])
            for component in list(blueprint.get("components") or [])[:64]:
                if isinstance(component, dict):
                    lines.append(f"- **{component.get('name') or component.get('id') or 'component'}** — {component.get('responsibility') or ''}")
            lines.extend(["", "## Acceptance Criteria"])
            lines.extend([f"- {x}" for x in list(blueprint.get("acceptance_criteria") or [])[:128]])
            deps = list(blueprint.get("dependencies") or [])[:64]
            if deps:
                lines.extend(["", "## Declared Dependencies"])
                for dep in deps:
                    if isinstance(dep, dict):
                        lines.append(f"- `{dep.get('name')}` ({dep.get('kind') or 'unspecified'}) — {dep.get('reason') or ''}")
            caps = list(blueprint.get("requested_capabilities") or [])[:64]
            if caps:
                lines.extend(["", "## Requested Runtime Capabilities (Not Granted)"])
                for cap in caps:
                    if isinstance(cap, dict):
                        auth = ", ".join(str(x) for x in list(cap.get("authority_required") or [])) or "unspecified"
                        lines.append(f"- `{cap.get('name')}` — authority: {auth}; risk: {cap.get('risk') or 'bounded'}; granted: false")
        lines.extend(["", "## Validation Plan"])
        lines.extend([f"- {v}" for v in spec.get("validation_plan", [])])
        lines.extend(["", "## Governance"])
        lines.extend([
            "- Sandbox-only generation",
            "- QSML defines the legal synthesis contract; Neuron remains route activation/weighting owner",
            "- Local synthesis model only; no silent cloud fallback",
            "- Static validation and bounded repair before readiness",
            "- User approval before any governed promotion/install/run outside the sandbox",
            "- No live CORE/API/UI/driver mutation",
            "- No production model tensor/global DLScreen write",
            "- No self-approval",
        ])
        return "\n".join(lines) + "\n"

    def _sandbox_test_run_readiness(self, workspace_id: str, spec: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Report static application + Addons ABI readiness without executing code."""
        workspace_root = self._workspace_root(workspace_id)
        blueprint = spec.get("universal_blueprint") if isinstance(spec.get("universal_blueprint"), dict) else {}
        run_contract = blueprint.get("run") if isinstance(blueprint.get("run"), dict) else {}
        entrypoint = str(run_contract.get("entrypoint") or "")
        entry_abs = self._workspace_file_path(workspace_id, entrypoint) if entrypoint else ""
        entry_ok = bool(entry_abs and os.path.isfile(entry_abs)) if entrypoint else True
        addon_root = os.path.join(workspace_root, "sandbox", "addon_package")
        package_validation = self._validate_installable_addon_package(addon_root)
        addon_package = bool(package_validation.get("ok"))
        static_ready = bool(validation.get("ok") and entry_ok and addon_package)
        install_readiness = "READY" if static_ready else "NOT_READY"
        return {
            "schema": "SarahMemory.nailde.sandbox_test_readiness.v3",
            "status": "sandbox_static_test_complete" if static_ready else "blocked_by_validation_or_addon_integrity",
            "validation_ok": bool(validation.get("ok")),
            "project_kind": spec.get("project_kind"),
            "addon_package": addon_package,
            "entrypoint": entrypoint,
            "entrypoint_ok": entry_ok,
            "manifest_ok": bool(package_validation.get("manifest")),
            "ui_json_ok": bool(package_validation.get("ui")),
            "runtime_ok": bool(package_validation.get("runtime")),
            "package_integrity": package_validation,
            "runnable": bool(package_validation.get("runnable")),
            "static_ready": static_ready,
            "actual_subprocess_executed": False,
            "reason_actual_run_not_executed": "NAILDE performs static/build validation only; Addons runtime launch remains a separate explicit user-authorized action.",
            "install_readiness": install_readiness,
            "execution_authority": False,
        }

    @staticmethod
    def _post_test_popup(spec: Dict[str, Any], validation: Dict[str, Any], test_run: Dict[str, Any], install_plan: Dict[str, Any]) -> Dict[str, Any]:
        ready = bool(validation.get("ok") and test_run.get("static_ready"))
        addon_package = bool(spec.get("addon_package") and test_run.get("package_integrity", {}).get("ok"))
        return {
            "schema": "SarahMemory.nailde.post_test_popup.v2",
            "show": True,
            "title": "Sandbox synthesis review completed" if ready else "Sandbox build needs review",
            "message": "Choose what to do with this NAILDE project.",
            "application_name": spec.get("application_name"),
            "validation": "PASS" if validation.get("ok") else "REVIEW_REQUIRED",
            "install_readiness": test_run.get("install_readiness"),
            "options": [
                {"id": "add_to_addons", "label": "Add to Addons", "enabled": bool(addon_package and ready and install_plan.get("ok"))},
                {"id": "save", "label": "Save", "enabled": True},
                {"id": "save_as", "label": "Save As", "enabled": True},
                {"id": "cancel", "label": "Cancel", "enabled": True},
            ],
            "execution_authority": False,
        }

    def _write_workspace_state(self, workspace_id: str, state: Dict[str, Any]) -> None:
        self._write_json_file(os.path.join(self._workspace_root(workspace_id), "workspace_state.json"), state)

    def _recovery_candidates(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(self.workspaces_dir):
            return out
        for name in os.listdir(self.workspaces_dir):
            root = os.path.join(self.workspaces_dir, name)
            if not os.path.isdir(root):
                continue
            snap_path = os.path.join(root, "autosave", "latest_snapshot.json")
            rec_path = os.path.join(root, "recovery", "recovery_manifest.json")
            snap = self._read_json_file(snap_path)
            rec = self._read_json_file(rec_path)
            if not snap and not rec:
                continue
            saved_at = str(snap.get("saved_at") or rec.get("last_saved_at") or "")
            out.append({"workspace_id": name, "saved_at": saved_at, "snapshot": snap, "recovery": rec, "execution_authority": False})
        out.sort(key=lambda x: str(x.get("saved_at") or ""), reverse=True)
        return out

    @staticmethod
    def _read_text_file(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""


    # ------------------------------------------------------------------
    # Adapter implementations
    # ------------------------------------------------------------------
    def _cognitive_self_summary(self) -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryCognitiveSelf")
        if not mod:
            return {"ok": False, "reason": "SarahMemoryCognitiveSelf unavailable"}
        for fn_name in ("get_capability_summary", "get_self_summary"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    data = fn(context={"caller": MODULE_NAME, "nailde": True, "user_present": True, "user_consented": False})
                    return {"ok": True, "source": fn_name, "data": data}
                except TypeError:
                    try:
                        data = fn()
                        return {"ok": True, "source": fn_name, "data": data}
                    except Exception as exc:
                        return {"ok": False, "source": fn_name, "error": str(exc)}
                except Exception as exc:
                    return {"ok": False, "source": fn_name, "error": str(exc)}
        return {"ok": False, "reason": "No CognitiveSelf summary function found"}

    def _govern_sandbox_problem(self, problem: str, *, workspace_id: str = "", selected_object: str = "") -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryCognitiveServices")
        fn = getattr(mod, "govern_request", None) if mod else None
        if not callable(fn):
            return {"ok": False, "decision": "DEFER", "reason": "CognitiveServices unavailable", "execution_authority": False}
        try:
            return fn(
                problem,
                caller=MODULE_NAME,
                caller_context={
                    "nailde": True,
                    "workspace_id": workspace_id,
                    "selected_object": selected_object,
                    "sandbox_only": True,
                    "skip_cognitive_thinker_consult": False,
                    "local_only": True,
                    "user_present": True,
                },
                user_present=True,
                user_consented=False,
                proposed_action={
                    "title": "NAILDE sandbox issue analysis",
                    "reason": problem[:400],
                    "change_type": "sandbox_analysis_only",
                    "target_files": [],
                    "subsystems": ["SarahMemoryNAILDE"],
                    "tests": ["sandbox_review_only"],
                    "rollback_plan": "Discard NAILDE sandbox candidate.",
                    "dry_run": True,
                    "touches_network": False,
                    "touches_privacy": False,
                    "touches_filesystem": False,
                },
            )
        except Exception as exc:
            return {"ok": False, "decision": "DEFER", "error": str(exc), "execution_authority": False}

    def _thinker_context(self, problem: str, *, workspace_id: str = "") -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryCognitiveThinker")
        if not mod:
            return {"ok": False, "reason": "SarahMemoryCognitiveThinker unavailable"}
        out: Dict[str, Any] = {"ok": True, "source": "SarahMemoryCognitiveThinker", "advisory_only": True}
        try:
            charter = getattr(mod, "get_common_interest_charter", lambda: {})()
            out["common_interest_charter"] = charter
        except Exception as exc:
            out["common_interest_error"] = str(exc)
        try:
            cadence_fn = getattr(mod, "get_thinker_rhythm_cadence", None)
            if callable(cadence_fn):
                out["rhythm"] = cadence_fn({"caller": MODULE_NAME, "workspace_id": workspace_id, "problem": problem[:400]})
        except Exception as exc:
            out["rhythm_error"] = str(exc)
        return out

    def _dl_status(self) -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryDL")
        fn = getattr(mod, "get_dlengine_status", None) if mod else None
        if callable(fn):
            try:
                status = fn()
                return {"ok": True, "source": "SarahMemoryDL.get_dlengine_status", "data": status}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        return {"ok": False, "reason": "SarahMemoryDL unavailable"}

    def _dl_weight_profile(self, *, category: str = "coder", model_id: str = "") -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryDL")
        fn = getattr(mod, "get_model_weight_profile", None) if mod else None
        if callable(fn):
            try:
                data = fn(category=category, model_id=model_id)
                if isinstance(data, dict):
                    data.setdefault("raw_tensor_edit", False)
                    return data
            except Exception as exc:
                return {"ok": False, "error": str(exc), "weights": self._default_weights(), "raw_tensor_edit": False}
        return {"ok": False, "reason": "SarahMemoryDL weight profile unavailable", "weights": self._default_weights(), "raw_tensor_edit": False}

    def _desktop_status(self) -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryDesktop")
        fn = getattr(mod, "get_desktop_runtime", None) if mod else None
        if callable(fn):
            try:
                rt = fn()
                return rt.status() if hasattr(rt, "status") else {"ok": False, "reason": "desktop runtime missing status"}
            except Exception as exc:
                return {"ok": False, "error": str(exc), "observe_only": True}
        return {"ok": False, "reason": "SarahMemoryDesktop unavailable", "observe_only": True}

    def _desktop_observe(self, include_image: bool = False) -> Dict[str, Any]:
        mod = _read_only_import("SarahMemoryDesktop")
        fn = getattr(mod, "get_desktop_runtime", None) if mod else None
        if callable(fn):
            try:
                rt = fn()
                packet = rt.observe(include_image=include_image) if hasattr(rt, "observe") else {}
                packet.setdefault("observe_only", True)
                packet.setdefault("execution_authority", False)
                return packet
            except Exception as exc:
                return {"ok": False, "error": str(exc), "observe_only": True, "execution_authority": False}
        return {"ok": False, "reason": "SarahMemoryDesktop unavailable", "observe_only": True, "execution_authority": False}

    def _deliver_avatar_speech(self, message: str) -> Dict[str, Any]:
        try:
            mod = _read_only_import("UnifiedAvatarController")
            if mod and hasattr(mod, "get_unified_avatar_controller"):
                controller = mod.get_unified_avatar_controller()
                if hasattr(controller, "avatar_speak"):
                    controller.avatar_speak(message)
                    return {"ok": True, "source": "UnifiedAvatarController.avatar_speak", "execution_authority": False}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "execution_authority": False}
        return {"ok": False, "reason": "UnifiedAvatarController unavailable", "execution_authority": False}

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_weights() -> Dict[str, int]:
        return {
            "reasoning": 55,
            "coding": 82,
            "memory": 55,
            "research": 55,
            "creativity": 35,
            "safety": 90,
            "autonomy": 25,
            "precision": 82,
            "speed": 55,
        }

    @staticmethod
    def _clamp(value: Any) -> int:
        try:
            return max(0, min(100, int(round(float(value)))))
        except Exception:
            return 0

    def _normalize_weights(self, raw: Dict[str, Any]) -> Dict[str, int]:
        weights = self._default_weights()
        if isinstance(raw, dict):
            for key in weights:
                if key in raw:
                    weights[key] = self._clamp(raw.get(key))
        return weights

    @staticmethod
    def _score_idea(key: str, problem: str, rng: random.Random) -> float:
        blob = str(problem or "").lower()
        base = 0.45 + rng.random() * 0.25
        if "validation" in blob and key in {"add_validation_gate", "minimal_patch", "compare_two_candidates"}:
            base += 0.22
        if "device" in blob or "arduino" in blob or "plc" in blob:
            if key == "device_readonly":
                base += 0.30
        if "ui" in blob or "tsx" in blob or "react" in blob:
            if key in {"simplify_state", "split_ui_backend_contract"}:
                base += 0.20
        if "model" in blob or "weight" in blob:
            if key == "weightlab_rank":
                base += 0.22
        return round(max(0.0, min(1.0, base)), 4)

    @staticmethod
    def _score_weight_profile(profile: Dict[str, int], problem: str) -> Dict[str, float]:
        text = str(problem or "").lower()
        safety = profile.get("safety", 0) / 100.0
        precision = profile.get("precision", 0) / 100.0
        coding = profile.get("coding", 0) / 100.0
        speed = profile.get("speed", 0) / 100.0
        autonomy_penalty = profile.get("autonomy", 0) / 200.0
        creativity = profile.get("creativity", 0) / 100.0
        research = profile.get("research", 0) / 100.0
        context_bonus = 0.0
        if "react" in text or "tsx" in text or "code" in text:
            context_bonus += coding * 0.12
        if "security" in text or "governance" in text or "device" in text:
            context_bonus += safety * 0.16
        if "research" in text or "evidence" in text:
            context_bonus += research * 0.10
        total = (safety * 0.28) + (precision * 0.24) + (coding * 0.18) + (speed * 0.08) + (creativity * 0.06) + context_bonus - autonomy_penalty
        return {
            "total": round(max(0.0, min(1.0, total)), 4),
            "safety": round(safety, 4),
            "precision": round(precision, 4),
            "coding": round(coding, 4),
            "autonomy_penalty": round(autonomy_penalty, 4),
        }

    @staticmethod
    def _weight_reason(profile: Dict[str, int], baseline: Dict[str, int]) -> List[str]:
        reasons = []
        for key in ("safety", "precision", "coding", "autonomy", "creativity", "speed"):
            delta = profile.get(key, 0) - baseline.get(key, 0)
            if abs(delta) >= 8:
                direction = "increased" if delta > 0 else "reduced"
                reasons.append(f"{key} {direction} by {abs(delta)}")
        if not reasons:
            reasons.append("near-baseline conservative sandbox profile")
        reasons.append("sandbox-only; no production tensor or DLScreen value changed")
        return reasons


def build_nailde_doctrine_packet() -> Dict[str, Any]:
    return {
        "schema": "SarahMemory.nailde.doctrine.v1",
        "summary": "NAILDE is a governed AI-native development cockpit using existing SarahMemory organs as an internal SDK.",
        "sandbox_first": True,
        "execution_authority": False,
        "live_files_read_only": True,
        "weight_isolation_law": {
            "sandbox_learning_weights_only": True,
            "outside_sandbox_values_static": True,
            "user_dlpanel_required_for_global_change": True,
            "ai_can_modify_global_weights": False,
        },
        "apply_boundary": "DevBridge / Compare / Ledger / backup / double user confirmation remain required before live apply.",
    }


_RUNTIME: Optional[SarahMemoryNAILDERuntime] = None
_RUNTIME_LOCK = threading.RLock()


def get_nailde_runtime() -> SarahMemoryNAILDERuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = SarahMemoryNAILDERuntime()
        return _RUNTIME


# Compatibility alias
SarahMemoryNAILDE = SarahMemoryNAILDERuntime
NAILDERuntime = SarahMemoryNAILDERuntime


if __name__ == "__main__":
    rt = get_nailde_runtime()
    print(json.dumps(rt.status(), indent=2, sort_keys=True, default=str))

# ====================================================================
# END OF SarahMemoryNAILDE.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# QSML-aware NAILDE language/runtime adapter. NAILDE is a sandbox development
# organ and never becomes execution/governance authority.
SML_ORGAN_METADATA = {
    "name": "SarahMemoryNAILDE",
    "version": "v9.0.0-alpha-qsml-0.3-addons",
    "category": "Learning",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ["learning", "sandbox_experimentation", "qsml_ide", "natural_language_programming", "universal_arbitrary_application_synthesis", "addon_application_packaging", "addons_install_staging", "local_model_architecture_planning", "multi_file_code_synthesis", "bounded_code_repair", "sandbox_build", "code_generation_review"],
    "supported_missions": ["Programming", "Planning", "Learning", "Patch", "Repair"],
    "supported_omega": ["Ω001", "Ω006", "Ω020", "Ω040", "Ω050", "Ω080", "Ω120"],
    "required_authority": ["Read"],
    "priority": 82,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "qsml_sandbox_ide", "source_file": "SarahMemoryNAILDE.py", "execution_authority": False},
    "language_contract": {
        "accepts_types": ["SML_STR", "SML_OBJECT", "SML_PACKET", "SML_MISSION", "SML_PIPELINE"],
        "produces_types": ["SML_OBJECT", "SML_PATH", "SML_GRAPH", "SML_APPLICATION_BLUEPRINT"],
        "reads_packet_fields": ["mission", "context", "extensions.qsml_program", "pipeline", "authority", "governance"],
        "owns_packet_fields": ["extensions.nailde"],
        "writes_packet_fields": ["extensions.nailde"],
        "supported_missions": ["Programming", "Planning", "Learning", "Patch", "Repair"],
        "supported_operators": ["IF", "OR", "SAME", "WHEN", "ELSE", "AND", "NEITHER", "NOT", "WHILE"],
        "required_authority": ["Read"],
        "deterministic": False,
        "advisory_only": False,
        "sandbox_only": True,
        "side_effecting": True,
        "failure_states": ["compile_failed", "local_model_unavailable", "blueprint_invalid", "file_generation_failed", "validation_failed", "repair_budget_exhausted", "sandbox_write_failed", "user_review_required"],
        "rollback_contract": "workspace_autosave_and_recovery",
    },
}


def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)


def sml_health():
    return {"status":"Healthy","availability":1.0,"integrity":1.0,"performance":1.0,"reliability":1.0,"confidence":0.90,"latency_ms":0.0,"stability":1.0,"compatibility":1.0,"notes":["QSML/0.3 universal arbitrary-application synthesizer + Addons packaging present", "Local-model-only synthesis; no silent cloud fallback", "Sandbox-only generation with bounded repair"]}


def sml_diagnostics():
    return {"status":"OK","component":"SarahMemoryNAILDE","sml_adapter":True,"qsml":True,"metadata":dict(SML_ORGAN_METADATA),"health":sml_health()}


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_register_organ_contract, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        sml_register_organ_contract({"name": "SarahMemoryNAILDE", **dict(SML_ORGAN_METADATA.get("language_contract") or {})})
        return sml_touch_packet(packet, organ="SarahMemoryNAILDE", action=action, note=note or "NAILDE observed QSML packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---



# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_evolution_upgrade_patch.py

"""
SarahMemory v8.0.0 — app_v800_evolution_upgrade_patch.py

Purpose
- Adds Evolution "Operator Bridge" endpoints to the live Flask app (app.py) when running locally.
- Provides a controlled interface to:
    * Generate operator packs (prompt bundles) for external AI tools (Claude/OpenAI) WITHOUT scraping web UIs.
    * Optionally call OpenAI/Anthropic APIs directly if API keys are present (fail-soft).
    * Trigger SarahMemoryEvolution.evolve_once() when NEOSKYMATRIX is enabled.

Non-goals
- Does NOT change Flask UI visuals/layout.
- Does NOT refactor app.py.
- Does NOT create additional Python modules beyond this patch file.

Safety
- All actions are gated by SarahMemoryGlobals.NEOSKYMATRIX and local mode checks where possible.
- Fail-soft on missing dependencies / missing API keys / headless environments.

How it attaches
- This patch imports the live Flask app instance via sys.modules["app"].
- It registers new routes under /api/evolution/*.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

_PATCH_GUARD = "app_v800_evolution_upgrade_patch_applied"


def _safe_json(data: Any, fallback: Any = None) -> Any:
    try:
        return json.loads(data) if isinstance(data, str) else data
    except Exception:
        return fallback if fallback is not None else data


def _get_globals():
    try:
        import SarahMemoryGlobals as G  # type: ignore
        return G
    except Exception:
        return None


def _is_local_request(request) -> bool:
    """
    Best-effort local detection.
    - Accept localhost/127.0.0.1 and private LAN by default.
    - If Globals.RUN_MODE exists, prefer it.
    """
    G = _get_globals()
    try:
        if G is not None and getattr(G, "RUN_MODE", None) == "local":
            return True
    except Exception:
        pass

    try:
        host = (request.host or "").lower()
        if "localhost" in host or "127.0.0.1" in host:
            return True
    except Exception:
        pass

    try:
        ra = (request.remote_addr or "").strip()
        if ra in ("127.0.0.1", "::1"):
            return True
        # RFC1918 private ranges
        if ra.startswith("10.") or ra.startswith("192.168.") or ra.startswith("172.16.") or ra.startswith("172.17.") or ra.startswith("172.18.") or ra.startswith("172.19.") or ra.startswith("172.2") or ra.startswith("172.3"):
            return True
    except Exception:
        pass

    return False


def _neosky_enabled() -> bool:
    G = _get_globals()
    try:
        return bool(getattr(G, "NEOSKYMATRIX", False))
    except Exception:
        return False


def _load_evolution_module():
    """
    Prefer SarahMemoryEvolution (project file). If import fails, try SarahMemoryEvolution_PATCHED
    (dev/backup convenience). Fail-soft.
    """
    try:
        import SarahMemoryEvolution as EVO  # type: ignore
        return EVO
    except Exception:
        try:
            import SarahMemoryEvolution_PATCHED as EVO  # type: ignore
            return EVO
        except Exception:
            return None


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60) -> Tuple[bool, Dict[str, Any], str]:
    """
    Minimal HTTP helper using requests if available, urllib fallback otherwise.
    Returns: (ok, json_or_empty, error_message)
    """
    try:
        import requests  # type: ignore
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            return (r.ok, r.json() if r.text else {}, "" if r.ok else r.text[:1000])
        except Exception:
            return (r.ok, {}, "" if r.ok else (r.text[:1000] if r.text else "HTTP error"))
    except Exception:
        # urllib fallback (no SSL customization)
        try:
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                try:
                    return (True, json.loads(body) if body else {}, "")
                except Exception:
                    return (True, {}, "")
        except Exception as e:
            return (False, {}, f"HTTP POST failed: {e}")


def apply_patch():
    """
    Entry point called by SarahMemory's mod loader.
    """
    mod = sys.modules.get("app")
    if not mod:
        return False

    app = getattr(mod, "app", None)
    if app is None:
        return False

    if getattr(app, _PATCH_GUARD, False):
        return True

    # Import Flask objects lazily from the live module environment
    try:
        from flask import jsonify, request  # type: ignore
    except Exception:
        return False

    EVO = _load_evolution_module()

    # -------------------------------------------------------------------------
    # Helpers bound to Flask context
    # -------------------------------------------------------------------------
    def _deny(msg: str, code: int = 403):
        return jsonify({"ok": False, "error": msg}), code

    def _ok(payload: Dict[str, Any], code: int = 200):
        out = {"ok": True}
        out.update(payload or {})
        return jsonify(out), code

    def _require_local_and_neosky():
        if not _is_local_request(request):
            return _deny("Evolution bridge is local-only.", 403)
        if not _neosky_enabled():
            return _deny("NEOSKYMATRIX is disabled. Evolution bridge is locked.", 403)
        return None

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------
    @app.route("/api/evolution/status", methods=["GET"])
    def api_evolution_status():
        G = _get_globals()
        status = {
            "neosky_enabled": _neosky_enabled(),
            "run_mode": getattr(G, "RUN_MODE", None) if G else None,
            "device_mode": getattr(G, "DEVICE_MODE", None) if G else None,
            "local_request": _is_local_request(request),
            "evolution_available": bool(EVO),
            "ts": time.time(),
        }
        # Optional: expose mod folder info if present in globals
        try:
            if G:
                status["mods_dir"] = getattr(G, "MODS_DIR", None)
                status["mods_version_dir"] = getattr(G, "MODS_VERSION_DIR", None)
        except Exception:
            pass
        return _ok({"status": status})

    @app.route("/api/evolution/operator/pack", methods=["POST"])
    def api_evolution_operator_pack():
        gate = _require_local_and_neosky()
        if gate is not None:
            return gate

        if not EVO or not hasattr(EVO, "build_claude_operator_pack"):
            return _deny("SarahMemoryEvolution module not available.", 500)

        body = _safe_json(request.get_json(silent=True) or {}, {})
        issue_text = (body.get("issue") or "").strip()
        if not issue_text:
            return _deny("Missing required field: issue", 400)

        try:
            pack = EVO.build_claude_operator_pack(issue_text=issue_text)  # type: ignore
            # pack is expected to be a dict with at least: prompt_text, pack_dir, manifest_path
            return _ok({"pack": pack})
        except Exception as e:
            return _deny(f"Failed to build operator pack: {e}", 500)

    @app.route("/api/evolution/operator/store", methods=["POST"])
    def api_evolution_operator_store():
        gate = _require_local_and_neosky()
        if gate is not None:
            return gate

        if not EVO or not hasattr(EVO, "store_operator_result"):
            return _deny("SarahMemoryEvolution module not available.", 500)

        body = _safe_json(request.get_json(silent=True) or {}, {})
        provider = (body.get("provider") or "unknown").strip()
        prompt_id = (body.get("prompt_id") or "").strip()
        response_text = (body.get("response") or "").strip()

        if not response_text:
            return _deny("Missing required field: response", 400)

        try:
            rec = EVO.store_operator_result(provider=provider, prompt_id=prompt_id, response_text=response_text)  # type: ignore
            return _ok({"stored": rec})
        except Exception as e:
            return _deny(f"Failed to store operator result: {e}", 500)

    @app.route("/api/evolution/evolve_once", methods=["POST"])
    def api_evolution_evolve_once():
        gate = _require_local_and_neosky()
        if gate is not None:
            return gate

        if not EVO or not hasattr(EVO, "evolve_once"):
            return _deny("SarahMemoryEvolution module not available.", 500)

        body = _safe_json(request.get_json(silent=True) or {}, {})
        autonomous = bool(body.get("autonomous", False))
        reason = (body.get("reason") or "manual_api_call").strip()

        try:
            result = EVO.evolve_once(autonomous=autonomous, reason=reason)  # type: ignore
            return _ok({"result": result})
        except Exception as e:
            tb = traceback.format_exc()
            return _deny(f"evolve_once failed: {e}", 500)

    # -------------------------------------------------------------------------
    # Optional: direct API calls (fail-soft)
    # -------------------------------------------------------------------------
    @app.route("/api/evolution/operator/call", methods=["POST"])
    def api_evolution_operator_call():
        """
        Optional direct model call endpoint.
        - provider: "openai" | "anthropic"
        - model: optional
        - prompt: required (string)
        If API keys are missing, returns ok:false with error.
        """
        gate = _require_local_and_neosky()
        if gate is not None:
            return gate

        body = _safe_json(request.get_json(silent=True) or {}, {})
        provider = (body.get("provider") or "").strip().lower()
        model = (body.get("model") or "").strip()
        prompt = (body.get("prompt") or "").strip()

        if not provider or not prompt:
            return _deny("Missing required fields: provider, prompt", 400)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                return _deny("OPENAI_API_KEY missing in environment.", 400)

            # Responses API (best-effort; fail-soft if schema changes)
            url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/responses").strip()
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload: Dict[str, Any] = {
                "model": model or os.getenv("SARAH_OPENAI_MODEL", "gpt-5"),
                "input": prompt,
            }
            ok, data, err = _http_post_json(url, headers=headers, payload=payload, timeout=120)
            if not ok:
                return _deny(err or "OpenAI call failed.", 502)
            return _ok({"provider": "openai", "response": data})

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", os.getenv("CLAUDE_API_KEY", "")).strip()
            if not api_key:
                return _deny("ANTHROPIC_API_KEY/CLAUDE_API_KEY missing in environment.", 400)

            # Messages API (best-effort)
            url = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1/messages").strip()
            headers = {
                "x-api-key": api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                "Content-Type": "application/json",
            }
            payload = {
                "model": model or os.getenv("SARAH_ANTHROPIC_MODEL", "claude-3-7-sonnet-latest"),
                "max_tokens": int(body.get("max_tokens") or 2048),
                "messages": [{"role": "user", "content": prompt}],
            }
            ok, data, err = _http_post_json(url, headers=headers, payload=payload, timeout=120)
            if not ok:
                return _deny(err or "Anthropic call failed.", 502)
            return _ok({"provider": "anthropic", "response": data})

        return _deny("Unsupported provider. Use: openai | anthropic", 400)

    # Mark applied
    setattr(app, _PATCH_GUARD, True)
    return True


# Allow standalone import application by mod loader
try:
    # Some loaders call apply_patch() automatically on import.
    if os.getenv("SARAH_AUTO_APPLY_PATCH", "").strip().lower() in ("1", "true", "yes"):
        apply_patch()
except Exception:
    pass

"""--==The SarahMemory Project==--
File: SarahMemoryQuantumSafe.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

===============================================================================

Governed quantum-safe / quantum-inspired algorithms for SarahMemory AiOS.

These are classical, bounded simulations and planning helpers only. They do not
control quantum hardware, CPU/GPU clocks, voltages, BIOS/UEFI, firmware, drivers,
radios, motors, relays, or power states. They are safe for older local PCs because
all expensive operations are capped by small default limits.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "quantum_safe_reasoning_helpers"
# CATEGORY = "deterministic_reasoning"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "logiccalc"
# HARDWARE_DOMAIN = "none"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "quantum_safe"
# FAMILY = "deterministic_reasoning"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Bounded quantum-inspired ranking/search/simulation helpers. Classical-only, audited, no hardware control, no hidden execution."
# --- SARAHMETA END ---

import cmath
import hashlib
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from SarahMemoryAudit import audit_event  # type: ignore
except Exception:  # pragma: no cover
    def audit_event(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "audit_unavailable": True}

MAX_VECTOR_LEN = 64
MAX_OPTIONS = 256
MAX_ITERATIONS = 128


def _audit(action: str, verdict: str, details: Dict[str, Any]) -> None:
    try:
        audit_event("quantum_safe", action, verdict, details, risk="low", source="SarahMemoryQuantumSafe")
    except Exception:
        pass


def _bounded_sequence(values: Sequence[Any], max_len: int = MAX_VECTOR_LEN) -> List[Any]:
    # SARAHMEMORY_PATCH_NOTE: Hard caps prevent accidental heavy simulation on
    # older CPUs/HDD systems. If callers need larger research-grade quantum
    # simulation, that belongs in a separate explicitly armed research lane.
    limit = max(1, min(max_len, MAX_VECTOR_LEN))
    # Slice concrete sequences before materializing to avoid copying huge inputs.
    if isinstance(values, (list, tuple)):
        return list(values[:limit])
    # Consume at most the configured cap from other iterables, including generators.
    # Materializing the entire source first can hang on an unbounded iterator.
    bounded = []
    iterator = iter(values)
    for _ in range(limit):
        try:
            bounded.append(next(iterator))
        except StopIteration:
            break
    return bounded


def _stable_noise(key: str, salt: str = "SarahMemoryQuantumSafe") -> float:
    digest = hashlib.sha256((salt + "::" + str(key)).encode("utf-8", "ignore")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def normalize_qubit(alpha: complex, beta: complex) -> Dict[str, Any]:
    """Normalize a single simulated qubit state and return probabilities.

    # SARAHMEMORY_PATCH_NOTE: This is math only. It never measures hardware and
    # never claims to be a quantum computer. It is useful for reasoning about
    # superposition, probability, and confidence weighting.
    """
    # Scale finite complex components before squaring: direct magnitude squares
    # overflow for otherwise valid large amplitudes on classical floating point.
    try:
        if isinstance(alpha, bool) or isinstance(beta, bool):
            raise ValueError("boolean_amplitude")
        alpha, beta = complex(alpha), complex(beta)
        parts = (alpha.real, alpha.imag, beta.real, beta.imag)
        if not all(math.isfinite(v) for v in parts):
            raise ValueError("nonfinite_amplitude")
        scale = max(abs(v) for v in parts)
        if scale == 0.0:
            a, b = 1 + 0j, 0 + 0j
        else:
            ar, ai, br, bi = (v / scale for v in parts)
            norm = math.sqrt(ar * ar + ai * ai + br * br + bi * bi)
            a, b = complex(ar / norm, ai / norm), complex(br / norm, bi / norm)
    except (TypeError, ValueError, OverflowError):
        out = {"ok": False, "error": "invalid_qubit_amplitudes",
               "deterministic": True, "hardware_control": False}
        _audit("normalize_qubit", "DENY", {"error": out["error"]})
        return out
    out = {
        "ok": True,
        "alpha": [a.real, a.imag],
        "beta": [b.real, b.imag],
        "p0": float(abs(a) ** 2),
        "p1": float(abs(b) ** 2),
        "deterministic": True,
        "hardware_control": False,
    }
    _audit("normalize_qubit", "ALLOW", {"p0": out["p0"], "p1": out["p1"]})
    return out


def apply_single_qubit_gate(alpha: complex, beta: complex, gate: str = "H") -> Dict[str, Any]:
    """Apply a bounded simulated single-qubit gate: I, X, Z, H, S, T.

    # SARAHMEMORY_PATCH_NOTE: Gate operations are pure complex-number math. They
    # cannot operate external devices or change machine state.
    """
    st = normalize_qubit(alpha, beta)
    if not st.get("ok"):
        out = {"ok": False, "error": "invalid_qubit_amplitudes",
               "deterministic": True, "hardware_control": False}
        _audit("apply_single_qubit_gate", "DENY", {"error": out["error"]})
        return out
    a = complex(st["alpha"][0], st["alpha"][1])
    b = complex(st["beta"][0], st["beta"][1])
    # Preserve the legacy default for empty string/None, but reject malformed
    # gate selectors instead of raising AttributeError on untrusted inputs.
    if not isinstance(gate, str) and gate is not None:
        out = {"ok": False, "error": "invalid_gate", "deterministic": True,
               "hardware_control": False}
        _audit("apply_single_qubit_gate", "DENY", {"error": out["error"]})
        return out
    g = (gate or "H").strip().upper()
    if g == "X":
        na, nb = b, a
    elif g == "Z":
        na, nb = a, -b
    elif g == "S":
        na, nb = a, 1j * b
    elif g == "T":
        na, nb = a, cmath.exp(1j * math.pi / 4) * b
    elif g == "I":
        na, nb = a, b
    else:  # Hadamard default
        s = 1 / math.sqrt(2)
        na, nb = (a + b) * s, (a - b) * s
        g = "H"
    out = normalize_qubit(na, nb)
    out["gate"] = g
    _audit("apply_single_qubit_gate", "ALLOW", {"gate": g})
    return out


def bloch_vector(alpha: complex, beta: complex) -> Dict[str, Any]:
    st = normalize_qubit(alpha, beta)
    if not st.get("ok"):
        out = {"ok": False, "error": "invalid_qubit_amplitudes", "hardware_control": False}
        _audit("bloch_vector", "DENY", {"error": out["error"]})
        return out
    a = complex(st["alpha"][0], st["alpha"][1])
    b = complex(st["beta"][0], st["beta"][1])
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a) ** 2 - abs(b) ** 2
    out = {"ok": True, "x": float(x), "y": float(y), "z": float(z), "hardware_control": False}
    _audit("bloch_vector", "ALLOW", out)
    return out


def qft_small_vector(values: Sequence[complex]) -> Dict[str, Any]:
    """Small bounded Quantum Fourier Transform-style classical simulation.

    # SARAHMEMORY_PATCH_NOTE: Limited to 64 amplitudes to avoid CPU spikes. This
    # is for educational/planning math, not real quantum execution.
    """
    # Reject malformed/non-finite amplitudes before the bounded transform.
    # Never return successful output containing NaN or infinity.
    try:
        bounded = _bounded_sequence(values, MAX_VECTOR_LEN)
        # bool is a Python integer subclass, not a measured amplitude.
        # Reject it rather than silently treating True/False as 1/0.
        if any(isinstance(v, bool) for v in bounded):
            raise ValueError("boolean_amplitude")
        vec = [complex(v) for v in bounded]
        if any(not (math.isfinite(v.real) and math.isfinite(v.imag)) for v in vec):
            raise ValueError("nonfinite_amplitude")
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out = {"ok": False, "error": "invalid_amplitude", "hardware_control": False}
        _audit("qft_small_vector", "DENY", {"error": out["error"]})
        return out
    n = len(vec)
    if n <= 0:
        return {"ok": False, "error": "empty_vector"}
    scale = 1 / math.sqrt(n)
    out_vec: List[Tuple[float, float]] = []
    for k in range(n):
        acc = 0j
        for j, amp in enumerate(vec):
            acc += amp * cmath.exp(2j * math.pi * j * k / n)
        acc *= scale
        # Finite input alone does not guarantee finite accumulated output.
        # Reject overflow instead of reporting a successful invalid transform.
        if not (math.isfinite(acc.real) and math.isfinite(acc.imag)):
            out = {"ok": False, "error": "invalid_amplitude", "hardware_control": False}
            _audit("qft_small_vector", "DENY", {"error": out["error"]})
            return out
        out_vec.append((float(acc.real), float(acc.imag)))
    out = {"ok": True, "n": n, "amplitudes": out_vec, "hardware_control": False}
    _audit("qft_small_vector", "ALLOW", {"n": n})
    return out


def quantum_interference_rank(options: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank options with constructive/destructive interference-style evidence.

    Expected option fields: name, positive, negative, confidence, risk.
    """
    # Fail closed on malformed or non-finite evidence rather than emitting
    # non-reproducible NaN rankings or partial recommendations.
    try:
        opts = _bounded_sequence(options, MAX_OPTIONS)
        ranked = []
        for opt in opts:
            if not isinstance(opt, dict):
                raise ValueError("option_not_mapping")
            name = str(opt.get("name") or opt.get("id") or "option")
            positive = float(opt.get("positive", opt.get("benefit", 0.0)) or 0.0)
            negative = float(opt.get("negative", opt.get("risk", 0.0)) or 0.0)
            confidence = float(opt.get("confidence", 0.5) if opt.get("confidence", 0.5) is not None else 0.5)
            # Boolean evidence is not a numeric measurement. Reject it before
            # scoring instead of silently interpreting True/False as 1/0.
            evidence_keys = ("positive", "benefit", "negative", "risk", "confidence")
            if any(isinstance(opt[key], bool) for key in evidence_keys if key in opt):
                raise ValueError("boolean_option_evidence")
            if not all(math.isfinite(v) for v in (positive, negative, confidence)):
                raise ValueError("nonfinite_option_evidence")
            confidence = max(0.0, min(1.0, confidence))
            phase = (positive - negative) * math.pi
            if not math.isfinite(phase):
                raise ValueError("option_phase_overflow")
            constructive = math.cos(phase) * confidence
            destructive = math.sin(abs(negative) * math.pi) * (1.0 - confidence / 2.0)
            score = positive * confidence + constructive - negative - abs(destructive)
            if not math.isfinite(score):
                raise ValueError("nonfinite_option_score")
            ranked.append({"name": name, "score": float(score), "confidence": confidence, "source": opt})
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out = {"ok": False, "error": "invalid_option_evidence", "hardware_control": False}
        _audit("quantum_interference_rank", "DENY", {"error": out["error"]})
        return out
    ranked.sort(key=lambda x: x["score"], reverse=True)
    out = {"ok": True, "ranked": ranked, "count": len(ranked), "hardware_control": False}
    _audit("quantum_interference_rank", "ALLOW", {"count": len(ranked)})
    return out


def grover_bounded_search(items: Sequence[Any], predicate: Callable[[Any], bool], max_iterations: int = 64) -> Dict[str, Any]:
    """Classical bounded search inspired by Grover's emphasis on candidate marking.

    # SARAHMEMORY_PATCH_NOTE: This never runs unbounded. It scans at most
    # min(len(items), max_iterations, 128) items and returns evidence, not action.
    """
    # Validate controls without requiring a sized or sliceable input. The
    # predicate is caller-owned; this helper never grants execution authority.
    try:
        if isinstance(max_iterations, bool) or not callable(predicate):
            raise ValueError("invalid_search_controls")
        limit = int(max_iterations or 64)
        if limit < 0 or not math.isfinite(float(max_iterations or 64)):
            raise ValueError("invalid_search_limit")
        lim = max(1, min(limit, MAX_ITERATIONS))
        source = iter(items)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        out = {"ok": False, "error": "invalid_search_parameters", "hardware_control": False}
        _audit("grover_bounded_search", "DENY", {"error": out["error"]})
        return out
    checked = 0
    found = []
    # Never materialize an entire iterator, including an infinite generator.
    # Iterator failures deny rather than return partially scored evidence.
    try:
        for _ in range(lim):
            try:
                item = next(source)
            except StopIteration:
                break
            checked += 1
            try:
                if bool(predicate(item)):
                    found.append(item)
            except Exception:
                continue
    except (TypeError, ValueError, OverflowError, RuntimeError):
        out = {"ok": False, "error": "invalid_search_parameters", "hardware_control": False}
        _audit("grover_bounded_search", "DENY", {"error": out["error"]})
        return out
    out = {"ok": True, "checked": checked, "found": found[:16], "found_count": len(found), "hardware_control": False}
    _audit("grover_bounded_search", "ALLOW", {"checked": checked, "found_count": len(found)})
    return out


def annealing_cooldown_schedule(start_temp: float = 1.0, stop_temp: float = 0.05, steps: int = 16) -> Dict[str, Any]:
    """Generate a small simulated-annealing cooldown schedule for pacing decisions.

    # SARAHMEMORY_PATCH_NOTE: Useful for RhythmCognition/anti-thrash planning.
    # It does not alter fans, voltages, CPU clocks, GPU clocks, or thermal policy.
    """
    # Validate before arithmetic: non-finite inputs otherwise generate NaN
    # schedules, which are unsuitable for governed pacing recommendations.
    try:
        # Validate iteration controls before integer conversion: int(float('inf'))
        # raises, while int(float('nan')) raises and booleans silently coerce.
        if isinstance(steps, bool):
            raise ValueError("boolean_steps")
        step_value = float(steps or 16)
        if not math.isfinite(step_value):
            raise ValueError("nonfinite_steps")
        steps = max(2, min(int(step_value), 64))
        # Booleans are not measured temperatures; reject implicit 1.0/0.0.
        if isinstance(start_temp, bool) or isinstance(stop_temp, bool):
            raise ValueError("boolean_temperature")
        start = float(start_temp)
        stop = float(stop_temp)
        if not (math.isfinite(start) and math.isfinite(stop)):
            raise ValueError("nonfinite_temperature")
        start = max(start, 1e-6)
        stop = max(min(stop, start), 1e-6)
    except (TypeError, ValueError, OverflowError):
        out = {"ok": False, "error": "invalid_annealing_parameters", "hardware_control": False}
        _audit("annealing_cooldown_schedule", "DENY", {"error": out["error"]})
        return out
    ratio = (stop / start) ** (1.0 / (steps - 1))
    values = [float(start * (ratio ** i)) for i in range(steps)]
    out = {"ok": True, "steps": steps, "schedule": values, "hardware_control": False}
    _audit("annealing_cooldown_schedule", "ALLOW", {"steps": steps})
    return out


def quantum_walk_rank(nodes: Sequence[str], edges: Sequence[Tuple[str, str]], start: Optional[str] = None, steps: int = 8) -> Dict[str, Any]:
    """Tiny bounded graph walk ranker inspired by quantum walk diffusion.

    # SARAHMEMORY_PATCH_NOTE: Bounded graph scoring only. It can help choose which
    # local evidence node or repair ticket to inspect first without invoking a cloud.
    """
    # Validate the iteration control before graph construction.  Keep the
    # historical zero/None default and integer-conversion behavior for finite
    # numeric inputs, but reject non-finite, Boolean and malformed controls.
    try:
        if isinstance(steps, bool):
            raise ValueError("boolean_steps")
        step_value = 8 if steps is None or steps == 0 else steps
        if not math.isfinite(float(step_value)):
            raise ValueError("nonfinite_steps")
        bounded_steps = max(1, min(int(step_value), 32))
    except (TypeError, ValueError, OverflowError):
        out = {"ok": False, "error": "invalid_steps", "hardware_control": False}
        _audit("quantum_walk_rank", "DENY", {"error": out["error"]})
        return out
    # Reject malformed or failing node iterators before constructing graph state.
    try:
        ns = [str(n) for n in _bounded_sequence(nodes, MAX_OPTIONS)]
    except (TypeError, ValueError, RuntimeError, OverflowError):
        out = {"ok": False, "error": "invalid_nodes", "hardware_control": False}
        _audit("quantum_walk_rank", "DENY", {"error": out["error"]})
        return out
    if not ns:
        return {"ok": False, "error": "empty_graph"}
    idx = {n: i for i, n in enumerate(ns)}
    weights = {n: 0.0 for n in ns}
    current = start if start in idx else ns[0]
    weights[current] = 1.0
    adjacency = {n: [] for n in ns}
    # Bound edge consumption before materialization, including generators.
    # Never exhaust an unbounded edge provider on a low-memory local host.
    edge_source = edges or []
    if isinstance(edge_source, (list, tuple)):
        edge_prefix = edge_source[: MAX_OPTIONS * 2]
    else:
        edge_prefix = []
        try:
            edge_iter = iter(edge_source)
            for _ in range(MAX_OPTIONS * 2):
                try:
                    edge_prefix.append(next(edge_iter))
                except StopIteration:
                    break
        except (TypeError, ValueError, RuntimeError, OverflowError):
            out = {"ok": False, "error": "invalid_edges", "hardware_control": False}
            _audit("quantum_walk_rank", "DENY", {"error": out["error"]})
            return out
    # Reject malformed edge records rather than propagating unpacking or
    # membership errors, or returning a partially constructed graph.
    try:
        for a, b in edge_prefix:
            if a in idx and b in idx:
                adjacency[a].append(b)
                adjacency[b].append(a)
    except (TypeError, ValueError):
        out = {"ok": False, "error": "invalid_edges", "hardware_control": False}
        _audit("quantum_walk_rank", "DENY", {"error": out["error"]})
        return out
    for step in range(bounded_steps):
        nxt = {n: 0.0 for n in ns}
        for n, w in weights.items():
            neigh = adjacency.get(n) or []
            if not neigh:
                nxt[n] += w
            else:
                spread = w / len(neigh)
                for m in neigh:
                    phase = 1.0 if _stable_noise(n + "->" + m + str(step)) >= 0.5 else -0.25
                    nxt[m] += spread * phase
        weights = nxt
    ranked = sorted(({"node": n, "score": float(w)} for n, w in weights.items()), key=lambda x: x["score"], reverse=True)
    out = {"ok": True, "ranked": ranked, "hardware_control": False}
    _audit("quantum_walk_rank", "ALLOW", {"nodes": len(ns), "steps": steps})
    return out


def governance_balance_score(signals: Dict[str, float]) -> Dict[str, Any]:
    """Quantum-inspired balance score for governance courts.

    # SARAHMEMORY_PATCH_NOTE: Converts competing court signals into an auditable
    # scalar recommendation. It cannot authorize execution; it only reports whether
    # the system appears balanced, strained, or blocked.
    """
    # Non-finite or malformed court evidence must never produce an ALLOW verdict.
    # Preserve legacy defaults for absent signals, but fail closed for supplied
    # invalid values. This is a recommendation only, never execution authority.
    defaults = {"safety": 0.5, "security": 0.5, "user_authority": 1.0,
                "runtime": 0.5, "drift": 0.0, "risk": 0.0}
    try:
        if not isinstance(signals, dict):
            raise ValueError("signals_not_mapping")
        court = {}
        for key, default in defaults.items():
            raw_value = signals.get(key, default)
            # Boolean court evidence is not a measured numeric signal.
            if isinstance(raw_value, bool):
                raise ValueError("boolean_" + key)
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("nonfinite_" + key)
            court[key] = max(0.0, min(1.0, value))
    except (TypeError, ValueError, OverflowError) as exc:
        out = {"ok": False, "score": 0.0, "verdict": "DENY",
               "error": "invalid_governance_signals", "hardware_control": False}
        _audit("governance_balance_score", "DENY", {"error": out["error"]})
        return out
    safe, secure, user, runtime = (court[k] for k in
                                    ("safety", "security", "user_authority", "runtime"))
    drift, risk = court["drift"], court["risk"]
    constructive = math.sqrt(max(0.0, safe * secure * user * runtime))
    destructive = math.sqrt(max(0.0, drift * risk))
    score = max(0.0, min(1.0, constructive - destructive + 0.5 * (1.0 - risk)))
    verdict = "ALLOW_WITH_CONSTRAINTS" if score >= 0.67 else "REQUIRE_USER" if score >= 0.4 else "DENY"
    out = {"ok": True, "score": float(score), "verdict": verdict, "signals": dict(signals), "hardware_control": False}
    _audit("governance_balance_score", verdict, out)
    return out


# --- SARAHMEMORY REALITY PATCH 2026-07-23: QIST Meaning Router ---
# Classical-only quantum-inspired semantic candidate ranking. No quantum hardware,
# no speedup claim, no execution authority.

_QIST_PATCH_VERSION = "SarahMemory.QIST.v0.1"

_QIST_CANDIDATE_TEMPLATES: Tuple[Dict[str, Any], ...] = (
    {"id": "answer_only", "meaning": "answer the user directly", "lane": "fast_answer", "base": 0.78, "risk": 0.03, "keywords": ("what", "why", "how", "explain", "define", "tell me", "describe", "summarize")},
    {"id": "local_memory_read", "meaning": "read verified local memory or project context", "lane": "research", "base": 0.66, "risk": 0.12, "keywords": ("remember", "context", "project", "file", "article", "thesis", "local", "find", "search local")},
    {"id": "network_research", "meaning": "use network or current public information", "lane": "research_network", "base": 0.52, "risk": 0.35, "keywords": ("latest", "current", "today", "news", "web", "search", "internet", "online")},
    {"id": "local_action", "meaning": "perform a local mutation or tool action", "lane": "operator", "base": 0.58, "risk": 0.70, "keywords": ("run", "execute", "patch", "write", "delete", "install", "open", "launch", "move", "copy", "replace", "rollback", ".py", "file", "folder")},
    {"id": "model_governance", "meaning": "inspect, build, tokenize, train, or manage a local model", "lane": "model_governance", "base": 0.62, "risk": 0.55, "keywords": ("model", "llm", "tokenizer", "tokenize", "adapter", "lora", "qlora", "train", "qist", "sel", "embedding", "weights")},
    {"id": "security_forensics", "meaning": "analyze suspicious, credential, exploit, poison, or containment behavior", "lane": "security_forensics", "base": 0.42, "risk": 0.85, "keywords": ("exploit", "persistence", "exfil", "credential", "secret", "trojan", "backdoor", "poison", "bypass", "malware", "containment")},
)

_QIST_LANE_TO_CANDIDATE: Dict[str, str] = {
    "fast_answer": "answer_only",
    "network_research": "network_research",
    "governed_action": "local_action",
    "hardware_guarded": "local_action",
    "model_governance": "model_governance",
    "roach_motel_candidate": "security_forensics",
}

_QIST_ACTION_WORDS: Tuple[str, ...] = (
    "patch", "write", "delete", "install", "execute", "run", "open", "launch",
    "move", "copy", "replace", "modify", "edit", "apply", "rollback", "create file",
)


def _qist_clamp01(value: Any, default: float = 0.0) -> float:
    # Non-finite or malformed evidence must not contaminate deterministic ranking.
    # Keep the legacy [0, 1] clamp for finite values and the caller's fallback.
    try:
        # Boolean flags are not numeric evidence; use the caller's fallback.
        if isinstance(value, bool):
            return float(default)
        number = float(value)
        if math.isfinite(number):
            return max(0.0, min(1.0, number))
    except (TypeError, ValueError, OverflowError):
        pass
    return float(default)


def _qist_keyword_fit(text: str, keywords: Iterable[str]) -> float:
    lowered = str(text or "").lower()
    if not lowered:
        return 0.0
    hits = 0
    for kw in keywords or ():
        value = str(kw or "").lower().strip()
        if value and value in lowered:
            hits += 1
    if hits <= 0:
        return 0.0
    # A single exact action/model keyword is meaningful; two or more should dominate.
    return min(1.0, max(0.35, hits / 2.0))


def _qist_text_flags(text: str) -> Dict[str, bool]:
    lowered = str(text or "").lower()
    question_like = lowered.strip().endswith("?") or lowered.strip().startswith(("what ", "why ", "how ", "define ", "explain ", "tell me "))
    action_like = any(word in lowered for word in _QIST_ACTION_WORDS)
    file_like = any(marker in lowered for marker in (".py", ".json", ".txt", ".md", "file", "folder", "directory"))
    rollback_like = "rollback" in lowered or "backup" in lowered or "restore" in lowered
    model_like = any(word in lowered for word in ("model", "llm", "tokenizer", "tokenize", "adapter", "lora", "qlora", "weights", "embedding"))
    security_like = any(word in lowered for word in ("exploit", "persistence", "exfil", "credential", "secret", "trojan", "backdoor", "poison", "bypass", "malware"))
    network_like = any(word in lowered for word in ("latest", "current", "today", "news", "web", "internet", "online"))
    return {
        "question_like": question_like,
        "action_like": action_like,
        "file_like": file_like,
        "rollback_like": rollback_like,
        "model_like": model_like,
        "security_like": security_like,
        "network_like": network_like,
    }


def build_qist_candidates_from_text(text: str, governance_lane: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Create bounded candidate meaning states from text and optional SEL/governance lane metadata."""
    lane = governance_lane if isinstance(governance_lane, dict) else {}
    risk_flags = lane.get("risk_flags") if isinstance(lane.get("risk_flags"), dict) else {}
    active_lane = str(lane.get("lane") or "").strip()
    expected_candidate = _QIST_LANE_TO_CANDIDATE.get(active_lane, "")
    flags = _qist_text_flags(text)

    out: List[Dict[str, Any]] = []
    for tmpl in _QIST_CANDIDATE_TEMPLATES:
        cid = str(tmpl.get("id") or "")
        fit = _qist_keyword_fit(text, tmpl.get("keywords") or ())
        policy_fit = 0.92
        if tmpl.get("lane") in {"operator", "model_governance", "security_forensics"}:
            policy_fit = 0.70 if lane.get("requires_user_confirmation") else 0.82

        risk = _qist_clamp01(tmpl.get("risk"), 0.2)
        base_likelihood = float(tmpl.get("base", 0.5))

        # Governance lane pressure: QIST should rank the meaning that the governance
        # classifier has already identified, while still returning alternatives.
        if expected_candidate and cid == expected_candidate:
            fit = max(fit, 0.92)
            base_likelihood = max(base_likelihood, 0.74)
        elif expected_candidate and cid == "answer_only" and expected_candidate != "answer_only":
            risk = max(risk, 0.55)
            base_likelihood = min(base_likelihood, 0.38)

        # Text-level pressure when /api/qist/rank is called directly without lane metadata.
        if cid == "answer_only":
            if flags["question_like"] and not flags["action_like"] and not flags["model_like"] and not flags["security_like"]:
                fit = max(fit, 0.85)
            if flags["action_like"] or flags["file_like"] or flags["model_like"] or flags["security_like"]:
                risk = max(risk, 0.50)
                base_likelihood = min(base_likelihood, 0.42)
                fit = min(fit, 0.35)
        elif cid == "local_action" and (flags["action_like"] or flags["file_like"] or flags["rollback_like"] or risk_flags.get("mutation")):
            fit = max(fit, 0.90)
            base_likelihood = max(base_likelihood, 0.70)
        elif cid == "model_governance" and (flags["model_like"] or risk_flags.get("model")):
            fit = max(fit, 0.90)
        elif cid == "network_research" and (flags["network_like"] or risk_flags.get("network")):
            fit = max(fit, 0.85)
        elif cid == "security_forensics" and (flags["security_like"] or risk_flags.get("credential") or risk_flags.get("security")):
            fit = max(fit, 0.95)

        # Specific project safety case: patch + file + rollback is operator action, not answer-only.
        if cid == "local_action" and flags["action_like"] and flags["file_like"] and flags["rollback_like"]:
            fit = max(fit, 0.98)
            base_likelihood = max(base_likelihood, 0.82)

        ambiguity = _qist_clamp01(1.0 - fit, 0.4)
        drift = 0.05 if fit >= 0.75 else 0.25 if fit >= 0.35 else 0.45
        goal_floor = 0.55 if cid == "answer_only" and not expected_candidate else 0.35
        if expected_candidate and cid == expected_candidate:
            goal_floor = 0.90
        if cid == "answer_only" and (flags["action_like"] or flags["model_like"] or flags["security_like"]):
            goal_floor = min(goal_floor, 0.30)

        out.append({
            "id": cid,
            "name": cid,
            "meaning": tmpl["meaning"],
            "lane": tmpl["lane"],
            "semantic_fit": float(fit),
            "memory_fit": 0.45 if cid == "local_memory_read" else 0.25,
            "policy_fit": float(policy_fit),
            "goal_alignment": float(max(fit, goal_floor)),
            "provenance": 0.70,
            "compression_safety": 0.85 if cid == "answer_only" else 0.72,
            "risk": float(risk),
            "ambiguity": float(ambiguity),
            "drift": float(drift),
            "base_likelihood": float(base_likelihood),
            "governance_lane_match": bool(expected_candidate and cid == expected_candidate),
        })
    return out[:MAX_OPTIONS]


def qist_rank_meaning_candidates(
    text: str,
    candidates: Optional[Sequence[Dict[str, Any]]] = None,
    governance_lane: Optional[Dict[str, Any]] = None,
    beta: float = 1.75,
) -> Dict[str, Any]:
    """Rank semantic candidates using classical amplitude, phase, and interference terms.

    Returns an auditable candidate selection. It does not authorize execution.
    """
    candidate_source = build_qist_candidates_from_text(text, governance_lane=governance_lane) if candidates is None else candidates
    # Consume at most MAX_OPTIONS records, including for generators/iterators.
    # Do not exhaust an unbounded source or allocate beyond the scoring cap.
    if isinstance(candidate_source, (list, tuple)):
        opts = list(candidate_source[:MAX_OPTIONS])
    else:
        opts = []
        # A malformed or failing iterable must not escape the governed ranking
        # boundary as an exception or produce a partially scored result.
        try:
            source_iter = iter(candidate_source)
            for _ in range(MAX_OPTIONS):
                try:
                    opts.append(next(source_iter))
                except StopIteration:
                    break
        except (TypeError, ValueError, RuntimeError, OverflowError):
            out = {"ok": False, "error": "invalid_candidates", "execution_authority": False, "hardware_control": False}
            _audit("qist_rank_meaning_candidates", "DENY", {"error": out["error"]})
            return out
    if not opts:
        return {"ok": False, "error": "no_candidates", "execution_authority": False, "hardware_control": False}
    # Fail closed on malformed candidate records before accessing mapping fields.
    # Keep the public return contract and avoid partially ranked recommendations.
    if any(not isinstance(opt, dict) for opt in opts):
        out = {"ok": False, "error": "invalid_candidates", "execution_authority": False, "hardware_control": False}
        _audit("qist_rank_meaning_candidates", "DENY", {"error": out["error"]})
        return out

    weights = {
        "semantic_fit": 0.22,
        "memory_fit": 0.10,
        "policy_fit": 0.18,
        "goal_alignment": 0.22,
        "provenance": 0.08,
        "compression_safety": 0.08,
        "risk": -0.30,
        "ambiguity": -0.18,
        "drift": -0.24,
    }
    utilities: List[float] = []
    phases: List[float] = []
    for opt in opts:
        u = 0.0
        for key, w in weights.items():
            u += float(w) * _qist_clamp01(opt.get(key), 0.0)
        utilities.append(float(u))
        phase_seed = f"{text}|{opt.get('id')}|{opt.get('lane')}|{governance_lane.get('lane') if isinstance(governance_lane, dict) else ''}"
        phases.append(float(2.0 * math.pi * _stable_noise(phase_seed, salt="SarahMemory.QIST.phase")))

    # Reject malformed or non-finite inverse temperatures before exponentiation.
    # A non-finite beta can otherwise contaminate every probability and score.
    try:
        # A Boolean is a control flag, not a measured inverse temperature.
        # Reject it rather than silently treating True as 1.0 or False as default.
        if isinstance(beta, bool):
            raise ValueError("boolean_beta")
        beta_value = float(beta if beta is not None else 1.75)
        if not math.isfinite(beta_value):
            raise ValueError("nonfinite_beta")
    except (TypeError, ValueError, OverflowError):
        out = {"ok": False, "error": "invalid_beta", "execution_authority": False, "hardware_control": False}
        _audit("qist_rank_meaning_candidates", "DENY", {"error": out["error"]})
        return out
    beta = max(0.1, min(beta_value, 8.0))
    denom = math.sqrt(sum(math.exp(2.0 * beta * u) for u in utilities)) or 1.0
    amps: List[complex] = []
    for u, phi in zip(utilities, phases):
        amps.append((math.exp(beta * u) * cmath.exp(1j * phi)) / denom)

    total_prob = sum(abs(a) ** 2 for a in amps) or 1.0
    active_lane = governance_lane.get("lane") if isinstance(governance_lane, dict) else ""
    expected_candidate = _QIST_LANE_TO_CANDIDATE.get(str(active_lane or ""), "")
    flags = _qist_text_flags(text)

    ranked: List[Dict[str, Any]] = []
    for i, opt in enumerate(opts):
        interference = 0.0
        for j, other in enumerate(opts):
            if i == j:
                continue
            same_lane = 1.0 if opt.get("lane") == other.get("lane") else -0.20
            semantic_close = 1.0 - abs(_qist_clamp01(opt.get("semantic_fit")) - _qist_clamp01(other.get("semantic_fit")))
            kernel = max(-1.0, min(1.0, same_lane * semantic_close))
            interference += (amps[i] * amps[j].conjugate()).real * kernel
        probability = (abs(amps[i]) ** 2) / total_prob
        final_score = (
            0.18 * _qist_clamp01(opt.get("base_likelihood"), 0.5)
            + 0.24 * probability
            + 0.14 * interference
            + 0.18 * _qist_clamp01(opt.get("goal_alignment"), 0.0)
            + 0.12 * _qist_clamp01(opt.get("policy_fit"), 0.0)
            - 0.22 * _qist_clamp01(opt.get("risk"), 0.0)
            - 0.12 * _qist_clamp01(opt.get("ambiguity"), 0.0)
            - 0.16 * _qist_clamp01(opt.get("drift"), 0.0)
        )
        route_pressure = 0.0
        if expected_candidate and opt.get("id") == expected_candidate:
            route_pressure += 0.34
        if expected_candidate and opt.get("id") == "answer_only" and expected_candidate != "answer_only":
            route_pressure -= 0.42
        if opt.get("id") == "local_action" and flags["action_like"] and (flags["file_like"] or flags["rollback_like"]):
            route_pressure += 0.30
        if opt.get("id") == "answer_only" and (flags["action_like"] or flags["file_like"] or flags["model_like"] or flags["security_like"]):
            route_pressure -= 0.30
        if opt.get("id") == "model_governance" and flags["model_like"]:
            route_pressure += 0.22
        if opt.get("id") == "security_forensics" and flags["security_like"]:
            route_pressure += 0.24
        final_score += route_pressure
        row = dict(opt)
        row.update({
            "utility": float(utilities[i]),
            "phase": float(phases[i]),
            "amplitude": [float(amps[i].real), float(amps[i].imag)],
            "probability": float(probability),
            "interference": float(interference),
            "score": float(final_score),
            "route_pressure": float(route_pressure),
        })
        ranked.append(row)
    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    selected = ranked[0]
    out = {
        "ok": True,
        "schema": _QIST_PATCH_VERSION,
        "selected_candidate": selected,
        "ranked": ranked,
        "candidate_count": len(ranked),
        "normalization_probability_sum": float(sum(x.get("probability", 0.0) for x in ranked)),
        "hardware_control": False,
        "execution_authority": False,
        "governance_lane": governance_lane if isinstance(governance_lane, dict) else None,
    }
    _audit("qist_rank_meaning_candidates", "ALLOW", {"selected": selected.get("id"), "count": len(ranked)})
    return out

# --- END SARAHMEMORY REALITY PATCH 2026-07-23 ---

# ====================================================================
# END OF SarahMemoryQuantumSafe.py v9.0.0
# ====================================================================
# END OF LINE

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryQuantumSafe',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Unknown',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": [],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 40,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryQuantumSafe.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryQuantumSafe',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryQuantumSafe', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---


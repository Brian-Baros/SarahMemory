"""
--==The SarahMemory Project==--
File: SarahMemoryLLM.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com
===============================================================================

Purpose: Model verification, download manager, and lightweight routing helpers.
Notes:
- Headless-safe: never requires GUI.
- Only downloads models when missing/invalid locally.
- Uses category-based stacks + hardware tier recommendations from SarahMemoryGlobals.
"""

import os
import json
import logging
import subprocess
import time
import urllib.request
import zlib
import shutil
import threading
import queue
import hashlib
from typing import Dict, List, Tuple, Optional

# Optional deps (never hard-crash)
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except Exception:
    AutoTokenizer = None
    AutoModel = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    from huggingface_hub import snapshot_download, HfApi  # type: ignore
except Exception:
    snapshot_download = None
    HfApi = None

import SarahMemoryGlobals as G

BASE_DIR = getattr(G, "BASE_DIR", os.getcwd())
MODELS_DIR = getattr(G, "MODELS_DIR", os.path.join(BASE_DIR, "data", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging
LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "model_download.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("SarahMemoryLLM")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)

# Imports from Globals (safe defaults)
MODEL_CONFIG = getattr(G, "MODEL_CONFIG", {})
OBJECT_MODEL_CONFIG = getattr(G, "OBJECT_MODEL_CONFIG", {})
MODEL_CATALOG = getattr(G, "MODEL_CATALOG", {})
MODEL_META = getattr(G, "MODEL_META", {})

MULTI_STACK_ENABLED = bool(getattr(G, "MULTI_STACK_ENABLED", False))
resolve_model_repo = getattr(G, "resolve_model_repo", lambda x: x)
get_stack_primary_repo = getattr(G, "get_stack_primary_repo", lambda category, text="", meta=None: "")
recommend_model_tier = getattr(G, "recommend_model_tier", lambda category="reasoning", metrics=None: "low")

model_policy_allows_download = getattr(G, "model_policy_allows_download", lambda size_gb, models_dir=None: {"ok": True, "prompt_required": False})
get_models_storage_usage_gb = getattr(G, "get_models_storage_usage_gb", lambda models_dir=None: 0.0)
is_headless_runtime = getattr(G, "is_headless_runtime", lambda: True)
is_interactive_tty = getattr(G, "is_interactive_tty", lambda: False)

# ---------------------------------------------------------------------------
# Local model folder conventions
# ---------------------------------------------------------------------------

def repo_to_local_dir(repo: str) -> str:
    return os.path.join(MODELS_DIR, (repo or "").replace("/", "_"))

def _nonempty_dir(path: str) -> bool:
    try:
        return os.path.isdir(path) and any(os.scandir(path))
    except Exception:
        return False

def model_files_valid(local_dir: str) -> bool:
    """Best-effort: detect a usable local model install without strict assumptions."""
    try:
        if not os.path.isdir(local_dir):
            return False

        # Optional integrity check (best-effort)
        if not verify_model_manifest(local_dir):
            return False

        # HF snapshot style folders
        for p in ("snapshots", "refs"):
            if os.path.isdir(os.path.join(local_dir, p)):
                return True

        files = [f.name for f in os.scandir(local_dir) if f.is_file()]
        if not files:
            return False

        # common weights formats
        weight_exts = (".bin", ".safetensors", ".pth", ".pt", ".onnx")
        if any(f.endswith(weight_exts) for f in files):
            for f in files:
                fp = os.path.join(local_dir, f)
                if os.path.isfile(fp) and os.path.getsize(fp) == 0:
                    return False
            return True

        # sentence-transformers often has modules.json
        if "modules.json" in files:
            return True

        # fall back: if config exists and directory is non-empty
        if any(f in files for f in ("config.json", "tokenizer.json", "tokenizer_config.json")):
            return True

        return False
    except Exception as e:
        logger.warning("VALIDATION ERROR %s: %s", local_dir, e)
        return False

def is_repo_installed(repo: str) -> bool:
    repo = resolve_model_repo(repo)
    return model_files_valid(repo_to_local_dir(repo))

def local_dir_size_gb(path: str) -> float:
    total = 0
    try:
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
    except Exception:
        return 0.0
    return float(total) / (1024**3)

# ---------------------------------------------------------------------------
# Optional integrity verification (CRC32) for local model installs
# ---------------------------------------------------------------------------
# NOTE:
# - CRC32 is best-effort and bounded to avoid hashing multi-GB weights by default.
# - Enable with: SARAH_MODEL_VERIFY_CRC32=1
# - Limit hashed file size with: SARAH_MODEL_CRC32_MAX_MB=64  (default)
# - Manifest stored per-model as: _sarah_manifest.json
#
# This is NOT a cryptographic guarantee; it is a lightweight corruption detector.

CRC32_VERIFY_ENABLED = str(os.getenv("SARAH_MODEL_VERIFY_CRC32", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
CRC32_MAX_MB = float(os.getenv("SARAH_MODEL_CRC32_MAX_MB", "64") or 64)
_MANIFEST_NAME = "_sarah_manifest.json"

def _crc32_stream(path: str, max_mb: float) -> Optional[int]:
    """Compute CRC32 for a file up to max_mb. Returns None if skipped."""
    try:
        size = os.path.getsize(path)
        if size <= 0:
            return None
        if (size / (1024**2)) > max_mb:
            return None
        crc = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
        return crc & 0xFFFFFFFF
    except Exception:
        return None

def _manifest_path(local_dir: str) -> str:
    return os.path.join(local_dir, _MANIFEST_NAME)

def write_model_manifest(local_dir: str) -> None:
    if not CRC32_VERIFY_ENABLED:
        return
    try:
        man = {"version": 1, "max_mb": CRC32_MAX_MB, "files": {}}
        for root, _, files in os.walk(local_dir):
            for fn in files:
                if fn == _MANIFEST_NAME:
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, local_dir).replace("\\", "/")
                try:
                    st = os.stat(fp)
                    crc = _crc32_stream(fp, CRC32_MAX_MB)
                    man["files"][rel] = {
                        "size": int(st.st_size),
                        "mtime": float(st.st_mtime),
                        "crc32": crc,
                    }
                except Exception:
                    pass
        with open(_manifest_path(local_dir), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Manifest write failed for %s: %s", local_dir, e)

def verify_model_manifest(local_dir: str) -> bool:
    """Verify files in a model dir against its manifest (best-effort)."""
    if not CRC32_VERIFY_ENABLED:
        return True
    mp = _manifest_path(local_dir)
    if not os.path.isfile(mp):
        return True  # no manifest => do not fail installs
    try:
        with open(mp, "r", encoding="utf-8") as f:
            man = json.load(f) or {}
        files = man.get("files") or {}
        max_mb = float(man.get("max_mb") or CRC32_MAX_MB)
        for rel, meta in files.items():
            fp = os.path.join(local_dir, rel.replace("/", os.sep))
            if not os.path.isfile(fp):
                return False
            try:
                st = os.stat(fp)
                if int(meta.get("size") or -1) != int(st.st_size):
                    return False
                expected_crc = meta.get("crc32", None)
                if expected_crc is not None:
                    got = _crc32_stream(fp, max_mb)
                    if got is None or int(expected_crc) != int(got):
                        return False
            except Exception:
                return False
        return True
    except Exception:
        return True

# ---------------------------------------------------------------------------
# Policy + size estimation helpers
# ---------------------------------------------------------------------------

def _console_confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
        return ans in ("y", "yes")
    except Exception:
        return False

def _safe_confirm_large_model(repo: str, size_gb: float, policy: dict) -> bool:
    """Confirm large downloads. Headless-safe; no GUI prompts."""
    if is_headless_runtime():
        logger.info("[HEADLESS_SKIP_CONFIRM] %s size_gb=%.3f reason=%s", repo, size_gb, policy.get("reason"))
        return False
    if is_interactive_tty():
        return _console_confirm(
            f"Model {repo} is ~{size_gb:.2f} GB (threshold {policy.get('max_single_gb')} GB). Download? (Y/N): "
        )
    return False

def _estimate_hf_repo_size_gb(repo: str) -> Optional[float]:
    """Best-effort remote repo size estimate using HfApi (if available)."""
    if HfApi is None:
        return None
    try:
        api = HfApi()
        info = api.model_info(repo)
        siblings = getattr(info, "siblings", None) or []
        total = 0
        for s in siblings:
            sz = getattr(s, "size", None)
            if isinstance(sz, int):
                total += sz
        if total <= 0:
            return None
        return float(total) / (1024**3)
    except Exception:
        return None

def _disk_free_gb(path: str) -> Optional[float]:
    try:
        import shutil
        du = shutil.disk_usage(path)
        return float(du.free) / (1024**3)
    except Exception:
        return None

def _policy_check_before_download(repo: str, local_dir: str) -> bool:
    """Return True if allowed to download based on budget + large model policy + disk free."""
    repo = resolve_model_repo(repo)
    est = _estimate_hf_repo_size_gb(repo)
    pol = model_policy_allows_download(est, models_dir=MODELS_DIR)

    if not pol.get("ok"):
        print(f"[SKIP] {repo}: model storage budget exceeded (used {pol.get('used_gb'):.1f} / {pol.get('budget_gb'):.1f} GB).")
        logger.info("[SKIP_BUDGET] %s used_gb=%.3f budget_gb=%.3f est_gb=%s", repo, pol.get("used_gb",0.0), pol.get("budget_gb",0.0), str(est))
        return False

    free = _disk_free_gb(MODELS_DIR) or _disk_free_gb(local_dir)
    if free is not None and est is not None:
        if free < (est + 5.0):
            print(f"[SKIP] {repo}: insufficient disk free ({free:.1f} GB free, need ~{est+5.0:.1f} GB).")
            logger.info("[SKIP_DISK] %s free_gb=%.3f est_gb=%.3f", repo, free, est)
            return False

    if pol.get("prompt_required") and est is not None:
        if not _safe_confirm_large_model(repo, est, pol):
            print(f"[SKIP] {repo}: declined large download.")
            logger.info("[SKIP_DECLINED] %s est_gb=%.3f", repo, est)
            return False

    return True

def _meta_line(repo: str) -> str:
    r = resolve_model_repo(repo)
    meta = (MODEL_META or {}).get(r) or {}
    disk = meta.get("disk_gb_est")
    vram = meta.get("vram_gb_est")
    tier = meta.get("tier")
    speed = meta.get("speed")
    parts = []
    if tier: parts.append(str(tier))
    if speed: parts.append(str(speed))
    if disk is not None: parts.append("disk~%sGB" % disk)
    if vram is not None: parts.append("vram~%sGB" % vram)
    return " | ".join(parts) if parts else "n/a"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def retry(func, retries: int = 6, wait: int = 15):
    """
    Retry helper for downloads.

    IMPORTANT:
    - Never raises on max retries; returns False instead.
    - Keeps the CLI menu alive (no hard exit).
    """
    last_err = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_err = e
            logger.error("Retry %s/%s failed: %s", attempt + 1, retries, e)
            try:
                time.sleep(wait)
            except Exception:
                pass
    logger.error("Max retries reached: %s", last_err)
    return False

def pip_install_if_needed(package_name: str):
    try:
        subprocess.run(["pip", "install", package_name], check=True)
        logger.info("[INSTALLED] %s", package_name)
        print(f"[INSTALLED] {package_name}")
    except Exception as e:
        logger.error("Pip install failed for %s: %s", package_name, e)

def download_hf_model(repo: str, local_dir: str, use_sentence_transformer: bool = False):
    repo = resolve_model_repo(repo)

    def inner():
        if model_files_valid(local_dir):
            print(f"[✔] {repo} already present.")
            return True

        print(f"[→] Downloading {repo} → {local_dir}")

        if not _policy_check_before_download(repo, local_dir):
            return False

        os.makedirs(local_dir, exist_ok=True)

        if use_sentence_transformer:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed.")
            SentenceTransformer(repo, cache_folder=local_dir)
        else:
            if snapshot_download is not None:
                snapshot_download(repo, local_dir=local_dir, ignore_patterns=["*.md", "*.txt"])
            else:
                if AutoTokenizer is None or AutoModel is None:
                    raise RuntimeError("transformers not installed.")
                AutoTokenizer.from_pretrained(repo, cache_dir=local_dir)
                AutoModel.from_pretrained(repo, cache_dir=local_dir)

        if not model_files_valid(local_dir):
            raise RuntimeError(f"Download completed but validation failed: {repo}")

        write_model_manifest(local_dir)
        print(f"[✔] Completed {repo}")
        return True

    return retry(inner)

def download_object_model(name: str, cfg: dict):
    repo_dir = (cfg.get("repo") or "").strip()
    hf_repo = (cfg.get("hf_repo") or "").strip()
    req = (cfg.get("require") or "").strip() or None
    weights = cfg.get("weights", []) or []

    if not repo_dir or not hf_repo:
        print(f"[SKIP] {name} missing repo/hf_repo.")
        return

    if req:
        pip_install_if_needed(req)

    local_dir = os.path.join(MODELS_DIR, repo_dir.replace("/", "_"))

    def inner():
        os.makedirs(local_dir, exist_ok=True)

        if _nonempty_dir(local_dir) and model_files_valid(local_dir):
            print(f"[✔] {name} already present.")
            return True

        print(f"[→] Downloading {name} from {hf_repo}")

        if not _policy_check_before_download(hf_repo, local_dir):
            return False

        if hf_repo.startswith("http"):
            if not _nonempty_dir(local_dir):
                subprocess.run(["git", "clone", hf_repo, local_dir], check=False)
        else:
            if snapshot_download is None:
                raise RuntimeError("huggingface_hub not installed.")
            snapshot_download(hf_repo, local_dir=local_dir, ignore_patterns=["*.md", "*.txt"])

        for w in weights:
            try:
                dest_path = os.path.join(local_dir, w["filename"])
                if not os.path.exists(dest_path):
                    print(f"[↓] Downloading weight {w['filename']} for {name}")
                    urllib.request.urlretrieve(w["url"], dest_path)
            except Exception as e:
                logger.warning("Weight download failed for %s: %s", name, e)

        write_model_manifest(local_dir)
        print(f"[✔] Completed {name}")
        return True

    return retry(inner)

# ---------------------------------------------------------------------------
# Router helpers (lightweight; execution happens elsewhere)
# ---------------------------------------------------------------------------

def infer_category_from_text(text: str, meta: Optional[dict] = None) -> str:
    meta = meta or {}
    t = (text or "").lower()

    if meta.get("image") or meta.get("frame") or meta.get("has_image") or meta.get("has_frame"):
        return "vision"
    if any(k in t for k in ["photo", "picture", "image", "screenshot", "what is in this", "what's in this"]):
        return "vision"

    if any(k in t for k in ["say this", "speak", "read this", "tts", "voice"]):
        return "tts"

    if any(k in t for k in ["generate an image", "make an image", "create an image", "draw", "render"]):
        return "image_generation"

    if any(k in t for k in ["code", "bug", "traceback", "stack trace", "error", "patch", "syntaxerror", "exception"]):
        return "coder"

    if any(k in t for k in ["math", "calculate", "proof", "logic", "reason", "derivative", "integral"]):
        return "reasoning"

    return "embeddings"

def route_to_primary_model_repo(text: str, meta: Optional[dict] = None) -> Tuple[str, str]:
    category = infer_category_from_text(text, meta)
    repo = get_stack_primary_repo(category, text=text, meta=meta) if MULTI_STACK_ENABLED else ""
    repo = resolve_model_repo(repo)
    return category, repo

def route_to_best_available_model_repo(text: str, meta: Optional[dict] = None) -> Tuple[str, str]:
    category, primary = route_to_primary_model_repo(text, meta)
    if primary and is_repo_installed(primary):
        return category, primary

    tier = recommend_model_tier(category)
    cat = category.lower()
    if cat in ("embedding", "embeddings", "semantic", "memory"):
        cat = "embeddings"

    c = (MODEL_CATALOG or {}).get(cat, {})
    tier_order = [tier] + (["mid", "low"] if tier == "beast" else ["low"] if tier == "mid" else [])
    for t in tier_order:
        for repo in (c.get(t, []) or []):
            repo = resolve_model_repo(repo)
            if is_repo_installed(repo):
                return category, repo

    return category, primary

# ---------------------------------------------------------------------------
# Update checks (HF best-effort)
# ---------------------------------------------------------------------------

_STAMP_FILE = os.path.join(MODELS_DIR, "_hf_model_stamps.json")

def _load_stamps() -> dict:
    try:
        if os.path.exists(_STAMP_FILE):
            with open(_STAMP_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def _save_stamps(stamps: dict) -> None:
    try:
        with open(_STAMP_FILE, "w", encoding="utf-8") as f:
            json.dump(stamps, f, indent=2, sort_keys=True)
    except Exception:
        pass

def check_hf_updates(repos: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if HfApi is None:
        return out
    api = HfApi()
    stamps = _load_stamps()
    for r in repos:
        repo = resolve_model_repo(r)
        local_dir = repo_to_local_dir(repo)
        installed = model_files_valid(local_dir)

        remote_last = None
        try:
            info = api.model_info(repo)
            remote_last = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
            remote_last = str(remote_last) if remote_last is not None else None
        except Exception:
            remote_last = None

        local_seen = stamps.get(repo, {}).get("lastModified")
        update_avail = bool(installed and remote_last and local_seen and (remote_last != local_seen))

        out[repo] = {
            "installed": installed,
            "remote_last_modified": remote_last,
            "local_last_seen": local_seen,
            "update_available": update_avail,
        }
    return out

def stamp_installed_repo(repo: str) -> None:
    if HfApi is None:
        return
    repo = resolve_model_repo(repo)
    stamps = _load_stamps()
    try:
        api = HfApi()
        info = api.model_info(repo)
        remote_last = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
        if remote_last is not None:
            stamps[repo] = {"lastModified": str(remote_last), "stamped_at": time.time()}
            _save_stamps(stamps)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# CLI Menu (categorized; headless-safe)
# ---------------------------------------------------------------------------

CATEGORY_ORDER = ["reasoning", "coder", "embeddings", "vision", "image_generation", "tts"]

def _print_hardware_summary():
    try:
        hs = getattr(G, "hardware_score", lambda: {"score": None, "tier": None, "metrics": {}})()
        m = hs.get("metrics", {}) or {}
        print("\n--- Hardware Score ---")
        print("Score: %s  Tier: %s" % (hs.get("score"), hs.get("tier")))
        print("RAM:  %s MB (avail %s MB)" % (m.get("ram_total_mb"), m.get("ram_avail_mb")))
        print("VRAM: %s MB (free %s MB) GPU: %s" % (m.get("gpu_vram_total_mb"), m.get("gpu_vram_free_mb"), m.get("gpu_name")))
        print("Disk free: %s GB  CPU%%: %s" % (m.get("disk_free_gb"), m.get("cpu_pct")))
        print("Models used: %.1f GB / budget %.1f GB" % (get_models_storage_usage_gb(MODELS_DIR), float(getattr(G, "SARAH_MODELS_BUDGET_GB", 0) or 0)))
        print("")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Hydra Model Manager (Prototype)
# ---------------------------------------------------------------------------
# Concept:
# - Control plane: Neuron/Globals decide policy; Hydra executes safe model lifecycle ops.
# - Data plane: actual inference engines are outside this class.
# - Supports: detect -> isolate -> replace(sync/acquire) -> verify -> promote -> rollback
# - Background worker for async updates, resumable across restarts.
#
# NOTE:
# - No new global flags are introduced. All policy gates reuse existing helpers and Globals.
# - SAFE default: will only SYNC installed repos unless caller enables acquire_missing=True.
# ---------------------------------------------------------------------------

class SarahMemoryHydraModelManager:
    """Prototype self-healing model lifecycle manager backed by Hugging Face snapshots.

    Primary responsibilities
    - Maintain a deterministic set of "active" repos by category (and optional tier).
    - Keep installed repos up-to-date using incremental HF snapshots.
    - Enforce disk/budget gates via existing SarahMemoryLLM policy helpers.
    - Quarantine corrupt installs and promote verified revisions atomically.
    - Persist state so interrupted downloads resume on next boot.

    This class intentionally does NOT run inference. It produces routing decisions and
    maintains "active" pointers that other components (Neuron/Runner) can consume.
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        hf_token_env: str = "HF_TOKEN",
        state_subdir: str = ".hydra",
        allow_remote_mesh_default: bool = False,
        allow_external_api_default: bool = False,
    ) -> None:
        self.models_dir = models_dir
        self.hf_token_env = hf_token_env
        self.allow_remote_mesh_default = allow_remote_mesh_default
        self.allow_external_api_default = allow_external_api_default

        self._root = os.path.join(self.models_dir, state_subdir)
        self._state_path = os.path.join(self._root, "hydra_state.json")
        self._active_path = os.path.join(self._root, "active_models.json")
        self._lock_path = os.path.join(self._root, "hydra.lock")
        self._quarantine_dir = os.path.join(self._root, "quarantine")
        self._staging_dir = os.path.join(self._root, "staging")
        self._export_dir = os.path.join(self._root, "exports")

        os.makedirs(self._root, exist_ok=True)
        os.makedirs(self._quarantine_dir, exist_ok=True)
        os.makedirs(self._staging_dir, exist_ok=True)
        os.makedirs(self._export_dir, exist_ok=True)

        self._q: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        # Load persisted state (resumable)
        self._state = self._load_json(self._state_path, default={
            "version": "8.0.0",
            "pending": [],
            "inflight": None,
            "last_gc_ts": 0,
        })
        self._active = self._load_json(self._active_path, default={
            "version": "8.0.0",
            "active": {},   # category -> {repo, local_dir, ts}
            "rollback": {}, # category -> {repo, local_dir, ts}
        })

        # Rehydrate pending queue
        for job in list(self._state.get("pending") or []):
            self._q.put(job)

        # If we crashed mid-download, keep inflight as priority job
        inflight = self._state.get("inflight")
        if inflight:
            self._q.put(inflight)

    # -----------------------------
    # Public API
    # -----------------------------

    def start_background(self) -> None:
        """Start the background worker once. Safe to call multiple times."""
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="SarahMemoryHydraModelManager", daemon=True)
        self._worker.start()

    def stop_background(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    def enqueue_recommended_sync(self, acquire_missing: bool = False) -> int:
        """Plan sync jobs for the current hardware tier across all categories.

        acquire_missing=False => SAFE: only sync repos already installed.
        acquire_missing=True  => EXPAND: also acquire missing repos in tier plan.
        """
        planned = self.plan_recommended_jobs(acquire_missing=acquire_missing)
        for j in planned:
            self._q.put(j)
        self._persist_state()
        return len(planned)

    def plan_recommended_jobs(self, acquire_missing: bool = False) -> List[dict]:
        """Build a deterministic job list based on the catalog and the current tier."""
        jobs: List[dict] = []

        # Respect POOR tier: no auto-acquire; sync only if caller has installed repos.
        tier_rating = "Poor"
        try:
            hs = getattr(G, "hardware_score", lambda metrics=None: {})()
            tier_rating = str(hs.get("tier_rating") or "Poor")
        except Exception:
            tier_rating = "Poor"

        for category in (MODEL_CATALOG or {}).keys():
            tier = str(recommend_model_tier(category=category) or "low").lower().strip()

            # Normalize tier to include 'high'
            if tier not in ("low", "mid", "high", "beast"):
                tier = "low"

            # Choose candidate repos for this category + tier fallback
            repo = self._choose_repo_for_category(category=category, tier=tier, prefer_installed=True)
            if not repo:
                continue

            local_dir = repo_to_local_dir(repo)
            installed = is_repo_installed(repo)

            if tier_rating.lower() == "poor":
                if installed:
                    jobs.append(self._job_sync(repo, category=category, local_dir=local_dir))
                continue

            if installed:
                jobs.append(self._job_sync(repo, category=category, local_dir=local_dir))
            elif acquire_missing:
                jobs.append(self._job_acquire(repo, category=category, local_dir=local_dir))

        # De-dup by repo
        seen = set()
        out: List[dict] = []
        for j in jobs:
            r = j.get("repo")
            if not r or r in seen:
                continue
            seen.add(r)
            out.append(j)
        return out

    def resolve_inference_ladder(
        self,
        category: str = "reasoning",
        privacy: str = "default",
        allow_remote_mesh: Optional[bool] = None,
        allow_external_api: Optional[bool] = None,
    ) -> dict:
        """Return a routing decision for Neuron/Runner: baseline/local/mesh/external.

        - baseline: hardcoded deterministic fallback (Tier-0).
        - local: use installed active repo (Tier-1).
        - mesh: use trusted remote model pool (Tier-2).
        - external: use external API (Tier-3, optional).
        """
        allow_remote_mesh = self.allow_remote_mesh_default if allow_remote_mesh is None else bool(allow_remote_mesh)
        allow_external_api = self.allow_external_api_default if allow_external_api is None else bool(allow_external_api)

        # Tier-0 always exists
        decision = {"route": "baseline", "category": category, "reason": "tier0_baseline"}

        active = (self._active.get("active") or {}).get(category) or {}
        if active.get("repo") and active.get("local_dir") and model_files_valid(active.get("local_dir")):
            return {
                "route": "local",
                "category": category,
                "repo": active.get("repo"),
                "local_dir": active.get("local_dir"),
                "reason": "tier1_local_active",
            }

        # If we have any installed candidate from routing, prefer that before mesh
        repo = self._choose_repo_for_category(category=category, tier=str(recommend_model_tier(category=category) or "low"), prefer_installed=True)
        if repo and is_repo_installed(repo):
            local_dir = repo_to_local_dir(repo)
            if model_files_valid(local_dir):
                return {"route": "local", "category": category, "repo": repo, "local_dir": local_dir, "reason": "tier1_local_installed"}

        if allow_remote_mesh and privacy.lower() not in ("strict_local", "local_only"):
            return {"route": "mesh", "category": category, "reason": "tier2_mesh_allowed"}

        if allow_external_api and privacy.lower() not in ("strict_local", "local_only"):
            return {"route": "external", "category": category, "reason": "tier3_external_allowed"}

        return decision

    def get_active_map(self) -> dict:
        return dict(self._active)

    def force_gc(self) -> dict:
        return self._gc_orphans(force=True)

    # -----------------------------
    # Internal: repo selection helpers (category/tier aware)
    # -----------------------------

    def _primary_repo_for_category(self, category: str) -> str:
        """Return the configured primary repo for a category (may be empty)."""
        try:
            r = get_stack_primary_repo(category, text="", meta=None) if MULTI_STACK_ENABLED else ""
            return resolve_model_repo(r) if r else ""
        except Exception:
            return ""

    def _catalog_candidates_for_category(self, category: str, tier: str) -> List[str]:
        cat = (category or "").strip().lower()
        if cat in ("embedding", "semantic", "memory"):
            cat = "embeddings"
        tiers = (MODEL_CATALOG or {}).get(cat, {}) or {}

        # Tier fallback order (includes high)
        t = (tier or "low").strip().lower()
        if t == "beast":
            order = ["beast", "high", "mid", "low"]
        elif t == "high":
            order = ["high", "mid", "low"]
        elif t == "mid":
            order = ["mid", "low"]
        else:
            order = ["low"]

        out: List[str] = []
        for k in order:
            for r in (tiers.get(k) or []):
                rr = resolve_model_repo(r)
                if rr and rr not in out:
                    out.append(rr)
        return out

    def _choose_repo_for_category(self, category: str, tier: str, prefer_installed: bool = False) -> str:
        """Pick the best repo for a category/tier, optionally preferring installed repos."""
        primary = self._primary_repo_for_category(category)
        candidates = self._catalog_candidates_for_category(category, tier)

        # Ensure primary is considered first (if present)
        if primary and primary not in candidates:
            candidates = [primary] + candidates
        elif primary:
            # move primary to front
            candidates = [primary] + [c for c in candidates if c != primary]

        if prefer_installed:
            for r in candidates:
                if is_repo_installed(r):
                    return r

        return candidates[0] if candidates else ""

    # -----------------------------
    # Internal: worker loop
    # -----------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except Exception:
                # periodic GC
                self._gc_orphans(force=False)
                continue

            try:
                self._state["inflight"] = job
                self._persist_state()
                self._execute_job(job)
            except Exception as e:
                logger.warning("[Hydra] job failed: %s", e)
            finally:
                self._state["inflight"] = None
                self._persist_state()
                self._q.task_done()

    # -----------------------------
    # Internal: job definitions
    # -----------------------------

    def _job_sync(self, repo: str, category: str, local_dir: str) -> dict:
        return {"type": "sync", "repo": repo, "category": category, "local_dir": local_dir, "ts": time.time()}

    def _job_acquire(self, repo: str, category: str, local_dir: str) -> dict:
        return {"type": "acquire", "repo": repo, "category": category, "local_dir": local_dir, "ts": time.time()}

    def _execute_job(self, job: dict) -> None:
        jtype = str(job.get("type") or "")
        repo = str(job.get("repo") or "")
        category = str(job.get("category") or "reasoning")
        local_dir = str(job.get("local_dir") or repo_to_local_dir(repo))

        if not repo:
            return

        # Detect -> isolate if invalid
        if os.path.isdir(local_dir) and not model_files_valid(local_dir):
            self._quarantine(local_dir, reason="invalid_pre")

        # Replace (sync/acquire)
        if jtype in ("sync", "acquire"):
            ok = self._hf_snapshot(repo=repo, target_dir=local_dir)
            if not ok:
                return

            # Verify
            if not self._verify_repo_install(repo, local_dir):
                self._quarantine(local_dir, reason="verify_fail")
                # Rollback if we had one
                self._rollback(category)
                return

            # Promote
            self._promote(category=category, repo=repo, local_dir=local_dir)

            # Cleanup governed retention
            self._gc_category(category)

    # -----------------------------
    # Internal: HF snapshot + verify
    # -----------------------------

    def _hf_snapshot(self, repo: str, target_dir: str) -> bool:
        if snapshot_download is None:
            logger.warning("[Hydra] huggingface_hub not available; cannot sync %s", repo)
            return False

        # Policy gate: best-effort size lookup; still enforce disk/budget
        size_gb = float(MODEL_META.get(repo, {}).get("size_gb") or 0.0)
        if size_gb <= 0.0:
            try:
                size_gb = float(_estimate_hf_repo_size_gb(repo) or 0.0)
            except Exception:
                size_gb = 0.0

        if not _policy_check_before_download(repo, target_dir):
            logger.info("[Hydra] policy blocked %s", repo)
            return False
        if not pol.get("ok"):
            logger.info("[Hydra] policy blocked %s: %s", repo, pol.get("reason"))
            return False

        # Use HF resumable download; keep portable (no symlinks)
        token = os.getenv(self.hf_token_env) or None

        # Ensure parent exists
        os.makedirs(target_dir, exist_ok=True)

        try:
            snapshot_download(
                repo_id=repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
            )
            # Mark installed stamp for existing menus/tools
            try:
                stamp_installed_repo(repo)
            except Exception:
                pass

            # Write/refresh manifest (optional)
            try:
                write_model_manifest(target_dir)
            except Exception:
                pass

            return True
        except Exception as e:
            logger.warning("[Hydra] snapshot_download failed for %s: %s", repo, e)
            return False

    def _verify_repo_install(self, repo: str, local_dir: str) -> bool:
        # Corruption check via manifest if enabled
        try:
            if not model_files_valid(local_dir):
                return False
        except Exception:
            return False

        # Optional lightweight load check (no heavy GPU inference)
        # We only verify tokenizer/config presence where possible.
        try:
            if AutoTokenizer is not None:
                # Will succeed even without GPU; may take time but is deterministic.
                AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            # tokenizer load isn't mandatory for "installed"; do not hard fail
            pass

        return True

    # -----------------------------
    # Internal: promote / rollback / quarantine
    # -----------------------------

    def _promote(self, category: str, repo: str, local_dir: str) -> None:
        now = time.time()

        current = (self._active.get("active") or {}).get(category)
        if current and current.get("local_dir") and current.get("local_dir") != local_dir:
            # store rollback pointer
            self._active.setdefault("rollback", {})[category] = dict(current)

        self._active.setdefault("active", {})[category] = {
            "repo": repo,
            "local_dir": local_dir,
            "ts": now,
        }
        self._save_json_atomic(self._active_path, self._active)

    def _rollback(self, category: str) -> bool:
        rb = (self._active.get("rollback") or {}).get(category)
        if not rb:
            return False
        if rb.get("local_dir") and model_files_valid(rb.get("local_dir")):
            self._active.setdefault("active", {})[category] = dict(rb)
            self._save_json_atomic(self._active_path, self._active)
            return True
        return False

    def _quarantine(self, local_dir: str, reason: str = "unknown") -> None:
        if not os.path.isdir(local_dir):
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.basename(local_dir.rstrip("/\\")).strip()
        dst = os.path.join(self._quarantine_dir, f"{base}__{reason}__{ts}")
        try:
            shutil.move(local_dir, dst)
            logger.warning("[Hydra] quarantined %s -> %s", local_dir, dst)
        except Exception as e:
            logger.warning("[Hydra] quarantine move failed: %s", e)

    # -----------------------------
    # Internal: governed cleanup
    # -----------------------------

    def _gc_category(self, category: str) -> None:
        """Keep active + rollback only for a category (best-effort)."""
        active = (self._active.get("active") or {}).get(category) or {}
        rollback = (self._active.get("rollback") or {}).get(category) or {}
        keep_dirs = set()
        if active.get("local_dir"):
            keep_dirs.add(os.path.abspath(active.get("local_dir")))
        if rollback.get("local_dir"):
            keep_dirs.add(os.path.abspath(rollback.get("local_dir")))

        # Only consider model dirs referenced by our repo naming convention (repo_to_local_dir)
        try:
            for entry in os.scandir(self.models_dir):
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue
                p = os.path.abspath(entry.path)
                if p in keep_dirs:
                    continue
                # Skip if not in our managed set (best-effort)
                # If it's installed but not referenced, leave it alone.
        except Exception:
            return

    def _gc_orphans(self, force: bool = False) -> dict:
        """Global cleanup: remove abandoned staging leftovers; keep quarantine."""
        now = time.time()
        last = float(self._state.get("last_gc_ts") or 0)
        if (not force) and (now - last) < 3600:
            return {"ok": True, "skipped": True}

        removed = 0
        # Remove empty staging directories older than 1h
        try:
            for entry in os.scandir(self._staging_dir):
                if not entry.is_dir():
                    continue
                try:
                    st = entry.stat()
                    age = now - st.st_mtime
                    if age > 3600 and not any(os.scandir(entry.path)):
                        shutil.rmtree(entry.path, ignore_errors=True)
                        removed += 1
                except Exception:
                    pass
        except Exception:
            pass

        self._state["last_gc_ts"] = now
        self._persist_state()
        return {"ok": True, "removed": removed, "force": force}

    # -----------------------------
    # Internal: persistence helpers
    # -----------------------------

    def _persist_state(self) -> None:
        # Persist queue snapshot + inflight marker
        try:
            pending = []
            # snapshot queue without draining (best-effort)
            with self._q.mutex:
                pending = list(self._q.queue)
            self._state["pending"] = pending
        except Exception:
            pass
        self._save_json_atomic(self._state_path, self._state)

    def _load_json(self, path: str, default: dict) -> dict:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return dict(default)

    def _save_json_atomic(self, path: str, data: dict) -> None:
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

def cli_menu():
    _print_hardware_summary()
    print("""===== SarahMemory Model Manager (Categorized) =====
[1] Auto: Download Recommended Stack (per hardware tier)
[2] Download by Category (Low/Mid/Beast options)
[3] Legacy: Download SELECTED Models (MODEL_CONFIG toggles)
[4] Download Vision/Object Models (OBJECT_MODEL_CONFIG toggles)
[5] Check HF updates (best-effort)
[6] Exit
""")
    return input("Enter choice (1-6): ").strip()

def _format_installed(repo: str) -> str:
    return "✔" if is_repo_installed(repo) else "❌"

def choose_one(title: str, items: List[str]) -> Optional[str]:
    if not items:
        return None
    print("\n" + title)
    for i, it in enumerate(items, 1):
        print("[%d] %s (%s) - %s" % (i, it, _format_installed(it), _meta_line(it)))
    print("[0] Cancel\n")
    while True:
        c = input("Select #: ").strip()
        if c == "0":
            return None
        try:
            idx = int(c) - 1
            if 0 <= idx < len(items):
                return items[idx]
        except Exception:
            pass
        print("Invalid selection.")

def download_recommended_stack():
    for cat in CATEGORY_ORDER:
        tier = recommend_model_tier(cat)
        repo = get_stack_primary_repo(cat, text="", meta=None)
        repo = resolve_model_repo(repo)
        if repo and is_repo_installed(repo):
            print("[✔] %s primary already installed." % repo)
            continue

        pick = getattr(G, "pick_catalog_model", lambda c,t, fallback_tiers=None: None)(cat if cat!="image_generation" else "image_generation", tier)
        repo_to_get = resolve_model_repo(pick or repo)
        if not repo_to_get:
            continue

        use_st = (cat == "embeddings")
        ok = download_hf_model(repo_to_get, repo_to_local_dir(repo_to_get), use_sentence_transformer=use_st)
        if ok:
            stamp_installed_repo(repo_to_get)
        else:
            print(f"[ERROR] Failed to download {repo_to_get}. Returning to menu.")

def download_by_category():
    print("\nCategories:")
    for i, c in enumerate(CATEGORY_ORDER, 1):
        print("[%d] %s" % (i, c))
    print("[0] Cancel\n")
    c = input("Select category #: ").strip()
    if c == "0":
        return
    try:
        cat = CATEGORY_ORDER[int(c)-1]
    except Exception:
        print("Invalid.")
        return

    tiers = ["low","mid","beast"]
    print("\nTiers:")
    for i, t in enumerate(tiers, 1):
        print("[%d] %s" % (i, t))
    tc = input("Select tier #: ").strip()
    try:
        tier = tiers[int(tc)-1]
    except Exception:
        print("Invalid.")
        return

    cat_key = "embeddings" if cat == "embeddings" else cat
    choices = (MODEL_CATALOG or {}).get(cat_key, {}).get(tier, []) or []
    repo = choose_one("Models for %s / %s:" % (cat, tier), choices)
    if not repo:
        return

    use_st = (cat == "embeddings")
    ok = download_hf_model(repo, repo_to_local_dir(repo), use_sentence_transformer=use_st)
    if ok:
        stamp_installed_repo(repo)
    else:
        print(f"[ERROR] Failed to download {repo}. Returning to menu.")

def legacy_download_selected():
    for name, enabled in (MODEL_CONFIG or {}).items():
        if not enabled:
            continue
        repo = resolve_model_repo(name)
        use_st = True
        ok = download_hf_model(repo, repo_to_local_dir(repo), use_sentence_transformer=use_st)
        if ok:
            stamp_installed_repo(repo)
        else:
            print(f"[ERROR] Failed to download {repo}. Continuing.")

def download_object_enabled():
    for name, cfg in (OBJECT_MODEL_CONFIG or {}).items():
        if not cfg.get("enabled"):
            continue
        ok = download_object_model(name, cfg)
        if ok is False:
            print(f"[ERROR] Failed to download object model {name}. Continuing.")

def check_updates_menu():
    repos = []
    for cat in CATEGORY_ORDER:
        r = resolve_model_repo(get_stack_primary_repo(cat, text="", meta=None))
        if r:
            repos.append(r)
    for cat, tiers in (MODEL_CATALOG or {}).items():
        for t, arr in (tiers or {}).items():
            repos.extend(arr or [])
    repos = sorted(set(repos))
    updates = check_hf_updates(repos)
    if not updates:
        print("[INFO] No update data available (huggingface_hub not installed or offline).")
        return
    print("\n--- HF Update Check ---")
    for repo, info in updates.items():
        if info.get("installed"):
            flag = "UPDATE" if info.get("update_available") else "OK"
            print("%s: %s (remote %s / local %s)" % (repo, flag, info.get("remote_last_modified"), info.get("local_last_seen")))
    print("")

def main():
    while True:
        try:
            choice = cli_menu()
        except Exception as e:
            logger.error('Menu error: %s', e)
            print('[ERROR] Menu error. Returning to menu.')
            continue
        try:
            if choice == "1":
                download_recommended_stack()
            elif choice == "2":
                download_by_category()
            elif choice == "3":
                legacy_download_selected()
            elif choice == "4":
                download_object_enabled()
            elif choice == "5":
                check_updates_menu()
            elif choice == "6":
                break
            else:
                print("Invalid choice.")
        except Exception as e:
            logger.error('Action error: %s', e)
            print('[ERROR] Action failed. Returning to menu.')

if __name__ == "__main__":
    main()
from __future__ import annotations

"""
--== The SarahMemory Project ==--
File: SarahMemoryUISelfAware.py
Part of the SarahMemory AiOS Platform
Version: v8.0.0
Date: 2026-03-13
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

PURPOSE:
--------
Sandboxed UI self-observation and frontend solution generator.

This file is a SIDE TOOL meant to run in its own console while SarahMemory is
already running locally. It does NOT modify the live frontend source tree.
It studies the current frontend, API bridge files, and backend core modules,
then writes proposed frontend solution files into ./data/ui/temp/solution and
presence/integration audit reports into ./data/ui/temp/audits.

DESIGN INTENT:
--------------
- Discover current frontend surfaces from real source code.
- Discover backend capability intent from Python modules and API bridge files.
- Detect when a capability already exists in the frontend under a direct name,
  friendly label, alias label, or family shell.
- Only generate sandbox frontend solutions for modules that appear to be
  user-facing, not already represented, and not core/internal infrastructure.
- Never tell the system to seek a specific target module by name.
"""
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "ui_tool"
# CATEGORY = "ui_awareness"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "sandbox"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "ui_self_aware"
# FAMILY = "ui_governance"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Sandboxed UI self-observation and frontend solution generator. Studies frontend, API bridge files, and backend modules; writes proposals to temp instead of modifying live UI."
# --- SARAHMETA END ---
import ast
import json
import logging
import os
import re
import shutil
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("SarahMemoryUISelfAware")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

@dataclass
class FrontendFileInfo:
    path: str
    rel_path: str
    extension: str
    guessed_kind: str = "unknown"
    exports: List[str] = field(default_factory=list)
    route_paths: List[str] = field(default_factory=list)
    ui_labels: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    api_refs: List[str] = field(default_factory=list)


@dataclass
class FrontendAnalysis:
    exists: bool
    root: str
    total_files: int
    inferred_dirs: Dict[str, str]
    files: List[FrontendFileInfo] = field(default_factory=list)
    feature_index: Dict[str, List[str]] = field(default_factory=dict)
    label_index: Dict[str, List[str]] = field(default_factory=dict)
    export_index: Dict[str, List[str]] = field(default_factory=dict)
    route_index: Dict[str, List[str]] = field(default_factory=dict)
    family_shells: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ApiEndpoint:
    file: str
    route: str
    methods: List[str]
    handler: str
    handler_tokens: List[str] = field(default_factory=list)
    route_tokens: List[str] = field(default_factory=list)
    imported_modules: List[str] = field(default_factory=list)
    evidence_lines: List[str] = field(default_factory=list)


@dataclass
class BackendModule:
    module_name: str
    rel_path: str
    feature_name: str
    docstring: str
    classes: List[str]
    functions: List[str]
    imports: List[str]
    tokens: List[str]
    category: str
    frontend_hits: List[str]
    alias_hits: List[str]
    family_hits: List[str]
    endpoint_matches: List[str]
    classification: str
    confidence: str
    candidate_reason: str
    should_generate: bool


@dataclass
class GapCandidate:
    module_name: str
    feature_name: str
    slug: str
    rel_file: str
    category: str
    confidence: str
    endpoint_matches: List[str]
    candidate_reason: str
    proposed_kind: str


# -----------------------------------------------------------------------------
# Heuristics
# -----------------------------------------------------------------------------

STOPWORDS = {
    "the", "and", "for", "from", "with", "this", "that", "true", "false", "none", "self",
    "data", "core", "main", "base", "file", "class", "function", "result", "value", "items",
    "item", "json", "http", "https", "local", "system", "tool", "module", "project", "python",
    "default", "state", "panel", "screen", "page", "component", "service", "route",
}

EXCLUSION_EXACT = {
    "globals", "ledger", "neuron", "reply", "advcu", "gui", "gui2", "uiupdater", "updater",
    "rankingupdater", "diagnostics", "database", "sync", "gittalk", "cognitiveservices",
    "initialization", "integration", "main", "voice", "network", "compare", "systemindexer",
    "filesystem", "api", "websym", "selfaware", "startup", "logiccalc", "smapi", "sobje",
}

EXCLUSION_PARTIAL = {
    "global", "ledger", "neuron", "reply", "gui", "bootstrap", "boot", "init", "update",
    "ranking", "diagnostic", "config", "govern", "kernel", "router", "governor", "guarddog",
    "orchestr", "semantic", "compare", "index", "vector", "memory", "db", "database",
    "migration", "cleanup", "evolution", "vault", "crypto", "wallet", "presence", "broker",
}

CATEGORY_KEYWORDS: Dict[str, Set[str]] = {
    "communication": {
        "email", "mail", "mailbox", "inbox", "smtp", "imap", "pop3", "compose", "draft", "drafts",
        "sender", "recipient", "attachment", "attachments", "forward", "thread", "threads", "sent",
        "trash", "outbox",
    },
    "terminal": {
        "terminal", "console", "command", "commands", "shell", "bash", "cmd", "powershell",
        "developer", "developersmode", "execute", "execution", "workdir", "session",
    },
    "browser": {
        "browser", "research", "reader", "search", "fetch", "web", "website", "page", "pages",
        "url", "duckduckgo", "reader", "livebrowser",
    },
    "addons": {
        "addon", "addons", "launcher", "plugin", "plugins", "extension", "extensions",
    },
    "studio": {
        "studio", "creative", "canvas", "image", "music", "video", "voice", "lyrics", "song", "comm",
    },
}

DIRECT_UI_CATEGORIES = {"communication", "terminal", "browser", "addons"}

# Category-level shells or labels that imply a UI family exists, but not necessarily
# that a specific feature inside that category exists.
CATEGORY_ALIASES: Dict[str, Set[str]] = {
    "browser": {"browser", "research", "reader", "live browser", "search"},
    "addons": {"addons", "addon launcher", "launcher", "extensions", "plugins"},
    "studio": {"studio", "creative studios", "creative", "image", "music", "voice", "video", "comm"},
}

# Feature-specific aliases. These are only used when they remain feature-faithful.
FEATURE_ALIASES: Dict[str, Set[str]] = {
    "browser": {"research", "reader", "livebrowser", "live browser"},
    "addons": {"addons", "addon launcher", "launcher"},
    "email": {"mail", "mailbox", "inbox", "compose", "drafts", "sent", "outbox"},
    "terminal": {"console", "shell", "command terminal", "developer terminal"},
}

FAMILY_CATEGORIES = {"studio"}

ROUTE_RE = re.compile(r'path\s*[:=]\s*[\'"]([^\'"]+)[\'"]')
API_REF_RE = re.compile(r'/api/[A-Za-z0-9_\-/<>{}:]+')
EXPORT_RE = re.compile(
    r'(?:export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)|'
    r'export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)|'
    r'export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=|'
    r'function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\()'
)

LABEL_PATTERNS = [
    re.compile(r'>\s*([A-Za-z][A-Za-z0-9 /&+\-]{2,80})\s*<'),
    re.compile(r'\b(?:label|title|name|heading|placeholder)\s*[:=]\s*[\'"]([^\'"]{2,80})[\'"]'),
    re.compile(r'\bsetCurrentScreen\([\'"]([^\'"]+)[\'"]\)'),
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "SarahMemoryGlobals.py").exists():
            return cur
        if (cur / "data").exists() and (cur / "api").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def _slugify(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name or "")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"


def _camel_to_words(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split()).strip()


def _normalize_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _module_feature_name(module_name: str) -> str:
    return module_name[len("SarahMemory"):] if module_name.startswith("SarahMemory") else module_name


def _tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text or "")
    out: List[str] = []
    seen: Set[str] = set()
    for token in raw:
        parts = re.split(r"[_\-]+", token)
        expanded: List[str] = []
        for part in parts:
            expanded.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|\d+", part))
        if not expanded:
            expanded = [token]
        for part in expanded:
            low = part.lower().strip()
            if len(low) < 2 or low in STOPWORDS:
                continue
            if low in seen:
                continue
            seen.add(low)
            out.append(low)
    return out


def _guess_frontend_kind(path: Path) -> str:
    p = path.as_posix().lower()
    if any(x in p for x in ("screen", "/pages/", "/views/")):
        return "screen"
    if any(x in p for x in ("panel", "/panels/")):
        return "panel"
    if any(x in p for x in ("service", "/services/", "api.ts", "/client/")):
        return "service"
    if any(x in p for x in ("route", "/routes/", "router")):
        return "route"
    if any(x in p for x in ("nav", "menu", "dock", "sidebar")):
        return "navigation"
    if path.suffix.lower() == ".tsx":
        return "component"
    return "unknown"


def _extract_labels(text: str) -> List[str]:
    labels: List[str] = []
    seen: Set[str] = set()
    for rx in LABEL_PATTERNS:
        for match in rx.findall(text):
            val = match if isinstance(match, str) else next((m for m in match if m), "")
            val = " ".join(str(val).split()).strip()
            if len(val) < 2:
                continue
            low = val.lower()
            if low in seen:
                continue
            if low.startswith("http") or low.startswith("/"):
                continue
            if len(val) > 80:
                continue
            seen.add(low)
            labels.append(val)
    return labels[:50]


def _extract_exports(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for match in EXPORT_RE.findall(text):
        for item in match:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return out[:30]


def _score_category(tokens: Iterable[str]) -> Tuple[str, int]:
    token_set = set(tokens)
    best_cat = "generic"
    best_score = 0
    for cat, words in CATEGORY_KEYWORDS.items():
        score = len(token_set & words)
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat, best_score


def _confidence_from_score(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


# -----------------------------------------------------------------------------
# Main engine
# -----------------------------------------------------------------------------

class SarahMemoryUISelfAware:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = _find_project_root(base_dir or Path.cwd())
        self.data_ui_dir = self.base_dir / "data" / "ui"
        self.frontend_src_dir = self.data_ui_dir / "V8_ui_src" / "src"
        self.api_server_dir = self.base_dir / "api" / "server"
        self.temp_root = self.data_ui_dir / "temp"

        self.snapshot_src_dir = self.temp_root / "src"
        self.snapshot_api_dir = self.temp_root / "api" / "server"
        self.snapshot_backend_dir = self.temp_root / "backend"
        self.analysis_dir = self.temp_root / "analysis"
        self.solution_dir = self.temp_root / "solution"
        self.audit_dir = self.temp_root / "audits"

    # ------------------------------------------------------------------
    # Workspace / snapshots
    # ------------------------------------------------------------------

    def prepare_workspace(self) -> None:
        for p in (
            self.snapshot_src_dir,
            self.snapshot_api_dir,
            self.snapshot_backend_dir,
            self.analysis_dir,
            self.solution_dir,
            self.audit_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def _clear_dir(self, path: Path) -> None:
        if path.exists():
            for item in path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    try:
                        item.unlink()
                    except Exception:
                        pass
        path.mkdir(parents=True, exist_ok=True)

    def refresh_snapshot(self) -> Dict[str, int]:
        logger.info("Refreshing sandbox snapshots")
        for p in (
            self.snapshot_src_dir,
            self.snapshot_api_dir,
            self.snapshot_backend_dir,
            self.solution_dir,
            self.audit_dir,
        ):
            self._clear_dir(p)

        counts = {"frontend": 0, "api": 0, "backend": 0}

        if self.frontend_src_dir.exists():
            for path in sorted(self.frontend_src_dir.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(self.frontend_src_dir)
                    target = self.snapshot_src_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    counts["frontend"] += 1

        if self.api_server_dir.exists():
            for path in sorted(self.api_server_dir.glob("app*.py")):
                if path.is_file():
                    target = self.snapshot_api_dir / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    counts["api"] += 1

        for path in sorted(self.base_dir.glob("SarahMemory*.py")):
            if not path.is_file() or path.name == "SarahMemoryUISelfAware.py":
                continue
            target = self.snapshot_backend_dir / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            counts["backend"] += 1

        _write_json(self.analysis_dir / "snapshot_counts.json", counts)
        return counts

    # ------------------------------------------------------------------
    # Frontend analysis
    # ------------------------------------------------------------------

    def analyze_frontend(self) -> FrontendAnalysis:
        files: List[FrontendFileInfo] = []
        feature_index: Dict[str, List[str]] = {}
        label_index: Dict[str, List[str]] = {}
        export_index: Dict[str, List[str]] = {}
        route_index: Dict[str, List[str]] = {}
        inferred_dirs = {
            "screens": "src/pages",
            "services": "src/services",
            "types": "src/types",
            "routes": "src/routes",
            "navigation": "src/components/navigation",
        }

        for path in sorted(self.snapshot_src_dir.rglob("*")):
            if not path.is_file():
                continue
            text = _safe_read_text(path)
            rel_path = path.relative_to(self.snapshot_src_dir).as_posix()
            tokens = sorted(set(_tokenize(rel_path)))
            exports = _extract_exports(text)
            route_paths = sorted(set(ROUTE_RE.findall(text)))
            api_refs = sorted(set(API_REF_RE.findall(text)))
            labels = _extract_labels(text)
            info = FrontendFileInfo(
                path=str(path),
                rel_path=rel_path,
                extension=path.suffix.lower(),
                guessed_kind=_guess_frontend_kind(path),
                exports=exports,
                route_paths=route_paths,
                ui_labels=labels,
                tokens=tokens,
                api_refs=api_refs,
            )
            files.append(info)

            for tok in tokens:
                feature_index.setdefault(tok, []).append(rel_path)
            for exp in exports:
                for tok in _tokenize(exp):
                    export_index.setdefault(tok, []).append(rel_path)
            for rp in route_paths:
                for tok in _tokenize(rp):
                    route_index.setdefault(tok, []).append(rel_path)
            for lbl in labels:
                for tok in _tokenize(lbl):
                    label_index.setdefault(tok, []).append(rel_path)

        family_shells = self._learn_family_shells(files)
        analysis = FrontendAnalysis(
            exists=self.snapshot_src_dir.exists(),
            root=str(self.snapshot_src_dir),
            total_files=len(files),
            inferred_dirs=inferred_dirs,
            files=files,
            feature_index={k: sorted(set(v)) for k, v in feature_index.items()},
            label_index={k: sorted(set(v)) for k, v in label_index.items()},
            export_index={k: sorted(set(v)) for k, v in export_index.items()},
            route_index={k: sorted(set(v)) for k, v in route_index.items()},
            family_shells=family_shells,
        )
        _write_json(self.analysis_dir / "frontend_analysis.json", asdict(analysis))
        return analysis

    def _learn_family_shells(self, files: Sequence[FrontendFileInfo]) -> Dict[str, List[str]]:
        shells: Dict[str, List[str]] = {}
        for info in files:
            label_tokens = set(tok for lbl in info.ui_labels for tok in _tokenize(lbl))
            path_tokens = set(info.tokens)
            export_tokens = {tok for exp in info.exports for tok in _tokenize(exp)}
            combined = label_tokens | path_tokens | export_tokens
            for family in FAMILY_CATEGORIES:
                if len(combined & CATEGORY_ALIASES.get(family, set())) >= 2:
                    shells.setdefault(family, []).append(info.rel_path)
        return {k: sorted(set(v)) for k, v in shells.items()}

    # ------------------------------------------------------------------
    # API analysis
    # ------------------------------------------------------------------

    def analyze_api_endpoints(self) -> List[ApiEndpoint]:
        out: List[ApiEndpoint] = []
        route_re = re.compile(r'@(?:app|bp|bp2|\w+)\.(?:route|get|post|put|delete|patch)\(\s*[\'"]([^\'"]+)[\'"](?:\s*,\s*methods\s*=\s*\[([^\]]+)\])?')
        fn_re = re.compile(r'^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', re.MULTILINE)
        import_re = re.compile(r'(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))')
        call_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(')

        for path in sorted(self.snapshot_api_dir.glob("app*.py")):
            text = _safe_read_text(path)
            lines = text.splitlines()
            imported_modules = sorted(set(x for tup in import_re.findall(text) for x in tup if x))
            call_names = sorted(set(call_re.findall(text)))
            for m in route_re.finditer(text):
                route = m.group(1)
                methods_raw = m.group(2) or "'GET'"
                methods = [x.strip().strip("'\"") for x in methods_raw.split(",") if x.strip()]
                tail = text[m.end():m.end() + 1200]
                fn = fn_re.search(tail)
                handler = fn.group(1) if fn else "unknown_handler"
                line_no = text[:m.start()].count("\n")
                evidence = [lines[i].strip() for i in range(max(0, line_no - 1), min(len(lines), line_no + 8))]
                imported_plus_calls = list(imported_modules) + call_names
                out.append(ApiEndpoint(
                    file=path.name,
                    route=route,
                    methods=methods,
                    handler=handler,
                    handler_tokens=_tokenize(handler),
                    route_tokens=_tokenize(route),
                    imported_modules=imported_plus_calls,
                    evidence_lines=evidence,
                ))
        _write_json(self.analysis_dir / "api_endpoint_map.json", [asdict(x) for x in out])
        return out

    # ------------------------------------------------------------------
    # Backend analysis / classification
    # ------------------------------------------------------------------

    def analyze_backend_modules(self, frontend: FrontendAnalysis, endpoints: Sequence[ApiEndpoint]) -> List[BackendModule]:
        modules: List[BackendModule] = []

        for path in sorted(self.snapshot_backend_dir.glob("SarahMemory*.py")):
            text = _safe_read_text(path)
            module_name = path.stem
            feature_name = _module_feature_name(module_name)
            doc, classes, functions, imports = self._parse_python_module(text)
            tokens = sorted(set(
                _tokenize(module_name)
                + _tokenize(feature_name)
                + _tokenize(doc)
                + _tokenize(" ".join(classes))
                + _tokenize(" ".join(functions))
            ))
            category, _ = _score_category(tokens)
            frontend_hits = self._find_frontend_direct_hits(feature_name, category, frontend)
            alias_hits = self._find_frontend_feature_alias_hits(feature_name, category, frontend)
            family_hits = list(frontend.family_shells.get(category, [])) if category in FAMILY_CATEGORIES else []
            endpoint_matches = self._match_module_to_endpoints(module_name, feature_name, category, tokens, imports, endpoints)
            classification, confidence, reason, should_generate = self._classify_module(
                module_name=module_name,
                feature_name=feature_name,
                category=category,
                tokens=tokens,
                doc=doc,
                frontend_hits=frontend_hits,
                alias_hits=alias_hits,
                family_hits=family_hits,
                endpoint_matches=endpoint_matches,
            )
            modules.append(BackendModule(
                module_name=module_name,
                rel_path=path.name,
                feature_name=feature_name,
                docstring=doc,
                classes=classes,
                functions=functions,
                imports=imports,
                tokens=tokens,
                category=category,
                frontend_hits=frontend_hits,
                alias_hits=alias_hits,
                family_hits=family_hits,
                endpoint_matches=endpoint_matches,
                classification=classification,
                confidence=confidence,
                candidate_reason=reason,
                should_generate=should_generate,
            ))

        _write_json(self.analysis_dir / "backend_module_map.json", [asdict(x) for x in modules])
        return modules

    def _parse_python_module(self, text: str) -> Tuple[str, List[str], List[str], List[str]]:
        doc = ""
        classes: List[str] = []
        functions: List[str] = []
        imports: List[str] = []
        try:
            tree = ast.parse(text)
            doc = ast.get_docstring(tree) or ""
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        except Exception as exc:
            logger.warning("AST parse failed for module parse: %s", exc)
        return doc, sorted(set(classes)), sorted(set(functions)), sorted(set(imports))

    def _find_frontend_direct_hits(self, feature_name: str, category: str, frontend: FrontendAnalysis) -> List[str]:
        hits: Set[str] = set()
        tokens = set(_tokenize(feature_name)) | {feature_name.lower(), _slugify(feature_name).replace("-", "")}
        exact_only = category in DIRECT_UI_CATEGORIES
        for tok in list(tokens):
            norm = _normalize_token(tok)
            if not norm or len(norm) < 3:
                continue
            for index in (frontend.feature_index, frontend.export_index, frontend.route_index):
                for key, rels in index.items():
                    key_norm = _normalize_token(key)
                    if key_norm == norm:
                        hits.update(rels)
                    elif not exact_only and len(norm) >= 5 and (norm in key_norm or key_norm in norm):
                        hits.update(rels)
        return sorted(hits)

    def _find_frontend_feature_alias_hits(self, feature_name: str, category: str, frontend: FrontendAnalysis) -> List[str]:
        hits: Set[str] = set()
        feature_norm = _normalize_token(feature_name)

        alias_terms: Set[str] = set()
        alias_terms.update(FEATURE_ALIASES.get(feature_norm, set()))
        alias_terms.update(FEATURE_ALIASES.get(feature_name.lower(), set()))

        # Do not allow category aliases to stand in for feature aliases for direct UI categories.
        # That is the main bug fix preventing "communications UI exists" from implying "email UI exists".
        if category in FAMILY_CATEGORIES and not alias_terms:
            alias_terms.update(CATEGORY_ALIASES.get(category, set()))

        for alias in alias_terms:
            for tok in _tokenize(alias):
                hits.update(frontend.label_index.get(tok, []))
                hits.update(frontend.export_index.get(tok, []))
                hits.update(frontend.route_index.get(tok, []))
                hits.update(frontend.feature_index.get(tok, []))
        return sorted(hits)

    def _match_module_to_endpoints(
        self,
        module_name: str,
        feature_name: str,
        category: str,
        tokens: Sequence[str],
        imports: Sequence[str],
        endpoints: Sequence[ApiEndpoint],
    ) -> List[str]:
        out: List[str] = []
        feature_tokens = set(_tokenize(feature_name))
        category_tokens = set(CATEGORY_KEYWORDS.get(category, set()))
        module_norm = _normalize_token(module_name)
        feature_norm = _normalize_token(feature_name)
        import_norms = {_normalize_token(x.split(".")[-1]) for x in imports}
        explicit_endpoint_aliases = {_normalize_token(x) for x in FEATURE_ALIASES.get(feature_norm, set())}

        for ep in endpoints:
            ep_tokens = set(ep.handler_tokens + ep.route_tokens + _tokenize(" ".join(ep.imported_modules)))
            reasons: List[str] = []
            score = 0

            route_norm = _normalize_token(ep.route)
            handler_norm = _normalize_token(ep.handler)
            imported_norms = {_normalize_token(x.split(".")[-1]) for x in ep.imported_modules}

            if module_norm and module_norm in imported_norms:
                score += 8
                reasons.append("endpoint_imports_module")
            if feature_norm and feature_norm in imported_norms:
                score += 7
                reasons.append("endpoint_imports_feature")
            if feature_norm and (feature_norm in route_norm or feature_norm in handler_norm):
                score += 6
                reasons.append("exact_feature_route_or_handler")

            explicit_overlap = feature_tokens & set(ep.handler_tokens + ep.route_tokens)
            if explicit_overlap:
                score += min(4, len(explicit_overlap))
                reasons.append("feature_token_overlap")

            alias_overlap = explicit_endpoint_aliases & {_normalize_token(x) for x in ep.handler_tokens + ep.route_tokens}
            if alias_overlap:
                score += min(3, len(alias_overlap))
                reasons.append("feature_alias_overlap")

            category_overlap = category_tokens & ep_tokens
            if len(category_overlap) >= 2:
                score += 1
                reasons.append("category_overlap")

            import_overlap = import_norms & imported_norms
            if import_overlap:
                score += 2
                reasons.append("shared_import")

            strong_feature_evidence = any(x in reasons for x in (
                "endpoint_imports_module", "endpoint_imports_feature",
                "exact_feature_route_or_handler", "feature_token_overlap", "feature_alias_overlap",
            ))
            if not strong_feature_evidence:
                continue

            if score >= 7:
                out.append(f"{ep.route} [{','.join(ep.methods)}] via {ep.file}:{ep.handler} | score={score} | {','.join(reasons)}")

        return sorted(set(out))

    def _classify_module(
        self,
        module_name: str,
        feature_name: str,
        category: str,
        tokens: Sequence[str],
        doc: str,
        frontend_hits: Sequence[str],
        alias_hits: Sequence[str],
        family_hits: Sequence[str],
        endpoint_matches: Sequence[str],
    ) -> Tuple[str, str, str, bool]:
        feature_norm = _normalize_token(feature_name)
        low_blob = (module_name + "\n" + doc).lower()

        if feature_norm in EXCLUSION_EXACT or any(part in feature_norm for part in EXCLUSION_PARTIAL):
            return (
                "internal_only_no_ui",
                "high",
                "Suppressed because module appears to be core/internal/governance/runtime infrastructure, not a direct Web UI capability.",
                False,
            )

        if "no ui" in low_blob and category not in DIRECT_UI_CATEGORIES:
            return (
                "internal_only_no_ui",
                "high",
                "Suppressed because module documentation explicitly indicates no direct UI exposure.",
                False,
            )

        if frontend_hits:
            return (
                "frontend_present_backend_partial" if endpoint_matches else "frontend_present",
                "high",
                "Frontend already contains a direct screen/panel/route/service/export match for this capability.",
                False,
            )

        if family_hits:
            return (
                "frontend_present_backend_partial" if endpoint_matches else "frontend_present",
                "high",
                "Frontend already contains a family shell for this capability category.",
                False,
            )

        if alias_hits:
            return (
                "frontend_present_backend_partial" if endpoint_matches else "frontend_present_alias",
                "high",
                "Frontend already contains a feature-faithful alias/friendly-label presentation surface for this capability.",
                False,
            )

        if category not in DIRECT_UI_CATEGORIES:
            return (
                "insufficient_evidence",
                "medium",
                "Module is not internal, but it does not yet read as a clear direct user-facing frontend surface.",
                False,
            )

        score = 0
        reasons: List[str] = []
        category_words = CATEGORY_KEYWORDS.get(category, set())
        overlap = set(tokens) & category_words
        if overlap:
            overlap_score = min(6, len(overlap) + 2)
            score += overlap_score
            reasons.append(f"category_intent:{category}+{overlap_score}")
        if endpoint_matches:
            score += 2
            reasons.append("endpoint_evidence+2")
        if any(x in low_blob for x in ("purpose", "supported path", "security model", "service")):
            score += 1
            reasons.append("structured_doc+1")

        confidence = _confidence_from_score(score)
        return (
            "missing_frontend_module",
            confidence,
            "; ".join(reasons) if reasons else "User-facing backend capability appears present but no frontend surface was detected.",
            True,
        )

    # ------------------------------------------------------------------
    # Candidate conversion / outputs
    # ------------------------------------------------------------------

    def detect_gap_candidates(self, modules: Sequence[BackendModule]) -> List[GapCandidate]:
        out: List[GapCandidate] = []
        for mod in modules:
            if mod.classification != "missing_frontend_module" or not mod.should_generate:
                continue
            out.append(GapCandidate(
                module_name=mod.module_name,
                feature_name=mod.feature_name,
                slug=_slugify(mod.feature_name),
                rel_file=mod.rel_path,
                category=mod.category,
                confidence=mod.confidence,
                endpoint_matches=list(mod.endpoint_matches),
                candidate_reason=mod.candidate_reason,
                proposed_kind=self._choose_kind(mod.category),
            ))
        out.sort(key=lambda x: (x.confidence != "high", x.module_name.lower()))
        _write_json(self.analysis_dir / "ui_gap_candidates.json", [asdict(x) for x in out])
        self._write_gap_report(modules, out)
        return out

    def _choose_kind(self, category: str) -> str:
        if category == "communication":
            return "communication_screen"
        if category == "terminal":
            return "tool_screen"
        if category == "browser":
            return "workspace_screen"
        if category == "addons":
            return "workspace_screen"
        return "screen"

    def _write_gap_report(self, modules: Sequence[BackendModule], candidates: Sequence[GapCandidate]) -> None:
        lines = [
            "# UI Self-Aware Classification Report",
            "",
            f"Generated: `{_now_iso()}`",
            "",
            "## Generated Candidates",
            "",
            "| Module | Category | Classification | Confidence | Reason |",
            "|---|---|---|---|---|",
        ]
        for c in candidates:
            lines.append(f"| `{c.module_name}` | `{c.category}` | `missing_frontend_module` | {c.confidence} | {c.candidate_reason.replace('|', '/')} |")
        lines.extend([
            "",
            "## Full Backend Classification",
            "",
            "| Module | Category | Classification | Confidence | Direct Hits | Alias Hits | Family Hits | Endpoint Matches | Reason |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for m in modules:
            lines.append(
                f"| `{m.module_name}` | `{m.category}` | `{m.classification}` | {m.confidence} | {len(m.frontend_hits)} | {len(m.alias_hits)} | {len(m.family_hits)} | {len(m.endpoint_matches)} | {m.candidate_reason.replace('|', '/')} |"
            )
        _write_text(self.analysis_dir / "ui_gap_report.md", "\n".join(lines) + "\n")

    def generate_solutions(self, frontend: FrontendAnalysis, candidates: Sequence[GapCandidate]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for cand in candidates:
            root = self.solution_dir / cand.module_name
            screen_dir = root / frontend.inferred_dirs.get("screens", "src/pages")
            service_dir = root / frontend.inferred_dirs.get("services", "src/services")
            types_dir = root / frontend.inferred_dirs.get("types", "src/types")
            route_dir = root / frontend.inferred_dirs.get("routes", "src/routes")
            nav_dir = root / frontend.inferred_dirs.get("navigation", "src/components/navigation")
            notes_dir = root / "notes"

            screen_name = f"{cand.feature_name}Screen"
            type_name = f"{cand.feature_name}Record"
            api_func = f"fetch{cand.feature_name}Data"
            api_file_name = f"{cand.feature_name[:1].lower() + cand.feature_name[1:]}Api.ts" if cand.feature_name else "featureApi.ts"
            endpoint_stub = self._choose_endpoint_stub(cand)

            screen_path = screen_dir / f"{screen_name}.tsx"
            api_path = service_dir / api_file_name
            types_path = types_dir / f"{cand.slug}.ts"
            route_path = route_dir / f"{cand.slug}.route.tsx"
            nav_path = nav_dir / f"{cand.slug}.nav.ts"
            notes_path = notes_dir / "solution_summary.md"

            screen_import_api = self._rel_import(screen_path, api_path)
            screen_import_type = self._rel_import(screen_path, types_path)
            api_import_type = self._rel_import(api_path, types_path)
            route_import_screen = self._rel_import(route_path, screen_path)

            _write_text(screen_path, self._render_screen_tsx(cand, screen_name, type_name, api_func, screen_import_api, screen_import_type, endpoint_stub))
            _write_text(api_path, self._render_api_ts(cand, type_name, api_func, api_import_type, endpoint_stub))
            _write_text(types_path, self._render_types_ts(type_name))
            _write_text(route_path, self._render_route_stub(cand, screen_name, route_import_screen))
            _write_text(nav_path, self._render_nav_stub(cand))
            _write_text(notes_path, self._render_solution_notes(cand, screen_path, api_path, types_path, route_path, nav_path, endpoint_stub))

            results.append({
                "module_name": cand.module_name,
                "solution_root": str(root),
                "endpoint_stub": endpoint_stub,
                "files": [
                    str(screen_path.relative_to(root)),
                    str(api_path.relative_to(root)),
                    str(types_path.relative_to(root)),
                    str(route_path.relative_to(root)),
                    str(nav_path.relative_to(root)),
                    str(notes_path.relative_to(root)),
                ],
            })

        _write_json(self.analysis_dir / "generated_solution_index.json", results)
        return results

    def write_audits(self, modules: Sequence[BackendModule]) -> List[Dict[str, Any]]:
        audits: List[Dict[str, Any]] = []
        for mod in modules:
            if mod.classification not in {"frontend_present", "frontend_present_alias", "frontend_present_backend_partial"}:
                continue
            root = self.audit_dir / mod.module_name
            note_path = root / "audit_summary.md"
            hit_lines = []
            for label, items in (("direct", mod.frontend_hits), ("alias", mod.alias_hits), ("family", mod.family_hits)):
                if items:
                    hit_lines.append(f"### {label.title()} Frontend Evidence")
                    for item in items:
                        hit_lines.append(f"- `{item}`")
                    hit_lines.append("")
            endpoint_lines = [f"- `{ep}`" for ep in mod.endpoint_matches] or ["- No endpoint evidence recorded."]
            content = textwrap.dedent(f"""
                # Frontend Presence Audit

                - Generated at: `{_now_iso()}`
                - Module: `{mod.module_name}`
                - Feature: `{mod.feature_name}`
                - Category: `{mod.category}`
                - Classification: `{mod.classification}`
                - Confidence: `{mod.confidence}`

                ## Why this was not generated

                {mod.candidate_reason}

                {chr(10).join(hit_lines).strip()}

                ## Endpoint Evidence

                {chr(10).join(endpoint_lines)}
            """).strip() + "\n"
            _write_text(note_path, content)
            audits.append({"module_name": mod.module_name, "audit_root": str(root), "file": str(note_path.relative_to(root))})
        _write_json(self.analysis_dir / "audit_index.json", audits)
        return audits

    def _choose_endpoint_stub(self, cand: GapCandidate) -> str:
        if cand.endpoint_matches:
            primary = cand.endpoint_matches[0].split(" [", 1)[0].strip()
            if primary.startswith("/api/"):
                return primary
        if cand.category == "communication":
            return f"/api/{cand.slug}/status"
        if cand.category == "terminal":
            return f"/api/{cand.slug}/status"
        if cand.category == "browser":
            return f"/api/{cand.slug}/fetch"
        if cand.category == "addons":
            return f"/api/{cand.slug}/list"
        return f"/api/{cand.slug}"

    def _rel_import(self, from_file: Path, to_file: Path) -> str:
        rel = os.path.relpath(to_file, start=from_file.parent).replace("\\", "/")
        if not rel.startswith("."):
            rel = "./" + rel
        return re.sub(r"\.(ts|tsx|js|jsx)$", "", rel)

    def _render_screen_tsx(self, cand: GapCandidate, screen_name: str, type_name: str, api_func: str, api_import: str, type_import: str, endpoint_stub: str) -> str:
        title = _camel_to_words(cand.feature_name)
        return textwrap.dedent(f"""
            import React, {{ useEffect, useMemo, useState }} from 'react';
            import {{ {api_func} }} from '{api_import}';
            import type {{ {type_name} }} from '{type_import}';

            type ViewState = 'idle' | 'loading' | 'ready' | 'error';

            export default function {screen_name}() {{
              const [items, setItems] = useState<{type_name}[]>([]);
              const [status, setStatus] = useState<ViewState>('idle');
              const [error, setError] = useState<string>('');

              useEffect(() => {{
                let mounted = true;
                setStatus('loading');
                {api_func}()
                  .then((rows) => {{
                    if (!mounted) return;
                    setItems(Array.isArray(rows) ? rows : []);
                    setStatus('ready');
                  }})
                  .catch((err) => {{
                    if (!mounted) return;
                    setError(err instanceof Error ? err.message : 'Unknown error');
                    setStatus('error');
                  }});
                return () => {{ mounted = false; }};
              }}, []);

              const meta = useMemo(() => ({{
                endpoint: '{endpoint_stub}',
                confidence: '{cand.confidence}',
                proposedKind: '{cand.proposed_kind}',
                total: items.length,
              }}), [items]);

              return (
                <div className="sm-screen sm-{cand.slug}-screen">
                  <header className="sm-screen__header">
                    <h1>{title}</h1>
                    <p>Sandbox-generated proposal for {cand.module_name}.</p>
                    <p>{cand.candidate_reason}</p>
                  </header>

                  <section className="sm-screen__meta">
                    <div><strong>Status:</strong> {{meta.endpoint ? status : 'idle'}}</div>
                    <div><strong>Endpoint:</strong> {{meta.endpoint}}</div>
                    <div><strong>Confidence:</strong> {{meta.confidence}}</div>
                    <div><strong>Proposed kind:</strong> {{meta.proposedKind}}</div>
                    <div><strong>Total items:</strong> {{meta.total}}</div>
                  </section>

                  {{status === 'loading' && <p>Loading {title} data...</p>}}
                  {{status === 'error' && <p>Unable to load data: {{error}}</p>}}
                  {{status === 'ready' && <pre>{{JSON.stringify(items, null, 2)}}</pre>}}
                </div>
              );
            }}
        """).strip() + "\n"

    def _render_api_ts(self, cand: GapCandidate, type_name: str, api_func: str, type_import: str, endpoint_stub: str) -> str:
        method = "POST" if any("[POST]" in x or "[POST," in x for x in cand.endpoint_matches) else "GET"
        return textwrap.dedent(f"""
            import type {{ {type_name} }} from '{type_import}';

            const DEFAULT_ENDPOINT = '{endpoint_stub}';

            export async function {api_func}(): Promise<{type_name}[]> {{
              const response = await fetch(DEFAULT_ENDPOINT, {{
                method: '{method}',
                headers: {{ 'Content-Type': 'application/json' }},
              }});

              if (!response.ok) {{
                throw new Error(`Request failed with status ${{response.status}}`);
              }}

              const payload = await response.json().catch(() => null);
              if (Array.isArray(payload)) return payload as {type_name}[];
              if (payload && Array.isArray((payload as any).items)) return (payload as any).items as {type_name}[];
              if (payload && Array.isArray((payload as any).data)) return (payload as any).data as {type_name}[];
              return payload ? [payload as {type_name}] : [];
            }}
        """).strip() + "\n"

    def _render_types_ts(self, type_name: str) -> str:
        return textwrap.dedent(f"""
            export default interface {type_name} {{
              id?: string | number;
              name?: string;
              title?: string;
              status?: string;
              message?: string;
              created_at?: string;
              updated_at?: string;
              [key: string]: unknown;
            }}
        """).strip() + "\n"

    def _render_route_stub(self, cand: GapCandidate, screen_name: str, screen_import: str) -> str:
        return textwrap.dedent(f"""
            import React from 'react';
            import {screen_name} from '{screen_import}';

            export const {cand.slug}Route = {{
              path: '/{cand.slug}',
              element: <{screen_name} />,
              title: '{_camel_to_words(cand.feature_name)}',
              sourceModule: '{cand.module_name}',
            }};
        """).strip() + "\n"

    def _render_nav_stub(self, cand: GapCandidate) -> str:
        return textwrap.dedent(f"""
            export const {cand.slug}NavItem = {{
              key: '{cand.slug}',
              label: '{_camel_to_words(cand.feature_name)}',
              route: '/{cand.slug}',
              sourceModule: '{cand.module_name}',
              generatedBy: 'SarahMemoryUISelfAware',
              generatedAt: '{_now_iso()}',
            }};
        """).strip() + "\n"

    def _render_solution_notes(self, cand: GapCandidate, screen_path: Path, api_path: Path, types_path: Path, route_path: Path, nav_path: Path, endpoint_stub: str) -> str:
        endpoint_md = "\n".join(f"- `{ep}`" for ep in cand.endpoint_matches) or "- No verified endpoint was found; a conservative stub was generated."
        files_md = "\n".join([
            f"- `{screen_path.as_posix()}`",
            f"- `{api_path.as_posix()}`",
            f"- `{types_path.as_posix()}`",
            f"- `{route_path.as_posix()}`",
            f"- `{nav_path.as_posix()}`",
        ])
        return textwrap.dedent(f"""
            # Sandbox Solution Summary

            - Generated at: `{_now_iso()}`
            - Source backend module: `{cand.module_name}`
            - Relative source file: `{cand.rel_file}`
            - Feature name: `{cand.feature_name}`
            - Category: `{cand.category}`
            - Proposed kind: `{cand.proposed_kind}`
            - Confidence: `{cand.confidence}`
            - Endpoint stub: `{endpoint_stub}`

            ## Why this solution was generated

            {cand.candidate_reason}

            ## Endpoint evidence

            {endpoint_md}

            ## Generated files

            {files_md}

            ## Review guidance

            - Confirm the backend capability truly needs a standalone frontend surface.
            - Compare folder placement against the live frontend conventions.
            - Replace conservative endpoint stubs with the final API contract if needed.
            - Promote only after manual review.
        """).strip() + "\n"

    # ------------------------------------------------------------------
    # Summary / run
    # ------------------------------------------------------------------

    def write_summary(
        self,
        frontend: FrontendAnalysis,
        endpoints: Sequence[ApiEndpoint],
        modules: Sequence[BackendModule],
        candidates: Sequence[GapCandidate],
        solutions: Sequence[Dict[str, Any]],
        audits: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = {
            "completed_at": _now_iso(),
            "base_dir": str(self.base_dir),
            "frontend_total_files": frontend.total_files,
            "api_endpoint_count": len(endpoints),
            "backend_module_count": len(modules),
            "candidate_count": len(candidates),
            "solution_count": len(solutions),
            "audit_count": len(audits),
            "top_candidates": [asdict(x) for x in list(candidates[:20])],
            "classifications": [
                {
                    "module_name": m.module_name,
                    "feature_name": m.feature_name,
                    "category": m.category,
                    "classification": m.classification,
                    "confidence": m.confidence,
                    "direct_hits": len(m.frontend_hits),
                    "alias_hits": len(m.alias_hits),
                    "family_hits": len(m.family_hits),
                    "endpoint_matches": len(m.endpoint_matches),
                    "reason": m.candidate_reason,
                }
                for m in modules
            ],
            "solutions": list(solutions),
            "audits": list(audits),
        }
        _write_json(self.analysis_dir / "run_summary.json", summary)
        return summary

    def run(self) -> Dict[str, Any]:
        logger.info("Starting SarahMemoryUISelfAware in %s", self.base_dir)
        self.prepare_workspace()
        self.refresh_snapshot()
        frontend = self.analyze_frontend()
        endpoints = self.analyze_api_endpoints()
        modules = self.analyze_backend_modules(frontend, endpoints)
        candidates = self.detect_gap_candidates(modules)
        solutions = self.generate_solutions(frontend, candidates)
        audits = self.write_audits(modules)
        summary = self.write_summary(frontend, endpoints, modules, candidates, solutions, audits)
        logger.info("Completed. Candidates=%s Solutions=%s Audits=%s", summary["candidate_count"], summary["solution_count"], summary["audit_count"])
        logger.info("Analysis dir: %s", self.analysis_dir)
        logger.info("Solution dir: %s", self.solution_dir)
        logger.info("Audit dir: %s", self.audit_dir)
        return summary


def main() -> int:
    try:
        tool = SarahMemoryUISelfAware()
        summary = tool.run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.exception("SarahMemoryUISelfAware failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

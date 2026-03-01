"""
SarahMemory - RSS Research Source (v8.0.0)
Drop into: ../data/mods/v800/sm_v800_research_rssfeedread.py

What it does:
- Adds RSS/Atom feed scanning as an optional web research source.
- SarahMemoryResearch.WebResearch.fetch_sources to include RSS digest.
- Uses built-ins (ElementTree) + BeautifulSoup (already used by SarahMemoryResearch.py)
- Keeps existing behavior/caching/synthesis intact, just adds one more "raw" source: "RSS".

Notes:
- RSS is polling-based, not truly realtime. You control “freshness” via polling interval elsewhere.
- This patch does NOT create background schedulers; it only fetches RSS when research is requested.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

# Import the module we are monkey-patching
import SarahMemoryResearch as SMR


# ------------------------------
# Defaults (safe fallbacks)
# ------------------------------
DEFAULT_RSS_FEEDS: Dict[str, List[str]] = {
    # World / Top
    "world": [
        # Add your preferred feeds here (examples intentionally omitted)
        # "https://example.com/rss",
    ],
    "tech": [],
    "finance": [],
    "science": [],
    "sports": [],
    "us": [],
}


def _get_bool(name: str, default: bool) -> bool:
    try:
        return bool(getattr(SMR.config, name))
    except Exception:
        return default


def _get_rss_feeds() -> Dict[str, List[str]]:
    """
    Optional config override:
      - config.RSS_FEEDS = { "world": [...], "tech": [...], ... }
    """
    try:
        feeds = getattr(SMR.config, "RSS_FEEDS", None)
        if isinstance(feeds, dict):
            # ensure list values
            out: Dict[str, List[str]] = {}
            for k, v in feeds.items():
                if isinstance(v, (list, tuple)):
                    out[str(k)] = [str(x) for x in v if x]
            return out or DEFAULT_RSS_FEEDS
    except Exception:
        pass
    return DEFAULT_RSS_FEEDS


def _strip_html(text: str) -> str:
    try:
        return SMR.BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)
    except Exception:
        return (text or "").strip()


def _parse_rss_or_atom(xml_text: str, limit: int = 10) -> List[Dict]:
    """
    Returns list of items:
      { "title": str, "link": str, "summary": str, "ts": float|None, "source": str|None }
    """
    items: List[Dict] = []
    if not xml_text:
        return items

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # RSS 2.0: <rss><channel><item>...</item></channel></rss>
    channel = root.find("channel")
    if channel is not None:
        for it in channel.findall("item")[:limit]:
            title = _strip_html((it.findtext("title") or "").strip())
            link = (it.findtext("link") or "").strip()
            desc = _strip_html((it.findtext("description") or "").strip())
            pub = (it.findtext("pubDate") or "").strip()

            ts = None
            if pub:
                try:
                    ts = parsedate_to_datetime(pub).timestamp()
                except Exception:
                    ts = None

            if title or link or desc:
                items.append({"title": title, "link": link, "summary": desc, "ts": ts})
        return items

    # Atom: namespaces
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns)[:limit]:
        title = _strip_html((e.findtext("a:title", default="", namespaces=ns) or "").strip())

        link_el = e.find("a:link", ns)
        link = ""
        if link_el is not None:
            link = (link_el.get("href") or "").strip()

        summ = _strip_html((e.findtext("a:summary", default="", namespaces=ns) or "").strip())
        upd = (e.findtext("a:updated", default="", namespaces=ns) or "").strip()

        ts = None
        if upd:
            try:
                ts = parsedate_to_datetime(upd).timestamp()
            except Exception:
                ts = None

        if title or link or summ:
            items.append({"title": title, "link": link, "summary": summ, "ts": ts})

    return items


def _looks_like_news_query(query: str) -> bool:
    q = (query or "").lower()
    news_terms = (
        "news", "headlines", "breaking", "latest", "today", "this week",
        "world news", "current events", "what happened", "update"
    )
    return any(t in q for t in news_terms)


async def _fetch_one_feed(url: str, timeout: int) -> Optional[str]:
    # Reuse SMR.WebResearch.fetch_html so we keep consistent session behavior and timeouts
    try:
        return await SMR.WebResearch.fetch_html(url, timeout=timeout)
    except Exception:
        return None


async def _fetch_rss_digest(
    query: str,
    category: str = "world",
    per_feed_limit: int = 6,
    max_items: int = 12,
) -> Optional[str]:
    feeds_map = _get_rss_feeds()
    feeds = feeds_map.get(category) or feeds_map.get("world") or []
    feeds = [f for f in feeds if isinstance(f, str) and f.strip()]

    if not feeds:
        return None

    # Keep it light: short timeout
    timeout = getattr(SMR, "HTTP_TIMEOUT_SHORT", 8)

    tasks = [_fetch_one_feed(u, timeout=timeout) for u in feeds]
    pages = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: List[Dict] = []
    for p in pages:
        if isinstance(p, Exception) or not p:
            continue
        all_items.extend(_parse_rss_or_atom(p, limit=per_feed_limit))

    if not all_items:
        return None

    # Dedupe by title+link
    seen = set()
    dedup: List[Dict] = []
    for it in all_items:
        key = hashlib.md5(((it.get("title") or "") + (it.get("link") or "")).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)

    # Prefer newest
    dedup.sort(key=lambda x: x.get("ts") or 0, reverse=True)

    top = dedup[:max_items]
    lines: List[str] = []
    for i, it in enumerate(top, 1):
        title = (it.get("title") or "").strip()
        summ = (it.get("summary") or "").strip()
        link = (it.get("link") or "").strip()

        if title and summ:
            lines.append(f"{i}. {title} — {summ} ({link})" if link else f"{i}. {title} — {summ}")
        elif title:
            lines.append(f"{i}. {title} ({link})" if link else f"{i}. {title}")
        elif link:
            lines.append(f"{i}. {link}")

    return "\n".join(lines).strip() if lines else None


# -----------------------------------------------------------------------------
# Monkey patch: WebResearch.fetch_sources
# -----------------------------------------------------------------------------
_ORIG_fetch_sources = SMR.WebResearch.fetch_sources


async def _PATCHED_fetch_sources(query: str) -> Optional["SMR.ResearchResult"]:
    """
    Replacement for WebResearch.fetch_sources with RSS integrated as an optional source.
    Preserves:
      - cache behavior
      - other web sources behavior
      - synthesis behavior
    """
    start_time = time.time()

    # Respect master toggle: if web research is off, don't do RSS either
    if not _get_bool("WEB_RESEARCH_ENABLED", True):
        return await _ORIG_fetch_sources(query)

    # Check cache first (use the same cache map)
    cache_key = hashlib.md5((query or "").encode("utf-8")).hexdigest()
    try:
        if cache_key in SMR.research_cache:
            cached_result, expiry = SMR.research_cache[cache_key]
            if time.time() < expiry:
                return cached_result
    except Exception:
        pass

    clean_query = SMR.WebResearch.preprocess_query(query)
    SMR.logger.info(f"[WEB RESEARCH][RSS PATCH] Preprocessed Query: '{clean_query}'")

    raw: Dict[str, str] = {}
    tasks: List[asyncio.Future] = []

    # Keep original sources (mirrors SarahMemoryResearch.py logic)
    if _get_bool("WIKIPEDIA_RESEARCH_ENABLED", True):
        tasks.append(SMR.WebResearch._fetch_wikipedia(clean_query))

    if _get_bool("DUCKDUCKGO_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_duckduckgo(clean_query))

    if _get_bool("FREE_DICTIONARY_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_dictionary(clean_query))

    if _get_bool("OPENLIBRARY_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_openlibrary(clean_query))

    if _get_bool("STACKOVERFLOW_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_stackoverflow(clean_query))

    if _get_bool("REDDIT_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_reddit(clean_query))

    if _get_bool("WIKIHOW_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_wikihow(clean_query))

    if _get_bool("QUORA_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_quora(clean_query))

    if _get_bool("INTERNET_ARCHIVE_RESEARCH_ENABLED", False):
        tasks.append(SMR.WebResearch._fetch_archive(clean_query))

    # RSS: optional + smart-gated (news-like queries by default)
    rss_enabled = _get_bool("RSS_RESEARCH_ENABLED", True)
    rss_gate = _get_bool("RSS_NEWS_ONLY", True)

    if rss_enabled and (not rss_gate or _looks_like_news_query(query)):
        # category: allow override via config.RSS_DEFAULT_CATEGORY
        category = getattr(SMR.config, "RSS_DEFAULT_CATEGORY", "world") if hasattr(SMR, "config") else "world"
        tasks.append(_fetch_rss_digest(query=query, category=str(category)))

    # Run tasks concurrently
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                SMR.logger.warning(f"[WEB][RSS PATCH] Task {i} failed: {result}")
                continue

            # Original fetchers return Tuple[str,str]
            if isinstance(result, tuple) and len(result) == 2:
                source_name, content = result
                if source_name and isinstance(content, str) and content.strip():
                    raw[str(source_name)] = content.strip()
                continue

            # RSS digest returns str
            if isinstance(result, str) and result.strip():
                raw["RSS"] = result.strip()

    # Synthesize like original if we got anything
    if raw:
        combined_raw_text = "\n\n".join(
            [f"[{source}] {content}" for source, content in raw.items() if isinstance(content, str)]
        )

        synthesized = SMR.WebSemanticSynthesizer.synthesize_response(combined_raw_text, query)

        if synthesized and "still researching" not in synthesized.lower():
            latency_ms = int((time.time() - start_time) * 1000)

            out = SMR.ResearchResult(
                source=SMR.ResearchSource.WEB_WIKIPEDIA,  # keep original primary source
                content=synthesized,
                confidence=0.80,
                query=query,
                intent="research",
                latency_ms=latency_ms,
                metadata={"sources": list(raw.keys()), "raw_count": len(raw), "rss_included": ("RSS" in raw)},
            )

            # Cache result (use the module's cache_result helper)
            try:
                SMR.cache_result(query, out)
            except Exception:
                pass

            return out

        # If synthesis failed, return first valid raw result (prefer RSS if present)
        for preferred in ("RSS",):
            if preferred in raw and isinstance(raw[preferred], str) and raw[preferred].strip():
                latency_ms = int((time.time() - start_time) * 1000)
                return SMR.ResearchResult(
                    source=SMR.ResearchSource.WEB_DUCKDUCKGO,
                    content=raw[preferred],
                    confidence=0.70,
                    query=query,
                    intent="research",
                    latency_ms=latency_ms,
                    metadata={"sources": list(raw.keys()), "rss_included": True},
                )

        for val in raw.values():
            if val and isinstance(val, str):
                latency_ms = int((time.time() - start_time) * 1000)
                return SMR.ResearchResult(
                    source=SMR.ResearchSource.WEB_DUCKDUCKGO,
                    content=val,
                    confidence=0.70,
                    query=query,
                    intent="research",
                    latency_ms=latency_ms,
                    metadata={"sources": list(raw.keys()), "rss_included": ("RSS" in raw)},
                )

    SMR.logger.warning("[WEB][RSS PATCH] No web results found")
    return None


def apply_patch() -> None:
    """
    Call this once during startup after SarahMemoryResearch is imported.
    """
    SMR.WebResearch.fetch_sources = staticmethod(_PATCHED_fetch_sources)
    SMR.logger.info("[MONKEY PATCH] RSS Research patch applied to WebResearch.fetch_sources")


# Auto-apply if desired
AUTO_APPLY = True
if AUTO_APPLY:
    try:
        apply_patch()
    except Exception as e:
        try:
            SMR.logger.warning(f"[MONKEY PATCH] RSS Research patch failed to apply: {e}")
        except Exception:
            pass

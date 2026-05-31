"""Multi-tier live web data: SearxNG (primary) → Wikipedia API (fallback)."""

from __future__ import annotations

import html
import logging
import re
from typing import Any, TypedDict
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

SEARXNG_URLS = [
    "https://searx.be/search",
    "https://searx.tiekoetter.com/search",
    "https://search.mdn.eu/search",
]

SEARXNG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    ),
}

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT_SECONDS = 3
MAX_SNIPPET_CHARS = 500
MAX_EVIDENCE_ITEMS = 5

UNAVAILABLE_MESSAGE = "Live web search currently unavailable."


class EvidenceItem(TypedDict):
    title: str
    source: str
    url: str
    snippet: str


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities from Wikipedia snippets."""
    decoded = html.unescape(text)
    without_tags = re.sub(r"<[^>]+>", "", decoded)
    return re.sub(r"\s+", " ", without_tags).strip()


def _truncate_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _wikipedia_article_url(title: str) -> str:
    slug = quote(title.replace(" ", "_"), safe="/")
    return f"https://en.wikipedia.org/wiki/{slug}"


def _format_context_from_evidence(items: list[EvidenceItem]) -> str:
    chunks: list[str] = []
    for index, item in enumerate(items, start=1):
        chunks.append(f"[{index}] {item['title']}\n{item['snippet']}".strip())
    return "\n\n".join(chunks)


def _fetch_searxng_evidence(query: str) -> list[EvidenceItem]:
    """Tier 1: public SearxNG instances (structured evidence rows)."""
    params = {"q": query, "format": "json"}
    evidence: list[EvidenceItem] = []

    for url in SEARXNG_URLS:
        try:
            response = requests.get(
                url,
                headers=SEARXNG_HEADERS,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            print(f"SearxNG timeout at {url}: {exc}")
            logger.warning("SearxNG timeout at %s: %s", url, exc)
            continue
        except requests.RequestException as exc:
            print(f"SearxNG request failed at {url}: {exc}")
            logger.warning("SearxNG request failed at %s: %s", url, exc)
            continue

        if response.status_code != 200:
            print(
                f"SearxNG non-200 at {url}: status={response.status_code}",
            )
            logger.warning(
                "SearxNG non-200 at %s: status=%s",
                url,
                response.status_code,
            )
            continue

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            print(f"SearxNG invalid JSON at {url}: {exc}")
            logger.warning("SearxNG invalid JSON at %s: %s", url, exc)
            continue

        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            print(f"SearxNG returned no results at {url}")
            continue

        for result in results:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "Untitled").strip()
            snippet = _truncate_snippet(
                str(result.get("content") or "").strip(),
            )
            page_url = str(result.get("url") or "").strip()
            if not title or not snippet or not page_url:
                continue
            evidence.append(
                {
                    "title": title,
                    "source": "Web",
                    "url": page_url,
                    "snippet": snippet,
                },
            )
            if len(evidence) >= MAX_EVIDENCE_ITEMS:
                break

        if evidence:
            print(f"SearxNG succeeded at {url} ({len(evidence)} evidence items)")
            logger.info("SearxNG success at %s for query '%s'", url, query[:80])
            return evidence

        print(f"SearxNG had unparseable results at {url}")

    print("SearxNG failed on all instances")
    logger.error("All SearxNG instances failed for query '%s'", query[:80])
    return []


def _fetch_wikipedia_evidence(query: str) -> list[EvidenceItem]:
    """Tier 2: Wikipedia search API (structured evidence rows)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": MAX_EVIDENCE_ITEMS,
    }

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            headers=SEARXNG_HEADERS,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print(f"Wikipedia request failed: {exc}")
        logger.warning("Wikipedia request failed: %s", exc)
        return []

    if response.status_code != 200:
        print(f"Wikipedia non-200: status={response.status_code}")
        logger.warning("Wikipedia non-200: status=%s", response.status_code)
        return []

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        print(f"Wikipedia invalid JSON: {exc}")
        logger.warning("Wikipedia invalid JSON: %s", exc)
        return []

    search_results = payload.get("query", {}).get("search", [])
    if not isinstance(search_results, list) or not search_results:
        print("Wikipedia returned no search results")
        return []

    evidence: list[EvidenceItem] = []
    for item in search_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Untitled").strip()
        snippet = _truncate_snippet(_clean_html(str(item.get("snippet") or "")))
        if not title or not snippet:
            continue
        evidence.append(
            {
                "title": title,
                "source": "Wikipedia",
                "url": _wikipedia_article_url(title),
                "snippet": snippet,
            },
        )

    if evidence:
        print(f"Wikipedia succeeded ({len(evidence)} evidence items)")
        logger.info("Wikipedia success for query '%s'", query[:80])
        return evidence

    print("Wikipedia results had no usable snippets")
    return []


def _fallback_evidence(query: str) -> list[EvidenceItem]:
    message = (
        f"Limited live data for '{query}'. "
        "No SearxNG or Wikipedia results were retrieved; agents should "
        "reason carefully from the user premise and general knowledge."
    )
    return [
        {
            "title": f"Context: {query[:80]}",
            "source": "Shoal Engine",
            "url": "https://shoal.ai",
            "snippet": _truncate_snippet(message),
        },
    ]


def scrape_for_premise(query: str) -> tuple[str, list[EvidenceItem]]:
    """
    Fetch live context and structured evidence for a premise.
    Tier 1: SearxNG → Tier 2: Wikipedia → minimal placeholder.
    """
    trimmed = query.strip()
    if not trimmed:
        print("scrape_for_premise called with empty query")
        logger.warning("scrape_for_premise called with empty query")
        return UNAVAILABLE_MESSAGE, []

    print(f"scrape_for_premise starting for query: {trimmed[:120]}")
    logger.info("Starting multi-tier search for: %s", trimmed[:120])

    try:
        searx_evidence = _fetch_searxng_evidence(trimmed)
        if searx_evidence:
            return _format_context_from_evidence(searx_evidence), searx_evidence
    except Exception as exc:
        print(f"SearxNG tier raised unexpected error: {exc}")
        logger.exception("SearxNG tier unexpected error")

    print("SearxNG failed, falling back to Wikipedia...")
    try:
        wiki_evidence = _fetch_wikipedia_evidence(trimmed)
        if wiki_evidence:
            return _format_context_from_evidence(wiki_evidence), wiki_evidence
    except Exception as exc:
        print(f"Wikipedia tier raised unexpected error: {exc}")
        logger.exception("Wikipedia tier unexpected error")

    print("Wikipedia failed; returning minimal fallback context")
    logger.error("All scraper tiers failed for query '%s'", trimmed[:80])
    fallback = _fallback_evidence(trimmed)
    return _format_context_from_evidence(fallback), fallback


def search_web(query: str) -> str:
    """Backward-compatible helper: returns combined web context text only."""
    web_data, _ = scrape_for_premise(query)
    return web_data

"""Live web data: Tavily API (primary) → deep page fetch → Wikipedia → placeholder."""

from __future__ import annotations

import html
import logging
import os
import re
from typing import Any, TypedDict
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 3
DEEP_FETCH_URL_COUNT = 2

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ShoalAI-Engine/1.0; +https://shoal.ai)"
    ),
}
REQUEST_TIMEOUT_SECONDS = 12
PAGE_FETCH_TIMEOUT_SECONDS = 15
MAX_SNIPPET_CHARS = 500
MAX_EVIDENCE_SNIPPET_CHARS = 1800
MAX_PAGE_TEXT_CHARS = 4000
MAX_EVIDENCE_ITEMS = 5

UNAVAILABLE_MESSAGE = "Live web search currently unavailable."


class EvidenceItem(TypedDict):
    title: str
    source: str
    url: str
    snippet: str


def _get_tavily_api_key() -> str | None:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return key or None


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    decoded = html.unescape(text)
    without_tags = re.sub(r"<[^>]+>", "", decoded)
    return re.sub(r"\s+", " ", without_tags).strip()


def _truncate_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_fetchable_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _extract_page_text(html_content: str) -> str:
    """Extract readable body text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag_name in ("script", "style", "noscript", "header", "footer", "nav", "aside", "form"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    return _normalize_whitespace(text)


def _fetch_url_page_text(url: str) -> str | None:
    """Fetch and extract text from a single URL."""
    if not _is_fetchable_http_url(url):
        return None

    try:
        response = requests.get(
            url,
            headers=HTTP_HEADERS,
            timeout=PAGE_FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.Timeout:
        logger.warning("Deep fetch timed out for %s", url)
        return None
    except requests.RequestException as exc:
        logger.warning("Deep fetch failed for %s: %s", url, exc)
        return None

    if response.status_code != 200:
        logger.warning("Deep fetch non-200 for %s: status=%s", url, response.status_code)
        return None

    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        logger.debug("Skipping non-HTML content at %s (%s)", url, content_type)
        return None

    text = _extract_page_text(response.text)
    if len(text) < 120:
        logger.debug("Deep fetch returned too little text for %s", url)
        return None

    return _truncate_snippet(text, MAX_PAGE_TEXT_CHARS)


def _enrich_evidence_with_deep_fetch(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """Fetch full page text for the top N Tavily URLs."""
    enriched: list[EvidenceItem] = []

    for index, item in enumerate(evidence):
        updated: EvidenceItem = {
            "title": item["title"],
            "source": item["source"],
            "url": item["url"],
            "snippet": item["snippet"],
        }

        if index < DEEP_FETCH_URL_COUNT:
            page_text = _fetch_url_page_text(item["url"])
            if page_text:
                combined = (
                    f"Search preview: {item['snippet']}\n\n"
                    f"Deep page extract:\n{page_text}"
                )
                updated["snippet"] = _truncate_snippet(
                    combined,
                    MAX_EVIDENCE_SNIPPET_CHARS,
                )
                updated["source"] = "Web (Tavily + Deep Fetch)"
                logger.info(
                    "Deep fetch enriched evidence [%s] %s (%s chars)",
                    index + 1,
                    item["url"][:80],
                    len(page_text),
                )

        enriched.append(updated)

    return enriched


def _wikipedia_article_url(title: str) -> str:
    slug = quote(title.replace(" ", "_"), safe="/")
    return f"https://en.wikipedia.org/wiki/{slug}"


def _format_context_from_evidence(items: list[EvidenceItem]) -> str:
    chunks: list[str] = []
    for index, item in enumerate(items, start=1):
        chunks.append(
            (
                f"[Source {index}] {item['title']}\n"
                f"URL: {item['url']}\n"
                f"{item['snippet']}"
            ).strip(),
        )
    return "\n\n".join(chunks)


def _map_tavily_results(results: list[Any]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "Untitled").strip()
        snippet = _truncate_snippet(str(result.get("content") or "").strip())
        page_url = str(result.get("url") or "").strip()
        if not title or not snippet or not page_url:
            continue
        evidence.append(
            {
                "title": title,
                "source": "Web (Tavily)",
                "url": page_url,
                "snippet": snippet,
            },
        )
        if len(evidence) >= TAVILY_MAX_RESULTS:
            break

    return evidence


def _fetch_tavily_evidence(query: str) -> list[EvidenceItem]:
    """Tier 1: Tavily real-time search API + deep fetch on top URLs."""
    api_key = _get_tavily_api_key()
    if not api_key:
        logger.warning("TAVILY_API_KEY is not set; skipping Tavily search")
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": TAVILY_MAX_RESULTS,
    }

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        logger.warning("Tavily request timed out: %s", exc)
        return []
    except requests.RequestException as exc:
        logger.warning("Tavily request failed: %s", exc)
        return []

    if response.status_code != 200:
        body_preview = response.text[:300] if response.text else ""
        logger.warning(
            "Tavily non-200: status=%s body=%s",
            response.status_code,
            body_preview,
        )
        return []

    try:
        data: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning("Tavily invalid JSON: %s", exc)
        return []

    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        logger.info("Tavily returned no results for query '%s'", query[:80])
        return []

    evidence = _map_tavily_results(results)
    if not evidence:
        return []

    evidence = _enrich_evidence_with_deep_fetch(evidence)
    logger.info(
        "Tavily success for query '%s' (%s items, deep fetch on top %s)",
        query[:80],
        len(evidence),
        min(DEEP_FETCH_URL_COUNT, len(evidence)),
    )
    return evidence


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
            headers=HTTP_HEADERS,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("Wikipedia request failed: %s", exc)
        return []

    if response.status_code != 200:
        logger.warning("Wikipedia non-200: status=%s", response.status_code)
        return []

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning("Wikipedia invalid JSON: %s", exc)
        return []

    search_results = payload.get("query", {}).get("search", [])
    if not isinstance(search_results, list) or not search_results:
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
        logger.info("Wikipedia success for query '%s'", query[:80])
    return evidence


def _fallback_evidence(query: str) -> list[EvidenceItem]:
    """Tier 3: graceful placeholder so agents can still deliberate."""
    message = (
        f"Limited live data for '{query}'. "
        "Tavily and Wikipedia did not return usable results; agents should "
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
    Tier 1: Tavily (+ deep page fetch) → Tier 2: Wikipedia → placeholder.
    """
    trimmed = query.strip()
    if not trimmed:
        logger.warning("scrape_for_premise called with empty query")
        return UNAVAILABLE_MESSAGE, []

    logger.info("Starting deep Tavily-first search for: %s", trimmed[:120])

    try:
        tavily_evidence = _fetch_tavily_evidence(trimmed)
        if tavily_evidence:
            return _format_context_from_evidence(tavily_evidence), tavily_evidence
    except Exception:
        logger.exception("Tavily tier unexpected error")

    logger.info("Tavily unavailable or failed, falling back to Wikipedia...")
    try:
        wiki_evidence = _fetch_wikipedia_evidence(trimmed)
        if wiki_evidence:
            return _format_context_from_evidence(wiki_evidence), wiki_evidence
    except Exception:
        logger.exception("Wikipedia tier unexpected error")

    logger.error("All scraper tiers failed for query '%s'", trimmed[:80])
    fallback = _fallback_evidence(trimmed)
    return _format_context_from_evidence(fallback), fallback


def search_web(query: str) -> str:
    """Backward-compatible helper: returns combined web context text only."""
    web_data, _ = scrape_for_premise(query)
    return web_data

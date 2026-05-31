"""Multi-tier live web data: SearxNG (primary) → Wikipedia API (fallback)."""

import html
import logging
import re
from typing import Any

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

UNAVAILABLE_MESSAGE = "Live web search currently unavailable."


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities from Wikipedia snippets."""
    decoded = html.unescape(text)
    without_tags = re.sub(r"<[^>]+>", "", decoded)
    return re.sub(r"\s+", " ", without_tags).strip()


def _search_searxng(query: str) -> str | None:
    """Tier 1: public SearxNG instances (3s timeout per request)."""
    params = {"q": query, "format": "json"}

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

        chunks: list[str] = []
        for index, result in enumerate(results[:3], start=1):
            if not isinstance(result, dict):
                continue
            title = result.get("title", "Untitled")
            content = result.get("content", "")
            chunks.append(f"[{index}] {title}\n{content}".strip())

        if chunks:
            combined = "\n\n".join(chunks)
            print(f"SearxNG succeeded at {url} ({len(chunks)} results)")
            logger.info("SearxNG success at %s for query '%s'", url, query[:80])
            return combined

        print(f"SearxNG had unparseable results at {url}")

    print("SearxNG failed on all instances")
    logger.error("All SearxNG instances failed for query '%s'", query[:80])
    return None


def _search_wikipedia(query: str) -> str | None:
    """Tier 2: Wikipedia search API (top 2 snippets)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": 2,
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
        return None

    if response.status_code != 200:
        print(f"Wikipedia non-200: status={response.status_code}")
        logger.warning("Wikipedia non-200: status=%s", response.status_code)
        return None

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        print(f"Wikipedia invalid JSON: {exc}")
        logger.warning("Wikipedia invalid JSON: %s", exc)
        return None

    search_results = payload.get("query", {}).get("search", [])
    if not isinstance(search_results, list) or not search_results:
        print("Wikipedia returned no search results")
        return None

    chunks: list[str] = []
    for index, item in enumerate(search_results[:2], start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title", "Untitled")
        snippet = _clean_html(item.get("snippet", ""))
        if snippet:
            chunks.append(f"[{index}] {title}\n{snippet}")

    if not chunks:
        print("Wikipedia results had no usable snippets")
        return None

    combined = "\n\n".join(chunks)
    print(f"Wikipedia succeeded ({len(chunks)} results)")
    logger.info("Wikipedia success for query '%s'", query[:80])
    return combined


def search_web(query: str) -> str:
    """
    Fetch live context for a premise.
    Tier 1: SearxNG → Tier 2: Wikipedia → minimal placeholder string.
    """
    trimmed = query.strip()
    if not trimmed:
        print("search_web called with empty query")
        logger.warning("search_web called with empty query")
        return UNAVAILABLE_MESSAGE

    print(f"search_web starting for query: {trimmed[:120]}")
    logger.info("Starting multi-tier search for: %s", trimmed[:120])

    try:
        searx_data = _search_searxng(trimmed)
        if searx_data:
            return searx_data
    except Exception as exc:
        print(f"SearxNG tier raised unexpected error: {exc}")
        logger.exception("SearxNG tier unexpected error")

    print("SearxNG failed, falling back to Wikipedia...")
    try:
        wiki_data = _search_wikipedia(trimmed)
        if wiki_data:
            return wiki_data
    except Exception as exc:
        print(f"Wikipedia tier raised unexpected error: {exc}")
        logger.exception("Wikipedia tier unexpected error")

    print("Wikipedia failed; returning minimal fallback context")
    logger.error("All scraper tiers failed for query '%s'", trimmed[:80])

    return (
        f"Limited live data for '{trimmed}'. "
        "No SearxNG or Wikipedia results were retrieved; agents should "
        "reason carefully from the user premise and general knowledge."
    )

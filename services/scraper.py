"""Live web search via public SearxNG instances."""

import logging
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

UNAVAILABLE_MESSAGE = "Live web search currently unavailable."


def search_web(query: str) -> str:
    """Fetch top 3 live web results via public SearxNG instances."""
    trimmed = query.strip()
    if not trimmed:
        logger.warning("search_web called with empty query")
        return UNAVAILABLE_MESSAGE

    params = {"q": trimmed, "format": "json"}
    logger.info("Starting SearxNG search for query: %s", trimmed[:120])

    for url in SEARXNG_URLS:
        try:
            response = requests.get(
                url,
                headers=SEARXNG_HEADERS,
                params=params,
                timeout=5,
            )
        except requests.Timeout as exc:
            logger.warning("SearxNG timeout at %s: %s", url, exc)
            continue
        except requests.RequestException as exc:
            logger.warning("SearxNG request failed at %s: %s", url, exc)
            continue

        if response.status_code != 200:
            logger.warning(
                "SearxNG non-200 at %s: status=%s body_preview=%s",
                url,
                response.status_code,
                response.text[:200],
            )
            continue

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            logger.warning("SearxNG invalid JSON at %s: %s", url, exc)
            continue

        results = payload.get("results", [])
        if not isinstance(results, list):
            logger.warning(
                "SearxNG unexpected results shape at %s: type=%s",
                url,
                type(results).__name__,
            )
            continue

        top_results = results[:3]
        if not top_results:
            logger.info("SearxNG returned zero results at %s", url)
            continue

        chunks: list[str] = []
        for index, result in enumerate(top_results, start=1):
            if not isinstance(result, dict):
                logger.debug(
                    "Skipping non-dict result at %s index %s", url, index
                )
                continue
            title = result.get("title", "Untitled")
            content = result.get("content", "")
            chunks.append(f"[{index}] {title}\n{content}".strip())

        if not chunks:
            logger.warning("SearxNG had results but none were parseable at %s", url)
            continue

        combined = "\n\n".join(chunks)
        logger.info(
            "SearxNG success at %s: %s results for query '%s'",
            url,
            len(chunks),
            trimmed[:80],
        )
        return combined

    logger.error(
        "All SearxNG instances failed for query '%s'",
        trimmed[:80],
    )
    return UNAVAILABLE_MESSAGE

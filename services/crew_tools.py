"""CrewAI tool wiring for Shoal engine."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def build_tavily_search_tool():
    """Tavily search tool for worker agents (requires TAVILY_API_KEY)."""
    if not os.getenv("TAVILY_API_KEY", "").strip():
        logger.warning("TAVILY_API_KEY missing; workers will run without live search tool")
        return None

    try:
        from crewai_tools import TavilySearchTool
    except ImportError:
        try:
            from crewai_tools.tools.tavily_search_tool.tavily_search_tool import (
                TavilySearchTool,
            )
        except ImportError as exc:
            raise RuntimeError(
                "crewai-tools TavilySearchTool is not installed. "
                "Run: pip install 'crewai[tools]' tavily-python",
            ) from exc

    return TavilySearchTool(
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

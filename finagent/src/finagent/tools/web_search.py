from __future__ import annotations

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from tenacity import retry, stop_after_attempt, wait_exponential

from finagent.config import get_settings

_search_client: TavilySearch | None = None


def _get_search_client() -> TavilySearch | None:
    global _search_client
    if _search_client is None:
        settings = get_settings()
        if not settings.tavily_api_key:
            return None  # key not configured
        _search_client = TavilySearch(
            max_results=5,
            tavily_api_key=settings.tavily_api_key,
        )
    return _search_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=False)
def _invoke_search(client: TavilySearch, query: str) -> dict | list:
    return client.invoke({"query": query})


@tool
def web_search(query: str) -> str:
    """Search the web for recent news and information about a company or topic.

    Args:
        query: Search query string (e.g. 'AAPL earnings Q4 2024 results')
    """
    try:
        client = _get_search_client()
        if client is None:
            return "Web search unavailable (TAVILY_API_KEY not configured). Skipping web research."
        raw = _invoke_search(client, query)
        # TavilySearch returns a dict: {"results": [...], "query": ..., ...}
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if isinstance(results, list):
            formatted = []
            for r in results:
                formatted.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', '')[:300]}\n"
                )
            return "\n---\n".join(formatted)
        return str(raw)
    except Exception as exc:
        return f"Web search failed: {exc}"

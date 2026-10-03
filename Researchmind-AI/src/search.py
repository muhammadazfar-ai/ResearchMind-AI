import os

from dotenv import load_dotenv

load_dotenv()


class SearchError(Exception):
    """Raised when web search cannot run or returns nothing usable."""


def _get_key() -> str:
    return os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'")


def search_topic(topic: str, max_results: int = 6, search_depth: str = "advanced") -> list:
    """Searches the web with Tavily and returns [{title, content, url}, ...].

    Raises SearchError with a user-friendly message if the key is missing,
    the request fails, or no usable results come back.
    """
    key = _get_key()
    if not key:
        raise SearchError("`TAVILY_API_KEY` is missing in your `.env` file (or Space secrets).")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        response = client.search(
            query=topic,
            search_depth=search_depth,
            max_results=max_results,
        )
    except Exception as e:
        print(f"[DEBUG ERROR] Tavily search failed for '{topic}': {e}")
        raise SearchError(f"Web search failed: `{e}`")

    results, seen_urls = [], set()
    for item in response.get("results", []):
        url = item.get("url", "")
        content = (item.get("content") or "").strip()
        if not content or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({
            "title": (item.get("title") or url or "Untitled source").strip(),
            "content": content,
            "url": url,
        })

    if not results:
        raise SearchError("The web search returned no usable results for this topic.")
    return results

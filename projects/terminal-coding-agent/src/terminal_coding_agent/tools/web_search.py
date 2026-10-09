"""Web search tool: DuckDuckGo text results, no page fetching.

`ddgs` is imported lazily inside `_search`: the tool is off by default, Harbor's
container has no `ddgs`, and this module must still be importable there.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool

from terminal_coding_agent.config import WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT_SECONDS
from terminal_coding_agent.tools.display import tool_output
from terminal_coding_agent.tools.truncate import truncate


class WebSearchError(Exception):
    """One search attempt failed; the message is already user-facing."""


def _search(query: str, limit: int) -> list[dict[str, Any]]:
    """One DuckDuckGo text search. An empty list means zero hits; raises `ImportError` without `ddgs`.

    ddgs signals zero hits by raising `DDGSException("No results found.")` rather than
    returning `[]`, so that one case is normalised to the empty list: the caller then has
    a single zero-hit path. No typed exception distinguishes it, hence the text match.
    """
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    try:
        return DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS).text(query, max_results=limit)
    except TimeoutException as exc:
        raise WebSearchError(f"search timed out after {WEB_SEARCH_TIMEOUT_SECONDS}s") from exc
    except RatelimitException as exc:
        raise WebSearchError(f"search rate limited: {exc}") from exc
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise WebSearchError(f"search failed: {exc}") from exc


def _format(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item.get('title', '')}")
        lines.append(f"   {item.get('href', '')}")
        lines.append(f"   {item.get('body', '')}")
    return "\n".join(lines)


def build_web_search() -> BaseTool:
    def _display(params: dict, result: str) -> tuple[str, str]:
        if result.startswith("No results"):
            return "no results", f"```text\n{result}\n```"
        count = sum(1 for line in result.splitlines() if line[:1].isdigit() and line[1:2] == ".")
        unit = "result" if count == 1 else "results"
        return f"{count} {unit}", f"```text\n{result}\n```"

    @tool
    @tool_output(_display)
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the public web with DuckDuckGo and return titles, URLs and snippets.

        Text search results only; this tool never fetches the result pages.

        Args:
            query: The search query.
            max_results: How many results to return (1-10; default 5).

        Returns:
            A numbered list of title / url / snippet, truncated to ~4000 tokens.
        """
        limit = max(1, min(max_results, WEB_SEARCH_MAX_RESULTS))
        try:
            results = _search(query, limit)
        except ImportError:
            return truncate("Error: ddgs not installed")
        except WebSearchError as exc:
            return truncate(f"Error: {exc}")

        if not results:
            # Zero hits is an outcome, not a failure: no `Error:` prefix, and the query is
            # kept so the model can rephrase instead of concluding the tool is broken.
            return truncate(f"No results for: {query}")
        return truncate(_format(results[:limit]))

    return web_search

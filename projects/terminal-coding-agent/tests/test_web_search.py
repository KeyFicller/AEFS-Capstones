"""`web_search` wiring: no network, `_search` is monkeypatched."""

import pytest
from terminal_coding_agent.tools import web_search as web_search_module
from terminal_coding_agent.tools.web_search import WebSearchError, build_web_search


def _fake_search(monkeypatch, *, results=None, error=None, seen=None):
    def fake(query, limit):
        if seen is not None:
            seen.append((query, limit))
        if error is not None:
            raise error
        return results

    monkeypatch.setattr(web_search_module, "_search", fake)


def test_results_are_numbered_with_url_and_snippet(monkeypatch) -> None:
    _fake_search(
        monkeypatch,
        results=[
            {"title": "Ripgrep guide", "href": "https://example.com/rg", "body": "how rg works"},
            {"title": "Ast docs", "href": "https://example.com/ast", "body": "the ast module"},
        ],
    )

    out = build_web_search().invoke({"query": "ripgrep"})

    assert "1. Ripgrep guide" in out
    assert "https://example.com/rg" in out
    assert "how rg works" in out
    assert "2. Ast docs" in out


def test_an_empty_result_list_is_reported_as_no_results(monkeypatch) -> None:
    """Zero hits is an outcome, not a failure: no `Error:` prefix, and the query is kept."""
    _fake_search(monkeypatch, results=[])
    out = build_web_search().invoke({"query": "nothing here"})
    assert out == "No results for: nothing here"


def test_a_missing_ddgs_dependency_is_reported(monkeypatch) -> None:
    _fake_search(monkeypatch, error=ImportError("no ddgs"))
    out = build_web_search().invoke({"query": "x"})
    assert out == "Error: ddgs not installed"


def test_a_search_failure_is_reported_not_raised(monkeypatch) -> None:
    _fake_search(monkeypatch, error=WebSearchError("search timed out after 15s"))
    out = build_web_search().invoke({"query": "x"})
    assert out == "Error: search timed out after 15s"


def test_max_results_is_capped(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []
    _fake_search(monkeypatch, results=[{"title": "t", "href": "u", "body": "b"}], seen=seen)

    build_web_search().invoke({"query": "q", "max_results": 100})

    assert seen == [("q", 10)]


def test_a_zero_max_results_still_asks_for_one(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []
    _fake_search(monkeypatch, results=[{"title": "t", "href": "u", "body": "b"}], seen=seen)

    build_web_search().invoke({"query": "q", "max_results": 0})

    assert seen == [("q", 1)]


def test_a_huge_body_is_truncated(monkeypatch) -> None:
    _fake_search(monkeypatch, results=[{"title": "t", "href": "u", "body": "x" * 100_000}])
    out = build_web_search().invoke({"query": "q"})
    assert "truncated" in out


def test_the_ddgs_no_results_sentinel_is_not_an_error(monkeypatch) -> None:
    """ddgs reports zero hits by raising; it must surface as a plain no-results line."""
    pytest.importorskip("ddgs")

    _install_fake_ddgs(monkeypatch, raised="NoResultsFound")

    out = build_web_search().invoke({"query": "obscure thing"})

    assert out == "No results for: obscure thing"
    assert not out.startswith("Error:")


def _install_fake_ddgs(monkeypatch, *, raised: str) -> None:
    """Install a fake `ddgs` package whose `text()` always raises the named exception."""
    import sys
    import types

    ddgs_module = types.ModuleType("ddgs")
    exceptions_module = types.ModuleType("ddgs.exceptions")

    class DDGSException(Exception):  # noqa: N818 - mirrors ddgs's real class name
        pass

    class TimeoutException(DDGSException):
        pass

    class RatelimitException(DDGSException):
        pass

    exception_class, message = {
        "NoResultsFound": (DDGSException, "No results found."),
        "TimeoutException": (TimeoutException, "boom"),
        "RatelimitException": (RatelimitException, "boom"),
        "DDGSException": (DDGSException, "boom"),
    }[raised]

    class FakeDDGS:
        def __init__(self, timeout=None, **kwargs):
            self._timeout = timeout

        def text(self, query, **kwargs):
            raise exception_class(message)

    ddgs_module.DDGS = FakeDDGS
    exceptions_module.DDGSException = DDGSException
    exceptions_module.TimeoutException = TimeoutException
    exceptions_module.RatelimitException = RatelimitException
    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exceptions_module)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        ("TimeoutException", "Error: search timed out after 15s"),
        ("RatelimitException", "Error: search rate limited: boom"),
        ("DDGSException", "Error: search failed: boom"),
    ],
)
def test_ddgs_exceptions_map_to_their_own_messages(monkeypatch, raised, expected) -> None:
    """Specific subclasses must be caught before the base class."""
    pytest.importorskip("ddgs")

    _install_fake_ddgs(monkeypatch, raised=raised)

    out = build_web_search().invoke({"query": "x"})

    assert out == expected

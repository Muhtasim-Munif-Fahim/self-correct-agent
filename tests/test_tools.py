"""Tests for self_correct.tools module."""

import pytest
from unittest.mock import MagicMock, patch
from self_correct.tools import DuckDuckGoSearchTool, SearchResult, Tool


def test_search_result_dataclass() -> None:
    """SearchResult should store title, snippet, url."""
    r = SearchResult(title="Test", snippet="A test result", url="https://example.com")
    assert r.title == "Test"
    assert r.snippet == "A test result"
    assert r.url == "https://example.com"


def test_tool_is_abstract() -> None:
    """Cannot instantiate Tool directly."""
    with pytest.raises(TypeError):
        Tool()


def test_ddg_search_requires_package() -> None:
    """DuckDuckGoSearchTool.search() raises ImportError if duckduckgo-search missing."""
    tool = DuckDuckGoSearchTool()
    assert tool.name == "DuckDuckGo Web Search"

    with patch.dict("sys.modules", {"duckduckgo_search": None}):
        with pytest.raises(ImportError, match="duckduckgo-search"):
            tool.search("test query")


def test_ddg_search_parses_results() -> None:
    """When duckduckgo-search is available, results should be parsed correctly."""
    tool = DuckDuckGoSearchTool()

    mock_results = [
        {"title": "Result 1", "body": "Snippet 1", "href": "https://a.com"},
        {"title": "Result 2", "body": "Snippet 2", "href": "https://b.com"},
    ]

    # Mock the DDGS context manager and its .text() method
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text.return_value = mock_results

    mock_ddgs_class = MagicMock(return_value=mock_ddgs_instance)

    mock_module = MagicMock()
    mock_module.DDGS = mock_ddgs_class

    with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
        # Need to reimport to pick up the mock
        import importlib
        import self_correct.tools
        importlib.reload(self_correct.tools)
        from self_correct.tools import DuckDuckGoSearchTool as ReloadedTool

        tool = ReloadedTool()
        results = tool.search("test query", max_results=2)

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[1].url == "https://b.com"


def test_ddg_search_handles_errors_gracefully() -> None:
    """If the DDG API throws, search() should return empty list, not crash."""
    tool = DuckDuckGoSearchTool()

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text.side_effect = RuntimeError("API down")

    mock_ddgs_class = MagicMock(return_value=mock_ddgs_instance)

    mock_module = MagicMock()
    mock_module.DDGS = mock_ddgs_class

    with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
        import importlib
        import self_correct.tools
        importlib.reload(self_correct.tools)
        from self_correct.tools import DuckDuckGoSearchTool as ReloadedTool

        tool = ReloadedTool()
        results = tool.search("test query")
        assert results == []

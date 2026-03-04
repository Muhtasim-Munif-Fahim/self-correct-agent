"""Pluggable verification tools for self-correct-agent.

Users can implement the ``Tool`` interface to add custom
verification backends (e.g., Google Scholar, Wolfram Alpha).
A default ``DuckDuckGoSearchTool`` ships out of the box.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result returned by a verification tool."""

    title: str
    snippet: str
    url: str


class Tool(ABC):
    """Base interface for verification tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the tool."""

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """
        Execute a search query and return results.

        Parameters
        ----------
        query : str
            The search query string.
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        List[SearchResult]
            A list of search results. May be empty if no results found.
        """


class DuckDuckGoSearchTool(Tool):
    """
    Web search verification tool using DuckDuckGo.

    Requires the ``duckduckgo-search`` package::

        pip install duckduckgo-search
    """

    @property
    def name(self) -> str:
        return "DuckDuckGo Web Search"

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """
        Search DuckDuckGo for a query.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        List[SearchResult]
            Parsed search results.

        Raises
        ------
        ImportError
            If ``duckduckgo-search`` is not installed.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "DuckDuckGoSearchTool requires the 'duckduckgo-search' package. "
                "Install it with: pip install duckduckgo-search"
            )

        results: List[SearchResult] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        url=r.get("href", ""),
                    ))
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s: %s", type(exc).__name__, exc)

        return results

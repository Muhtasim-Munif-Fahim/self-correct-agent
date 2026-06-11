"""Pluggable verification tools for self-correct-agent.

Users can implement the ``Tool`` interface to add custom
verification backends (e.g., Google Scholar, Wolfram Alpha).
A default ``DuckDuckGoSearchTool`` ships out of the box.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

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
        if not query or not query.strip():
            logger.warning("Empty search query provided. Returning no results.")
            return results
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


class WikipediaSearchTool(Tool):
    """
    Fact-check claims using Wikipedia article summaries.
    
    Requires the ``wikipedia`` package::
    
        pip install wikipedia
    """
    
    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    @property
    def name(self) -> str:
        return "Wikipedia"

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """
        Search Wikipedia for a query and return article summaries.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int
            Maximum number of article results to return.

        Returns
        -------
        List[SearchResult]
            Parsed Wikipedia article summaries.

        Raises
        ------
        ImportError
            If ``wikipedia`` is not installed.
        """
        try:
            import wikipedia as wp
        except ImportError:
            raise ImportError(
                "WikipediaSearchTool requires the 'wikipedia' package. "
                "Install it with: pip install wikipedia"
            )

        wp.set_lang(self.lang)
        results: List[SearchResult] = []
        if not query or not query.strip():
            logger.warning("Empty search query provided. Returning no results.")
            return results

        try:
            search_titles = wp.search(query, results=max_results)
            for title in search_titles[:max_results]:
                try:
                    page = wp.page(title, auto_suggest=False)
                    results.append(SearchResult(
                        title=page.title,
                        snippet=page.summary[:600],
                        url=page.url,
                    ))
                except (wp.exceptions.DisambiguationError, wp.exceptions.PageError) as exc:
                    logger.debug("Wikipedia page lookup skipped: %s", exc)
                    continue
        except Exception as exc:
            logger.warning("Wikipedia search failed: %s: %s", type(exc).__name__, exc)

        return results


class StaticKnowledgeTool(Tool):
    """
    Verify claims against a user-provided static knowledge base.

    The knowledge base is a dictionary mapping topics or entities
    (lowercased for matching) to their verified descriptions.

    Example::

        knowledge = {
            "population of tokyo": "Tokyo has a population of ~14 million "
                                    "(ward area) or ~37 million (metro area).",
            "einstein": "Albert Einstein developed the theory of relativity.",
        }
        tool = StaticKnowledgeTool(knowledge, name="Custom KB")
    """

    def __init__(
        self,
        knowledge: Dict[str, str],
        name: str = "Knowledge Base",
    ) -> None:
        """
        Initialize with a knowledge dictionary.

        Parameters
        ----------
        knowledge : Dict[str, str]
            Mapping of topic/entity to verified description.
        name : str
            Human-readable name for the tool.
        """
        self._knowledge = {k.lower().strip(): v for k, v in knowledge.items()}
        self._name = name

    @classmethod
    def from_json(cls, path: str, name: str = "Knowledge Base") -> "StaticKnowledgeTool":
        """
        Load a knowledge base from a JSON file.

        The JSON should be an object of ``{ "topic": "description", ... }``.

        Parameters
        ----------
        path : str
            Path to the JSON file.
        name : str
            Human-readable name for the tool.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(knowledge=data, name=name)

    @classmethod
    def from_json_url(cls, url: str, name: str = "Knowledge Base") -> "StaticKnowledgeTool":
        """
        Load a knowledge base from a JSON URL.

        Parameters
        ----------
        url : str
            URL to the JSON file.
        name : str
            Human-readable name for the tool.
        """
        import json, urllib.request
        with urllib.request.urlopen(url) as f:
            data = json.loads(f.read().decode("utf-8"))
        return cls(knowledge=data, name=name)

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """
        Match query against the knowledge base.

        Looks for entries whose keys are substrings of the query,
        or vice versa.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        List[SearchResult]
            Matching knowledge entries.
        """
        results: List[SearchResult] = []
        query_lower = query.lower().strip()
        if not query_lower:
            return results

        matches: List[tuple[str, str, float]] = []
        for topic, desc in self._knowledge.items():
            # Score: overlap of words between query and topic
            query_words = set(query_lower.split())
            topic_words = set(topic.split())
            common = query_words & topic_words
            if common:
                score = len(common) / max(len(query_words), len(topic_words))
                matches.append((topic, desc, score))

        # Sort by relevance score descending
        matches.sort(key=lambda x: x[2], reverse=True)

        for topic, desc, score in matches[:max_results]:
            results.append(SearchResult(
                title=topic,
                snippet=desc[:600],
                url="",
            ))

        return results
"""Public package exports for self-correct-agent."""

from .core import AntiHallucinator, AntiHallucinationResponse, TokenUsage
from .tools import DuckDuckGoSearchTool, SearchResult, StaticKnowledgeTool, Tool, WikipediaSearchTool

__version__ = "0.2.1"

__all__ = [
    "AntiHallucinator",
    "AntiHallucinationResponse",
    "DuckDuckGoSearchTool",
    "SearchResult",
    "StaticKnowledgeTool",
    "__version__",
    "TokenUsage",
    "Tool",
    "WikipediaSearchTool",
]

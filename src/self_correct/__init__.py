"""Public package exports for self-correct-agent."""

from .core import AntiHallucinator, AntiHallucinationResponse, TokenUsage
from .tools import DuckDuckGoSearchTool, SearchResult, Tool

__version__ = "0.1.0"

__all__ = [
    "AntiHallucinator",
    "AntiHallucinationResponse",
    "DuckDuckGoSearchTool",
    "SearchResult",
    "__version__",
    "TokenUsage",
    "Tool",
]

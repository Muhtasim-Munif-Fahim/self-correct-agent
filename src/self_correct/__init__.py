"""Public package exports for self-correct-agent."""

from .core import (
    AntiHallucinator,
    AntiHallucinationResponse,
    TokenUsage,
    VerificationDecision,
    VerificationPolicy,
)
from .tools import DuckDuckGoSearchTool, SearchResult, StaticKnowledgeTool, Tool, WikipediaSearchTool

__version__ = "0.2.4"

__all__ = [
    "AntiHallucinator",
    "AntiHallucinationResponse",
    "DuckDuckGoSearchTool",
    "SearchResult",
    "StaticKnowledgeTool",
    "__version__",
    "TokenUsage",
    "VerificationDecision",
    "VerificationPolicy",
    "Tool",
    "WikipediaSearchTool",
]

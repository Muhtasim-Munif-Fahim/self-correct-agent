"""Public package exports for self-correct-agent."""

from .core import (
    AntiHallucinator,
    AntiHallucinationResponse,
    ContentCheck,
    RegexContentCheck,
    TokenUsage,
    VerificationDecision,
    VerificationPolicy,
    load_content_checks,
)
from .tools import DuckDuckGoSearchTool, SearchResult, StaticKnowledgeTool, Tool, WikipediaSearchTool

__version__ = "0.2.4"

__all__ = [
    "AntiHallucinator",
    "AntiHallucinationResponse",
    "ContentCheck",
    "RegexContentCheck",
    "DuckDuckGoSearchTool",
    "SearchResult",
    "StaticKnowledgeTool",
    "__version__",
    "TokenUsage",
    "VerificationDecision",
    "VerificationPolicy",
    "load_content_checks",
    "Tool",
    "WikipediaSearchTool",
]

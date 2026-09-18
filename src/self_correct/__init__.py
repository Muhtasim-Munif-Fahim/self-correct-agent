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
from .providers import AnthropicMessagesClient, build_client
from .structured import StructuredClaim, StructuredVerificationResult
from .tools import DuckDuckGoSearchTool, SearchResult, StaticKnowledgeTool, Tool, WikipediaSearchTool

__version__ = "0.2.4"

__all__ = [
    "AnthropicMessagesClient",
    "AntiHallucinator",
    "AntiHallucinationResponse",
    "ContentCheck",
    "RegexContentCheck",
    "DuckDuckGoSearchTool",
    "SearchResult",
    "StaticKnowledgeTool",
    "StructuredClaim",
    "StructuredVerificationResult",
    "__version__",
    "TokenUsage",
    "VerificationDecision",
    "VerificationPolicy",
    "build_client",
    "load_content_checks",
    "Tool",
    "WikipediaSearchTool",
]

"""LLM provider clients used by the CLI and library.

The verification pipeline only needs ``client.chat.completions.create()``.
OpenAI, Ollama, and other OpenAI-compatible servers already expose that.
Anthropic's Messages API does not, so this module adapts it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MAX_TOKENS = 4096

#: Default endpoint for each named provider. ``custom`` has none by design.
PROVIDER_BASE_URLS = {
    "openai": None,
    "ollama": OLLAMA_DEFAULT_BASE_URL,
    "anthropic": ANTHROPIC_DEFAULT_BASE_URL,
    "custom": None,
}

_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "ollama": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "custom": "OPENAI_API_KEY",
}

Transport = Callable[
    [str, Dict[str, Any], Mapping[str, str], Optional[float]],
    Dict[str, Any],
]


def content_to_text(content: Any) -> str:
    """Flatten OpenAI or Anthropic message content into a single string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "content" in block:
                    parts.append(content_to_text(block.get("content")))
        return "".join(parts)
    return str(content)


def messages_endpoint(base_url: str) -> str:
    """Return the Anthropic Messages URL for ``base_url``.

    Accepts either ``https://api.anthropic.com`` or ``.../v1``, including
    local proxies that already include the version prefix.
    """

    root = (base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
    if root.endswith("/v1"):
        return root + "/messages"
    return root + "/v1/messages"


def openai_messages_to_anthropic(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Split OpenAI chat messages into Anthropic ``system`` + ``messages``."""

    system_parts: List[str] = []
    converted: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        text = content_to_text(message.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        converted.append({"role": role, "content": text})
    system = "\n\n".join(system_parts) or None
    return system, converted


def build_messages_payload(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    """Build a POST /v1/messages JSON body from OpenAI-style arguments."""

    system, converted = openai_messages_to_anthropic(messages)
    if not converted:
        raise ValueError("Anthropic requests need at least one non-system message")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": converted,
        "max_tokens": int(max_tokens) if max_tokens else DEFAULT_MAX_TOKENS,
    }
    if system is not None:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def completion_from_anthropic(payload: Mapping[str, Any]) -> SimpleNamespace:
    """Adapt an Anthropic Messages response to OpenAI chat-completion shape."""

    text = content_to_text(payload.get("content"))
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return SimpleNamespace(
        id=payload.get("id") or "",
        model=payload.get("model") or "",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=payload.get("stop_reason") or "stop",
                message=SimpleNamespace(role="assistant", content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    timeout: Optional[float],
) -> Dict[str, Any]:
    """POST JSON and return the decoded object, or raise ``RuntimeError``."""

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic API request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Anthropic API returned non-JSON: {body[:200]}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Anthropic API returned a non-object JSON payload")
    return decoded


class _Completions:
    """``client.chat.completions`` namespace for :class:`AnthropicMessagesClient`."""

    def __init__(self, client: "AnthropicMessagesClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> SimpleNamespace:
        if extra:
            raise TypeError("unknown parameter: " + ", ".join(sorted(extra)))
        payload = build_messages_payload(model, messages, temperature, max_tokens)
        body = self._client.transport(
            messages_endpoint(self._client.base_url),
            payload,
            self._client.headers(),
            self._client.timeout,
        )
        return completion_from_anthropic(body)


class _Chat:
    """``client.chat`` namespace for :class:`AnthropicMessagesClient`."""

    def __init__(self, client: "AnthropicMessagesClient") -> None:
        self.completions = _Completions(client)


class AnthropicMessagesClient:
    """OpenAI-shaped wrapper around Anthropic's native Messages API.

    ``AntiHallucinator`` calls ``client.chat.completions.create()``. This
    client accepts those arguments, POSTs ``/v1/messages``, and returns an
    object with ``choices[0].message.content`` and OpenAI-style ``usage``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "AnthropicMessagesClient requires an API key; pass api_key "
                "or set $ANTHROPIC_API_KEY"
            )
        self.api_key = key
        self.base_url = (base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.transport = transport or post_json
        self.chat = _Chat(self)

    def headers(self) -> Dict[str, str]:
        """Headers required by the Anthropic Messages API."""

        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }


def resolve_api_key(provider: str, api_key_env: Optional[str] = None) -> str:
    """Return the API key for ``provider``, or raise ``ValueError``."""

    if provider == "ollama":
        if api_key_env:
            return os.environ.get(api_key_env) or "ollama"
        return os.environ.get("OPENAI_API_KEY") or "ollama"

    key_env = api_key_env or _DEFAULT_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.environ.get(key_env)
    if api_key:
        return api_key
    raise ValueError(
        f"No API key found in ${key_env}. Set it, or pass --api-key-env "
        "to name a different variable."
    )


def build_client(
    provider: str = "openai",
    *,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Construct a chat-completions client for a named provider.

    ``openai``, ``ollama``, and ``custom`` use the OpenAI SDK. ``anthropic``
    uses :class:`AnthropicMessagesClient` against the native Messages API.
    """

    provider = provider or "openai"
    resolved_base = base_url or PROVIDER_BASE_URLS.get(provider)
    if provider == "custom" and not resolved_base:
        raise ValueError("--provider custom requires --base-url")

    key = api_key if api_key is not None else resolve_api_key(provider, api_key_env)

    if provider == "anthropic":
        return AnthropicMessagesClient(
            api_key=key,
            base_url=resolved_base,
            timeout=timeout,
        )

    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": key}
    if resolved_base:
        kwargs["base_url"] = resolved_base
    if timeout:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)

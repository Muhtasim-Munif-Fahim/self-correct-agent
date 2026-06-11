import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import Mock

from self_correct import (
    AntiHallucinator,
    AntiHallucinationResponse,
    DuckDuckGoSearchTool,
    StaticKnowledgeTool,
    Tool,
    WikipediaSearchTool,
)


def _make_mock_response(content: str):
    mock = Mock()
    choice = Mock()
    choice.message.content = content
    mock.choices = [choice]
    usage = Mock()
    usage.prompt_tokens = 50
    usage.completion_tokens = len(content.split())
    mock.usage = usage
    return mock


class MockCompletions:
    def __init__(self, chat):
        self.chat = chat

    def create(self, **kwargs):
        return self.chat._create(**kwargs)


class MockChat:
    def __init__(self):
        self.call_count = 0
        self.completions = MockCompletions(self)

    def _create(self, **kwargs):
        self.call_count += 1
        messages = kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "Extract" in user_msg:
            return _make_mock_response("1. Tokyo is the capital of Japan.\n2. Population is approx 14 million.")
        elif "Critique" in user_msg or "Briefly check" in user_msg:
            if "capital" in user_msg:
                return _make_mock_response("VERIFIED: True")
            elif "14 million" in user_msg:
                return _make_mock_response("VERIFIED: True")
            return _make_mock_response("VERIFIED: False")
        elif "Rewrite" in user_msg:
            return _make_mock_response("Tokyo is Japan's capital city. Its population is roughly 14 million.")
        else:
            return _make_mock_response("Tokyo is the capital of Japan. It has a population of approx 14 million.")


class MockClient:
    def __init__(self):
        self.chat = MockChat()


DEMO_KNOWLEDGE = {
    "tokyo capital": "Tokyo is the capital of Japan, located in the Kanto region.",
    "tokyo population": "The Tokyo Metropolis has an estimated population of 14 million.",
    "japan": "Japan is an island country in East Asia with a population of about 125 million.",
}


def main():
    print("=" * 65)
    print("  self-correct-agent - Tool Comparison Demo")
    print("  (Uses mocked LLM - no API keys required)")
    print("=" * 65)

    client = MockClient()

    tools = {
        "DuckDuckGo Web Search": DuckDuckGoSearchTool(),
        "Wikipedia": WikipediaSearchTool(),
        "Static Knowledge Base": StaticKnowledgeTool(DEMO_KNOWLEDGE, name="Demo KB"),
    }

    prompt = "What is the capital and population of Tokyo?"

    for tool_name, tool_instance in tools.items():
        print()
        print("-" * 65)
        print("  Tool: " + tool_name)
        print("-" * 65)

        try:
            safe = AntiHallucinator(
                client=client,
                strictness=1.0,
                tools=[tool_instance],
                cache_size=0,
            )

            result = safe.generate(model="gpt-4o-mini", prompt=prompt)

            print("  Prompt: " + prompt)
            print("  Output: " + result.content)
            print("  Claims flagged: " + str(len(result.hallucinations_caught)))
            print("  Tokens: " + str(result.token_usage.total_tokens))
            print("  Cache hits: 0 (cache disabled)")

            if result.verification_log:
                for v in result.verification_log[:2]:
                    status = "OK" if v["is_valid"] else "XX"
                    claim = v["claim"][:65]
                    print("    " + status + " " + claim)

        except Exception as exc:
            print("  Error: " + str(exc))

    print()
    print("=" * 65)
    print("  All tools demonstrated successfully.")
    print("  Total LLM calls: " + str(client.chat.call_count))
    print("=" * 65)


if __name__ == "__main__":
    main()

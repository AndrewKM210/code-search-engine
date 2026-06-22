from unittest.mock import MagicMock

from cse.agent.llm import (
    LLMClient,
    _extract_tool_call,
    _format_tool_instructions,
)
from cse.agent.schema import ToolCall

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Searches the indexed codebase.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _client_with_mock_llm(*responses):
    """Builds an LLMClient whose self.llm.invoke yields the given contents in order."""
    client = object.__new__(LLMClient)
    client.llm = MagicMock()
    client.llm.invoke.side_effect = [MagicMock(content=r) for r in responses]
    return client


def test_parse_tool_calls_single():
    raw = [{"name": "search_code", "args": {"query": "device"}, "id": "1"}]

    calls = LLMClient._parse_tool_calls(raw)

    assert calls == [
        ToolCall(name="search_code", arguments={"query": "device"})
    ]


def test_parse_tool_calls_multiple():
    raw = [
        {"name": "read_file", "args": {"path": "a.py"}},
        {"name": "grep", "args": {"pattern": "def foo"}},
    ]

    calls = LLMClient._parse_tool_calls(raw)

    assert [c.name for c in calls] == ["read_file", "grep"]
    assert calls[1].arguments == {"pattern": "def foo"}


def test_parse_tool_calls_empty():
    assert LLMClient._parse_tool_calls([]) == []


def test_parse_tool_calls_defaults_missing_args_to_empty_dict():
    raw = [{"name": "list_directory"}]

    calls = LLMClient._parse_tool_calls(raw)

    assert calls[0].arguments == {}


def test_format_tool_instructions_lists_each_tool():
    text = _format_tool_instructions(TOOLS)

    assert "search_code(query)" in text
    assert "Searches the indexed codebase." in text


def test_extract_tool_call_parses_plain_json():
    call = _extract_tool_call(
        '{"name": "search_code", "arguments": {"query": "x"}}'
    )

    assert call == ToolCall(name="search_code", arguments={"query": "x"})


def test_extract_tool_call_parses_fenced_json():
    text = '```json\n{"name": "grep", "arguments": {"pattern": "foo"}}\n```'

    call = _extract_tool_call(text)

    assert call == ToolCall(name="grep", arguments={"pattern": "foo"})


def test_extract_tool_call_returns_none_for_plain_text():
    assert _extract_tool_call("Hello, how can I help?") is None


def test_call_with_tools_fallback_parses_valid_json_first_try():
    client = _client_with_mock_llm(
        '{"name": "search_code", "arguments": {"query": "device"}}'
    )

    content, calls = client.call_with_tools_fallback(
        [("user", "find the device resolver")], TOOLS
    )

    assert calls == [
        ToolCall(name="search_code", arguments={"query": "device"})
    ]
    assert client.llm.invoke.call_count == 1


def test_call_with_tools_fallback_returns_plain_text_without_retry():
    client = _client_with_mock_llm("Hi there, no tool needed.")

    content, calls = client.call_with_tools_fallback([("user", "hello")], TOOLS)

    assert calls == []
    assert content == "Hi there, no tool needed."
    assert client.llm.invoke.call_count == 1


def test_call_with_tools_fallback_retries_malformed_json():
    client = _client_with_mock_llm(
        '{"name": "search_code"',  # malformed on first attempt
        '{"name": "search_code", "arguments": {"query": "device"}}',
    )

    content, calls = client.call_with_tools_fallback(
        [("user", "find the device resolver")], TOOLS
    )

    assert calls == [
        ToolCall(name="search_code", arguments={"query": "device"})
    ]
    assert client.llm.invoke.call_count == 2


def test_call_with_tools_fallback_gives_up_after_max_retries():
    client = _client_with_mock_llm('{"broken', '{"broken', '{"broken')

    content, calls = client.call_with_tools_fallback(
        [("user", "find the device resolver")], TOOLS, max_retries=2
    )

    assert calls == []
    assert content == '{"broken'
    assert client.llm.invoke.call_count == 3

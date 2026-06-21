from cse.agent.llm import LLMClient
from cse.agent.schema import ToolCall


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

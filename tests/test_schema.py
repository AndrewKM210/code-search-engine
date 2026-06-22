import pytest
from pydantic import ValidationError

from cse.agent.schema import ToolCall


def test_tool_call_validates_dict():
    call = ToolCall.model_validate(
        {"name": "search_code", "arguments": {"query": "foo"}}
    )

    assert call.name == "search_code"
    assert call.arguments == {"query": "foo"}


def test_tool_call_validates_json_string():
    call = ToolCall.model_validate_json(
        '{"name": "read_file", "arguments": {"path": "a.py"}}'
    )

    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}


def test_tool_call_defaults_arguments_to_empty_dict():
    call = ToolCall.model_validate({"name": "list_directory"})

    assert call.arguments == {}


def test_tool_call_rejects_missing_name():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"arguments": {"path": "a.py"}})


def test_tool_call_rejects_malformed_json():
    with pytest.raises(ValidationError):
        ToolCall.model_validate_json("not valid json")


def test_tool_call_rejects_wrong_arguments_type():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"name": "grep", "arguments": "pattern"})

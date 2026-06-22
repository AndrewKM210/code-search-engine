from unittest.mock import MagicMock

from cse.agent.core import ToolCallingAgent
from cse.agent.schema import ToolCall


def test_solve_answers_directly_when_no_tool_call():
    llm = MagicMock()
    llm.call_with_tools_auto.return_value = ("the answer", [])
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={})

    steps = list(agent.solve("what does this repo do?"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "the answer"
    assert llm.call_with_tools_auto.call_count == 1


def test_solve_runs_tool_then_answers():
    engine = MagicMock()
    engine.search.return_value = [
        {"code_id": 1, "score": 0.9, "payload": {"content": "def foo(): pass"}}
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("Found it: def foo(): pass", []),
    ]
    agent = ToolCallingAgent(engine, llm, llm_config={})

    steps = list(agent.solve("what does foo do?"))

    assert [s.step_type for s in steps] == [
        "start",
        "plan",
        "tool_call",
        "tool_result",
        "plan",
        "answer",
    ]
    assert "def foo(): pass" in steps[3].content
    assert steps[-1].content == "Found it: def foo(): pass"


def test_solve_passes_tool_result_back_in_conversation():
    engine = MagicMock()
    engine.search.return_value = [
        {"code_id": 1, "score": 0.9, "payload": {"content": "def foo(): pass"}}
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("answer", []),
    ]
    agent = ToolCallingAgent(engine, llm, llm_config={})

    list(agent.solve("what does foo do?"))

    second_call_messages = llm.call_with_tools_auto.call_args_list[1].args[0]
    assert any(
        "def foo(): pass" in content for _, content in second_call_messages
    )


def test_solve_errors_after_max_steps_without_answer():
    llm = MagicMock()
    llm.call_with_tools_auto.return_value = (
        "",
        [ToolCall(name="list_directory", arguments={})],
    )
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={}, max_steps=2)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "error"
    assert llm.call_with_tools_auto.call_count == 2


def test_run_tool_reports_unknown_tool_name():
    agent = ToolCallingAgent(MagicMock(), MagicMock(), llm_config={})

    result = agent._run_tool(ToolCall(name="delete_everything", arguments={}))

    assert result.startswith("Error: unknown tool")


def test_run_tool_reports_invalid_arguments():
    agent = ToolCallingAgent(MagicMock(), MagicMock(), llm_config={})

    result = agent._run_tool(
        ToolCall(name="grep", arguments={"not_a_real_arg": "x"})
    )

    assert result.startswith("Error: invalid arguments for 'grep'")

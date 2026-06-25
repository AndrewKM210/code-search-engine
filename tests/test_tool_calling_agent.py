from unittest.mock import MagicMock

from cse.agent.core import ToolCallingAgent
from cse.agent.schema import ToolCall


def test_solve_nudges_model_to_use_a_tool_before_accepting_an_answer():
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("the answer", []),  # tries to answer without using a tool first
        ("", [ToolCall(name="list_directory", arguments={})]),
        ("final answer", []),  # accepted now that a tool has been used
    ]
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={}, max_steps=5)

    steps = list(agent.solve("what does this repo do?"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "final answer"
    assert llm.call_with_tools_auto.call_count == 3


def test_solve_errors_if_model_never_calls_a_tool():
    llm = MagicMock()
    llm.call_with_tools_auto.return_value = ("the answer", [])
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={}, max_steps=3)

    steps = list(agent.solve("what does this repo do?"))

    assert steps[-1].step_type == "error"
    assert llm.call_with_tools_auto.call_count == 3


def test_solve_runs_tool_then_answers():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {"content": "def foo(): pass", "source": "foo.py"},
        }
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("Found it in foo.py: def foo(): pass", []),
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
    assert steps[-1].content == "Found it in foo.py: def foo(): pass"


def test_solve_passes_tool_result_back_in_conversation():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {"content": "def foo(): pass", "source": "foo.py"},
        }
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("answer, see foo.py", []),
    ]
    agent = ToolCallingAgent(engine, llm, llm_config={})

    list(agent.solve("what does foo do?"))

    second_call_messages = llm.call_with_tools_auto.call_args_list[1].args[0]
    assert any(
        "def foo(): pass" in content for _, content in second_call_messages
    )


def test_solve_rejects_raw_json_answer_and_retries():
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="list_directory", arguments={})]),
        ('{"name": "list_directory", "parameters": {}}', []),
        ("Here's the answer in plain English, see foo.py", []),
    ]
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={}, max_steps=5)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "Here's the answer in plain English, see foo.py"
    assert llm.call_with_tools_auto.call_count == 3


def test_solve_rejects_answer_missing_citation_and_retries():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {"content": "def foo(): pass", "source": "foo.py"},
        }
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("foo does nothing.", []),  # doesn't cite foo.py
        ("foo does nothing, per foo.py.", []),
    ]
    agent = ToolCallingAgent(engine, llm, llm_config={}, max_steps=5)

    steps = list(agent.solve("what does foo do?"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "foo does nothing, per foo.py."
    assert llm.call_with_tools_auto.call_count == 3


def test_solve_rejects_narrated_tool_call_and_retries():
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="list_directory", arguments={})]),
        (
            "I will call another tool to find this.\n\n"
            "Calling tool search_code({'query': 'chunk_size'})",
            [],
        ),
        ("The answer is in config/main_config.yaml.", []),
    ]
    agent = ToolCallingAgent(MagicMock(), llm, llm_config={}, max_steps=5)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "The answer is in config/main_config.yaml."
    assert llm.call_with_tools_auto.call_count == 3


def test_solve_does_not_treat_an_errored_path_as_a_citable_source():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {"content": "def foo(): pass", "source": "foo.py"},
        }
    ]
    llm = MagicMock()
    llm.call_with_tools_auto.side_effect = [
        ("", [ToolCall(name="search_code", arguments={"query": "foo"})]),
        ("", [ToolCall(name="read_file", arguments={"path": "missing.py"})]),
        (
            "found it in missing.py",
            [],
        ),  # cites the failed path, not the real one
        ("found it in foo.py", []),
    ]
    agent = ToolCallingAgent(engine, llm, llm_config={}, max_steps=5)
    agent.tools["read_file"] = lambda path: (
        "Error: 'missing.py' does not exist."
    )

    steps = list(agent.solve("what does foo do?"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "found it in foo.py"
    assert llm.call_with_tools_auto.call_count == 4


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

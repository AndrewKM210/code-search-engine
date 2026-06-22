from cse.agent.core import AgentStep
from cse.agent.presenter import describe_step


def test_describe_step_known_type():
    label, text = describe_step(AgentStep("plan", "Generated query"))

    assert label == "PLAN"
    assert text == "Generated query"


def test_describe_step_tool_call_and_result_labels():
    assert describe_step(AgentStep("tool_call", "x"))[0] == "TOOL CALL"
    assert describe_step(AgentStep("tool_result", "y"))[0] == "TOOL RESULT"


def test_describe_step_unknown_type_falls_back_to_uppercased_type():
    label, text = describe_step(AgentStep("mystery", "huh"))

    assert label == "MYSTERY"
    assert text == "huh"

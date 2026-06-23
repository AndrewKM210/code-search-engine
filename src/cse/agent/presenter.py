from cse.agent.core import AgentStep

# Short display label for each AgentStep.step_type, shared by the CLI and Streamlit app
_STEP_LABELS = {
    "start": "START",
    "plan": "PLAN",
    "search": "SEARCH",
    "critique": "CRITIQUE",
    "tool_call": "TOOL CALL",
    "tool_result": "TOOL RESULT",
    "answer": "ANSWER",
    "error": "ERROR",
}


def describe_step(step: AgentStep) -> tuple[str, str]:
    """
    Maps an AgentStep to a (label, text) pair for display.

    Args:
        step (AgentStep): A step yielded by an agent's solve() loop.

    Returns:
        tuple[str, str]: (label, text), where label is a short uppercase tag
            (e.g. "PLAN", "TOOL CALL") and text is the step's content. Each
            caller decides how to render the pair (print, st.markdown, etc.).
    """
    label = _STEP_LABELS.get(step.step_type, step.step_type.upper())
    return label, step.content

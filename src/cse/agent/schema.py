from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    A structured request from the LLM to invoke one tool.

    Used as the common shape both native tool-calling (parsed from the
    model's own tool_calls response) and the prompt+JSON fallback
    (parsed from raw text) are validated into, so the agent loop's
    dispatch logic doesn't need to know which path produced it.

    Args:
        name (str): The tool's name, e.g. "search_code".
        arguments (dict): Keyword arguments to call the tool with.
    """

    name: str
    arguments: dict[str, Any] = {}

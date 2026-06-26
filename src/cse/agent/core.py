import json
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cse.agent.llm import LLMClient
from cse.agent.schema import ToolCall
from cse.agent.tools import (
    TOOL_SPECS,
    grep,
    list_directory,
    read_file,
    search_code,
)
from cse.search_engine.engine import CodeSearchEngine

# Matches the "(source: <path>, score: ...)" tag search_code puts on each result
_SOURCE_TAG_RE = re.compile(r"\(source:\s*([^,]+),\s*score:")

TOOL_LOOP_SYSTEM_MSG = (
    "You are a coding assistant exploring a repository. Use the available "
    "tools (search_code, read_file, list_directory, grep) to find the "
    "information needed to answer the user's question. Once you have "
    "enough information, answer in plain English (never JSON), citing the "
    "specific file path the information came from."
)

# Nudge used when the model tries to answer before ever calling a tool,
# since small local models often skip retrieval and guess from prior knowledge
NO_TOOL_USED_YET_MSG = (
    "Answer the question using one of the available tools (search_code, "
    "read_file, list_directory, grep) first, instead of answering directly."
)

# Nudge used when the model's "answer" is a bare JSON object instead of prose
RAW_JSON_ANSWER_MSG = (
    "Don't respond with a JSON object. Answer in one or two plain English "
    "sentences, citing the file path the information came from."
)

# Nudge used when the model's answer never names any file it actually looked at
MISSING_CITATION_MSG = (
    "Rewrite your answer, explicitly naming the file path you found this "
    "information in."
)

# Nudge used when the model narrates "calling a tool" in prose instead of
# actually issuing one (it tends to echo this loop's own "Calling tool
# name(...)" conversation history back as if it were a real action)
FAKE_TOOL_NARRATION_MSG = (
    "Don't describe calling a tool in your response. Either actually use a "
    "tool, or answer the question directly in plain English with a citation."
)

# Re-stated after every tool result, since small local models tend to lose
# track of the system-level "stop and answer" instruction after a few turns
ANSWER_IF_SUFFICIENT_MSG = (
    "If this answers the question, respond now in plain English citing the "
    "file it came from. Otherwise, call another tool."
)


def _looks_like_raw_tool_call(content: str) -> bool:
    """
    Detects a JSON tool-call object masquerading as the final answer.

    Args:
        content (str): The model's would-be final answer.

    Returns:
        bool: True if content is just a JSON object shaped like a tool
            call, e.g. {"name": ..., "parameters": {...}}.
    """
    text = content.strip()
    if not text.startswith("{"):
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and "name" in parsed


def _extract_sources(tool_result: str) -> set[str]:
    """
    Pulls the file paths search_code tagged its results with out of its output.

    Args:
        tool_result (str): Output of the search_code tool.

    Returns:
        set[str]: Source file paths tagged in the result, if any.
    """
    return set(_SOURCE_TAG_RE.findall(tool_result))


def _narrates_a_tool_call(content: str) -> bool:
    """
    Detects the model describing a tool call in prose instead of issuing one.

    Args:
        content (str): The model's would-be final answer.

    Returns:
        bool: True if content echoes the "Calling tool name(...)" phrasing
            this loop's own conversation history uses, instead of actually
            using the tool-call mechanism.
    """
    return "calling tool" in content.lower()


def _mentions_any_source(content: str, seen_sources: set[str]) -> bool:
    """
    Checks whether an answer cites at least one file it was grounded in.

    Args:
        content (str): The model's would-be final answer.
        seen_sources (set[str]): File paths the agent actually looked at
            this run (from search_code results, or read_file/grep/
            list_directory paths).

    Returns:
        bool: True if content mentions a full path or basename from
            seen_sources, or if seen_sources is empty (nothing to cite).
    """
    if not seen_sources:
        return True
    return any(
        source in content or Path(source).name in content
        for source in seen_sources
    )


def _reject_answer(
    content: str, has_called_tool: bool, seen_sources: set[str]
) -> tuple[str, str] | None:
    """
    Decides whether a would-be final answer should be rejected and retried.

    Args:
        content (str): The model's would-be final answer.
        has_called_tool (bool): Whether any tool has been called this run.
        seen_sources (set[str]): File paths looked at so far this run.

    Returns:
        tuple[str, str] | None: (status message, corrective nudge) if the
            answer should be rejected, or None if it's acceptable as-is.
    """
    if not has_called_tool:
        return (
            "Model answered without using a tool, asking it to search first...",
            NO_TOOL_USED_YET_MSG,
        )
    if _looks_like_raw_tool_call(content):
        return (
            "Model answered with a raw JSON object, asking it to use plain English...",
            RAW_JSON_ANSWER_MSG,
        )
    if _narrates_a_tool_call(content):
        return (
            "Model narrated a tool call instead of using one, asking it to "
            "actually call it or answer directly...",
            FAKE_TOOL_NARRATION_MSG,
        )
    if not _mentions_any_source(content, seen_sources):
        return (
            "Model's answer didn't cite a source file, asking it to add one...",
            MISSING_CITATION_MSG,
        )
    return None


@dataclass
class AgentStep:
    """Data structure to pass state updates to the UI/Console."""

    step_type: str  # 'plan', 'search', 'critique', 'answer', 'error'
    content: str  # Text to display
    data: Any = None  # Optional raw data (e.g., the list of code results)


class CodingAgent:
    def __init__(self, engine: CodeSearchEngine, llm: LLMClient):
        self.engine = engine
        self.llm = llm
        self.max_retries = 2

    def solve(self, user_query: str) -> Generator[AgentStep, None, None]:
        """
        The main Agentic Loop. Yields steps for real-time rendering.

        Args:
            user_query (str): Input query.

        Returns:
            Generator[AgentStep]: Contains all the steps the agent has done.
        """
        attempt = 0
        previous_search_query = None

        # Yield initial state
        yield AgentStep("start", f"Processing request: {user_query}")

        while attempt <= self.max_retries:
            # --- Step 1: Plan / Generate Query ---
            yield AgentStep("plan", "Reasoning about search strategy...")

            # If it's a retry, the LLM generates a new query based on failure
            search_query = self.llm.generate_search_query(
                user_query, previous_search_query
            )
            previous_search_query = search_query

            yield AgentStep("plan", f"Generated Search Query: '{search_query}'")

            # --- Step 2: Retrieve / Search ---
            yield AgentStep("search", f"Querying Qdrant for: {search_query}")

            # Query the vector database
            results = self.engine.search(search_query, k=3)
            if not results:
                yield AgentStep("search", "No results found in database.")
                attempt += 1
                continue

            # Format context for the LLM
            context_str = ""
            for i, res in enumerate(results):
                code_content = res["payload"].get("content", "") or res[
                    "payload"
                ].get("code_content", "")
                source = res["payload"].get("source", "Unknown")
                context_str += f"\n--- Snippet {i + 1} (Source: {source}) ---\n{code_content}\n"

            # Pass raw data
            yield AgentStep(
                "search", "Retrieved candidates from database.", data=results
            )

            # --- Step 3: Reason & Critique ---
            yield AgentStep("critique", "Analyzing code relevance...")
            is_sufficient, analysis = self.llm.analyze_and_answer(
                user_query, context_str
            )

            if is_sufficient:
                # Success, yield final answer and break loop
                yield AgentStep("answer", analysis)
                return
            else:
                # Failure, yield critique and loop back
                yield AgentStep(
                    "critique", f"Relevance check failed: {analysis}"
                )
                yield AgentStep("critique", "Refining search strategy...")
                attempt += 1

        # Fallback if retries exhausted
        yield AgentStep(
            "error", "Could not find relevant code after multiple attempts."
        )


class ToolCallingAgent:
    """Agentic loop where the LLM chooses which tool to call until it can answer."""

    def __init__(
        self,
        engine: CodeSearchEngine,
        llm: LLMClient,
        llm_config,
        base_dir: str = ".",
        max_steps: int = 6,
        use_nudges: bool = True,
    ):
        self.llm = llm
        self.llm_config = llm_config
        self.max_steps = max_steps
        self.use_nudges = use_nudges
        self.tools: dict[str, Callable[..., str]] = {
            "search_code": lambda query: search_code(query, engine),
            "read_file": lambda path: read_file(path, base_dir),
            "list_directory": lambda path=".": list_directory(path, base_dir),
            "grep": lambda pattern, path=".": grep(pattern, path, base_dir),
        }

    def solve(self, user_query: str) -> Generator[AgentStep, None, None]:
        """
        The tool-choosing agentic loop. Yields steps for real-time rendering.

        Args:
            user_query (str): Input query.

        Returns:
            Generator[AgentStep]: Contains all the steps the agent has done.
        """
        yield AgentStep("start", f"Processing request: {user_query}")
        conversation = [
            ("system", TOOL_LOOP_SYSTEM_MSG),
            ("user", user_query),
        ]
        has_called_tool = False
        seen_sources: set[str] = set()

        for _ in range(self.max_steps):
            yield AgentStep("plan", "Deciding next action...")
            content, tool_calls = self.llm.call_with_tools_auto(
                conversation, TOOL_SPECS, self.llm_config
            )

            if not tool_calls:
                if self.use_nudges:
                    rejection = _reject_answer(
                        content, has_called_tool, seen_sources
                    )
                    if rejection is not None:
                        status, nudge = rejection
                        yield AgentStep("plan", status)
                        conversation.append(("assistant", content))
                        conversation.append(("user", nudge))
                        continue

                yield AgentStep("answer", content)
                return

            for call in tool_calls:
                yield AgentStep(
                    "tool_call", f"Calling {call.name}({call.arguments})"
                )
                result = self._run_tool(call)
                yield AgentStep("tool_result", result)

                # Only a successful result counts as having "used a tool":
                # a failed call hasn't actually grounded anything yet
                if not result.startswith("Error:"):
                    has_called_tool = True
                    seen_sources |= _extract_sources(result)
                    path = call.arguments.get("path")
                    if (
                        call.name in ("read_file", "grep", "list_directory")
                        and path
                    ):
                        seen_sources.add(path)

                conversation.append(
                    ("assistant", f"Calling tool {call.name}({call.arguments})")
                )
                result_msg = f"Result of {call.name}:\n{result}"
                if self.use_nudges:
                    result_msg += f"\n\n{ANSWER_IF_SUFFICIENT_MSG}"
                conversation.append(("user", result_msg))

        yield AgentStep(
            "error", "Could not produce an answer within the step limit."
        )

    def _run_tool(self, call: ToolCall) -> str:
        """
        Runs a single tool call, isolating bad arguments from the loop.

        Args:
            call (ToolCall): The tool name and arguments requested by the LLM.

        Returns:
            str: The tool's output, or a human-readable error message
                starting with "Error:" if the tool name or arguments are bad.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            return f"Error: unknown tool '{call.name}'."

        try:
            return tool(**call.arguments)
        except TypeError as e:
            return f"Error: invalid arguments for '{call.name}': {e}."

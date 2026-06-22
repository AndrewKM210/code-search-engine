from collections.abc import Callable, Generator
from dataclasses import dataclass
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

TOOL_LOOP_SYSTEM_MSG = (
    "You are a coding assistant exploring a repository. Use the available "
    "tools (search_code, read_file, list_directory, grep) to find the "
    "information needed to answer the user's question. Once you have "
    "enough information, answer the question directly in plain text "
    "instead of calling another tool."
)


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
    ):
        self.llm = llm
        self.llm_config = llm_config
        self.max_steps = max_steps
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

        for _ in range(self.max_steps):
            yield AgentStep("plan", "Deciding next action...")
            content, tool_calls = self.llm.call_with_tools_auto(
                conversation, TOOL_SPECS, self.llm_config
            )

            if not tool_calls:
                yield AgentStep("answer", content)
                return

            for call in tool_calls:
                yield AgentStep(
                    "tool_call", f"Calling {call.name}({call.arguments})"
                )
                result = self._run_tool(call)
                yield AgentStep("tool_result", result)

                conversation.append(
                    ("assistant", f"Calling tool {call.name}({call.arguments})")
                )
                conversation.append(
                    ("user", f"Result of {call.name}:\n{result}")
                )

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

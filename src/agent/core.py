from dataclasses import dataclass
from typing import Any, Generator
from src.agent.llm import LLMClient
from src.search_engine.engine import CodeSearchEngine


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
            search_query = self.llm.generate_search_query(user_query, previous_search_query)
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
                code_content = res["payload"].get("content", "") or res["payload"].get("code_content", "")
                source = res["payload"].get("source", "Unknown")
                context_str += f"\n--- Snippet {i + 1} (Source: {source}) ---\n{code_content}\n"

            # Pass raw data
            yield AgentStep("search", "Retrieved candidates from database.", data=results)

            # --- Step 3: Reason & Critique ---
            yield AgentStep("critique", "Analyzing code relevance...")
            is_sufficient, analysis = self.llm.analyze_and_answer(user_query, context_str)

            if is_sufficient:
                # Success, yield final answer and break loop
                yield AgentStep("answer", analysis)
                return
            else:
                # Failure, yield critique and loop back
                yield AgentStep("critique", f"Relevance check failed: {analysis}")
                yield AgentStep("critique", "Refining search strategy...")
                attempt += 1

        # Fallback if retries exhausted
        yield AgentStep("error", "Could not find relevant code after multiple attempts.")

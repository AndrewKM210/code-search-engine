import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from cse.agent.schema import ToolCall

# Matches a ```json ... ``` or ``` ... ``` fence wrapping the JSON payload
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


class LLMClient:
    """Wraps the local Ollama model to handle reasoning and generation."""

    def __init__(self, model_name="phi3", temperature=0.1):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            validate_model_on_init=True,
        )
        self.output_parser = StrOutputParser()

    def call_with_tools(
        self, messages: list, tools: list
    ) -> tuple[str, list[ToolCall]]:
        """
        Sends messages to the model with tools available via native tool calling.

        Binds the tool definitions so a tool-capable model (e.g.
        llama3.2:3b) can decide to call one. Models without native tool
        support need the prompt+JSON fallback path instead.

        Args:
            messages (list): Conversation so far, as LangChain messages or
                (role, content) tuples.
            tools (list): Tool definitions to expose, in any form
                ChatOllama.bind_tools accepts (functions, Pydantic models or
                OpenAI-style dicts).

        Returns:
            tuple: (content, tool_calls), where tool_calls is empty when the
                model answers directly instead of requesting a tool.
        """
        response = self.llm.bind_tools(tools).invoke(messages)
        return response.content, self._parse_tool_calls(response.tool_calls)

    @staticmethod
    def _parse_tool_calls(
        raw_tool_calls: list[dict[str, Any]],
    ) -> list[ToolCall]:
        """
        Normalizes LangChain tool-call dicts into validated ToolCall objects.

        Args:
            raw_tool_calls (list): Each item is a dict with "name" and "args",
                as found on an AIMessage's tool_calls attribute.

        Returns:
            list[ToolCall]: One validated ToolCall per requested invocation.
        """
        return [
            ToolCall(name=tc["name"], arguments=tc.get("args") or {})
            for tc in raw_tool_calls
        ]

    def call_with_tools_fallback(
        self, messages: list, tools: list, max_retries: int = 2
    ) -> tuple[str, list[ToolCall]]:
        """
        Sends messages to a model without reliable native tool calling,
        asking it to emit a tool call as JSON text instead.

        Some models (e.g. phi3, qwen2.5-coder:3b) either lack native tool
        support or return the call as JSON text rather than populating
        tool_calls. This prepends a system message describing the tools and
        the expected JSON shape, then parses the model's raw text response.
        If the response looks like a failed JSON attempt, it retries with a
        corrective message instead of giving up immediately.

        Args:
            messages (list): Conversation so far, as (role, content) tuples.
            tools (list): Tool definitions, in the same OpenAI-style dicts
                used by call_with_tools.
            max_retries (int): Extra attempts allowed after a malformed
                response, before giving up and returning plain text.

        Returns:
            tuple: (content, tool_calls), where tool_calls is empty when the
                model answers directly or never produces a valid call.
        """
        conversation = [("system", _format_tool_instructions(tools)), *messages]

        for _ in range(max_retries + 1):
            content = self.llm.invoke(conversation).content.strip()

            tool_call = _extract_tool_call(content)
            if tool_call is not None:
                return content, [tool_call]

            if not content.lstrip().startswith(("{", "```")):
                return content, []

            conversation.append(("assistant", content))
            conversation.append(
                (
                    "user",
                    "That wasn't valid JSON for a tool call. Respond with "
                    'ONLY a JSON object: {"name": "...", "arguments": {...}}.',
                )
            )

        return content, []

    def call_with_tools_auto(
        self, messages: list, tools: list, llm_config, max_retries: int = 2
    ) -> tuple[str, list[ToolCall]]:
        """
        Calls a tool, picking native or prompt+JSON fallback based on config.

        Looks up this client's model in llm_config to avoid callers having
        to know or hardcode which models support native tool calling.

        Args:
            messages (list): Conversation so far, as (role, content) tuples.
            tools (list): Tool definitions, in the same OpenAI-style dicts
                used by call_with_tools.
            llm_config: Loaded llm_config.yaml (OmegaConf), with a "models"
                map of model name to capability flags.
            max_retries (int): Forwarded to call_with_tools_fallback.

        Returns:
            tuple: (content, tool_calls), where tool_calls is empty when the
                model answers directly or never produces a valid call.
        """
        if resolve_native_tool_calling(self.model_name, llm_config):
            return self.call_with_tools(messages, tools)
        return self.call_with_tools_fallback(
            messages, tools, max_retries=max_retries
        )

    def generate_search_query(
        self, user_input: str, previous_attempt: str = None
    ) -> str:
        """
        Feeds the user's query to the LLM to create the vector DB query.

        Args:
            user_input (str): User's input query.
            previous_attempt (str): Previous failed query (if there is, otherwise `None`).

        Returns:
            str: Query for the database.
        """
        system_msg = "You are an expert code retrieval assistant. Your job is to generate ONE concise search query for a vector database."

        if previous_attempt:
            user_msg = (
                "The previous query '{previous_attempt}' yielded no relevant "
                "results. Generate a DIFFERENT, better search query to find "
                "code for: {user_input}"
                "\nOutput ONLY the query string, no quotes or explanations."
            )
        else:
            user_msg = (
                "Generate a search query to find code relevant to: {user_input}"
                "\nOutput ONLY the query string, no quotes or explanations."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("user", user_msg),
            ]
        )

        # Use template variables so literal braces in user input aren't parsed as placeholders
        chain = prompt | self.llm | self.output_parser
        return chain.invoke(
            {"user_input": user_input, "previous_attempt": previous_attempt}
        ).strip()

    def analyze_and_answer(
        self, user_input: str, retrieved_context: str
    ) -> tuple:
        """
        Uses the LLM to check if context is sufficient and generate answer or rejection signal.

        Args:
            user_input (str): User's input query.
            retrieved_context (str): Context obtained from querying the vector DB.

        Returns:
            tuple: (is_sufficient: bool, content: str)
        """
        system_msg = """
        You are a senior software engineer. You must answer the user's question strictly based on the provided retrieved code context.
        
        If the retrieved code contains the answer:
        1. Start your response with "MATCH:".
        2. Explain the solution clearly using the code.
        
        If the retrieved code is irrelevant or does not help:
        1. Reply ONLY with "MISSING: The retrieved code is about [Topic of code], but the user asked for [User intent]."
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "user",
                    "User Question: {user_input}\n\nRetrieved Code Context:\n{retrieved_context}",
                ),
            ]
        )

        # Use template variables so literal braces in retrieved code aren't parsed as placeholders
        chain = prompt | self.llm | self.output_parser
        response = chain.invoke(
            {"user_input": user_input, "retrieved_context": retrieved_context}
        )

        if response.startswith("MATCH:"):
            return True, response.replace("MATCH:", "").strip()
        else:
            return False, response.replace("MISSING:", "").strip()


def resolve_native_tool_calling(model_name: str, llm_config) -> bool:
    """
    Looks up whether a model supports native tool calling.

    Args:
        model_name (str): Ollama model name, e.g. "llama3.2:3b".
        llm_config: Loaded llm_config.yaml (OmegaConf), with a "models" map
            of model name to capability flags.

    Returns:
        bool: True if the model is listed with native_tool_calling: true.
            Defaults to False (use the JSON fallback) for unlisted models.
    """
    model_cfg = llm_config.models.get(model_name)
    if model_cfg is None:
        return False
    return bool(model_cfg.get("native_tool_calling", False))


def _format_tool_instructions(tools: list[dict]) -> str:
    """
    Renders OpenAI-style tool specs into a plain-text system instruction.

    Args:
        tools (list): Tool definitions in the OpenAI-style dict shape used
            by call_with_tools (each with a "function" key holding name,
            description and parameters).

    Returns:
        str: Instructions telling the model how to request a tool as JSON,
            followed by one line per available tool.
    """
    lines = [
        "You can call one of the following tools to help answer the user. "
        "If a tool is useful, respond with ONLY a JSON object of the form "
        '{"name": "<tool_name>", "arguments": {<arg_name>: <value>, ...}}. '
        "If no tool is needed, answer normally in plain text.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        spec = tool["function"]
        params = ", ".join(spec["parameters"]["properties"])
        lines.append(f"- {spec['name']}({params}): {spec['description']}")

    return "\n".join(lines)


def _extract_tool_call(text: str) -> ToolCall | None:
    """
    Parses a model's raw text response into a ToolCall, if it is one.

    Args:
        text (str): The model's response, optionally wrapped in a markdown
            code fence around the JSON payload.

    Returns:
        ToolCall | None: The parsed tool call, or None if the text isn't a
            valid ToolCall JSON object.
    """
    text = text.strip()
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return ToolCall.model_validate_json(text)
    except ValidationError:
        return None

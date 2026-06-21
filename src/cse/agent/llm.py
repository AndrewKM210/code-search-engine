from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from cse.agent.schema import ToolCall


class LLMClient:
    """Wraps the local Ollama model to handle reasoning and generation."""

    def __init__(self, model_name="phi3", temperature=0.1):
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

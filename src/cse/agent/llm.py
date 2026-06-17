from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class LLMClient:
    """Wraps the local Ollama model to handle reasoning and generation."""

    def __init__(self, model_name="phi3", temperature=0.1):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()

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
            prompt_text = f"The previous query '{previous_attempt}' yielded no relevant results. Generate a DIFFERENT, better search query to find code for: {user_input}"
        else:
            prompt_text = f"Generate a search query to find code relevant to: {user_input}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "user",
                    prompt_text
                    + "\nOutput ONLY the query string, no quotes or explanations.",
                ),
            ]
        )

        chain = prompt | self.llm | self.output_parser
        return chain.invoke({}).strip()

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
                    f"User Question: {user_input}\n\nRetrieved Code Context:\n{retrieved_context}",
                ),
            ]
        )

        chain = prompt | self.llm | self.output_parser
        response = chain.invoke({})

        if response.startswith("MATCH:"):
            return True, response.replace("MATCH:", "").strip()
        else:
            return False, response.replace("MISSING:", "").strip()

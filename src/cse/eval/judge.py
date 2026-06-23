import re

from pydantic import ValidationError

from cse.agent.llm import LLMClient
from cse.eval.schema import JudgeVerdict

# Matches a ```json ... ``` or ``` ... ``` fence wrapping the JSON payload
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)

JUDGE_SYSTEM_MSG = (
    "You are an impartial judge evaluating an AI coding assistant's answer "
    "to a question about a software repository.\n\n"
    "Score two independent dimensions:\n"
    "- correctness: true if the candidate answer conveys the same key facts "
    "as the reference answer (minor wording differences are fine), false "
    "otherwise.\n"
    "- faithfulness: true if every claim in the candidate answer is "
    "supported by the retrieved context, false if it makes claims not "
    "grounded in that context (hallucination), regardless of correctness.\n\n"
    "Respond with ONLY a JSON object of the form "
    '{"correctness": true|false, "faithfulness": true|false, '
    '"rationale": "one sentence explaining both scores"}.'
)

_RETRY_MSG = (
    "That wasn't valid JSON. Respond with ONLY a JSON object: "
    '{"correctness": true|false, "faithfulness": true|false, "rationale": "..."}.'
)


class LLMJudge:
    """Scores a candidate answer for correctness and faithfulness using an LLM."""

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def score(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        retrieved_context: str,
    ) -> JudgeVerdict:
        """
        Scores a candidate answer against a reference answer.

        Args:
            question (str): The gold question being answered.
            reference_answer (str): The known-correct answer to compare against.
            candidate_answer (str): The agent's final answer to evaluate.
            retrieved_context (str): Tool/search output the agent actually
                saw while answering, used to check faithfulness. Pass a
                placeholder like "(no tools were called)" if the agent
                answered without retrieving anything.

        Returns:
            JudgeVerdict: correctness, faithfulness and a one-line rationale.
                Both scores default to False if the judge never produces
                valid JSON within max_retries attempts.
        """
        conversation = [
            ("system", JUDGE_SYSTEM_MSG),
            (
                "user",
                f"Question: {question}\n\n"
                f"Reference answer: {reference_answer}\n\n"
                f"Retrieved context the assistant had access to:\n{retrieved_context}\n\n"
                f"Candidate answer to evaluate:\n{candidate_answer}",
            ),
        ]

        content = ""
        for _ in range(self.max_retries + 1):
            content = self.llm.llm.invoke(conversation).content.strip()
            verdict = _extract_verdict(content)
            if verdict is not None:
                return verdict

            conversation.append(("assistant", content))
            conversation.append(("user", _RETRY_MSG))

        return JudgeVerdict(
            correctness=False,
            faithfulness=False,
            rationale=f"Judge failed to produce valid JSON, last response: {content!r}",
        )


def _extract_verdict(text: str) -> JudgeVerdict | None:
    """
    Parses a judge's raw text response into a JudgeVerdict, if it is one.

    Args:
        text (str): The model's response, optionally wrapped in a markdown
            code fence around the JSON payload.

    Returns:
        JudgeVerdict | None: The parsed verdict, or None if the text isn't a
            valid JudgeVerdict JSON object.
    """
    text = text.strip()
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return JudgeVerdict.model_validate_json(text)
    except ValidationError:
        return None

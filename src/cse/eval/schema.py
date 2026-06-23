from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    """
    An LLM judge's evaluation of a candidate answer against a reference answer.

    Args:
        correctness (bool): True if the candidate conveys the same key facts
            as the reference answer.
        faithfulness (bool): True if every claim in the candidate answer is
            grounded in the retrieved context, with no unsupported claims.
        rationale (str): One-line explanation for both scores.
    """

    correctness: bool
    faithfulness: bool
    rationale: str

from unittest.mock import MagicMock

from cse.agent.llm import LLMClient
from cse.eval.judge import LLMJudge, _extract_verdict
from cse.eval.schema import JudgeVerdict


def _judge_with_mock_llm(*responses):
    """Builds an LLMJudge whose underlying llm.invoke yields the given contents in order."""
    client = object.__new__(LLMClient)
    client.model_name = "phi3"
    client.llm = MagicMock()
    client.llm.invoke.side_effect = [MagicMock(content=r) for r in responses]
    return LLMJudge(client)


def test_extract_verdict_parses_plain_json():
    verdict = _extract_verdict(
        '{"correctness": true, "faithfulness": false, "rationale": "close enough"}'
    )

    assert verdict == JudgeVerdict(
        correctness=True, faithfulness=False, rationale="close enough"
    )


def test_extract_verdict_parses_fenced_json():
    text = (
        '```json\n{"correctness": false, "faithfulness": true, '
        '"rationale": "wrong fact"}\n```'
    )

    verdict = _extract_verdict(text)

    assert verdict == JudgeVerdict(
        correctness=False, faithfulness=True, rationale="wrong fact"
    )


def test_extract_verdict_returns_none_for_plain_text():
    assert _extract_verdict("I think this is correct.") is None


def test_score_parses_valid_json_first_try():
    judge = _judge_with_mock_llm(
        '{"correctness": true, "faithfulness": true, "rationale": "matches"}'
    )

    verdict = judge.score(
        question="What distance metric does Qdrant use?",
        reference_answer="Cosine distance.",
        candidate_answer="It uses cosine distance.",
        retrieved_context="models.Distance.COSINE",
    )

    assert verdict == JudgeVerdict(
        correctness=True, faithfulness=True, rationale="matches"
    )
    assert judge.llm.llm.invoke.call_count == 1


def test_score_retries_malformed_json():
    judge = _judge_with_mock_llm(
        "not json at all",
        '{"correctness": true, "faithfulness": false, "rationale": "ok"}',
    )

    verdict = judge.score(
        question="q",
        reference_answer="r",
        candidate_answer="c",
        retrieved_context="ctx",
    )

    assert verdict == JudgeVerdict(
        correctness=True, faithfulness=False, rationale="ok"
    )
    assert judge.llm.llm.invoke.call_count == 2


def test_score_defaults_to_false_after_max_retries():
    judge = _judge_with_mock_llm("nope", "still nope", "nope again")
    judge.max_retries = 2

    verdict = judge.score(
        question="q",
        reference_answer="r",
        candidate_answer="c",
        retrieved_context="ctx",
    )

    assert verdict.correctness is False
    assert verdict.faithfulness is False
    assert "nope again" in verdict.rationale
    assert judge.llm.llm.invoke.call_count == 3


def test_score_includes_question_and_context_in_prompt():
    judge = _judge_with_mock_llm(
        '{"correctness": true, "faithfulness": true, "rationale": "ok"}'
    )

    judge.score(
        question="What does foo do?",
        reference_answer="It returns 1.",
        candidate_answer="foo returns 1.",
        retrieved_context="def foo(): return 1",
    )

    sent_messages = judge.llm.llm.invoke.call_args_list[0].args[0]
    user_msg = sent_messages[1][1]
    assert "What does foo do?" in user_msg
    assert "def foo(): return 1" in user_msg

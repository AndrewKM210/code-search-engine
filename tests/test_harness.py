import csv
from unittest.mock import MagicMock

from cse.agent.core import AgentStep
from cse.eval.harness import (
    NO_ANSWER_PLACEHOLDER,
    NO_CONTEXT_PLACEHOLDER,
    EvalResult,
    GoldItem,
    load_gold_set,
    run_agent,
    run_eval,
    save_results,
    subset_gold_set,
    summarize,
)
from cse.eval.schema import JudgeVerdict


def _agent_with_steps(*steps):
    agent = MagicMock()
    agent.solve.return_value = iter(steps)
    return agent


def test_run_agent_collects_search_context_and_answer():
    agent = _agent_with_steps(
        AgentStep("start", "..."),
        AgentStep(
            "search",
            "...",
            data=[{"payload": {"content": "def foo(): pass"}}],
        ),
        AgentStep("answer", "foo does nothing"),
    )

    answer, context = run_agent(agent, "what does foo do?")

    assert answer == "foo does nothing"
    assert "def foo(): pass" in context


def test_run_agent_collects_tool_results_as_context():
    agent = _agent_with_steps(
        AgentStep("tool_call", "grep(foo)"),
        AgentStep("tool_result", "src/foo.py:1:def foo(): pass"),
        AgentStep("answer", "foo is defined in src/foo.py"),
    )

    answer, context = run_agent(agent, "where is foo defined?")

    assert answer == "foo is defined in src/foo.py"
    assert "src/foo.py:1:def foo(): pass" in context


def test_run_agent_falls_back_to_placeholders_when_nothing_retrieved():
    agent = _agent_with_steps(AgentStep("start", "..."))

    answer, context = run_agent(agent, "anything?")

    assert answer == NO_ANSWER_PLACEHOLDER
    assert context == NO_CONTEXT_PLACEHOLDER


def test_run_agent_reports_error_step_as_answer():
    agent = _agent_with_steps(AgentStep("error", "step limit exceeded"))

    answer, _ = run_agent(agent, "anything?")

    assert "step limit exceeded" in answer


def test_load_gold_set_parses_yaml(tmp_path):
    gold_path = tmp_path / "gold_qa.yaml"
    gold_path.write_text(
        "- id: 1\n"
        "  category: lookup\n"
        "  question: q1\n"
        "  reference_answer: a1\n"
        "  evidence: [file.py]\n"
    )

    gold_set = load_gold_set(str(gold_path))

    assert gold_set == [
        GoldItem(
            id=1,
            category="lookup",
            question="q1",
            reference_answer="a1",
            evidence=["file.py"],
        )
    ]


def test_subset_gold_set_caps_each_category_independently():
    gold_set = [
        GoldItem(1, "lookup", "q1", "a1", []),
        GoldItem(2, "lookup", "q2", "a2", []),
        GoldItem(3, "lookup", "q3", "a3", []),
        GoldItem(4, "multi_hop", "q4", "a4", []),
        GoldItem(5, "multi_hop", "q5", "a5", []),
    ]

    subset = subset_gold_set(gold_set, limit_per_category=1)

    assert [item.id for item in subset] == [1, 4]


def test_subset_gold_set_returns_full_set_when_limit_is_none():
    gold_set = [GoldItem(1, "lookup", "q1", "a1", [])]

    assert subset_gold_set(gold_set, limit_per_category=None) == gold_set


def test_run_eval_scores_every_question_against_every_agent():
    gold_set = [
        GoldItem(
            id=1,
            category="lookup",
            question="q1",
            reference_answer="a1",
            evidence=[],
        ),
        GoldItem(
            id=2,
            category="lookup",
            question="q2",
            reference_answer="a2",
            evidence=[],
        ),
    ]

    def _agent_answering_every_question():
        agent = MagicMock()
        agent.solve.side_effect = lambda q: iter([AgentStep("answer", "ans")])
        return agent

    agents = {
        "baseline": _agent_answering_every_question(),
        "tool-loop": _agent_answering_every_question(),
    }

    judge = MagicMock()
    judge.score.return_value = JudgeVerdict(
        correctness=True, faithfulness=True, rationale="ok"
    )

    results = run_eval(gold_set, agents, judge)

    assert len(results) == 4
    assert {(r.id, r.agent) for r in results} == {
        (1, "baseline"),
        (1, "tool-loop"),
        (2, "baseline"),
        (2, "tool-loop"),
    }
    assert all(r.correctness and r.faithfulness for r in results)


def test_summarize_averages_per_agent():
    results = [
        EvalResult(1, "lookup", "q1", "baseline", "a", True, True, "ok"),
        EvalResult(2, "lookup", "q2", "baseline", "a", False, True, "ok"),
        EvalResult(1, "lookup", "q1", "tool-loop", "a", True, False, "ok"),
        EvalResult(2, "lookup", "q2", "tool-loop", "a", True, True, "ok"),
    ]

    summary = summarize(results)

    assert summary["baseline"] == {"correctness": 0.5, "faithfulness": 1.0}
    assert summary["tool-loop"] == {"correctness": 1.0, "faithfulness": 0.5}


def test_save_results_writes_csv(tmp_path):
    results = [
        EvalResult(1, "lookup", "q1", "baseline", "a1", True, False, "ok"),
    ]
    out_path = tmp_path / "out" / "results.csv"

    save_results(results, str(out_path))

    with out_path.open() as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["question"] == "q1"
    assert rows[0]["correctness"] == "True"

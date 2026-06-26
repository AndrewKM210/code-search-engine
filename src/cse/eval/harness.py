import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from cse.agent.core import CodingAgent, ToolCallingAgent
from cse.eval.judge import LLMJudge

NO_CONTEXT_PLACEHOLDER = "(no tools were called)"
NO_ANSWER_PLACEHOLDER = "(agent did not produce an answer)"


@dataclass
class GoldItem:
    """One gold question from data/eval/gold_qa.yaml."""

    id: int
    category: str
    question: str
    reference_answer: str
    evidence: list[str]


@dataclass
class EvalResult:
    """One agent's scored answer to one gold question."""

    id: int
    category: str
    question: str
    agent: str
    answer: str
    correctness: bool
    faithfulness: bool
    rationale: str


def load_gold_set(path: str) -> list[GoldItem]:
    """
    Loads the gold Q&A set from a YAML file.

    Args:
        path (str): Path to a gold_qa.yaml file shaped like data/eval/gold_qa.yaml.

    Returns:
        list[GoldItem]: One entry per gold question.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return [GoldItem(**item) for item in raw]


def subset_gold_set(
    gold_set: list[GoldItem], limit_per_category: int | None
) -> list[GoldItem]:
    """
    Caps the gold set to the first N items per category, for fast trial runs.

    Args:
        gold_set (list[GoldItem]): The full gold set, in file order.
        limit_per_category (int | None): Max items to keep per category, or
            None to keep the full gold set unchanged.

    Returns:
        list[GoldItem]: The capped gold set, in original order.
    """
    if limit_per_category is None:
        return gold_set

    kept_per_category: dict[str, int] = {}
    subset = []
    for item in gold_set:
        kept = kept_per_category.get(item.category, 0)
        if kept < limit_per_category:
            subset.append(item)
            kept_per_category[item.category] = kept + 1
    return subset


def run_agent(
    agent: CodingAgent | ToolCallingAgent, question: str
) -> tuple[str, str]:
    """
    Runs an agent to completion on one question.

    Args:
        agent (CodingAgent | ToolCallingAgent): The agent under evaluation.
        question (str): The gold question to ask it.

    Returns:
        tuple[str, str]: (answer, retrieved_context). answer falls back to a
            placeholder if the agent errors out instead of answering.
            retrieved_context concatenates everything the agent actually
            retrieved (search hits for CodingAgent, tool outputs for
            ToolCallingAgent), or a placeholder if it retrieved nothing.
    """
    context_chunks: list[str] = []
    answer = NO_ANSWER_PLACEHOLDER

    for step in agent.solve(question):
        if step.step_type == "search" and step.data:
            for res in step.data:
                payload = res.get("payload", {})
                content = payload.get("content") or payload.get(
                    "code_content", ""
                )
                if content:
                    context_chunks.append(content)
        elif step.step_type == "tool_result":
            context_chunks.append(step.content)
        elif step.step_type == "answer":
            answer = step.content
        elif step.step_type == "error":
            answer = f"{NO_ANSWER_PLACEHOLDER}: {step.content}"

    retrieved_context = "\n---\n".join(context_chunks) or NO_CONTEXT_PLACEHOLDER
    return answer, retrieved_context


def run_eval(
    gold_set: Iterable[GoldItem],
    agents: dict[str, CodingAgent | ToolCallingAgent],
    judge: LLMJudge,
) -> list[EvalResult]:
    """
    Runs every agent over every gold question and scores each answer.

    Args:
        gold_set (Iterable[GoldItem]): Gold questions to evaluate against.
        agents (dict): Maps agent label (e.g. "baseline", "tool-loop") to a
            built agent instance.
        judge (LLMJudge): Scorer used to evaluate each agent's answer.

    Returns:
        list[EvalResult]: One row per (gold question, agent) pair.
    """
    gold_set = list(gold_set)

    results = []
    for agent_label, agent in agents.items():
        for item in tqdm(gold_set, desc=agent_label):
            answer, retrieved_context = run_agent(agent, item.question)
            verdict = judge.score(
                question=item.question,
                reference_answer=item.reference_answer,
                candidate_answer=answer,
                retrieved_context=retrieved_context,
            )
            results.append(
                EvalResult(
                    id=item.id,
                    category=item.category,
                    question=item.question,
                    agent=agent_label,
                    answer=answer,
                    correctness=verdict.correctness,
                    faithfulness=verdict.faithfulness,
                    rationale=verdict.rationale,
                )
            )
    return results


def summarize(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """
    Aggregates per-agent task-success rates from raw eval results.

    Args:
        results (list[EvalResult]): Rows produced by run_eval.

    Returns:
        dict: Maps agent label to {"correctness": rate, "faithfulness": rate},
            each averaged over that agent's rows.
    """
    summary: dict[str, dict[str, float]] = {}
    for agent_label in {r.agent for r in results}:
        rows = [r for r in results if r.agent == agent_label]
        summary[agent_label] = {
            "correctness": sum(r.correctness for r in rows) / len(rows),
            "faithfulness": sum(r.faithfulness for r in rows) / len(rows),
        }
    return summary


def save_results(results: list[EvalResult], path: str) -> None:
    """
    Writes raw per-question eval results to a CSV file.

    Args:
        results (list[EvalResult]): Rows produced by run_eval.
        path (str): Destination CSV path; parent directory is created if missing.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

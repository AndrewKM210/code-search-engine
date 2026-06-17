from unittest.mock import MagicMock

from cse.agent.core import CodingAgent


def _make_llm(sufficient=True, query="search query"):
    llm = MagicMock()
    llm.generate_search_query.return_value = query
    llm.analyze_and_answer.return_value = (sufficient, "the answer")
    return llm


def test_solve_uses_content_key_when_present():
    engine = MagicMock()
    engine.search.return_value = [
        {"code_id": 1, "score": 0.9, "payload": {"content": "def from_dir(): pass", "source": "f.py"}}
    ]
    llm = _make_llm()
    agent = CodingAgent(engine, llm)

    list(agent.solve("how does from_dir work?"))

    context = llm.analyze_and_answer.call_args.args[1]
    assert "def from_dir(): pass" in context


def test_solve_falls_back_to_code_content_key():
    engine = MagicMock()
    engine.search.return_value = [
        {"code_id": 1, "score": 0.9, "payload": {"code_content": "def from_corpus(): pass"}}
    ]
    llm = _make_llm()
    agent = CodingAgent(engine, llm)

    list(agent.solve("how does from_corpus work?"))

    context = llm.analyze_and_answer.call_args.args[1]
    assert "def from_corpus(): pass" in context


def test_solve_yields_answer_step_on_success():
    engine = MagicMock()
    engine.search.return_value = [{"code_id": 1, "score": 0.9, "payload": {"content": "x"}}]
    llm = _make_llm(sufficient=True)
    agent = CodingAgent(engine, llm)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "answer"
    assert steps[-1].content == "the answer"


def test_solve_retries_then_errors_when_insufficient():
    engine = MagicMock()
    engine.search.return_value = [{"code_id": 1, "score": 0.9, "payload": {"content": "x"}}]
    llm = _make_llm(sufficient=False)
    agent = CodingAgent(engine, llm)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "error"
    assert llm.generate_search_query.call_count == agent.max_retries + 1


def test_solve_retries_when_no_search_results():
    engine = MagicMock()
    engine.search.return_value = []
    llm = _make_llm()
    agent = CodingAgent(engine, llm)

    steps = list(agent.solve("question"))

    assert steps[-1].step_type == "error"
    assert llm.analyze_and_answer.call_count == 0
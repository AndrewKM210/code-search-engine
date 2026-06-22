from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from cse.agent.core import CodingAgent, ToolCallingAgent
from cse.agent.setup import AgentOptions, build_agent

CONFIG = OmegaConf.create(
    {
        "model_name": "base-model",
        "finetuned_model_path": "finetuned-model",
        "device": "auto",
        "qdrant": {
            "storage_path": "./qdrant_storage",
            "full_collection": "cosqa_full",
            "self_repo_collection": "self_repo",
        },
    }
)
LLM_CONFIG = OmegaConf.create({"models": {}})


@pytest.fixture(autouse=True)
def _mock_dependencies(monkeypatch):
    """Replaces the heavy config load, search engine and LLM with mocks."""
    monkeypatch.setattr(
        "cse.agent.setup.OmegaConf.load",
        lambda path: CONFIG if "main_config" in path else LLM_CONFIG,
    )
    monkeypatch.setattr("cse.agent.setup.CodeSearchEngine", MagicMock())
    monkeypatch.setattr("cse.agent.setup.LLMClient", MagicMock())


def test_build_agent_yields_ready_baseline_agent():
    steps = list(build_agent(AgentOptions(agent_type="baseline")))

    assert [s.step_type for s in steps[:-1]] == ["status"] * (len(steps) - 1)
    assert steps[-1].step_type == "ready"
    assert isinstance(steps[-1].agent, CodingAgent)


def test_build_agent_yields_ready_tool_loop_agent():
    steps = list(build_agent(AgentOptions(agent_type="tool-loop")))

    assert steps[-1].step_type == "ready"
    assert isinstance(steps[-1].agent, ToolCallingAgent)


def test_build_agent_uses_finetuned_model_when_requested():
    steps = list(build_agent(AgentOptions(finetuned=True)))

    assert "finetuned-model" in steps[0].content


def test_build_agent_uses_full_collection_for_baseline():
    import cse.agent.setup as setup_module

    list(build_agent(AgentOptions(agent_type="baseline")))

    kwargs = setup_module.CodeSearchEngine.call_args.kwargs
    assert kwargs["db_collection"] == "cosqa_full"


def test_build_agent_uses_self_repo_collection_for_tool_loop():
    import cse.agent.setup as setup_module

    list(build_agent(AgentOptions(agent_type="tool-loop")))

    kwargs = setup_module.CodeSearchEngine.call_args.kwargs
    assert kwargs["db_collection"] == "self_repo"


def test_build_agent_yields_error_step_on_exception(monkeypatch):
    monkeypatch.setattr(
        "cse.agent.setup.CodeSearchEngine",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    steps = list(build_agent(AgentOptions()))

    assert steps[-1].step_type == "error"
    assert "boom" in steps[-1].content

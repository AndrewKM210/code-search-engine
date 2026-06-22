from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

from omegaconf import DictConfig, OmegaConf

from cse.agent.core import CodingAgent, ToolCallingAgent
from cse.agent.llm import LLMClient
from cse.search_engine.engine import CodeSearchEngine


@dataclass(frozen=True)
class AgentOptions:
    """User-facing choices needed to build an agent, shared by the CLI and Streamlit app."""

    finetuned: bool = False
    agent_type: str = "baseline"  # "baseline" or "tool-loop"
    model: str = "phi3"
    config_path: str = "config/main_config.yaml"
    llm_config_path: str = "config/llm_config.yaml"


@dataclass
class SetupStep:
    """Status update yielded while constructing the agent and its dependencies."""

    step_type: str  # "status", "ready" or "error"
    content: str
    agent: CodingAgent | ToolCallingAgent | None = None


def build_agent(
    options: AgentOptions,
) -> Generator[SetupStep, None, None]:
    """
    Loads config and constructs the search engine, LLM client and agent.

    Centralizes the model/collection selection and agent construction shared
    by the CLI and Streamlit app, so this logic only needs to change (and be
    tested) in one place.

    Args:
        options (AgentOptions): User-facing choices (model, agent type, etc.)

    Returns:
        Generator[SetupStep]: One "status" step per initialization stage,
            followed by either a "ready" step (with the built agent attached)
            or an "error" step if construction failed.
    """
    try:
        config = cast(DictConfig, OmegaConf.load(options.config_path))

        # Select the embedding model: base by default, fine-tuned only when requested
        model_name = (
            config.finetuned_model_path
            if options.finetuned
            else config.model_name
        )

        # The tool-loop agent explores this repo's own source via read_file/list_directory/grep,
        # so it must search the self-indexed collection rather than the CoSQA corpus
        db_collection = (
            config.qdrant.self_repo_collection
            if options.agent_type == "tool-loop"
            else config.qdrant.full_collection
        )

        yield SetupStep(
            "status",
            f"Loading Search Engine (Qdrant + SBERT), using "
            f"{'fine-tuned' if options.finetuned else 'base'} model: {model_name}",
        )
        engine = CodeSearchEngine(
            model_name=model_name,
            db_collection=db_collection,
            db_path=config.qdrant.storage_path,
            device=config.get("device", "auto"),
        )

        yield SetupStep("status", f"Loading LLM (Ollama/{options.model})")
        llm = LLMClient(model_name=options.model)

        if options.agent_type == "tool-loop":
            yield SetupStep("status", "Initializing Tool-Choosing Agent")
            llm_config = OmegaConf.load(options.llm_config_path)
            agent: CodingAgent | ToolCallingAgent = ToolCallingAgent(
                engine, llm, llm_config
            )
        else:
            yield SetupStep("status", "Initializing Self-Correcting Code Agent")
            agent = CodingAgent(engine, llm)

        yield SetupStep("ready", "System Ready.", agent=agent)

    except Exception as e:
        yield SetupStep("error", str(e))

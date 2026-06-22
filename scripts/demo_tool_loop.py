from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.agent.core import ToolCallingAgent
from cse.agent.llm import LLMClient
from cse.search_engine.engine import CodeSearchEngine

QUERIES = [
    "How does resolve_device pick a compute device?",
    # "Which files are in the src/cse/agent directory?",
]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--llm-config", type=str, default="config/llm_config.yaml"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    llm_config = OmegaConf.load(args.llm_config)

    engine = CodeSearchEngine(
        model_name=config.model_name,
        db_collection=config.qdrant.self_repo_collection,
        db_path=config.qdrant.storage_path,
        device=config.get("device", "auto"),
        quiet=True,
    )
    llm = LLMClient(model_name=args.model)
    agent = ToolCallingAgent(engine, llm, llm_config)

    for query in QUERIES:
        print(f"\n=== User: {query} ===")
        for step in agent.solve(query):
            print(f"[{step.step_type.upper()}] {step.content}")


if __name__ == "__main__":
    main()

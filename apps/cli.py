import time
from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.agent.core import CodingAgent
from cse.agent.llm import LLMClient
from cse.search_engine.engine import CodeSearchEngine


def main():
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Search using the fine-tuned model, defaults to the base model.",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Select the embedding model: base by default, fine-tuned only when requested
    model_name = (
        config.finetuned_model_path if args.finetuned else config.model_name
    )

    # Initialize components
    try:
        print(
            f"--- Loading Search Engine (Qdrant + SBERT), using {'fine-tuned' if args.finetuned else 'base'} model: {model_name} ---"
        )
        engine = CodeSearchEngine(
            model_name=model_name,
            db_collection=config.qdrant.full_collection,
            db_path=config.qdrant.storage_path,
            device=config.get("device", "auto"),
        )

        print("\n--- Loading LLM (Ollama/Phi-3) ---")
        llm = LLMClient(model_name="phi3")

        print("--- Initializing Self-Correcting Code Agent ---")
        agent = CodingAgent(engine, llm)
        print("System Ready.\n")

    except Exception as e:
        print(f"Initialization Failed: {e}")
        exit(-1)

    # Interactive loop
    while True:
        # Get user query
        user_query = input("\n>> Ask a coding question (or 'exit'/'quit'): ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("-" * 50)
        start_time = time.time()

        # Print all steps the agent takes to answer
        for step in agent.solve(user_query):
            if step.step_type == "plan":
                print(f"[PLAN] {step.content}")

            elif step.step_type == "search":
                print(f"[SEARCH] {step.content}")
                if step.data:
                    print(f"    Found {len(step.data)} snippets.")

            elif step.step_type == "critique":
                print(f"[CRITIQUE] {step.content}")

            elif step.step_type == "answer":
                print(f"\n[ANSWER]:\n{step.content}")

            elif step.step_type == "error":
                print(f"[ERROR] {step.content}")

        print(f"\n(Time elapsed: {time.time() - start_time:.2f}s)")
        print("-" * 50)


if __name__ == "__main__":
    main()

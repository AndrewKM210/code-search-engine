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
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Initialize components
    try:
        print("--- Loading Search Engine (Qdrant + SBERT) ---")
        engine = CodeSearchEngine(
            model_name=config.finetuned_model_path,
            db_collection=config.qdrant.full_collection,
            db_path=config.qdrant.storage_path,
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

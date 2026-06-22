import time
from argparse import ArgumentParser

from cse.agent.presenter import describe_step
from cse.agent.setup import AgentOptions, build_agent


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Search using the fine-tuned model, defaults to the base model.",
    )
    parser.add_argument(
        "--agent",
        choices=["baseline", "tool-loop"],
        default="baseline",
        help=(
            "baseline: fixed plan->search->critique pipeline. "
            "tool-loop: LLM chooses tools (search/read/list/grep) in a loop."
        ),
    )
    parser.add_argument(
        "--model", type=str, default="phi3", help="Ollama model for reasoning."
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default="config/llm_config.yaml",
        help="Per-model tool-calling capabilities, used by --agent tool-loop.",
    )
    args = parser.parse_args()

    options = AgentOptions(
        finetuned=args.finetuned,
        agent_type=args.agent,
        model=args.model,
        config_path=args.config,
        llm_config_path=args.llm_config,
    )

    # Initialize components
    agent = None
    for setup_step in build_agent(options):
        if setup_step.step_type == "status":
            print(f"--- {setup_step.content} ---")
        elif setup_step.step_type == "ready":
            agent = setup_step.agent
        elif setup_step.step_type == "error":
            print(f"Initialization Failed: {setup_step.content}")
            exit(-1)
    print("System Ready.\n")

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
            label, text = describe_step(step)
            if step.step_type == "search" and step.data:
                print(f"[{label}] {text}")
                print(f"    Found {len(step.data)} snippets.")
            elif step.step_type == "answer":
                print(f"\n[{label}]:\n{text}")
            else:
                print(f"[{label}] {text}")

        print(f"\n(Time elapsed: {time.time() - start_time:.2f}s)")
        print("-" * 50)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.agent.core import CodingAgent, ToolCallingAgent
from cse.agent.llm import LLMClient
from cse.eval.harness import (
    load_gold_set,
    run_eval,
    save_results,
    subset_gold_set,
    summarize,
)
from cse.eval.judge import LLMJudge
from cse.search_engine.engine import CodeSearchEngine


def main():
    # Parse arguments and config files
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--llm-config", type=str, default="config/llm_config.yaml"
    )
    parser.add_argument(
        "--gold-set", type=str, default="data/eval/gold_qa.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi3",
        help="Ollama model used by both agents.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Ollama model used as the judge, defaults to --model.",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Search using the fine-tuned embeddings model, defaults to the base model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max tool-call steps the tool-loop agent gets per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the gold set to the first N questions per category, for "
            "fast trial runs. Defaults to the full gold set."
        ),
    )
    parser.add_argument(
        "--out", type=str, default="results/generation_eval.csv"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    llm_config = OmegaConf.load(args.llm_config)

    # Select the embedding model: base by default, fine-tuned only when requested
    model_name = (
        config.finetuned_model_path if args.finetuned else config.model_name
    )
    print(
        f"--- Loading Search Engine, using "
        f"{'fine-tuned' if args.finetuned else 'base'} model: {model_name} ---"
    )
    engine = CodeSearchEngine(
        model_name=model_name,
        db_collection=config.qdrant.self_repo_collection,
        db_path=config.qdrant.storage_path,
        device=config.get("device", "auto"),
    )

    print(f"--- Loading LLM (Ollama/{args.model}) ---")
    llm = LLMClient(model_name=args.model)

    agents: dict[str, CodingAgent | ToolCallingAgent] = {
        "baseline": CodingAgent(engine, llm),
        "tool-loop": ToolCallingAgent(
            engine, llm, llm_config, max_steps=args.max_steps
        ),
    }

    # Reuse the agents' LLM as the judge unless a separate one is requested
    judge_model = args.judge_model or args.model
    print(f"--- Loading Judge (Ollama/{judge_model}) ---")
    judge_llm = (
        llm if judge_model == args.model else LLMClient(model_name=judge_model)
    )
    judge = LLMJudge(judge_llm)

    gold_set = subset_gold_set(load_gold_set(args.gold_set), args.limit)
    print(
        f"--- Running eval: {len(gold_set)} questions x {len(agents)} agents ---"
    )
    results = run_eval(gold_set, agents, judge)

    save_results(results, args.out)
    print(f"Saved {len(results)} rows to {args.out}")

    print("\n--- Task success ---")
    for agent_label, metrics in summarize(results).items():
        print(
            f"{agent_label}: correctness={metrics['correctness']:.2%}, "
            f"faithfulness={metrics['faithfulness']:.2%}"
        )


if __name__ == "__main__":
    main()

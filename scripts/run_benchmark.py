import csv
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

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
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--llm-config", type=str, default="config/llm_config.yaml"
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="config/benchmark_config.yaml",
    )
    parser.add_argument(
        "--gold-set", type=str, default="data/eval/gold_qa.yaml"
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Search using the fine-tuned embeddings model, defaults to the base model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override benchmark_config.limit: cap to N questions per category.",
    )
    parser.add_argument("--out-dir", type=str, default="results/")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    llm_config = OmegaConf.load(args.llm_config)
    bench = OmegaConf.load(args.benchmark_config)

    model_name = (
        config.finetuned_model_path if args.finetuned else config.model_name
    )
    print(
        f"--- Loading Search Engine "
        f"({'fine-tuned' if args.finetuned else 'base'} model: {model_name}) ---"
    )
    engine = CodeSearchEngine(
        model_name=model_name,
        db_collection=config.qdrant.self_repo_collection,
        db_path=config.qdrant.storage_path,
        device=config.get("device", "auto"),
    )

    limit = args.limit if args.limit is not None else bench.get("limit")
    gold_set = subset_gold_set(load_gold_set(args.gold_set), limit)
    print(f"--- Gold set: {len(gold_set)} questions ---")

    judge_model_name = bench.judge_model
    print(f"--- Loading Judge (Ollama/{judge_model_name}) ---")
    judge = LLMJudge(LLMClient(model_name=judge_model_name))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []

    for run in bench.runs:
        run_name = run.name
        model = run.model
        agent_type = run.agent
        nudges = run.get("nudges", True)

        print(
            f"\n--- Run: {run_name} "
            f"(model={model}, agent={agent_type}"
            + (f", nudges={nudges}" if agent_type == "tool-loop" else "")
            + ") ---"
        )
        llm = LLMClient(model_name=model)

        if agent_type == "baseline":
            agent = CodingAgent(engine, llm)
        else:
            agent = ToolCallingAgent(
                engine,
                llm,
                llm_config,
                max_steps=bench.max_steps,
                use_nudges=nudges,
            )

        results = run_eval(gold_set, {run_name: agent}, judge)
        all_results.extend(results)

        metrics = summarize(results)[run_name]
        summary_rows.append(
            {
                "timestamp": timestamp,
                "run": run_name,
                "model": model,
                "agent": agent_type,
                "nudges": ""
                if agent_type == "baseline"
                else str(nudges).lower(),
                "n_questions": len(results),
                "correctness_pct": round(metrics["correctness"] * 100, 1),
                "faithfulness_pct": round(metrics["faithfulness"] * 100, 1),
            }
        )
        print(
            f"  correctness={metrics['correctness']:.2%}, "
            f"faithfulness={metrics['faithfulness']:.2%}"
        )

    full_path = out_dir / f"{timestamp}_benchmark_full.csv"
    save_results(all_results, str(full_path))
    print(f"\nFull results saved to {full_path}")

    summary_path = out_dir / f"{timestamp}_benchmark.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to {summary_path}")

    print("\n--- Benchmark Summary ---")
    for row in summary_rows:
        print(
            f"  {row['run']}: "
            f"correctness={row['correctness_pct']}%, "
            f"faithfulness={row['faithfulness_pct']}%"
        )


if __name__ == "__main__":
    main()

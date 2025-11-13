import os
from argparse import ArgumentParser
import pandas as pd
from omegaconf import OmegaConf
from search_engine.evaluation import prepare_cosqa_data, run_evaluation


def main():
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Prepare data
    corpus, eval_queries = prepare_cosqa_data()

    # Evaluate base model
    base_metrics = run_evaluation(
        model_name=config.model_name,
        corpus=corpus,
        eval_queries=eval_queries,
        db_collection=config.qdrant.eval_base_collection,
        db_path=config.qdrant.storage_path,
    )

    # Evaluate fine-tuned model
    if not os.path.exists(config.finetuned_model_path):
        print(f"Fine-tuned model not found at '{config.finetuned_model_path}'.")
        print("Please run 'python fine_tune.py' first.")
        finetuned_metrics = {k: 0 for k in base_metrics}  # Empty results
    else:
        finetuned_metrics = run_evaluation(
            model_name=config.finetuned_model_path,
            corpus=corpus,
            eval_queries=eval_queries,
            db_collection=config.qdrant.eval_finetuned_collection,
            db_path=config.qdrant.storage_path,
        )

    # Compare and store results
    print("\n--- Final Comparison ---")
    df = pd.DataFrame([base_metrics, finetuned_metrics], index=["Base Model", "Fine-Tuned Model"])
    print(df.to_markdown(floatfmt=".4f"))
    out_path = "results/evaluation.csv"
    if not os.path.exists(os.path.split(out_path)[0]):
        os.mkdir(os.path.split(out_path)[0])
    df.to_csv(out_path, index_label="Model")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

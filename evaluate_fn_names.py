import os
from argparse import ArgumentParser
import pandas as pd
from omegaconf import OmegaConf
from src.search_engine.evaluation import prepare_cosqa_data, run_evaluation


def main():
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Prepare data without function names
    corpus, eval_queries = prepare_cosqa_data(fn_names=False)

    # Evaluate fine-tuned model
    if not os.path.exists(config.finetuned_model_path):
        print(f"Fine-tuned model not found at '{config.finetuned_model_path}'.")
        print("Please run 'python fine_tune.py' first.")
        return exit(-1)
    else:
        finetuned_metrics = run_evaluation(
            model_name=config.finetuned_model_path,
            corpus=corpus,
            eval_queries=eval_queries,
            db_collection=config.qdrant.eval_finetuned_collection,
            db_path=config.qdrant.storage_path,
        )

    # Prepare data with function names
    corpus, eval_queries = prepare_cosqa_data(fn_names=True)

    # Evaluate fine-tuned model on function names
    if not os.path.exists(config.finetuned_model_path):
        print(f"Fine-tuned model not found at '{config.finetuned_model_path}'.")
        print("Please run 'python fine_tune.py --fn_names' first.")
        return exit(-1)
    else:
        fn_names_metrics = run_evaluation(
            model_name=config.finetuned_model_path + "_fn_names",
            corpus=corpus,
            eval_queries=eval_queries,
            db_collection=config.qdrant.eval_finetuned_collection,
            db_path=config.qdrant.storage_path,
        )

    # Compare and store results
    print("\n--- Final Comparison ---")
    df = pd.DataFrame(
        [finetuned_metrics, fn_names_metrics], index=["Fine-Tuned Model", "Fine-Tuned Model - Only Function Names"]
    )
    print(df.to_markdown(floatfmt=".4f"))
    out_path = "results/evaluation_fn_names.csv"
    if not os.path.exists(os.path.split(out_path)[0]):
        os.mkdir(os.path.split(out_path)[0])
    df.to_csv(out_path, index_label="Model")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

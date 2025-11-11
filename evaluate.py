from argparse import ArgumentParser
from omegaconf import OmegaConf
import time
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from search_engine.engine import CodeSearchEngine
import pandas as pd

METRICS_K = 10  # Evaluate metrics @ 10


def prepare_cosqa_data() -> (dict, dict):
    """
    Loads the CoSQA 'validation' split and prepares it for retrieval evaluation.

    Returns:
        (dict, dict): Maps id -> code snippet, maps query -> id.
    """
    # TODO: ensure this is an equivalent dataset
    # dataset: list of {idx, doc, code, code_tokens, docstring_tokens, label}
    print("Loading CoSQA 'validation' split...")
    dataset = load_dataset("gonglinyuan/CoSQA", split="validation")

    print("Building unique code corpus...")
    code_snippets = set(item["code"] for item in tqdm(dataset, desc="Reading code"))

    corpus = {i: code for i, code in enumerate(code_snippets)}
    code_to_id = {code: i for i, code in corpus.items()}
    print(f"Prepared Corpus: {len(corpus)} unique code snippets.")

    eval_queries = {}

    print("Building validation set...")
    for item in tqdm(dataset, desc="Mapping queries"):
        if item["label"] == 1: # Only evaluate on positive samples
            query = item["doc"]
            code = item["code"]
            ground_truth_code_id = code_to_id[code]

            if query not in eval_queries:
                eval_queries[query] = []
            if ground_truth_code_id not in eval_queries[query]:
                eval_queries[query].append(ground_truth_code_id)

    print(f"Prepared Evaluation Set: {len(eval_queries)} unique queries.")
    return corpus, eval_queries


def run_evaluation(model_name: str, corpus: dict, eval_queries: dict, db_collection: str, db_path: str) -> dict:
    """
    Runs the full evaluation for a given model and returns metrics.

    Args:
        model_name (str): The name or path of the Hugging Face sentence-transformer model.
        corpus (dict): Maps id -> code snippet.
        eval_queries (dict): Maps query -> id.
        db_collection (str): The name of the Qdrant collection.
        db_path (str): The path for Qdrant's on-disk storage.

    Returns:
        dict: Maps metric -> value.
    """
    print(f"\n--- Starting Evaluation for: {model_name} ---")

    # Initialize and index the engine
    engine = CodeSearchEngine(model_name=model_name, db_collection=db_collection, db_path=db_path)
    engine.index_corpus(corpus)

    # Run evaluation loop
    mrr_at_k_scores = []
    ndcg_at_k_scores = []
    recall_at_k_scores = []

    total_queries = len(eval_queries)

    start_time = time.time()

    for query, ground_truth_ids in tqdm(eval_queries.items(), desc="Evaluating queries"):
        results = engine.search(query, k=METRICS_K)

        # MRR@k and Recall@k
        rank = 0
        for j, res in enumerate(results):
            if res["code_id"] in ground_truth_ids:
                rank = j + 1
                break

        mrr_at_k = (1 / rank) if rank > 0 else 0
        recall_at_k = 1 if rank > 0 else 0

        mrr_at_k_scores.append(mrr_at_k)
        recall_at_k_scores.append(recall_at_k)

        # nDCG@k
        dcg_at_k = 0
        relevance = [1 if res["code_id"] in ground_truth_ids else 0 for res in results]

        for j, rel in enumerate(relevance):
            dcg_at_k += rel / np.log2(j + 2)  # (j+1) rank, +1 for log base

        idcg_at_k = 0
        num_ground_truth = len(ground_truth_ids)
        num_ideal_hits = min(num_ground_truth, METRICS_K)
        for j in range(num_ideal_hits):
            idcg_at_k += 1 / np.log2(j + 2)

        ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0
        ndcg_at_k_scores.append(ndcg_at_k)

    end_time = time.time()

    # Calculate final mean metrics
    metrics = {
        f"MRR@{METRICS_K}": np.mean(mrr_at_k_scores),
        f"nDCG@{METRICS_K}": np.mean(ndcg_at_k_scores),
        f"Recall@{METRICS_K}": np.mean(recall_at_k_scores),
        "Avg. Query Time (ms)": ((end_time - start_time) / total_queries) * 1000,
    }

    return metrics


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
        db_collection=config.qdrant.eval_base_collection,
        corpus=corpus,
        eval_queries=eval_queries,
        db_path=config.qdrant.storage_path,
    )

    # Compare results
    print("\n--- Final Comparison ---")
    df = pd.DataFrame([base_metrics], index=["Base Model"])
    print(df.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()

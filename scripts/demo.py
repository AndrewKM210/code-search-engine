from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.search_engine.engine import CodeSearchEngine


def run_demo(
    code_dir, model_name, db_collection, db_path, chunk_size, chunk_overlap
):
    print("--- Starting Code Search Demo ---")

    # Initialize the search engine
    engine = CodeSearchEngine(
        model_name=model_name, db_collection=db_collection, db_path=db_path
    )

    # Index the collection of documents
    engine.index_from_directory(code_dir, chunk_size, chunk_overlap)

    # Define test queries
    test_queries = [
        "a python function for sorting a list",
        "how to get a user with javascript",
        "calculate the mean value in python",
    ]

    # Run test queries
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = engine.search(query, k=2)

        for i, res in enumerate(results):
            print(f"  Result {i + 1} (Score: {res['score']:.4f})")
            print(f"  Source: {res['payload']['source']}")
            print("  " + "-" * 30)
            preview = "\n  ".join(res["payload"]["content"].splitlines()[:5])
            print(f"  {preview}...")
            print("  " + "=" * 30)

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument("--sample_code", type=str, default="sample_code")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Run demo
    run_demo(
        args.sample_code,
        config.model_name,
        config.qdrant.demo_collection,
        config.qdrant.storage_path,
        config.splitter.chunk_size,
        config.splitter.chunk_overlap,
    )

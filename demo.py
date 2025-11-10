from search_engine.engine import CodeSearchEngine

# --- Configuration ---
# Use a standard, non-fine-tuned model for the demo
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_code_collection"
SAMPLE_CODE_DIR = "sample_code"


def run_demo():
    print("--- Starting Code Search Demo ---")

    # Initialize the Search Engine
    engine = CodeSearchEngine(
        model_name=MODEL_NAME, collection_name=COLLECTION_NAME, storage_path="./qdrant_storage_demo"
    )

    # Index the collection of documents
    engine.index_from_directory(SAMPLE_CODE_DIR)

    # Define Test Queries
    test_queries = [
        "a python function for sorting a list",
        "how to get a user with javascript",
        "calculate the mean value in python",
    ]

    # Run Test Queries
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
    run_demo()

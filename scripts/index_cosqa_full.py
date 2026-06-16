from argparse import ArgumentParser
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from cse.search_engine.engine import CodeSearchEngine


def prepare_cosqa_dataset() -> dict:
    """
    Loads the CoSQA dataset.

    Returns:
        dict: Maps id -> code snippet
    """
    # TODO: ensure this is an equivalent dataset
    # dataset: list of {idx, doc, code, code_tokens, docstring_tokens, label}
    print("Loading CoSQA 'train' split...")
    dataset = load_dataset("gonglinyuan/CoSQA", split="train")

    print("Building unique code corpus...")
    code_snippets = set(item["code"] for item in tqdm(dataset, desc="Reading code"))
    corpus = {i: code for i, code in enumerate(code_snippets)}

    print(f"Prepared Corpus: {len(corpus)} unique code snippets.")
    return corpus


def main():
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Index using the fine-tuned model, defaults to the base model.",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Select the embedding model: base by default, fine-tuned only when requested
    model_name = config.finetuned_model_path if args.finetuned else config.model_name
    print(f"Using {'fine-tuned' if args.finetuned else 'base'} model: {model_name}")

    print("--- Preparing Full CoSQA Dataset ---")
    corpus = prepare_cosqa_dataset()

    print("\n--- Initializing Code Search Engine ---")
    engine = CodeSearchEngine(
        model_name,
        db_collection=config.qdrant.full_collection,
        db_path=config.qdrant.storage_path,
    )

    print("\n--- Indexing Full CoSQA Dataset ---")
    engine.index_corpus(corpus)


if __name__ == "__main__":
    main()

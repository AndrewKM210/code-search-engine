from argparse import ArgumentParser

from omegaconf import OmegaConf

from cse.search_engine.engine import CodeSearchEngine


def main():
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=".",
        help="Path to this repository's root.",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Index using the fine-tuned model, defaults to the base model.",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Select the embedding model: base by default, fine-tuned only when requested
    model_name = (
        config.finetuned_model_path if args.finetuned else config.model_name
    )
    print(
        f"Using {'fine-tuned' if args.finetuned else 'base'} model: {model_name}"
    )

    print("\n--- Initializing Code Search Engine ---")
    engine = CodeSearchEngine(
        model_name,
        db_collection=config.qdrant.self_repo_collection,
        db_path=config.qdrant.storage_path,
        device=config.get("device", "auto"),
        # Always rebuild from scratch: this script has no incremental/diffing
        # logic, so a partial upsert would leave stale points around for any
        # file that was since renamed, moved or deleted
        db_recreate=True,
    )

    print("\n--- Indexing Repository Source (.py files) ---")
    engine.index_from_directory(
        args.repo_dir,
        chunk_size=config.splitter.chunk_size,
        chunk_overlap=config.splitter.chunk_overlap,
        glob="**/*.py",
    )


if __name__ == "__main__":
    main()

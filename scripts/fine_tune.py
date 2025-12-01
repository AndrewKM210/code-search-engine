import os
from argparse import ArgumentParser
import mlflow
import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from cse.search_engine.utils import extract_function_name


def get_fine_tuning_data(fn_names: bool = False):
    """
    Loads CoSQA train split and formats it for MNR loss.

    Returns:
        Dataset: Contains "anchor" and "positive" columns for training with MNR loss.
    """
    print("Loading CoSQA 'train' split for fine-tuning...")
    dataset = load_dataset("gonglinyuan/CoSQA", split="train")

    print("Formatting training examples...")
    dataset = dataset.filter(
        lambda example: example["label"] == 1,  # Only keep positive pairs
        num_proc=4,
    )

    if fn_names:
        # Extract function names
        print("Only using function names.")
        dataset = dataset.map(
            lambda example: {"function_name": extract_function_name(example["code"])},
            num_proc=4,
        )

        # Rename columns to what the loss function expects: "anchor" and "positive"
        dataset = dataset.rename_columns({"doc": "anchor", "function_name": "positive"}).select_columns(
            ["anchor", "positive"]
        )
    else:
        # Rename columns to what the loss function expects: "anchor" and "positive"
        print("Using whole code snippets.")
        dataset = dataset.rename_columns({"doc": "anchor", "code": "positive"}).select_columns(["anchor", "positive"])


    print(f"Created {len(dataset)} positive training examples.")
    return dataset


def run_fine_tuning(
    base_model: str,
    batch_size: int,
    num_epochs: int,
    max_train_steps: int,
    tuned_model_path: str,
    fn_names: bool = False,
):
    """
    Finetunes the base model and stores it.

    Args:
        base_model (str): Name of HuggingFace base model.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        max_train_steps (int): Maximum number of training steps per epoch.
        tuned_model_path (str): Path to store finetuned model.
    """
    print(f"Starting fine-tuning process based on '{base_model}'...")

    # Get training data
    train_dataset = get_fine_tuning_data(fn_names)

    # Load base model
    model = SentenceTransformer(base_model)

    # Define the loss function
    # TODO: test with other loss functions
    train_loss = MultipleNegativesRankingLoss(model=model)

    # Training the model
    mlflow.set_experiment("finetuning")
    with mlflow.start_run(run_name="finetune_run") as run:
        mlflow.log_param("base_model", base_model)
        print(f"\n--- Starting training... Logging to MLflow run: {run.info.run_id} ---")
        args = SentenceTransformerTrainingArguments(
            output_dir=tuned_model_path + "_checkpoints",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            max_steps=max_train_steps,
            warmup_ratio=0.1,
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MNR loss benefits from no duplicates
            logging_strategy="steps",
            logging_steps=10,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=train_loss,
            args=args,
        )
        trainer.train()

        trainer.save_model(tuned_model_path)
        print(f"\nFine-tuning complete. Model saved to '{tuned_model_path}'")

    store_loss_from_mlflow(run.info.run_id, fn_names)


def store_loss_from_mlflow(run_id, fn_names):
    """
    Fetches training loss from MLflow and stores it in results/losses.csv

    Args:
        run_id (str): The id of a mlflow run.
    """
    print(f"Fetching loss history for run: {run_id}")
    try:
        # Get the metrics from the MLflow client
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(run_id, "loss")
        if not metrics:
            print("Could not find 'loss' metric in MLflow.")
            return

        # Save loss to csv
        df = pd.DataFrame({"Step": [m.step for m in metrics], "Loss": [m.value for m in metrics]})
        out_path = "results/losses.csv" if not fn_names else "results/losses_fn_names.csv"
        if not os.path.exists(os.path.split(out_path)[0]):
            os.mkdir(os.path.split(out_path)[0])
        df.to_csv(out_path, index=False)
        print(f"Saved train losses to {out_path}.")

    except Exception as e:
        print(f"Error fetching from MLflow: {e}")


if __name__ == "__main__":
    # Parse arguments and config file
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main_config.yaml")
    parser.add_argument("--fn_names", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    finetuned_path = config.finetuned_model_path
    if args.fn_names:
        finetuned_path += "_fn_names"

    # Train model if it does not already exist
    if os.path.exists(finetuned_path):
        print("Fine-tuned model already exists. Skipping training.")
    else:
        run_fine_tuning(
            config.model_name,
            config.training.batch_size,
            config.training.num_epochs,
            config.training.max_train_steps,
            finetuned_path,
            args.fn_names,
        )

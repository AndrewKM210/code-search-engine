# Local AI Self-Correcting RAG Code Assistant

[![Tests](https://github.com/AndrewKM210/code-search-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/AndrewKM210/code-search-engine/actions/workflows/tests.yml)
[![Python 3.10.18](https://img.shields.io/badge/Python-3.10.18-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-2E8B57?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-F3E838?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-0096C6?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-385073?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

This project implements a Retrieval-Augmented Generation (RAG) agent designed to answer programming queries by utilizing a semantic vector database and a local Large Language Model (LLM) with a self-correction mechanism.

> **Active development:** this project is being grown from a one-shot RAG assistant into a
> full **agentic coding assistant**: an LLM that chooses tools (semantic search, read file,
> list directory, grep) to navigate a codebase, measured against the original pipeline as a
> baseline rather than assumed to be better. The work is tracked milestone by milestone in
> [ROADMAP.md](ROADMAP.md): M0 (harden the base) is done and M1 (agentic loop + tools) is in
> progress, with generation evals, an MCP server, a local/API model benchmark, and a QLoRA
> fine-tune capstone planned next.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/demo_dark.gif">
  <img src="assets/demo_light.gif" alt="Animated demonstration of the agent in action.">
</picture>

## Project Overview

The primary challenge in code Q&A is retrieving semantically relevant, high-quality code snippets from large, unstructured repositories. This project addresses this by:

1.  **Semantic Indexing:** Using a fine-tuned Sentence Transformer model to index the CoSQA dataset into a dense vector space (Qdrant).
2.  **Agentic Workflow:** Implementing a multi-step execution loop that generates initial search queries, evaluates the retrieval results, critiques itself, and refines the search query before synthesizing the final answer.
3.  **Local LLM Integration:** Utilizing **Ollama (Phi-3)** to ensure low-latency reasoning and data privacy without reliance on external commercial APIs.

### Technical Stack

| Category | Component | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | Python 3.10.18 | Core language |
| **Embedding Model**| SBERT / MiniLM-L6 | Generates high-dimensional semantic vectors |
| **Vector DB** | Qdrant | Vector storage and retrieval |
| **LLM Inference** | Ollama / Phi-3 | Local reasoning |
| **Agent Framework** | Custom Classes / LangChain | Manages the multi-step execution flow |
| **Frontend** | Streamlit | Interactive UI |
| **Packaging** | `pyproject.toml` / Setuptools | Dependency and project management |

### Architecture and Agentic Loop

The system operates based on a clear separation of concerns, managed by the central **`Agent`** class:

1.  **Plan:** The LLM receives the user query and generates an optimized search string for the vector database.
2.  **Retrieve:** The `CodeSearchEngine` queries Qdrant to retrieve top-k code snippets.
3.  **Critique:** The LLM receives the original query and the retrieved context. It determines if the context is sufficient to form a definitive answer.
4.  **Execute/Retry:**
    * **If Sufficient:** The LLM synthesizes the final, explained answer.
    * **If Insufficient:** The LLM generates a new, refined search query, and the process repeats (up to a defined limit).


## Getting Started

### Prerequisites
 
1.  **Ollama:** Install the Ollama model locally, advanced setup instructions can be found [here](https://docs.ollama.com/linux#manual-install).
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull phi3
    ```
2. **Python:** Create virtual environment with pyenv (suggested, but optional):
    ```bash
    pyenv virtualenv 3.10.18 code_search
    pyenv activate code_search
    ```

### Installation

Navigate to the project root directory and install the project in editable mode:

```bash
pip install -e .
```

For development (running tests, notebooks, plots), install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

### Initial Setup (Indexing the Data and Fine-Tuning the SBERT model)

Before running the agent, you must index the code dataset into Qdrant.

1.  **Configure:** Check and edit configuration in `config/main_config.yaml`.
2. **Index the CoSQA Dataset:**
    ```bash
      python scripts/index_cosqa_full.py
    ```
2.  **Execute Indexing Script:**
    ```bash
    python scripts/fine_tune.py
    ```

## Usage

### Running the Agent UI

Launch the Streamlit frontend to interact with the self-correcting agent:

```bash
streamlit run apps/streamlit_app.py
```

# Additional

This project also focuses on fine-tuning a pre-trained sentence transformer model (MiniLM-L6-v2) using the CoSQA (Code Search and Question Answering) dataset. The goal is to enhance the model's ability to semantically map natural language queries to relevant code snippets, outperforming the base model in retrieval metrics like MRR, nDCG and Recall. The retrieval engine is built using Qdrant for efficient vector indexing and search.

### Fine-tuning Model on CoSQA

The `fine_tune.py` script fine-tunes the sentence transformer model used in the demo on code snippets of the training split of the [CoSQA dataset](https://huggingface.co/datasets/gonglinyuan/CoSQA):

```bash
python scripts/fine_tune.py
```

Tracking is done with `mlflow` and the log of the loss during training will be stored in the `results/losses.csv` file. MLflow will automatically store all training parameters and metrics in `mlruns` and be visualized with the command:

```bash
mlflow ui
```

The loss can be plotted as shown in the `report.ipynb` notebook. An example loss plot of training for one epoch on the whole dataset is shown below. Even though the loss is noisy, the average loss does decrease over time.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/loss_ft_dark.png">
  <img src="assets/loss_ft_light.png" alt="Fine-tuning loss.">
</picture>

### CoSQA Evaluation

The `scripts/evaluate_finetuning.py` script evaluates both the base and finetuned model on the validation split of the CoSQA dataset, calculating metrics such as the MRR@10, nDCG@10 and Recall@10:

`python scripts/evaluate_finetuning.py`

The resulting metrics are stored in `results/evaluation.csv` and can be plotted for comparison as shown in the `report.ipynb` notebook. The following plot shows how the finetuned model outperforms the base model in all metrics, as well as the improvement gain in terms of percentage:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/eval_ft_dark.png">
  <img src="assets/eval_ft_light.png" alt="Fine-tuning evaluation.">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/improvement_ft_dark.png">
  <img src="assets/improvement_ft_light.png" alt="Fine-tuning improvement.">
</picture>


### Fine-tuning on Function Names

Out of curiosity, the model was also fine-tuned using the same CoSQA dataset but only on the function names. The fine-tuning and evaluation can be done by running the following commands:

```bash
python scripts/fine_tune.py --fn_names
python scipts/evaluate_fn_names.py
```

The results will be stored in `results/losses_fn_names.csv` and `results/evaluation_fn_names.csv`, and can be plotted as shown in the `report.ipynb` notebook. The following plots show that using only function names leads to a significantly higher training loss and worse evaluation metrics. However, given that the amount of tokens stored in the database is now lower, query time is reduced.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/loss_fn_dark.png">
  <img src="assets/loss_fn_light.png" alt="Fine-tuning with function names loss.">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/eval_fn_dark.png">
  <img src="assets/eval_fn_light.png" alt="Fine-tuning with function names evaluation.">
</picture>





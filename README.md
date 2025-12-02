# Local AI Self-Correcting RAG Code Assistant

[![Python 3.10.18](https://img.shields.io/badge/Python-3.10.18-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-2E8B57?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-F3E838?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-0096C6?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-385073?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

This project implements a Retrieval-Augmented Generation (RAG) agent designed to answer programming queries by utilizing a semantic vector database and a local Large Language Model (LLM) with a self-correction mechanism.

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

Navigate to the project root directory and install dependencies:

1.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install Project in Editable Mode:**
    ```bash
    pip install -e .
    ```

### Initial Setup (Indexing the Data and Fine-Tuning the SBERT model)

Before running the agent, you must index the code dataset into Qdrant.

1.  **Configure:** Check and edit configuration in `config/main_config.yaml`.
2. **Index the CoSQA dataset:**
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
streamlit run frontend.py
```





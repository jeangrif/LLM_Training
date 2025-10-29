> ⚠️ **Note**  
> This README was partially generated and refined with the help of an LLM (ChatGPT).  
> Always verify configuration paths and environment variables before running the pipeline.

## 1️⃣ Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Hydra** for configuration management.

It provides:
- A **data preparation workflow** (download → preprocessing → augmentation)
- A **Hydra-driven modular pipeline** for model inference or training
- A clean, configurable structure for reproducible experiments
## 2️⃣ Installation

This project uses **[uv](https://github.com/astral-sh/uv)** to manage Python environments efficiently.

```bash
# Create and synchronize the environment
uv sync

# Activate the environment
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
```
If you prefer to install manually with pip:
```bash
pip install -r requirements.txt
```
## 3️⃣ Project Structure Overview

```text
src/
├── components/                    # Core building blocks used by stages/pipeline
│   ├── embed/                     # Embedding backends & helpers
│   ├── llm/                       # LLM backends (base classes, llama.cpp, etc.)
│   ├── retriever/                 # Retrievers (dense FAISS, BM25, hybrid…)
│   ├── __init__.py
│   ├── generator.py               # LLM selection/wrapper exposing a unified `generate()`
│   └── reranker.py                # (Optional) re-ranking of retrieved passages
│
├── eval/                          # Evaluation logic (quality) & system performance
│   ├── metrics/                   # Exact match, semantic similarity, lexical overlap, faithfulness…
│   ├── performance/               # Latency meter / runtime KPIs
│   └── __init__.py
│
├── optimization/
│   └── optuna_search.py           # Optuna study for hyperparameter tuning
│
├── pipeline/
│   └── orchestrator.py            # Orchestrates the dynamic stage flow (Hydra-driven)
│
├── rag/
│   ├── __init__.py
│   └── engine.py                # RAG core: retrieve → (rerank) → generate for one query
│
├── scripts/                       # Ad-hoc utilities / one-off scripts (placeholder)
│
├── stages/                        # Executable units run by the orchestrator (sequentially)
│   ├── __init__.py
│   ├── check_models.py            # Verify/download required models
│   ├── index_manager.py           # Build/load search indexes
│   ├── rag_runner.py                  # Batch RAG run over dataset (uses rag.pipeline)
│   └── evaluator.py               # Compute evaluation metrics & persist results
│
├── ui/
│   └── app.py                     # Minimal app/CLI/UI to drive the pipeline
│
├── utils/
│
└── main.py                        #entry point                      
```
## 4️⃣ Data Preparation

Before running the Hydra pipeline, you must first prepare the dataset.  
This step downloads a dataset from **Hugging Face**, extracts **question–answer pairs**, and generates **paraphrased variations** for evaluation.

### 🔹 Run the full preparation process
```bash
python src/scripts/prepare_data.py
```
This command automatically executes the three preparation steps in sequence:

| Step | Script | Output |
|------|---------|--------|
| 1️⃣ Ingest dataset | `scripts/ingest_hf_dataset.py` | `data/raw/<dataset>_<split>.parquet` |
| 2️⃣ Build base eval | `scripts/build_eval_base.py` | `data/eval/base_questions.jsonl` |
| 3️⃣ Generate paraphrases | `scripts/generate_paraphrases.py` | `data/eval/augmented_questions.jsonl` |

Example output:

✅ data/raw/squad_train.parquet  
✅ data/eval/base_questions.jsonl  
✅ data/eval/augmented_questions.jsonl
## 5️⃣ Configuration for Data Preparation

Each script inside the `scripts/` folder contains a small configuration block named `CONFIG` at the very top of the file.  
This block centralizes all editable parameters — you can modify dataset names, paths, or model options directly here.

### 🔹 Example – `scripts/ingest_hf_dataset.py`
```bash
CONFIG = {
    "hf_dataset_name": "squad",
    "hf_dataset_split": "train",
    "text_field": "context",
    "raw_dir": Path("data/raw"),
}
```
### 🔹 Example – `scripts/generate_paraphrases.py`
```bash

CONFIG = {
    "model_name": "eugenesiow/bart-paraphrase",
    "input_path": Path("data/eval/base_questions.jsonl"),
    "output_path": Path("data/eval/augmented_questions.jsonl"),
    "limit": 200,
    "device_preference": "auto",
}
```
👉 No `.env` file or external configuration is required.  
All settings are explicit, versioned, and easy to track directly inside each script.
## 6️⃣ Running the Hydra Pipeline

Once the dataset is prepared, you can launch the main pipeline directly.

### 🔹 Default run
```bash
python -m src.main
```
Hydra will automatically handle:
- Output directory creation under `outputs/`
- Logging (`main.log`)
- Configuration management through YAML files
- Modular execution of each stage (Preprocessing, Model, Evaluation, etc.)
## 7️⃣ Modifying the Pipeline Configuration

All parameters are defined in the `configs/` directory.  
Edit `pipeline.yaml` or the submodules (e.g. `llm/settings.yaml`, `embed/settings.yaml`, `model/llama_cpp.yaml`) to adjust models, paths, or modes before running `python -m src.main`.
7️⃣ Modifying the Pipeline Configuration

----------------------------------------

All parameters are defined in the configs/ directory.The main file configs/pipeline.yaml defines the stages and modules of the pipeline.

### Example structure

*   **defaults** : lists all submodules (model, llm, embed, rerank, eval, latency)

*   **stages** : defines the execution order (check\_models, check\_data, run\_rag, evaluate)

### Stage overview

*   **check\_models** → verifies or downloads all required models

*   **check\_data** → checks or builds the FAISS index and embeddings

*   **run\_rag** → runs the main RAG process on the augmented questions

*   **evaluate** → computes and saves RAG performance metrics

You can modify any parameter (model paths, chunk size, overlap, top\_k, etc.) directly in configs/pipeline.yaml before running:python -m src.main

## 🧭 Next Steps — RAG Pipeline Improvements

| # | Category | Task | Status |
|:-:|-----------|-------|:------:|
| **1** | 🗂️ **Improve folder structure** |  | ☐ |
| **2** | ⚙️ **Improve config file** |  | ☐ |
| **3** | 🧠 **Improve Rerank module** |  | ☐ |
| **4** | 🧪 **Continue Optuna script** |  | ☐ |
| **5** | 🧩 **Improve App.py** |  | ☐ |

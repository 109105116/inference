# RNA-Ligand Inference Pipeline

This directory contains the scripts and tools for the RNA-Ligand binding affinity prediction pipeline. The pipeline handles feature generation, consolidation, and model training.

## Quick Start

### Run the full pipeline (Development Mode)
Run a fast test with small batch sizes and few epochs:
```bash
bash run_full_pipeline.sh --dev
```

### Run the full pipeline (Production)
```bash
bash run_full_pipeline.sh
```

### Run Inference on New Data
```bash
bash run_inference.sh \
    --input data/test_inference.csv \
    --output data/inference_results \
    --model-dir models/inference_ensemble
```
The output `predictions.csv` will contain predictions (pKd) and uncertainty (sigma).

## Directory Structure

### Core Scripts
- **`run_full_pipeline.sh`**: The main entry point. Orchestrates the entire flow:
    1. Sets up environments (`setup_colab.sh`).
    2. Generates features (`generate_all_features.py`).
    3. Consolidates features into efficiency pickles (`consolidate_features.py`).
    4. Trains the inference model (`train_inference_model.py`).
- **`setup_colab.sh`**: Handles Conda environment creation and caching. Designed for Colab but works locally.
- **`setup_envs.sh`**: Lower-level environment setup script.

### Python Modules
- **`generate_all_features.py`**: Orchestrator for feature generation. Calls specific scripts in `rna/` and `mol/`.
    - `rna/`: Scripts for RiNALMo, MXFold2, PSSM.
    - `mol/`: Scripts for UniMol, Graph Features.
- **`consolidate_features.py`**: Merges scattered feature files (npy, json) into unified pickle dictionaries for fast loading during training. Handles data cleaning (e.g., Unimol truncation fix).
- **`train_inference_model.py`**: Trains the `RMPred` ensemble model using the consolidated data.

### Environments
The pipeline uses isolated Conda environments to avoid dependency conflicts:
- `pipeline`: Orchestration, pandas, numpy.
- `rinalmo`: RiNALMo RNA embeddings (Flash Attention).
- `unimol`: UniMol molecule embeddings.
- `training`: PyTorch, PyG, RDKit for training the final model.

## Data Flow
1. **Input**: `data/union_dataset_final.csv` (SMILES, Sequences, Labels)
2. **Features**: `data/processed/` (Raw generated features)
3. **Consolidated**: `data/consolidated/` (Pickles: `rna_embed.pkl`, `mole_graph.pkl`, etc.)
4. **Model**: `models/inference_ensemble/` (Saved `.pt` weights)

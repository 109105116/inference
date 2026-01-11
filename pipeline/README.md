# RNA-Ligand Binding Affinity Prediction Pipeline

Complete feature generation pipeline for RNA-ligand binding affinity prediction.

## Features Generated

### RNA Features
| Feature | Description | Shape | File Format |
|---------|-------------|-------|-------------|
| **RiNALMo** | Pre-trained RNA language model embeddings | L × 1280 | `.npy` |
| **mxfold2** | Secondary structure (dot-bracket + base pairs) | - | `.json` |
| **PSSM** | Position-Specific Scoring Matrix from MSA | L × 4 | `.npy` |
| **One-hot** | Basic nucleotide encoding | L × 4 | `.npy` |

### Molecule Features
| Feature | Description | Shape | File Format |
|---------|-------------|-------|-------------|
| **UniMol CLS** | Global molecular representation | 512 | `.npy` |
| **UniMol Atomic** | Per-atom representations | N × 512 | `.npy` |
| **One-hot** | Atom type encoding | N × 14 | `.npy` |
| **Graph** | Atom features + bond edges | N × 22 | `.npz` |

## Quick Start

```bash
# 1. Setup environments (one-time)
cd pipeline
./setup_envs.sh

# 2. Download RNAcentral for PSSM (optional, ~9GB)
cd ../databases
nohup ./download_rnacentral.sh > download.log 2>&1 &

# 3. Generate all features
cd ../pipeline
python generate_all_features.py

# 4. Verify features
python verify_model_inputs.py
```

## Conda Environments

| Environment | Tool | Python | Key Dependencies |
|-------------|------|--------|------------------|
| `rinalmo` | RiNALMo | 3.11 | PyTorch 2.1, flash-attn 2.3.2 |
| `mxfold2` | mxfold2 | 3.10 | PyTorch, mxfold2 (local) |
| `mmseqs2` | MMseqs2 | 3.10 | mmseqs2 18.8 (bioconda) |
| `unimol` | UniMol | 3.10 | unimol_tools (pip) |

## Directory Structure

```
inference/
├── pipeline/
│   ├── generate_all_features.py  # Main orchestrator
│   ├── verify_model_inputs.py    # Verify & create manifest
│   ├── test_pssm.py              # Test PSSM generation
│   ├── setup_envs.sh             # Environment setup
│   ├── rna/
│   │   ├── generate_rinalmo_embeddings.py
│   │   ├── generate_secondary_structure.py
│   │   └── generate_msa_pssm.py
│   └── mol/
│       ├── generate_unimol_embeddings.py
│       └── generate_mol_features.py
├── databases/
│   ├── download_rnacentral.sh    # Database setup script
│   └── rnacentral_db/            # MMseqs2 database (after setup)
├── data/
│   ├── processed/                # Generated features
│   │   ├── rinalmo/
│   │   ├── mxfold2/
│   │   ├── pssm/
│   │   ├── rna_onehot/
│   │   ├── unimol/
│   │   └── mol_features/
│   ├── model_ready_dataset.csv   # Final training dataset
│   └── feature_manifest.json     # Feature paths
├── rinalmo/                      # RiNALMo source (patched)
├── mxfold2/                      # mxfold2 source
└── mmseq2/                       # MMseqs2 source (optional)
```

Verify all environments and input files are ready:

```bash
python run_pipeline.py --mode check
```

### 3. (Optional) Download RNAcentral Database

For MSA-based PSSM generation, you can download the RNAcentral database:

```bash
# Download RNAcentral (large, ~20GB)
wget https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
gunzip rnacentral_active.fasta.gz

# Create MMseqs2 database
conda activate mmseqs2_env
mmseqs createdb rnacentral_active.fasta rnacentral_db
mmseqs createindex rnacentral_db tmp
```

## Usage

### Generate All Features

```bash
python run_pipeline.py --mode all
```

### Generate RNA Features Only

```bash
python run_pipeline.py --mode rna
```

### Generate Molecule Features Only

```bash
python run_pipeline.py --mode mol
```

### Generate Specific Feature Types

```bash
python run_pipeline.py --mode rna-embed    # RiNALMo only
python run_pipeline.py --mode rna-ss       # Secondary structure only
python run_pipeline.py --mode rna-pssm     # PSSM only
python run_pipeline.py --mode mol-embed    # UniMol only
python run_pipeline.py --mode mol-graph    # Molecule graphs only
```

### With MMseqs2 Database for MSA

```bash
python run_pipeline.py --mode rna-pssm --mmseqs2-db /path/to/rnacentral_db
```

### Regenerate Features

```bash
python run_pipeline.py --mode all --overwrite
```

## Output

### Build Feature Manifest

After running the pipeline, build a manifest of all generated features:

```bash
python build_manifest.py --output manifest.json
```

This creates a JSON file with paths to all features and statistics on coverage.

### Output Directory Structure

```
data/
├── union_dataset.csv       # Input: RNA-ligand pairs with pKd
├── unique_rnas.csv         # Input: Unique RNA sequences
├── unique_molecules.csv    # Input: Unique molecules
├── manifest.json           # Feature manifest
└── processed/
    ├── rna/
    │   ├── embeddings/     # *.npy (L x D) RiNALMo embeddings
    │   ├── secondary/      # *.json with dot-bracket and edges
    │   ├── msa/            # *.a3m MSA files (if database used)
    │   ├── pssm/           # *.npy (L x 4) PSSM matrices
    │   └── onehot/         # *.npy (L x 4) one-hot encodings
    └── mol/
        ├── embeddings/
        │   ├── cls/        # *.npy (D,) molecule-level embeddings
        │   └── atomic/     # *.npy (N x D) atomic embeddings
        ├── onehot/         # *.npy (N x 14) atom type one-hot
        └── graph/          # *.npz with edges and features
```

## Individual Script Usage

Each script can be run independently within its environment:

### RiNALMo Embeddings
```bash
conda activate rinalmo_env
python rna/generate_rinalmo_embeddings.py \
    --input ../data/unique_rnas.csv \
    --output ../data/processed/rna/embeddings
```

### Secondary Structure
```bash
conda activate mxfold2_env
python rna/generate_secondary_structure.py \
    --input ../data/unique_rnas.csv \
    --output ../data/processed/rna/secondary
```

### PSSM
```bash
conda activate mmseqs2_env
python rna/generate_msa_pssm.py \
    --input ../data/unique_rnas.csv \
    --output-msa ../data/processed/rna/msa \
    --output-pssm ../data/processed/rna/pssm \
    --output-onehot ../data/processed/rna/onehot
```

### UniMol Embeddings
```bash
conda activate unimol_env
python mol/generate_unimol_embeddings.py \
    --input ../data/unique_molecules.csv \
    --output ../data/processed/mol/embeddings
```

### Molecule Features
```bash
conda activate unimol_env
python mol/generate_mol_features.py \
    --input ../data/unique_molecules.csv \
    --output-onehot ../data/processed/mol/onehot \
    --output-graph ../data/processed/mol/graph
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in individual scripts:
```bash
python rna/generate_rinalmo_embeddings.py --batch-size 2 ...
```

### Long RNA Sequences
RiNALMo handles sequences up to 4096 by default. Longer sequences are automatically chunked and merged. Adjust with:
```bash
python rna/generate_rinalmo_embeddings.py --max-length 2048 ...
```

### Missing MSA
If no RNAcentral database is available, PSSM will be generated from sequence only (essentially smoothed one-hot encoding).

## Next Steps

After generating features, use them to:
1. Train the model on the union dataset
2. Run inference on new RNA-ligand pairs
3. Evaluate binding affinity predictions with epistemic uncertainty

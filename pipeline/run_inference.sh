#!/bin/bash
set -e

# =============================================================================
# Run Inference on New Data
# =============================================================================
# Usage:
#   bash run_inference.sh --input test.csv --output results/ --model-dir models/inference_ensemble
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"

# Default defaults
INPUT_CSV=""
OUTPUT_DIR=""
MODEL_DIR=""
# For A100, increase BATCH_SIZE (e.g., 32 or 64) for faster inference.
BATCH_SIZE=2
GPU=0

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_CSV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Resolve absolute paths
INPUT_CSV=$(readlink -f "$INPUT_CSV")
OUTPUT_DIR=$(readlink -f "$OUTPUT_DIR")
MODEL_DIR=$(readlink -f "$MODEL_DIR")

if [ -z "$INPUT_CSV" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL_DIR" ]; then
    echo "Usage: bash run_inference.sh --input <csv> --output <dir> --model-dir <dir> [--batch-size <N>]"
    exit 1
fi

echo "=========================================="
echo "RNA-Ligand Inference Pipeline"
echo "=========================================="
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_DIR"
echo "Models: $MODEL_DIR"
echo ""

# Setup directories
TEMP_FEATURES="$OUTPUT_DIR/features"
TEMP_CONSOLIDATED="$OUTPUT_DIR/consolidated"
mkdir -p "$TEMP_FEATURES"
mkdir -p "$TEMP_CONSOLIDATED"

# Initialize conda
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
source "$CONDA_DIR/bin/activate"

# 1. Generate Features
echo "Step 1: Generating Features..."
conda activate pipeline
python -u "$SCRIPT_DIR/generate_all_features.py" \
    --input "$INPUT_CSV" \
    --output "$TEMP_FEATURES" \
    --gpu "$GPU" \
    --rinalmo-batch-size 2 \
    --unimol-batch-size 8 # Conservative defaults for inference

conda deactivate

# 2. Consolidate
echo ""
echo "Step 2: Consolidating Features..."
conda activate training
python -u "$SCRIPT_DIR/consolidate_features.py" \
    --input "$INPUT_CSV" \
    --features "$TEMP_FEATURES" \
    --output "$TEMP_CONSOLIDATED"

# Fix: Symlink PSSM directory to consolidated so loader can find it
# Google Drive doesn't support symlinks well, so we copy.
rm -rf "$TEMP_CONSOLIDATED/pssm"
cp -r "$TEMP_FEATURES/pssm" "$TEMP_CONSOLIDATED/pssm"

# 3. Predict output
echo ""
echo "Step 3: Running Prediction..."
# conda activate training (Already active)
python -u "$SCRIPT_DIR/predict_inference.py" \
    --data "$TEMP_CONSOLIDATED" \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU"

conda deactivate

echo ""
echo "✅ Inference Completed."
echo "Results: $OUTPUT_DIR/predictions.csv"

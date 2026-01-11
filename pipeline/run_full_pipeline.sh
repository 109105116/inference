#!/bin/bash
# =============================================================================
# One-Click Pipeline Runner
# =============================================================================
# This script sets up the environment (if needed) and runs the full feature
# generation pipeline. Perfect for Google Colab one-click execution.
#
# Usage:
#   bash run_full_pipeline.sh
#   bash run_full_pipeline.sh --skip-setup  # Skip env setup if already done
#
# =============================================================================

set -e

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$INFERENCE_DIR/data"
OUTPUT_DIR="$DATA_DIR/processed"

echo "=========================================="
echo "RNA-Ligand Feature Generation Pipeline"
echo "=========================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Parse arguments
SKIP_SETUP=false
for arg in "$@"; do
    case $arg in
        --skip-setup) SKIP_SETUP=true ;;
    esac
done

# -----------------------------------------------------------------------------
# Step 1: Environment Setup
# -----------------------------------------------------------------------------
if [ "$SKIP_SETUP" = false ]; then
    echo "Step 1: Setting up conda environments..."
    bash "$SCRIPT_DIR/setup_colab.sh"
else
    echo "Step 1: Skipping setup (--skip-setup)"
fi

# Initialize conda
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# -----------------------------------------------------------------------------
# Step 2: Generate All Features
# -----------------------------------------------------------------------------
echo ""
echo "Step 2: Generating all features..."
echo ""

# Check input files exist
if [ ! -f "$DATA_DIR/union_dataset.csv" ]; then
    echo "❌ Error: union_dataset.csv not found in $DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the feature generation
python "$SCRIPT_DIR/generate_all_features.py" \
    --input "$DATA_DIR/union_dataset_final.csv" \
    --output "$OUTPUT_DIR" \
    --gpu 0

# -----------------------------------------------------------------------------
# Step 3: Verify Features
# -----------------------------------------------------------------------------
echo ""
echo "Step 3: Verifying generated features..."
echo ""

# Count files in each directory
echo "Feature file counts:"
for dir in rinalmo mxfold2 pssm rna_onehot unimol mol_onehot mol_graph; do
    if [ -d "$OUTPUT_DIR/$dir" ]; then
        count=$(find "$OUTPUT_DIR/$dir" -type f -name "*.npy" -o -name "*.json" -o -name "*.npz" 2>/dev/null | wc -l)
        echo "  $dir: $count files"
    else
        echo "  $dir: (not found)"
    fi
done

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Features saved to: $OUTPUT_DIR"
echo ""

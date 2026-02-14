#!/bin/bash
# Setup script for creating isolated conda environments for each tool
# Run this script once to set up all required environments
#
# Usage:
#   bash setup_envs.sh           # Setup all environments
#   bash setup_envs.sh rinalmo   # Setup only rinalmo
#   bash setup_envs.sh unimol    # Setup only unimol
#   bash setup_envs.sh mxfold2   # Setup only mxfold2
#   bash setup_envs.sh mmseqs2   # Setup only mmseqs2

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
ENVS_DIR="$SCRIPT_DIR/envs"

echo "=========================================="
echo "RNA-Ligand Binding Pipeline Environment Setup"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Inference directory: $INFERENCE_DIR"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Parse arguments
SETUP_ALL=true
SETUP_RINALMO=false
SETUP_MXFOLD2=false
SETUP_UNIMOL=false
SETUP_MMSEQS2=false

if [ $# -gt 0 ]; then
    SETUP_ALL=false
    for arg in "$@"; do
        case $arg in
            rinalmo) SETUP_RINALMO=true ;;
            mxfold2) SETUP_MXFOLD2=true ;;
            unimol) SETUP_UNIMOL=true ;;
            mmseqs2) SETUP_MMSEQS2=true ;;
            all) SETUP_ALL=true ;;
            *) echo "Unknown environment: $arg"; exit 1 ;;
        esac
    done
fi

# ============================================
# 1. RiNALMo environment
# ============================================
setup_rinalmo() {
    echo ""
    echo "============================================"
    echo "Setting up RiNALMo environment..."
    echo "============================================"
    
    if conda env list | grep -q "^rinalmo "; then
        echo "rinalmo environment already exists."
        read -p "Recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n rinalmo -y
        else
            echo "Skipping rinalmo setup."
            return
        fi
    fi
    
    echo "Creating rinalmo environment..."
    conda env create -f "$ENVS_DIR/rinalmo_env.yml"
    
    echo "RiNALMo environment setup complete!"
}

# ============================================
# 2. mxfold2 environment
# ============================================
setup_mxfold2() {
    echo ""
    echo "============================================"
    echo "Setting up mxfold2 environment..."
    echo "============================================"
    
    if conda env list | grep -q "^mxfold2 "; then
        echo "mxfold2 environment already exists."
        read -p "Recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n mxfold2 -y
        else
            echo "Skipping mxfold2 setup."
            return
        fi
    fi
    
    echo "Creating mxfold2 environment..."
    conda env create -f "$ENVS_DIR/mxfold2_env.yml"
    
    echo "mxfold2 environment setup complete!"
}

# ============================================
# 3. UniMol environment
# ============================================
setup_unimol() {
    echo ""
    echo "============================================"
    echo "Setting up UniMol environment..."
    echo "============================================"
    
    if conda env list | grep -q "^unimol "; then
        echo "unimol environment already exists."
        read -p "Recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n unimol -y
        else
            echo "Skipping unimol setup."
            return
        fi
    fi
    
    echo "Creating UniMol environment..."
    conda env create -f "$ENVS_DIR/unimol_env.yml"
    
    echo "UniMol environment setup complete!"
}

# ============================================
# 4. MMseqs2 environment
# ============================================
setup_mmseqs2() {
    echo ""
    echo "============================================"
    echo "Setting up MMseqs2 environment..."
    echo "============================================"
    
    if conda env list | grep -q "^mmseqs2 "; then
        echo "mmseqs2 environment already exists."
        read -p "Recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n mmseqs2 -y
        else
            echo "Skipping mmseqs2 setup."
            return
        fi
    fi
    
    echo "Creating MMseqs2 environment..."
    conda env create -f "$ENVS_DIR/mmseqs2_env.yml"
    
    echo "MMseqs2 environment setup complete!"
}

# ============================================
# Run setup based on arguments
# ============================================
if [ "$SETUP_ALL" = true ]; then
    setup_rinalmo
    setup_mxfold2
    setup_unimol
    setup_mmseqs2
else
    [ "$SETUP_RINALMO" = true ] && setup_rinalmo
    [ "$SETUP_MXFOLD2" = true ] && setup_mxfold2
    [ "$SETUP_UNIMOL" = true ] && setup_unimol
    [ "$SETUP_MMSEQS2" = true ] && setup_mmseqs2
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available environments:"
echo "  - rinalmo  : RiNALMo RNA embeddings (Python 3.11, PyTorch 2.1, CUDA 11.8)"
echo "  - mxfold2  : mxfold2 secondary structure (Python 3.10, PyTorch 2.1)"
echo "  - unimol   : UniMol molecule embeddings (Python 3.10, PyTorch 2.1)"
echo "  - mmseqs2  : MMseqs2 MSA search (Python 3.10)"
echo ""
echo "To activate an environment, use:"
echo "  conda activate <env_name>"
echo ""
echo "To test the setup, run:"
echo "  python test_pipeline.py"
echo ""

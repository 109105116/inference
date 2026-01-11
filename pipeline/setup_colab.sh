#!/bin/bash
# =============================================================================
# Colab-Friendly Environment Setup Script
# =============================================================================
# This script sets up all required conda environments for the RNA-ligand 
# binding affinity prediction pipeline. Designed for Google Colab VMs which
# restart each session (only Drive files persist).
#
# Usage:
#   bash setup_colab.sh           # Setup all environments
#   bash setup_colab.sh rinalmo   # Setup only rinalmo
#   bash setup_colab.sh unimol    # Setup only unimol
#   bash setup_colab.sh mxfold2   # Setup only mxfold2
#   bash setup_colab.sh mmseqs2   # Setup only mmseqs2
#   bash setup_colab.sh check     # Just check what's installed
#
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
ENVS_DIR="$SCRIPT_DIR/envs"

# Conda installation path (persistent on Colab via Drive or local)
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"

echo "=========================================="
echo "RNA-Ligand Pipeline Setup"
echo "=========================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Inference directory: $INFERENCE_DIR"
echo "Conda directory: $CONDA_DIR"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check/Install Miniconda
# -----------------------------------------------------------------------------
install_miniconda() {
    if [ -f "$CONDA_BIN" ]; then
        echo "✅ Miniconda already installed at $CONDA_DIR"
        return 0
    fi
    
    echo "📦 Installing Miniconda..."
    
    # Download Miniconda installer
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
    
    # Install silently
    bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
    rm "$MINICONDA_INSTALLER"
    
    # Initialize conda
    eval "$($CONDA_BIN shell.bash hook)"
    $CONDA_BIN init bash 2>/dev/null || true
    
    echo "✅ Miniconda installed at $CONDA_DIR"
}

# Initialize conda for this shell session
init_conda() {
    if [ ! -f "$CONDA_BIN" ]; then
        # Check if conda is in PATH
        if command -v conda &> /dev/null; then
            CONDA_BIN=$(command -v conda)
            echo "✅ Using system conda at $CONDA_BIN"
        else
            echo "❌ Conda not found. Installing..."
            install_miniconda
        fi
    fi
    # Only eval if we haven't already
    if [ -z "$CONDA_PREFIX" ]; then
        eval "$($CONDA_BIN shell.bash hook)"
    fi
}

# Check if environment exists
env_exists() {
    $CONDA_BIN env list 2>/dev/null | grep -q "^$1 "
}

# -----------------------------------------------------------------------------
# Environment Setup Functions
# -----------------------------------------------------------------------------

setup_rinalmo() {
    echo ""
    echo "============================================"
    echo "Setting up RiNALMo environment..."
    echo "============================================"
    
    if env_exists "rinalmo"; then
        echo "rinalmo environment already exists. Skipping."
        return 0
    fi
    
    # Locate RiNALMo source
    # Check reference folder first
    if [ -f "$INFERENCE_DIR/reference/rinalmo/environment.yml" ]; then
        RINALMO_DIR="$INFERENCE_DIR/reference/rinalmo"
        echo "✅ Found RiNALMo in reference: $RINALMO_DIR"
    elif [ -f "$INFERENCE_DIR/rinalmo/environment.yml" ]; then
        RINALMO_DIR="$INFERENCE_DIR/rinalmo"
        echo "✅ Found RiNALMo in root: $RINALMO_DIR"
    else
        echo "⚠️  RiNALMo source not found in workspace. Cloning..."
        cd "$INFERENCE_DIR"
        git clone https://github.com/lbcb-sci/RiNALMo.git rinalmo
        RINALMO_DIR="$INFERENCE_DIR/rinalmo"
    fi
    
    # Create environment from RiNALMo's environment.yml
    echo "Creating rinalmo environment (Python 3.11, PyTorch 2.1, CUDA 11.8)..."
    # Note: We use the yml from the source
    $CONDA_BIN env create -f "$RINALMO_DIR/environment.yml" -y
    
    # Install RiNALMo package
    echo "Installing RiNALMo package..."
    $CONDA_BIN run -n rinalmo pip install "$RINALMO_DIR"
    
    echo "✅ RiNALMo environment setup complete!"
}

setup_mxfold2() {
    echo ""
    echo "============================================"
    echo "Setting up mxfold2 environment..."
    echo "============================================"
    
    if env_exists "mxfold2"; then
        echo "mxfold2 environment already exists. Skipping."
        return 0
    fi
    
    echo "Creating mxfold2 environment..."
    $CONDA_BIN env create -f "$ENVS_DIR/mxfold2_env.yml" -y
    
    # Install mxfold2 via pip 
    echo "Installing mxfold2 package..."
    $CONDA_BIN run -n mxfold2 pip install mxfold2
    
    echo "✅ mxfold2 environment setup complete!"
}

setup_unimol() {
    echo ""
    echo "============================================"
    echo "Setting up UniMol environment..."
    echo "============================================"
    
    if env_exists "unimol"; then
        echo "unimol environment already exists. Skipping."
        return 0
    fi
    
    echo "Creating UniMol environment..."
    $CONDA_BIN env create -f "$ENVS_DIR/unimol_env.yml" -y
    
    # unimol_tools is installed via pip in the yaml file, so we are good
    
    echo "✅ UniMol environment setup complete!"
}

setup_mmseqs2() {
    echo ""
    echo "============================================"
    echo "Setting up MMseqs2 environment..."
    echo "============================================"
    
    if env_exists "mmseqs2"; then
        echo "mmseqs2 environment already exists. Skipping."
        return 0
    fi
    
    echo "Creating MMseqs2 environment..."
    $CONDA_BIN env create -f "$ENVS_DIR/mmseqs2_env.yml" -y
    
    echo "✅ MMseqs2 environment setup complete!"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

# Initialize conda
init_conda

# Parse arguments
SETUP_ALL=true
if [ $# -gt 0 ]; then
    SETUP_ALL=false
    for arg in "$@"; do
        case $arg in
            rinalmo) setup_rinalmo ;;
            mxfold2) setup_mxfold2 ;;
            unimol) setup_unimol ;;
            mmseqs2) setup_mmseqs2 ;;
            check) $CONDA_BIN env list ;;
            all) SETUP_ALL=true ;;
        esac
    done
fi

if [ "$SETUP_ALL" = true ]; then
    setup_rinalmo
    setup_mxfold2
    setup_unimol
    setup_mmseqs2
fi

echo ""
echo "=========================================="
echo "Setup Process Finished!"
echo "=========================================="

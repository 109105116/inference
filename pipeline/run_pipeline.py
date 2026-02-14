#!/usr/bin/env python3
"""
Main orchestrator for the RNA-ligand binding affinity prediction pipeline.
This script coordinates feature generation across different conda environments.

Usage:
    python run_pipeline.py --mode all          # Generate all features
    python run_pipeline.py --mode rna          # Generate RNA features only
    python run_pipeline.py --mode mol          # Generate molecule features only
    python run_pipeline.py --mode rna-embed    # Generate RiNALMo embeddings only
    python run_pipeline.py --mode rna-ss       # Generate secondary structure only
    python run_pipeline.py --mode rna-pssm     # Generate PSSM only
    python run_pipeline.py --mode mol-embed    # Generate UniMol embeddings only
    python run_pipeline.py --mode mol-graph    # Generate molecule graphs only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import json
import shutil

# Determine script locations
SCRIPT_DIR = Path(__file__).parent.absolute()
RNA_SCRIPTS_DIR = SCRIPT_DIR / "rna"
MOL_SCRIPTS_DIR = SCRIPT_DIR / "mol"


@dataclass
class PipelinePaths:
    """Paths for the pipeline."""
    # Input data
    unique_rnas: Path
    unique_mols: Path
    union_dataset: Path
    
    # Output directories
    rna_embeddings: Path
    rna_secondary: Path
    rna_msa: Path
    rna_pssm: Path
    rna_onehot: Path
    mol_embeddings: Path
    mol_onehot: Path
    mol_graph: Path


def get_default_paths() -> PipelinePaths:
    """Get default paths based on directory structure."""
    inference_dir = SCRIPT_DIR.parent
    data_dir = inference_dir / "data"
    processed_dir = data_dir / "processed"
    
    return PipelinePaths(
        unique_rnas=data_dir / "unique_rnas.csv",
        unique_mols=data_dir / "unique_molecules.csv",
        union_dataset=data_dir / "union_dataset.csv",
        rna_embeddings=processed_dir / "rna" / "embeddings",
        rna_secondary=processed_dir / "rna" / "secondary",
        rna_msa=processed_dir / "rna" / "msa",
        rna_pssm=processed_dir / "rna" / "pssm",
        rna_onehot=processed_dir / "rna" / "onehot",
        mol_embeddings=processed_dir / "mol" / "embeddings",
        mol_onehot=processed_dir / "mol" / "onehot",
        mol_graph=processed_dir / "mol" / "graph",
    )


def run_in_conda_env(env_name: str, script: Path, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a Python script within a conda environment.
    
    Args:
        env_name: Name of the conda environment
        script: Path to the Python script
        args: Arguments to pass to the script
        check: Whether to raise exception on failure
    """
    # Build the command
    cmd = f"conda run -n {env_name} python {script} " + " ".join(args)
    
    print(f"\n{'='*60}")
    print(f"Running in conda env: {env_name}")
    print(f"Script: {script.name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=False,
        text=True,
    )
    
    if check and result.returncode != 0:
        raise RuntimeError(f"Script failed with return code {result.returncode}")
    
    return result


def generate_rinalmo_embeddings(paths: PipelinePaths, overwrite: bool = False):
    """Generate RiNALMo embeddings for RNA sequences."""
    script = RNA_SCRIPTS_DIR / "generate_rinalmo_embeddings.py"
    
    args = [
        "--input", str(paths.unique_rnas),
        "--output", str(paths.rna_embeddings),
    ]
    if overwrite:
        args.append("--overwrite")
    
    run_in_conda_env("rinalmo_env", script, args)


def generate_secondary_structure(paths: PipelinePaths, overwrite: bool = False):
    """Generate secondary structure predictions."""
    script = RNA_SCRIPTS_DIR / "generate_secondary_structure.py"
    
    args = [
        "--input", str(paths.unique_rnas),
        "--output", str(paths.rna_secondary),
    ]
    if overwrite:
        args.append("--overwrite")
    
    run_in_conda_env("mxfold2_env", script, args)


def generate_msa_pssm(paths: PipelinePaths, database_path: Optional[str] = None, overwrite: bool = False):
    """Generate MSA and PSSM for RNA sequences."""
    script = RNA_SCRIPTS_DIR / "generate_msa_pssm.py"
    
    args = [
        "--input", str(paths.unique_rnas),
        "--output-msa", str(paths.rna_msa),
        "--output-pssm", str(paths.rna_pssm),
        "--output-onehot", str(paths.rna_onehot),
    ]
    if database_path:
        args.extend(["--database", database_path])
    if overwrite:
        args.append("--overwrite")
    
    run_in_conda_env("mmseqs2_env", script, args)


def generate_unimol_embeddings(paths: PipelinePaths, overwrite: bool = False):
    """Generate UniMol embeddings for molecules."""
    script = MOL_SCRIPTS_DIR / "generate_unimol_embeddings.py"
    
    args = [
        "--input", str(paths.unique_mols),
        "--output", str(paths.mol_embeddings),
    ]
    if overwrite:
        args.append("--overwrite")
    
    run_in_conda_env("unimol_env", script, args)


def generate_mol_features(paths: PipelinePaths, overwrite: bool = False):
    """Generate molecule one-hot and graph features."""
    script = MOL_SCRIPTS_DIR / "generate_mol_features.py"
    
    args = [
        "--input", str(paths.unique_mols),
        "--output-onehot", str(paths.mol_onehot),
        "--output-graph", str(paths.mol_graph),
    ]
    if overwrite:
        args.append("--overwrite")
    
    # This can run in any environment with RDKit
    run_in_conda_env("unimol_env", script, args)


def check_conda_envs() -> List[str]:
    """Check which conda environments exist."""
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    
    existing = []
    for line in result.stdout.split("\n"):
        for env in ["rinalmo_env", "mxfold2_env", "mmseqs2_env", "unimol_env"]:
            if env in line:
                existing.append(env)
    
    return existing


def check_input_files(paths: PipelinePaths) -> bool:
    """Check if required input files exist."""
    missing = []
    if not paths.unique_rnas.exists():
        missing.append(str(paths.unique_rnas))
    if not paths.unique_mols.exists():
        missing.append(str(paths.unique_mols))
    
    if missing:
        print("ERROR: Missing input files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate RNA-ligand binding affinity prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  all          Generate all features (RNA + molecule)
  rna          Generate all RNA features
  mol          Generate all molecule features
  rna-embed    Generate RiNALMo embeddings only
  rna-ss       Generate secondary structure only
  rna-pssm     Generate MSA and PSSM only
  mol-embed    Generate UniMol embeddings only
  mol-graph    Generate molecule graphs and one-hot only
  check        Check environment setup

Examples:
  python run_pipeline.py --mode all
  python run_pipeline.py --mode rna --overwrite
  python run_pipeline.py --mode check
        """
    )
    parser.add_argument("--mode", "-m", required=True,
                        choices=["all", "rna", "mol", "rna-embed", "rna-ss", "rna-pssm", 
                                "mol-embed", "mol-graph", "check"],
                        help="Pipeline mode")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing features")
    parser.add_argument("--mmseqs2-db", help="Path to MMseqs2 database for MSA search")
    parser.add_argument("--data-dir", help="Custom data directory (default: ../data)")
    
    args = parser.parse_args()
    
    # Get paths
    paths = get_default_paths()
    
    # Check mode
    if args.mode == "check":
        print("Checking pipeline setup...")
        print("\nConda environments:")
        existing_envs = check_conda_envs()
        required_envs = ["rinalmo_env", "mxfold2_env", "mmseqs2_env", "unimol_env"]
        for env in required_envs:
            status = "✓" if env in existing_envs else "✗"
            print(f"  {status} {env}")
        
        print("\nInput files:")
        for name, path in [("unique_rnas", paths.unique_rnas), 
                          ("unique_mols", paths.unique_mols),
                          ("union_dataset", paths.union_dataset)]:
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {path}")
        
        print("\nTo set up missing environments, run:")
        print(f"  bash {SCRIPT_DIR / 'setup_envs.sh'}")
        return
    
    # Check input files
    if not check_input_files(paths):
        sys.exit(1)
    
    # Create output directories
    for path in [paths.rna_embeddings, paths.rna_secondary, paths.rna_msa,
                 paths.rna_pssm, paths.rna_onehot, paths.mol_embeddings,
                 paths.mol_onehot, paths.mol_graph]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Run requested pipeline components
    try:
        if args.mode in ["all", "rna", "rna-embed"]:
            generate_rinalmo_embeddings(paths, args.overwrite)
        
        if args.mode in ["all", "rna", "rna-ss"]:
            generate_secondary_structure(paths, args.overwrite)
        
        if args.mode in ["all", "rna", "rna-pssm"]:
            generate_msa_pssm(paths, args.mmseqs2_db, args.overwrite)
        
        if args.mode in ["all", "mol", "mol-embed"]:
            generate_unimol_embeddings(paths, args.overwrite)
        
        if args.mode in ["all", "mol", "mol-graph"]:
            generate_mol_features(paths, args.overwrite)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

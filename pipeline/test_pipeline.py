#!/usr/bin/env python3
"""
Test the pipeline components with a small sample of data.
This can be run to verify the pipeline structure without full environment setup.

For testing with actual tools, run individual scripts in their respective environments.
"""

import sys
import os
from pathlib import Path

# Add pipeline to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import numpy as np
import json


def test_config():
    """Test configuration loading."""
    print("Testing config...")
    from config import (
        PIPELINE_DIR, DATA_DIR, PROCESSED_DIR,
        ensure_directories, PipelineConfig
    )
    
    assert PIPELINE_DIR.exists(), f"Pipeline dir not found: {PIPELINE_DIR}"
    assert DATA_DIR.exists(), f"Data dir not found: {DATA_DIR}"
    
    config = PipelineConfig()
    print(f"  ✓ Config loaded: {config.device}")
    
    ensure_directories()
    print(f"  ✓ Directories created")


def test_rna_onehot():
    """Test RNA one-hot encoding (no external deps)."""
    print("Testing RNA one-hot encoding...")
    
    # Import function
    sys.path.insert(0, str(SCRIPT_DIR / "rna"))
    from generate_msa_pssm import sequence_to_onehot, generate_pssm_basic
    
    # Test sequence
    seq = "ACGUACGU"
    onehot = sequence_to_onehot(seq)
    
    assert onehot.shape == (8, 4), f"Wrong shape: {onehot.shape}"
    assert onehot.sum() == 8.0, f"Wrong sum: {onehot.sum()}"
    
    # Check specific positions
    assert onehot[0, 0] == 1.0, "A should be at index 0"  # A
    assert onehot[1, 1] == 1.0, "C should be at index 1"  # C
    assert onehot[2, 2] == 1.0, "G should be at index 2"  # G
    assert onehot[3, 3] == 1.0, "U should be at index 3"  # U
    
    print(f"  ✓ One-hot encoding works: {seq} -> shape {onehot.shape}")
    
    # Test PSSM
    pssm = generate_pssm_basic(seq)
    assert pssm.shape == (8, 4), f"Wrong PSSM shape: {pssm.shape}"
    print(f"  ✓ PSSM generation works: shape {pssm.shape}")


def test_secondary_structure_parsing():
    """Test secondary structure parsing."""
    print("Testing secondary structure parsing...")
    
    sys.path.insert(0, str(SCRIPT_DIR / "rna"))
    from generate_secondary_structure import parse_dot_bracket, dot_bracket_to_edges
    
    # Test simple hairpin
    struct = "(((...)))"
    pairs = parse_dot_bracket(struct)
    
    expected = [(0, 8), (1, 7), (2, 6)]
    assert set(pairs) == set(expected), f"Wrong pairs: {pairs}"
    print(f"  ✓ Dot-bracket parsing: {struct} -> {pairs}")
    
    # Test edge conversion
    edges = dot_bracket_to_edges(struct)
    assert edges.shape == (3, 2), f"Wrong edges shape: {edges.shape}"
    print(f"  ✓ Edge extraction: shape {edges.shape}")


def test_mol_features():
    """Test molecule feature generation."""
    print("Testing molecule features...")
    
    try:
        from rdkit import Chem
    except ImportError:
        print("  ⚠ RDKit not installed, skipping molecule tests")
        return
    
    sys.path.insert(0, str(SCRIPT_DIR / "mol"))
    from generate_mol_features import (
        smiles_to_mol, mol_to_onehot, mol_to_edges, mol_to_atom_features
    )
    
    # Test simple molecule (ethanol)
    smiles = "CCO"
    mol = smiles_to_mol(smiles, add_hs=False)
    
    assert mol is not None, "Failed to parse SMILES"
    assert mol.GetNumAtoms() == 3, f"Wrong atom count: {mol.GetNumAtoms()}"
    print(f"  ✓ SMILES parsing: {smiles} -> {mol.GetNumAtoms()} atoms")
    
    # Test with hydrogens
    mol_h = smiles_to_mol(smiles, add_hs=True)
    assert mol_h.GetNumAtoms() == 9, f"Wrong atom count with H: {mol_h.GetNumAtoms()}"
    print(f"  ✓ SMILES with H: {smiles} -> {mol_h.GetNumAtoms()} atoms")
    
    # Test one-hot
    onehot = mol_to_onehot(mol)
    assert onehot.shape[0] == 3, f"Wrong onehot shape: {onehot.shape}"
    print(f"  ✓ One-hot encoding: shape {onehot.shape}")
    
    # Test edges
    edges, edge_types = mol_to_edges(mol)
    assert edges.shape[0] > 0, "No edges found"
    print(f"  ✓ Edge extraction: {edges.shape[0]} edges")
    
    # Test atom features
    features = mol_to_atom_features(mol)
    assert features.shape == (3, 22), f"Wrong feature shape: {features.shape}"
    print(f"  ✓ Atom features: shape {features.shape}")


def test_data_loading():
    """Test loading the union dataset."""
    print("Testing data loading...")
    
    from config import DATA_DIR
    
    # Load unique RNAs
    rna_path = DATA_DIR / "unique_rnas.csv"
    if rna_path.exists():
        rna_df = pd.read_csv(rna_path)
        print(f"  ✓ Loaded {len(rna_df)} unique RNAs")
        print(f"    Columns: {list(rna_df.columns)}")
        
        # Check lengths
        rna_df['length'] = rna_df['rna_sequence'].str.len()
        print(f"    Length range: {rna_df['length'].min()} - {rna_df['length'].max()}")
    else:
        print("  ⚠ unique_rnas.csv not found")
    
    # Load unique molecules
    mol_path = DATA_DIR / "unique_molecules.csv"
    if mol_path.exists():
        mol_df = pd.read_csv(mol_path)
        print(f"  ✓ Loaded {len(mol_df)} unique molecules")
        print(f"    Columns: {list(mol_df.columns)}")
    else:
        print("  ⚠ unique_molecules.csv not found")
    
    # Load union dataset
    union_path = DATA_DIR / "union_dataset.csv"
    if union_path.exists():
        union_df = pd.read_csv(union_path)
        print(f"  ✓ Loaded {len(union_df)} binding pairs")
        print(f"    pKd range: {union_df['pKd'].min():.2f} - {union_df['pKd'].max():.2f}")
    else:
        print("  ⚠ union_dataset.csv not found")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Pipeline Component Tests")
    print("="*60)
    print()
    
    tests = [
        ("Configuration", test_config),
        ("RNA One-hot & PSSM", test_rna_onehot),
        ("Secondary Structure", test_secondary_structure_parsing),
        ("Molecule Features", test_mol_features),
        ("Data Loading", test_data_loading),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print()
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

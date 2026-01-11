#!/usr/bin/env python3
"""
Create unified dataset from All_sf and rsim-v2.
Extracts only the entries with valid affinity measurements.
"""

import pandas as pd
import numpy as np
import hashlib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")


def get_best_affinity_rsim(row):
    """Get best available affinity from rsim and convert to pKd.
    Priority: Kd > Ki > IC50 > EC50 > Ka (all in that order for reliability).
    """
    # Kd in M -> pKd = -log10(Kd)
    if row['Kd(M)'] > 0:
        return -np.log10(row['Kd(M)']), 'Kd'
    # Ki in M -> pKi = -log10(Ki)
    if row['Ki(M)'] > 0:
        return -np.log10(row['Ki(M)']), 'Ki'
    # IC50 in M -> pIC50 = -log10(IC50)
    if row['IC50(M)'] > 0:
        return -np.log10(row['IC50(M)']), 'IC50'
    # EC50 in M
    if row['EC50(M)'] > 0:
        return -np.log10(row['EC50(M)']), 'EC50'
    # Ka in M^-1 -> Kd = 1/Ka -> pKd = log10(Ka)
    if row['Ka(M-1)'] > 0:
        return np.log10(row['Ka(M-1)']), 'Ka'
    return np.nan, None


def generate_rna_id(sequence) -> str:
    """Generate a deterministic RNA ID based on sequence hash."""
    if pd.isna(sequence):
        return None
    h = hashlib.sha256(str(sequence).encode()).hexdigest()[:12]
    return f"RNA_{h}"


def generate_mol_id(smiles) -> str:
    """Generate a deterministic molecule ID based on SMILES hash."""
    if pd.isna(smiles):
        return None
    h = hashlib.sha256(str(smiles).encode()).hexdigest()[:12]
    return f"MOL_{h}"


def main():
    print("=" * 60)
    print("Creating Unified Dataset")
    print("=" * 60)
    
    # Load datasets
    all_sf = pd.read_csv(os.path.join(DATA_DIR, "All_sf_dataset_v1.csv"), sep='\t')
    rsim = pd.read_csv(os.path.join(DATA_DIR, "rsim-v2.csv"), sep='\t')
    
    print(f"\nLoaded All_sf: {len(all_sf)} entries")
    print(f"Loaded rsim: {len(rsim)} entries")
    
    # Convert rsim affinity columns to numeric
    for col in ['Ka(M-1)', 'Ki(M)', 'Kd(M)', 'IC50(M)', 'EC50(M)']:
        rsim[col] = pd.to_numeric(rsim[col], errors='coerce').fillna(0)
    
    # Calculate pKd for rsim
    rsim_results = rsim.apply(get_best_affinity_rsim, axis=1, result_type='expand')
    rsim['pKd'] = rsim_results[0]
    rsim['affinity_source'] = rsim_results[1]
    
    # Filter rsim to entries with valid pKd
    rsim_valid = rsim[rsim['pKd'].notna()].copy()
    print(f"rsim entries with valid pKd: {len(rsim_valid)}")
    
    # Normalize columns for union
    # All_sf already has pKd
    all_sf_normalized = all_sf[['Entry_ID', 'SMILES', 'Target_RNA_sequence', 
                                 'Molecule_name', 'Molecule_ID', 
                                 'Target_RNA_name', 'Target_RNA_ID', 'pKd']].copy()
    all_sf_normalized['source'] = 'All_sf'
    all_sf_normalized['affinity_source'] = 'pKd'  # Direct measurement
    
    rsim_normalized = rsim_valid[['Entry_ID', 'SMILES', 'Target_RNA_sequence',
                                   'Molecule_name', 'Molecule_ID',
                                   'Target_RNA_name', 'Target_RNA_ID', 'pKd', 
                                   'affinity_source']].copy()
    rsim_normalized['source'] = 'rsim'
    
    # Rename columns for consistency
    all_sf_normalized = all_sf_normalized.rename(columns={
        'Target_RNA_sequence': 'rna_sequence',
        'Target_RNA_name': 'rna_name', 
        'Target_RNA_ID': 'rna_id',
        'Molecule_name': 'mol_name',
        'Molecule_ID': 'mol_id',
        'SMILES': 'smiles',
        'Entry_ID': 'entry_id'
    })
    
    rsim_normalized = rsim_normalized.rename(columns={
        'Target_RNA_sequence': 'rna_sequence',
        'Target_RNA_name': 'rna_name',
        'Target_RNA_ID': 'rna_id', 
        'Molecule_name': 'mol_name',
        'Molecule_ID': 'mol_id',
        'SMILES': 'smiles',
        'Entry_ID': 'entry_id'
    })
    
    # Combine
    union = pd.concat([all_sf_normalized, rsim_normalized], ignore_index=True)
    print(f"\nCombined: {len(union)} entries")
    
    # Remove duplicates based on (smiles, rna_sequence) pairs
    # Keep the one with more reliable affinity source
    affinity_priority = {'pKd': 0, 'Kd': 1, 'Ki': 2, 'IC50': 3, 'EC50': 4, 'Ka': 5}
    union['affinity_rank'] = union['affinity_source'].map(affinity_priority)
    union = union.sort_values('affinity_rank')
    union = union.drop_duplicates(subset=['smiles', 'rna_sequence'], keep='first')
    union = union.drop(columns=['affinity_rank'])
    
    print(f"After deduplication: {len(union)} entries")
    
    # Create canonical IDs based on sequence/smiles hashes for consistency
    union['rna_canonical_id'] = union['rna_sequence'].apply(generate_rna_id)
    union['mol_canonical_id'] = union['smiles'].apply(generate_mol_id)
    
    # Reset entry IDs
    union = union.reset_index(drop=True)
    union['entry_id'] = union.index + 1
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("UNION DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries: {len(union)}")
    print(f"Unique RNAs: {union['rna_canonical_id'].nunique()}")
    print(f"Unique molecules: {union['mol_canonical_id'].nunique()}")
    print(f"\nSource breakdown:")
    print(union['source'].value_counts())
    print(f"\nAffinity source breakdown:")
    print(union['affinity_source'].value_counts())
    print(f"\npKd statistics:")
    print(union['pKd'].describe())
    
    # Check sequence length distribution
    union['rna_length'] = union['rna_sequence'].str.len()
    print(f"\nRNA sequence length statistics:")
    print(union['rna_length'].describe())
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "union_dataset.csv")
    union.to_csv(output_path, index=False)
    print(f"\n✓ Saved union dataset to: {output_path}")
    
    # Also create lookup tables for unique RNAs and molecules
    unique_rnas = union[['rna_canonical_id', 'rna_id', 'rna_name', 'rna_sequence']].drop_duplicates(
        subset=['rna_canonical_id']
    )
    unique_mols = union[['mol_canonical_id', 'mol_id', 'mol_name', 'smiles']].drop_duplicates(
        subset=['mol_canonical_id']
    )
    
    rna_path = os.path.join(OUTPUT_DIR, "unique_rnas.csv")
    mol_path = os.path.join(OUTPUT_DIR, "unique_molecules.csv")
    unique_rnas.to_csv(rna_path, index=False)
    unique_mols.to_csv(mol_path, index=False)
    
    print(f"✓ Saved {len(unique_rnas)} unique RNAs to: {rna_path}")
    print(f"✓ Saved {len(unique_mols)} unique molecules to: {mol_path}")
    
    return union


if __name__ == "__main__":
    main()

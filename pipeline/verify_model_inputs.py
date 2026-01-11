#!/usr/bin/env python3
"""
Verify that all generated features are ready for model training/inference.
Creates a manifest of valid pairs with all required features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    # Load the final cleaned dataset
    union_df = pd.read_csv(data_dir / 'union_dataset_final.csv')
    rna_df = pd.read_csv(data_dir / 'unique_rnas_final.csv')
    mol_df = pd.read_csv(data_dir / 'unique_molecules_final.csv')
    
    print("=" * 60)
    print("FEATURE VERIFICATION FOR MODEL TRAINING/INFERENCE")
    print("=" * 60)
    
    # Check RNA features
    print("\n[RNA Features]")
    rna_features = {
        'rinalmo': processed_dir / 'rinalmo',
        'mxfold2': processed_dir / 'mxfold2',
        'rna_onehot': processed_dir / 'rna_onehot',
        'pssm': processed_dir / 'pssm',  # Optional - may not exist yet
    }
    
    rna_valid = set(rna_df['rna_canonical_id'])
    for name, path in rna_features.items():
        if not path.exists():
            print(f"  {name}: NOT FOUND (optional)")
            continue
        if name == 'mxfold2':
            files = list(path.glob('*.json'))
        else:
            files = list(path.glob('*.npy'))
        ids = {f.stem for f in files}
        missing = rna_valid - ids
        coverage = len(rna_valid & ids) / len(rna_valid) * 100
        print(f"  {name}: {len(ids)} files, {coverage:.1f}% coverage, {len(missing)} missing")
        # Only require core features (rinalmo, mxfold2, rna_onehot)
        if missing and name in ['rinalmo', 'mxfold2', 'rna_onehot']:
            rna_valid &= ids
    
    print(f"  -> RNAs with ALL features: {len(rna_valid)}")
    
    # Check molecule features
    print("\n[Molecule Features]")
    mol_features = {
        'unimol_cls': processed_dir / 'unimol' / 'cls',
        'unimol_atomic': processed_dir / 'unimol' / 'atomic',
        'mol_onehot': processed_dir / 'mol_features' / 'onehot',
        'mol_graph': processed_dir / 'mol_features' / 'graph',
    }
    
    mol_valid = set(mol_df['mol_canonical_id'])
    for name, path in mol_features.items():
        files = list(path.glob('*.npy')) + list(path.glob('*.npz'))
        ids = {f.stem for f in files}
        missing = mol_valid - ids
        coverage = len(mol_valid & ids) / len(mol_valid) * 100
        print(f"  {name}: {len(ids)} files, {coverage:.1f}% coverage, {len(missing)} missing")
        if missing:
            mol_valid &= ids
    
    print(f"  -> Molecules with ALL features: {len(mol_valid)}")
    
    # Filter union dataset to only valid pairs
    print("\n[Creating Valid Pairs Dataset]")
    valid_pairs = union_df[
        (union_df['rna_canonical_id'].isin(rna_valid)) & 
        (union_df['mol_canonical_id'].isin(mol_valid))
    ].copy()
    
    print(f"  Original pairs: {len(union_df)}")
    print(f"  Valid pairs (all features): {len(valid_pairs)}")
    
    # Save valid pairs
    valid_pairs.to_csv(data_dir / 'model_ready_dataset.csv', index=False)
    print(f"  Saved to: data/model_ready_dataset.csv")
    
    # Create manifest with feature paths
    manifest = {
        'rna_features': {
            'rinalmo': str(processed_dir / 'rinalmo'),
            'mxfold2': str(processed_dir / 'mxfold2'),
            'rna_onehot': str(processed_dir / 'rna_onehot'),
            'pssm': str(processed_dir / 'pssm'),
            'msa': str(processed_dir / 'msa'),
        },
        'mol_features': {
            'unimol_cls': str(processed_dir / 'unimol' / 'cls'),
            'unimol_atomic': str(processed_dir / 'unimol' / 'atomic'),
            'mol_onehot': str(processed_dir / 'mol_features' / 'onehot'),
            'mol_graph': str(processed_dir / 'mol_features' / 'graph'),
        },
        'dataset': str(data_dir / 'model_ready_dataset.csv'),
        'stats': {
            'num_rnas': len(rna_valid),
            'num_molecules': len(mol_valid),
            'num_pairs': len(valid_pairs),
        }
    }
    
    with open(data_dir / 'feature_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to: data/feature_manifest.json")
    
    # Verify feature shapes
    print("\n[Feature Shape Verification]")
    
    # Sample one RNA
    sample_rna = list(rna_valid)[0]
    rinalmo = np.load(processed_dir / 'rinalmo' / f'{sample_rna}.npy')
    rna_oh = np.load(processed_dir / 'rna_onehot' / f'{sample_rna}.npy')
    with open(processed_dir / 'mxfold2' / f'{sample_rna}.json') as f:
        mxfold2 = json.load(f)
    
    print(f"  Sample RNA: {sample_rna}")
    print(f"    RiNALMo embedding: {rinalmo.shape} (L x 1280)")
    print(f"    RNA one-hot: {rna_oh.shape} (L x 4)")
    print(f"    mxfold2 structure: {mxfold2['length']} nt, {len(mxfold2['base_pairs'])} base pairs")
    
    # Sample one molecule
    sample_mol = list(mol_valid)[0]
    unimol_cls = np.load(processed_dir / 'unimol' / 'cls' / f'{sample_mol}.npy')
    unimol_atomic = np.load(processed_dir / 'unimol' / 'atomic' / f'{sample_mol}.npy')
    mol_oh = np.load(processed_dir / 'mol_features' / 'onehot' / f'{sample_mol}.npy')
    mol_graph = np.load(processed_dir / 'mol_features' / 'graph' / f'{sample_mol}.npz')
    
    print(f"  Sample Molecule: {sample_mol}")
    print(f"    UniMol CLS: {unimol_cls.shape} (512,)")
    print(f"    UniMol atomic: {unimol_atomic.shape} (N x 512)")
    print(f"    Mol one-hot: {mol_oh.shape} (N x 14)")
    print(f"    Mol graph: {mol_graph['num_atoms']} atoms, {mol_graph['edges'].shape[0]} edges, features {mol_graph['atom_features'].shape}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print(f"Ready for model training with {len(valid_pairs)} binding pairs")
    print("=" * 60)


if __name__ == '__main__':
    main()

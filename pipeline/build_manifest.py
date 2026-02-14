#!/usr/bin/env python3
"""
Build a unified feature manifest after running the pipeline.
This creates a consolidated view of all generated features for easy loading.

Usage:
    python build_manifest.py --output manifest.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).parent.absolute()
INFERENCE_DIR = SCRIPT_DIR.parent
DATA_DIR = INFERENCE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class RNAFeatures:
    """Container for RNA feature paths and metadata."""
    rna_id: str
    sequence: str
    length: int
    embedding_path: Optional[str] = None
    secondary_path: Optional[str] = None
    pssm_path: Optional[str] = None
    onehot_path: Optional[str] = None
    msa_path: Optional[str] = None


@dataclass
class MolFeatures:
    """Container for molecule feature paths and metadata."""
    mol_id: str
    smiles: str
    embedding_cls_path: Optional[str] = None
    embedding_atomic_path: Optional[str] = None
    onehot_path: Optional[str] = None
    graph_path: Optional[str] = None


def scan_rna_features() -> Dict[str, RNAFeatures]:
    """Scan all RNA feature directories and build feature map."""
    rna_dir = PROCESSED_DIR / "rna"
    
    # Load unique RNAs
    unique_rnas_path = DATA_DIR / "unique_rnas.csv"
    if not unique_rnas_path.exists():
        print("Warning: unique_rnas.csv not found")
        return {}
    
    rna_df = pd.read_csv(unique_rnas_path)
    
    # Build feature map
    features = {}
    for _, row in rna_df.iterrows():
        rna_id = row["rna_canonical_id"]
        seq = row["rna_sequence"]
        
        feat = RNAFeatures(
            rna_id=rna_id,
            sequence=seq,
            length=len(seq),
        )
        
        # Check embeddings
        embed_path = rna_dir / "embeddings" / f"{rna_id}.npy"
        if embed_path.exists():
            feat.embedding_path = str(embed_path.relative_to(INFERENCE_DIR))
        
        # Check secondary structure
        ss_path = rna_dir / "secondary" / f"{rna_id}.json"
        if ss_path.exists():
            feat.secondary_path = str(ss_path.relative_to(INFERENCE_DIR))
        
        # Check PSSM
        pssm_path = rna_dir / "pssm" / f"{rna_id}.npy"
        if pssm_path.exists():
            feat.pssm_path = str(pssm_path.relative_to(INFERENCE_DIR))
        
        # Check one-hot
        onehot_path = rna_dir / "onehot" / f"{rna_id}.npy"
        if onehot_path.exists():
            feat.onehot_path = str(onehot_path.relative_to(INFERENCE_DIR))
        
        # Check MSA
        msa_path = rna_dir / "msa" / f"{rna_id}.a3m"
        if msa_path.exists():
            feat.msa_path = str(msa_path.relative_to(INFERENCE_DIR))
        
        features[rna_id] = feat
    
    return features


def scan_mol_features() -> Dict[str, MolFeatures]:
    """Scan all molecule feature directories and build feature map."""
    mol_dir = PROCESSED_DIR / "mol"
    
    # Load unique molecules
    unique_mols_path = DATA_DIR / "unique_molecules.csv"
    if not unique_mols_path.exists():
        print("Warning: unique_molecules.csv not found")
        return {}
    
    mol_df = pd.read_csv(unique_mols_path)
    
    # Build feature map
    features = {}
    for _, row in mol_df.iterrows():
        mol_id = row["mol_canonical_id"]
        smiles = row["smiles"]
        
        feat = MolFeatures(
            mol_id=mol_id,
            smiles=smiles,
        )
        
        # Check CLS embedding
        cls_path = mol_dir / "embeddings" / "cls" / f"{mol_id}.npy"
        if cls_path.exists():
            feat.embedding_cls_path = str(cls_path.relative_to(INFERENCE_DIR))
        
        # Check atomic embedding
        atomic_path = mol_dir / "embeddings" / "atomic" / f"{mol_id}.npy"
        if atomic_path.exists():
            feat.embedding_atomic_path = str(atomic_path.relative_to(INFERENCE_DIR))
        
        # Check one-hot
        onehot_path = mol_dir / "onehot" / f"{mol_id}.npy"
        if onehot_path.exists():
            feat.onehot_path = str(onehot_path.relative_to(INFERENCE_DIR))
        
        # Check graph
        graph_path = mol_dir / "graph" / f"{mol_id}.npz"
        if graph_path.exists():
            feat.graph_path = str(graph_path.relative_to(INFERENCE_DIR))
        
        features[mol_id] = feat
    
    return features


def count_complete(features: Dict[str, Any], required_fields: List[str]) -> int:
    """Count entries with all required fields present."""
    count = 0
    for feat in features.values():
        if isinstance(feat, dict):
            complete = all(feat.get(f) is not None for f in required_fields)
        else:
            complete = all(getattr(feat, f) is not None for f in required_fields)
        if complete:
            count += 1
    return count


def build_manifest(output_path: Path):
    """Build and save the feature manifest."""
    print("Scanning RNA features...")
    rna_features = scan_rna_features()
    
    print("Scanning molecule features...")
    mol_features = scan_mol_features()
    
    # Calculate statistics
    rna_stats = {
        "total": len(rna_features),
        "with_embeddings": sum(1 for f in rna_features.values() if f.embedding_path),
        "with_secondary": sum(1 for f in rna_features.values() if f.secondary_path),
        "with_pssm": sum(1 for f in rna_features.values() if f.pssm_path),
        "with_onehot": sum(1 for f in rna_features.values() if f.onehot_path),
        "complete": count_complete(rna_features, 
            ["embedding_path", "secondary_path", "pssm_path", "onehot_path"]),
    }
    
    mol_stats = {
        "total": len(mol_features),
        "with_cls_embeddings": sum(1 for f in mol_features.values() if f.embedding_cls_path),
        "with_atomic_embeddings": sum(1 for f in mol_features.values() if f.embedding_atomic_path),
        "with_onehot": sum(1 for f in mol_features.values() if f.onehot_path),
        "with_graph": sum(1 for f in mol_features.values() if f.graph_path),
        "complete": count_complete(mol_features,
            ["embedding_cls_path", "embedding_atomic_path", "onehot_path", "graph_path"]),
    }
    
    # Build manifest
    manifest = {
        "version": "1.0",
        "base_dir": str(INFERENCE_DIR),
        "statistics": {
            "rna": rna_stats,
            "mol": mol_stats,
        },
        "rna_features": {k: asdict(v) for k, v in rna_features.items()},
        "mol_features": {k: asdict(v) for k, v in mol_features.items()},
    }
    
    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Feature Manifest Summary")
    print("="*60)
    print(f"\nRNA Features ({rna_stats['total']} sequences):")
    print(f"  Embeddings:        {rna_stats['with_embeddings']:4d} / {rna_stats['total']}")
    print(f"  Secondary struct:  {rna_stats['with_secondary']:4d} / {rna_stats['total']}")
    print(f"  PSSM:              {rna_stats['with_pssm']:4d} / {rna_stats['total']}")
    print(f"  One-hot:           {rna_stats['with_onehot']:4d} / {rna_stats['total']}")
    print(f"  Complete:          {rna_stats['complete']:4d} / {rna_stats['total']}")
    
    print(f"\nMolecule Features ({mol_stats['total']} molecules):")
    print(f"  CLS embeddings:    {mol_stats['with_cls_embeddings']:4d} / {mol_stats['total']}")
    print(f"  Atomic embeddings: {mol_stats['with_atomic_embeddings']:4d} / {mol_stats['total']}")
    print(f"  One-hot:           {mol_stats['with_onehot']:4d} / {mol_stats['total']}")
    print(f"  Graph:             {mol_stats['with_graph']:4d} / {mol_stats['total']}")
    print(f"  Complete:          {mol_stats['complete']:4d} / {mol_stats['total']}")
    
    print(f"\nManifest saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build feature manifest")
    parser.add_argument("--output", "-o", default="manifest.json", help="Output manifest file")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path
    
    build_manifest(output_path)


if __name__ == "__main__":
    main()

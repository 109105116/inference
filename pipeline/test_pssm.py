#!/usr/bin/env python3
"""
Test PSSM generation on a small sample of RNAs.
Run after RNAcentral database is downloaded and indexed.
"""

import sys
import pandas as pd
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from rna.generate_msa_pssm import generate_msa_and_pssm


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    db_path = base_dir / 'databases' / 'rnacentral_db'
    
    # Check database
    if not (db_path.parent / 'rnacentral_db.index').exists():
        print("ERROR: RNAcentral database not found.")
        print(f"Expected: {db_path}")
        print("Run: databases/download_rnacentral.sh first")
        return 1
    
    # Load sample of RNAs
    rna_df = pd.read_csv(data_dir / 'unique_rnas_final.csv')
    sample_df = rna_df.head(5)  # Test with 5 RNAs
    
    print(f"Testing PSSM generation on {len(sample_df)} RNAs...")
    print(f"Database: {db_path}")
    
    output_msa = data_dir / 'processed' / 'test_msa'
    output_pssm = data_dir / 'processed' / 'test_pssm'
    
    generate_msa_and_pssm(
        rna_data=sample_df,
        output_msa_dir=output_msa,
        output_pssm_dir=output_pssm,
        database_path=str(db_path),
        sensitivity=7.5,
        num_iterations=2,
        max_seqs=100,  # Smaller for testing
        overwrite=True,
    )
    
    # Verify outputs
    import numpy as np
    for _, row in sample_df.iterrows():
        rna_id = row['rna_canonical_id']
        pssm_file = output_pssm / f'{rna_id}.npy'
        if pssm_file.exists():
            pssm = np.load(pssm_file)
            print(f"  {rna_id}: PSSM shape {pssm.shape}")
        else:
            print(f"  {rna_id}: PSSM NOT FOUND")
    
    print("\n✅ PSSM test complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Test all feature generation tools with real data from the dataset.
Run this after setting up all environments to verify everything works.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Test sequences and molecules
TEST_RNA_SEQ = "GCUACGAUAGCUAGCUAGCUAGC"
TEST_RNA_NAME = "test_rna"
TEST_SMILES = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"]  # ethanol, aspirin
TEST_MOL_NAMES = ["mol_ethanol", "mol_aspirin"]


def run_conda_command(env_name: str, python_code: str):
    """Run Python code in a specific conda environment."""
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", python_code],
        capture_output=True,
        text=True
    )
    return result


def test_rinalmo():
    """Test RiNALMo embedding generation."""
    print("\n" + "="*60)
    print("Testing RiNALMo...")
    print("="*60)
    
    code = f'''
import os
import torch
from rinalmo.pretrained import get_pretrained_model

# Use second GPU if first is busy
DEVICE = "cuda:0"

model, alphabet = get_pretrained_model(model_name="giga-v1")
model = model.to(device=DEVICE)
model.eval()

seqs = ["{TEST_RNA_SEQ}"]
tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=DEVICE)

with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model(tokens)

print(f"Input: {TEST_RNA_SEQ}")
print(f"Embedding shape: {{outputs['representation'].shape}}")
print("RiNALMo test PASSED!")
'''
    
    # Use GPU 1 if GPU 0 is busy
    env = {"CUDA_VISIBLE_DEVICES": "1"}
    result = subprocess.run(
        ["conda", "run", "-n", "rinalmo", "python", "-c", code],
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env}
    )
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"FAILED: {result.stderr}")
        return False


def test_mxfold2():
    """Test mxfold2 secondary structure prediction."""
    print("\n" + "="*60)
    print("Testing mxfold2...")
    print("="*60)
    
    # Write test FASTA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        f.write(f">{TEST_RNA_NAME}\n{TEST_RNA_SEQ}\n")
        fasta_file = f.name
    
    result = subprocess.run(
        ["conda", "run", "-n", "mxfold2", "mxfold2", "predict", fasta_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
        # Parse output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('>'):
                print(f"RNA: {line[1:]}")
            elif '(' in line or ')' in line or '.' in line:
                parts = line.split()
                if len(parts) >= 2:
                    print(f"Structure: {parts[0]}")
                    print(f"Energy: {parts[1]}")
        print("mxfold2 test PASSED!")
        return True
    else:
        print(f"FAILED: {result.stderr}")
        return False


def test_unimol():
    """Test UniMol embedding generation."""
    print("\n" + "="*60)
    print("Testing UniMol...")
    print("="*60)
    
    code = f'''
from unimol_tools import UniMolRepr

clf = UniMolRepr(data_type='molecule', remove_hs=False)
smiles = {TEST_SMILES}
reprs = clf.get_repr(smiles, return_atomic_reprs=True)

for i, smi in enumerate(smiles):
    cls_shape = reprs['cls_repr'][i].shape
    atomic_shape = reprs['atomic_reprs'][i].shape
    print(f"Molecule: {{smi}}")
    print(f"  CLS embedding: {{cls_shape}}")
    print(f"  Atomic embeddings: {{atomic_shape}}")

print("UniMol test PASSED!")
'''
    
    result = run_conda_command("unimol", code)
    if result.returncode == 0:
        # Filter info messages
        for line in result.stdout.split('\n'):
            if 'Molecule:' in line or 'CLS' in line or 'Atomic' in line or 'PASSED' in line:
                print(line)
        return True
    else:
        print(f"FAILED: {result.stderr}")
        return False


def main():
    print("="*60)
    print("Feature Generation Tools Test Suite")
    print("="*60)
    
    results = {}
    
    # Test each tool
    results['RiNALMo'] = test_rinalmo()
    results['mxfold2'] = test_mxfold2()
    results['UniMol'] = test_unimol()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for tool, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {tool}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("All tests passed! Ready for feature generation.")
        return 0
    else:
        print("Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

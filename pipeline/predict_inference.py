import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from RMPred import RMPred
from RNAdataset import load_global_stores, make_dataloader
# RMPred/RNAdataset internals will now resolve 'config' from parent_dir first

def predict(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    ids_path = os.path.join(args.data, "ids.pkl")
    rna_embed_path = os.path.join(args.data, "rna_embed.pkl")
    rna_graph_path = os.path.join(args.data, "rna_graph.pkl")
    mole_embed_path = os.path.join(args.data, "mole_embed.pkl")
    mole_edge_path = os.path.join(args.data, "mole_graph.pkl")
    pssm_dir = os.path.join(args.data, "pssm")

    if not os.path.exists(ids_path):
        print(f"Error: ids.pkl not found in {args.data}. Did you run consolidate_features.py?")
        sys.exit(1)

    stores = load_global_stores(
        ids_path=ids_path,
        rna_embed_path=rna_embed_path,
        rna_graph_path=rna_graph_path,
        mole_embed_path=mole_embed_path,
        mole_edge_path=mole_edge_path,
        pssm_dir=pssm_dir
    )

    # Dataloader
    test_loader = make_dataloader(
        stores,
        batch_size=args.batch_size,
        shuffle=False,
        strict=False, # Allow missing items to be skipped (though for inference we prefer to know)
        max_rna_len=8192 # Support long inference
    )
    print(f"Dataloader created with {len(test_loader.dataset)} entries.")

    # 2. Load Models
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".pt")]
    if not model_files:
        print(f"Error: No .pt model files found in {args.model_dir}")
        sys.exit(1)
    
    print(f"Found {len(model_files)} models in ensemble.")
    
    models = []
    # Load one model to get architecture params if needed (assuming fixed config for now)
    # Reusing implicit config from train_inference_model.py / RMPred
    # We really should have saved config.json, but for now we hardcode what we know.
    
    # Check dimensions from data to be safe?
    # RMPred needs to know input dims.
    sample_rna = next(iter(stores.rna_embed.values()))
    sample_mol = next(iter(stores.mole_embed.values()))
    d_llm_rna = sample_rna.shape[1]
    d_llm_mole = sample_mol.shape[1]
    d_pssm_rna = 4 # Fixed
    
    for mf in model_files:
        path = os.path.join(args.model_dir, mf)
        
        # Reconstruct model (must match training params!)
        # Assuming we used the defaults + max_len=8192 from our recent changes.
        model = RMPred(
            d_llm_rna=d_llm_rna,
            c_onehot_rna=5,
            d_pssm_rna=d_pssm_rna,
            d_llm_mole=d_llm_mole,
            c_onehot_mole=14,
            d_model_inner=256,
            d_model_fusion=512,
            dropout=0.0, # No dropout for inference
            fusion_layers=2,
            rna_max_len=8192
        )
        
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    # 3. Inference Loop
    print("Running inference...")
    
    all_preds = []
    all_ids = []
    
    pkd_mean = stores.pkd_norm['mean']
    pkd_std = stores.pkd_norm['std']
    
    print("Running inference...")
    
    all_preds = []
    all_ids = []
    
    pkd_mean = stores.pkd_norm['mean']
    pkd_std = stores.pkd_norm['std']
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None: continue
            
            # Move to device
            rna_llm = batch['rna_llm'].to(device)
            rna_onehot = batch['rna_onehot'].to(device)
            rna_mask = batch['rna_mask'].to(device)
            rna_pssm = batch['rna_pssm'].to(device)
            rna_edges = [e.to(device) for e in batch['rna_edges']]
            
            mole_llm = batch['mole_llm'].to(device)
            mole_onehot = batch['mole_onehot'].to(device)
            mole_mask = batch['mole_mask'].to(device)
            mole_edges = [e.to(device) for e in batch['mole_edges']]
            
            ensemble_outs = []
            
            for model in models:
                out = model(
                    rna_llm=rna_llm, 
                    rna_onehot=rna_onehot, 
                    rna_edges=rna_edges, 
                    rna_pssm=rna_pssm, 
                    rna_mask=rna_mask,
                    mole_llm=mole_llm, 
                    mole_onehot=mole_onehot, 
                    mole_edges=mole_edges, 
                    mole_mask=mole_mask
                )
                # Un-normalize
                out = out * pkd_std + pkd_mean
                ensemble_outs.append(out.cpu().numpy())
            
            # ensemble_outs is list of (B,) arrays
            # Stack to (B, N_models)
            stacked = np.stack(ensemble_outs, axis=1)
            
            # Store ID and Preds
            for i, entry_id in enumerate(batch['entry_ids']):
                preds = stacked[i] # (N_models,)
                all_ids.append({
                    'entry_id': entry_id,
                    'rna_id': batch['rna_ids'][i],
                    'mol_id': batch['mol_ids'][i]
                })
                all_preds.append(preds)

    # 4. Save results
    final_rows = []
    for meta, preds in zip(all_ids, all_preds):
        mu = np.mean(preds)
        sigma = np.std(preds)
        
        row = meta.copy()
        row['pred_pKd'] = mu
        row['uncertainty_sigma'] = sigma
        # Optional: include individual model preds?
        # for i, p in enumerate(preds):
        #     row[f'model_{i}'] = p
        final_rows.append(row)
        
    df_out = pd.DataFrame(final_rows)
    out_path = os.path.join(args.output, "predictions.csv")
    os.makedirs(args.output, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    
    print(f"✅ Prediction complete. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory with consolidated features (ids.pkl, etc)")
    parser.add_argument("--model-dir", required=True, help="Directory containing .pt model files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    predict(args)

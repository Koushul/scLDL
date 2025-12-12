
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from scLDL.label_enhancer import HybridLEVI

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

def load_data_simple():
    filepath = 'data/scrna_kidney.h5ad'
    print(f"Loading {filepath}...")
    adata = sc.read_h5ad(filepath)
    sc.pp.subsample(adata, n_obs=10000) # Small subset for debug
    
    print("Preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    
    X = adata.X.toarray().astype('float32')
    
    y = adata.obs['cell_type'].values
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    enc = OneHotEncoder(sparse_output=False)
    L = enc.fit_transform(y_int.reshape(-1, 1))
    
    return X, L, len(le.classes_)

def debug_loss(gamma_val=1.0):
    print(f"\n--- DEBUGGING GAMMA = {gamma_val} ---")
    X, L, n_outputs = load_data_simple()
    n_features = X.shape[1]
    
    model = HybridLEVI(
        n_features=n_features,
        n_outputs=n_outputs,
        n_hidden=512,
        n_latent=256, # Increased latent size
        encoder_type='mlp',
        gamma=gamma_val, 
        device=device
    )
    
    X_tensor = torch.FloatTensor(X).to(device)
    L_tensor = torch.FloatTensor(L).to(device)
    dataset = TensorDataset(X_tensor, L_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    
    # Run 1 epoch and print components
    for batch_X, batch_L in dataloader:
        optimizer.zero_grad()
        # forward(X) -> mean, logvar, z, X_hat, evidence
        # Wait, original forward returns: mean, logvar, z, X_hat, evidence
        # Let's check src/scLDL/label_enhancer.py
        # Yes: mean, logvar, z, X_hat, evidence = self.forward(...)
        mean, logvar, z, X_hat, evidence = model.forward(batch_X)
        
        # Losses
        # KL
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        # Rec
        rec_loss = torch.mean(torch.mean((batch_X - X_hat)**2, dim=1))
        
        # CDL
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        m = alpha / S
        A = torch.sum((batch_L - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        cdl_loss = torch.mean(A + B)
        
        total_loss = rec_loss + gamma_val * cdl_loss + model.alpha * kl_loss
        
        print(f"Rec Loss: {rec_loss.item():.6f}")
        print(f"CDL Loss: {cdl_loss.item():.6f}")
        print(f"KL Loss: {kl_loss.item():.6f}")
        print(f"Gamma * CDL: {(gamma_val * cdl_loss).item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")
        
        break # Only need one batch

if __name__ == "__main__":
    debug_loss(gamma_val=1.0)
    debug_loss(gamma_val=10.0)
    debug_loss(gamma_val=100.0)

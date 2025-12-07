import torch
import numpy as np
import anndata
from torch.utils.data import DataLoader
import sys
import os

# Add src to path to import scLDL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.models.label_enhancer import LabelEnhancer
from scLDL.models.trainer import LabelEnhancerTrainer
from scLDL.utils.data import scDataset
from scLDL.utils.metrics import compute_metrics

def main():
    print("Generating synthetic single-cell data...")
    # 1. Generate synthetic data
    # 1000 cells, 500 genes, 3 cell types
    n_obs = 1000
    n_vars = 500
    n_classes = 3
    
    X = np.random.randn(n_obs, n_vars).astype(np.float32)
    # Add some signal
    labels = np.random.randint(0, n_classes, n_obs)
    for i in range(n_classes):
        X[labels == i, :50] += i * 2.0
        
    # Create AnnData
    adata = anndata.AnnData(X=X)
    adata.obs['cell_type'] = labels
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    # Add dummy spatial coordinates
    adata.obsm['spatial'] = np.random.rand(n_obs, 2).astype(np.float32)
    
    # Create Ground Truth Distributions (for testing metrics)
    y_true = np.zeros((n_obs, n_classes))
    y_true[np.arange(n_obs), labels] = 0.8
    y_true += 0.2 / n_classes # Add background noise
    y_true = y_true / y_true.sum(axis=1, keepdims=True) # Normalize
    
    print(f"Data shape: {adata.shape}")
    print(f"Labels: {adata.obs['cell_type'].unique()}")
    
    # 2. Create Dataset and DataLoader
    print("\nInitializing Dataset...")
    dataset = scDataset(adata, label_key='cell_type', spatial_key='spatial')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Initialize Model
    print("Initializing LabelEnhancer Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LabelEnhancer(
        x_dim=dataset.get_input_dim(),
        d_dim=dataset.get_num_classes(),
        h_dim=64,
        device=device
    )
    
    # 4. Train
    print("\nStarting Training...")
    trainer = LabelEnhancerTrainer(
        model, 
        lr=1e-3, 
        beta=0.001, 
        lambda_gap=1.0, 
        lambda_spatial=0.1 # Enable spatial regularization
    )
    
    trainer.train(train_loader, epochs=5, log_interval=1)
    
    # 5. Predict
    print("\nPredicting Label Distributions...")
    # Create a loader for prediction (no shuffle)
    pred_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    distributions = trainer.predict(pred_loader)
    
    print(f"Predictions shape: {distributions.shape}")
    print("Sample prediction (first 5 cells):")
    print(distributions[:5])
    
    # Save results to AnnData
    adata.obsm['X_label_enhanced'] = distributions
    print("\nSaved predictions to adata.obsm['X_label_enhanced']")
    
    # 6. Evaluate
    print("\nEvaluating Performance...")
    metrics = compute_metrics(y_true, distributions)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

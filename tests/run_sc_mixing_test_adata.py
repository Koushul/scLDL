
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scanpy as sc
import anndata
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.label_enhancer import ImprovedLEVI, ConcentrationLE, HybridLEVI

# Consistent Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# MLP Baseline (Same definition)
class MLPBaseline(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.BatchNorm1d(n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden // 2, n_classes)
        )
        
    def forward(self, x):
        return self.net(x)
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x).float().to(next(self.parameters()).device)
            logits = self(x_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

class MixingSCDataset(Dataset):
    def __init__(self, X, y, n_samples=50000, n_classes=22, n_mix=2):
        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_mix = n_mix
        self.samples = []
        
        print(f"Generating {n_samples} {n_mix}-way mixed scRNA-seq samples...")
        
        # Indices by label
        self.indices_by_label = [[] for _ in range(n_classes)]
        for idx, label in enumerate(y):
            self.indices_by_label[label].append(idx)
            
        for _ in tqdm(range(n_samples)):
            # Pick N distinct labels
            labels = np.random.choice(n_classes, n_mix, replace=False)
            
            # Pick weights
            if n_mix == 2:
                w = np.random.uniform(0.2, 0.8)
                weights = [w, 1-w]
            else:
                weights = np.random.dirichlet(np.ones(n_mix))
            
            # Pick indices
            indices = [np.random.choice(self.indices_by_label[l]) for l in labels]
            
            self.samples.append({
                'indices': indices,
                'labels': labels,
                'weights': weights
            })
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        indices = s['indices']
        weights = s['weights']
        
        vec_mix = np.zeros_like(self.X[0])
        for i, idx in enumerate(indices):
            vec_mix += weights[i] * self.X[idx]
            
        return torch.tensor(vec_mix).float(), torch.tensor(s['labels'])

def load_data_and_models():
    # Load Metadata
    with open('models/sc_adata/preprocessing_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    n_features = meta['n_features']
    n_classes = meta['n_classes']
    var_names = meta['var_names']
    
    # Load Data
    print("Loading data...")
    adata = anndata.read_h5ad('data/adata.h5ad')
    
    # Preprocess EXACTLY as training
    print("Preprocessing...")
    if np.max(adata.X) > 100:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    # Slice to same HVGs
    adata = adata[:, var_names].copy()
    
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    # Labels
    le = meta['le']
    y = le.transform(adata.obs['cell_type'].values)
    
    # Load Models
    models = {}
    
    print("Loading MLPBaseline...")
    m = MLPBaseline(n_features, n_classes).to(DEVICE)
    m.load_state_dict(torch.load('models/sc_adata/MLPBaseline.pth', map_location=DEVICE))
    m.eval()
    models['MLPBaseline'] = m
    
    print("Loading HybridLEVI...")
    m = HybridLEVI(n_features, n_classes, encoder_type='mlp', n_hidden=256, device=DEVICE)
    m.load_state_dict(torch.load('models/sc_adata/HybridLEVI.pth', map_location=DEVICE))
    m.eval()
    models['HybridLEVI'] = m
    
    print("Loading ConcentrationLE...")
    m = ConcentrationLE(n_features, n_classes, encoder_type='mlp', n_hidden=256, device=DEVICE)
    m.load_state_dict(torch.load('models/sc_adata/ConcentrationLE.pth', map_location=DEVICE))
    m.eval()
    models['ConcentrationLE'] = m
    
    print("Loading ImprovedLEVI...")
    m = ImprovedLEVI(n_features, n_classes, encoder_type='mlp', n_hidden=256, device=DEVICE)
    m.load_state_dict(torch.load('models/sc_adata/ImprovedLEVI.pth', map_location=DEVICE))
    m.eval()
    models['ImprovedLEVI'] = m
    
    return X, y, models, n_classes

def predict_batch(model, X_batch, n_classes):
    model.eval()
    with torch.no_grad():
        name = type(model).__name__
        if name == 'MLPBaseline':
            return model.predict(X_batch)
        elif name in ['ConcentrationLE', 'HybridLEVI']:
            if name == 'ConcentrationLE':
                _, alpha = model(X_batch)
            else:
                 _, _, _, _, evidence = model(X_batch)
                 alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            return (alpha / S).cpu().numpy()
        elif name == 'ImprovedLEVI':
            L_dummy = torch.ones(X_batch.size(0), n_classes).to(DEVICE) / n_classes
            return model.predict(X_batch.cpu().numpy(), L_dummy.cpu().numpy())
        else:
            return np.zeros((X_batch.size(0), n_classes))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mix', type=int, default=2, help='Number of classes to mix')
    args = parser.parse_args()
    
    X, y, models, n_classes = load_data_and_models()
    
    dataset = MixingSCDataset(X, y, n_samples=50000, n_classes=n_classes, n_mix=args.mix)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    results = {name: {'all_in_top3': 0} for name in models.keys()}
    
    print(f"\nStarting Evaluation ({args.mix}-Way Mixing scRNA-seq - adata)...")
    
    for batch_idx, (inputs, true_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        inputs = inputs.to(DEVICE)
        
        for name, model in models.items():
            probs = predict_batch(model, inputs, n_classes)
            
            top3_preds = np.argsort(probs, axis=1)[:, -3:]
            
            for i in range(len(inputs)):
                pred_set = set(top3_preds[i])
                true_set = set(true_labels[i].numpy())
                
                # Check if ALL true labels are in Top-3
                if true_set.issubset(pred_set):
                    results[name]['all_in_top3'] += 1
        
        # Plotting (First batch only)
        if batch_idx == 0:
            os.makedirs('reports/plots', exist_ok=True)
            n_plot = min(5, len(inputs))
            fig, axes = plt.subplots(n_plot, len(models), figsize=(3 * len(models), 3 * n_plot))
            
            # If n_plot=1 or len(models)=1, axes might handle differently, assuming typical case
            if len(models) == 1: axes = axes[:, None] # Ensure 2D
             
            for i in range(n_plot):
                true_lbls = true_labels[i].numpy()
                
                # Predictions
                for j, (name, model) in enumerate(models.items()):
                     p = predict_batch(model, inputs[i:i+1], n_classes)[0]
                     ax = axes[i, j]
                     ax.bar(range(n_classes), p, color='skyblue')
                     ax.set_ylim(0, 1)
                     ax.set_title(f"{name}\nTrue: {true_lbls}")
                     ax.set_xticks(range(n_classes))
                     ax.tick_params(labelsize=6)
            
            plt.tight_layout()
            plt.savefig(f'reports/plots/sc_adata_mixing_{args.mix}way.png')
            plt.close()

    print(f"\n=== scRNA-seq MIXING TEST RESULTS (adata.h5ad, 50k Samples, {args.mix}-Way) ===")
    print(f"{'Model':<20} | {'All In Top-3 %':<20}")
    print("-" * 45)
    
    for name, res in results.items():
        acc = (res['all_in_top3'] / len(dataset)) * 100
        print(f"{name:<20} | {acc:<19.2f}%")

if __name__ == "__main__":
    main()

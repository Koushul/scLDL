import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.ldl_models import ConcentrationLDL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Inline Baseline (Must match previous scripts) ---
class MLPBaseline(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def normalize_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    return adata

def main():
    # 1. Load & Prep
    print("Loading Data...")
    adata = anndata.read_h5ad('data/adata.h5ad')
    df = pd.read_csv('data/tonsil_cell_types.csv').set_index('NAME')
    common = adata.obs.index.intersection(df.index)
    adata = adata[common].copy()
    adata.obs['cell_type_fine'] = df.loc[common, 'cell_type_2']
    
    adata = normalize_data(adata)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['cell_type_fine'].values)
    classes = le.classes_
    n_classes = len(classes)
    
    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Teacher
    print("Training Teacher...")
    teacher = MLPBaseline(X.shape[1], n_classes).to(DEVICE)
    opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    ds_t = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    dl_t = DataLoader(ds_t, batch_size=64, shuffle=True)
    
    for _ in range(10): # 10 epochs
        teacher.train()
        for bx, by in dl_t:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_t.zero_grad()
            out = teacher(bx)
            loss = crit(out, by)
            loss.backward()
            opt_t.step()
            
    # 3. Distill
    teacher.eval()
    with torch.no_grad():
        logits = teacher(torch.tensor(X_train).float().to(DEVICE))
        soft_labels = torch.softmax(logits, dim=1).cpu().numpy()
        
    # 4. Train Student
    print("Training Student...")
    student = ConcentrationLDL(X.shape[1], n_classes, encoder_type='mlp', device=DEVICE).to(DEVICE)
    opt_s = optim.Adam(student.parameters(), lr=1e-3)
    
    ds_s = TensorDataset(torch.tensor(X_train).float(), torch.tensor(soft_labels).float())
    dl_s = DataLoader(ds_s, batch_size=64, shuffle=True)
    
    for epoch in range(20):
        student.train()
        kl_coeff = min(0.01, (epoch/10.0)*0.01)
        for bx, by in dl_s:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_s.zero_grad()
            loss, _, _, _ = student.compute_loss(bx, by, global_step=epoch, lambda_kl=kl_coeff)
            loss.backward()
            opt_s.step()
            
    # 5. Demixing Experiment
    print("Generating Demixing Curve...")
    
    # Pick two distinct classes
    # e.g. 'B_naive' and 'B_memory'
    class_a_name = 'B_naive'
    class_b_name = 'B_memory'
    
    if class_a_name not in classes or class_b_name not in classes:
        # Fallback to defaults if names mismatc
        class_a_name = classes[0]
        class_b_name = classes[1]
        
    idx_a = le.transform([class_a_name])[0]
    idx_b = le.transform([class_b_name])[0]
    
    # Get average profile for these classes to act as "pure" prototypes
    # Or just pick random single cells
    cells_a = X_test[y_test == idx_a]
    cells_b = X_test[y_test == idx_b]
    
    if len(cells_a) == 0 or len(cells_b) == 0:
        print("Error: Classes not found in test set")
        return

    # Use centroids for cleaner mixing trajectory (less noise from individual cell variation)
    proto_a = np.mean(cells_a, axis=0)
    proto_b = np.mean(cells_b, axis=0)
    
    # Generate mixing ratios
    ratios = np.linspace(0, 1, 21) # 0.0, 0.05, ..., 1.0
    
    mixed_data = []
    for r in ratios:
        # Mix: (1-r)*A + r*B
        # 0 -> A, 1 -> B
        mix = (1-r)*proto_a + r*proto_b
        mixed_data.append(mix)
        
    mixed_data = np.array(mixed_data)
    
    # Predict with Student
    student.eval()
    bx_mix = torch.tensor(mixed_data).float().to(DEVICE)
    beliefs, uncertainty = student.predict_evidence(bx_mix)
    beliefs = beliefs.cpu().numpy()
    uncertainty = uncertainty.cpu().numpy().flatten()
    
    # Extract prob for Class A and Class B
    prob_a = beliefs[:, idx_a]
    prob_b = beliefs[:, idx_b]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, prob_a, label=f'Belief: {class_a_name}', marker='o')
    plt.plot(ratios, prob_b, label=f'Belief: {class_b_name}', marker='o')
    plt.plot(ratios, uncertainty, label='Uncertainty (Background)', linestyle='--', color='gray', marker='x')
    
    # Sum of these 3
    # Note: sum(beliefs) + u = 1.
    # But we are only plotting Belief A and Belief B. There are other classes.
    # If the mix passes through regions closer to other classes, their beliefs might rise.
    # But usually it should be dominated by A and B.
    
    plt.title(f'Student LDL Demixing: {class_a_name} -> {class_b_name}')
    plt.xlabel(f'Mixing Ratio (0={class_a_name}, 1={class_b_name})')
    plt.ylabel('Mass (Belief / Uncertainty)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('reports/student_demixing_plot.png')
    print("Saved plot to reports/student_demixing_plot.png")

if __name__ == "__main__":
    main()

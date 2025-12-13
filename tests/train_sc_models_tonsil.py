
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scanpy as sc
import anndata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

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

# MLP Baseline
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

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, is_baseline=False):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            if is_baseline:
                # Standard classification
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            else:
                # Custom models fit method usually handles optimization, 
                # but here we are converting them to torch modules wrapper manually?
                # No, ImprovedLEVI etc are classes with .fit().
                # But to have a unified loop or control, we might need to access their internal module.
                # Actually, our LE models interact differently. 
                # Let's NOT use a generic loop for non-baselines. 
                # They implement .fit().
                pass

            if is_baseline:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # Validation
        if is_baseline:
             model.eval()
             correct = 0
             total = 0
             with torch.no_grad():
                 for batch_x, batch_y in test_loader:
                     batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                     outputs = model(batch_x)
                     _, predicted = torch.max(outputs.data, 1)
                     total += batch_y.size(0)
                     correct += (predicted == batch_y).sum().item()
             
             if (epoch+1) % 5 == 0:
                 print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}, Acc {100*correct/total:.2f}%")

def main():
    # 1. Load and Preprocess Data
    print("Loading data...")
    adata = anndata.read_h5ad('data/scrna_tonsil.h5ad')
    
    # Preprocessing
    print("Preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize and Log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Highly Variable Genes
    print("Selecting HVGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    # Labels
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['cell_type'].values)
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    
    print(f"Data shape: {X.shape}, Classes: {n_classes}")
    
    # Save Metadata
    os.makedirs('models/sc_tonsil', exist_ok=True)
    with open('models/sc_tonsil/preprocessing_metadata.pkl', 'wb') as f:
        pickle.dump({
            'le': le,
            'var_names': adata.var_names.tolist(), # To filter new data if needed
            'n_features': n_features,
            'n_classes': n_classes
        }, f)
        
    # Split indices first to keep track for strict testing
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    # Save Indices for Strict Mixing Test
    with open('models/sc_tonsil/split_indices.pkl', 'wb') as f:
        pickle.dump({'train': idx_train, 'test': idx_test}, f)
    
    # Tensors for Baseline
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # One-hot for LE models
    L_train = np.eye(n_classes)[y_train]
    
    # --- Train Models ---
    
    # 1. MLP Baseline
    print("\nTraining MLP Baseline...")
    mlp = MLPBaseline(n_features, n_classes).to(DEVICE)
    train_model(mlp, train_loader, test_loader, epochs=20, is_baseline=True)
    torch.save(mlp.state_dict(), 'models/sc_tonsil/MLPBaseline.pth')
    
    # 2. HybridLEVI
    print("\nTraining HybridLEVI...")
    hybrid = HybridLEVI(
        n_features=n_features, n_outputs=n_classes, encoder_type='mlp',
        n_hidden=256, epochs=20, batch_size=64, lr=1e-3, device=DEVICE
    )
    hybrid.fit(X_train, L_train)
    torch.save(hybrid.state_dict(), 'models/sc_tonsil/HybridLEVI.pth')
    # Eval
    probs = hybrid.predict(X_test, np.ones((len(X_test), n_classes))/n_classes) # dummy L
    acc = np.mean(np.argmax(probs, axis=1) == y_test)
    print(f"HybridLEVI Acc: {acc*100:.2f}%")
    
    # 3. ConcentrationLE
    print("\nTraining ConcentrationLE...")
    conc = ConcentrationLE(
        n_features=n_features, n_outputs=n_classes, encoder_type='mlp',
        n_hidden=256, epochs=20, batch_size=64, lr=1e-3, device=DEVICE
    )
    conc.fit(X_train, L_train)
    torch.save(conc.state_dict(), 'models/sc_tonsil/ConcentrationLE.pth')
    # Eval
    _, alpha = conc(torch.tensor(X_test).float().to(DEVICE))
    probs = (alpha / torch.sum(alpha, dim=1, keepdim=True)).detach().cpu().numpy()
    acc = np.mean(np.argmax(probs, axis=1) == y_test)
    print(f"ConcentrationLE Acc: {acc*100:.2f}%")
    
    # 4. ImprovedLEVI
    print("\nTraining ImprovedLEVI...")
    impr = ImprovedLEVI(
        n_features=n_features, n_outputs=n_classes, encoder_type='mlp',
        n_hidden=256, epochs=20, batch_size=64, lr=1e-3, device=DEVICE
    )
    impr.fit(X_train, L_train)
    torch.save(impr.state_dict(), 'models/sc_tonsil/ImprovedLEVI.pth')
    # Eval
    probs = impr.predict(X_test, np.ones((len(X_test), n_classes))/n_classes)
    acc = np.mean(np.argmax(probs, axis=1) == y_test)
    print(f"ImprovedLEVI Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    main()

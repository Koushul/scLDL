import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from scLDL.lesc.lesc import LESC

def load_sc_data(h5ad_path):
    print(f"Loading single-cell data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Features: Use normalized counts if available, else X
    if 'normalized_count' in adata.layers:
        X = adata.layers['normalized_count']
    else:
        X = adata.X
        
    if hasattr(X, 'toarray'):
        X = X.toarray()
        
    # Labels: cell_type
    y_labels = adata.obs['cell_type'].values
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    L = enc.fit_transform(y_labels.reshape(-1, 1))
    
    class_names = enc.categories_[0]
    y_indices = np.argmax(L, axis=1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {L.shape}")
    print(f"Classes: {class_names}")
    
    return X, L, y_indices, class_names

def main():
    h5ad_path = "/Users/koush/Projects/scLDL/data/adata.h5ad"
    if not os.path.exists(h5ad_path):
        print(f"Error: File not found at {h5ad_path}")
        return

    # Load Data
    X, L, y_indices, class_names = load_sc_data(h5ad_path)
    
    # Split Data
    X_train, X_test, L_train, L_test, y_train, y_test = train_test_split(
        X, L, y_indices, test_size=0.2, random_state=42, stratify=y_indices
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Check size for memory safety warning
    if X_train.shape[0] > 10000:
        print(f"WARNING: Training on {X_train.shape[0]} samples. Kernel matrix will be large.")
    
    # Initialize LESC
    # Heuristic for sigma
    print("Calculating mean distance for sigma heuristic...")
    # Taking a subsample for mean dist calculation if dataset is huge, to save time
    subsample_idx = np.random.choice(X_train.shape[0], min(2000, X_train.shape[0]), replace=False)
    X_sub = X_train[subsample_idx]
    mean_dist = np.mean(np.linalg.norm(X_sub - np.mean(X_sub, axis=0), axis=1))
    sigma = mean_dist * 2
    print(f"Using sigma={sigma:.4f}")
    
    lesc = LESC(lambda_param=0.1, beta=0.1, kernel='rbf', sigma=sigma)
    
    # Fit
    print("Training LESC...")
    start_time = time.time()
    try:
        lesc.fit(X_train, L_train)
    except MemoryError:
        print("Error: Out of Memory during training (likely kernel matrix computation).")
        return
        
    # Predict
    print("Predicting on Test data...")
    L_pred = lesc.predict(X_test)
    y_pred = np.argmax(L_pred, axis=1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"LESC Accuracy: {acc*100:.2f}%")
    print(f"Total Time: {elapsed_time:.2f}s")

if __name__ == "__main__":
    main()

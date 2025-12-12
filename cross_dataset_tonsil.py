
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import scanpy as sc
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import models
from scLDL.label_enhancer import LIBLE, ConcentrationLE, HybridLEVI

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

# Label Mapping (Train -> Test)
LABEL_MAP = {
    "DZ GC": "GC Dark Zone",
    "LZ GC": "GC Light Zone",
    "Naive": "B_naive",
    "MBC": "B_memory",
    "Plasmablast": "plasma",
    "TfH": "T_follicular_helper",
    "FDC": "FDC",
    # Add exact matches if any
    "GC Dark Zone": "GC Dark Zone",
    "GC Light Zone": "GC Light Zone",
    "B_naive": "B_naive",
    "B_memory": "B_memory",
    "plasma": "plasma", 
    "T_follicular_helper": "T_follicular_helper"
}

def preprocess_and_harmonize(train_path, test_path, test_csv, n_hvgs=3000):
    print(f"Loading Test: {test_path}")
    adata_test = sc.read_h5ad(test_path)
    
    # Merge CSV labels
    print(f"Loading Test Labels: {test_csv}")
    df_labels = pd.read_csv(test_csv, index_col=0)
    common = adata_test.obs_names.intersection(df_labels.index)
    adata_test = adata_test[common].copy()
    adata_test.obs['final_label'] = df_labels.loc[common, 'cell_type_2']
    
    # Filter Test Test to only include target classes
    target_classes = set(adata_test.obs['final_label'].unique())
    # But wait, we define classes based on intersection with Train mappable labels
    
    print(f"Loading Train: {train_path}")
    adata_train = sc.read_h5ad(train_path)
    
    # Map Train Labels
    if 'author_cell_type' not in adata_train.obs:
         # Try cell_type if author not present
         label_key = 'cell_type'
    else:
         label_key = 'author_cell_type'
         
    adata_train.obs['mapped_label'] = adata_train.obs[label_key].map(LABEL_MAP)
    
    # Drop unmapped
    adata_train = adata_train[~adata_train.obs['mapped_label'].isna()].copy()
    print(f"Train subset after mapping: {adata_train.shape}")

    # Use feature_name as index for Train if available
    if 'feature_name' in adata_train.var:
        print("Using 'feature_name' as gene symbols for Train.")
        adata_train.var_names = adata_train.var['feature_name'].astype(str)
        adata_train.var_names_make_unique()
    
    # Classes (intersection)
    train_classes = set(adata_train.obs['mapped_label'].unique())
    test_classes = set(adata_test.obs['final_label'].unique())
    common_classes = sorted(list(train_classes.intersection(test_classes)))
    
    print(f"Common Classes ({len(common_classes)}): {common_classes}")
    
    # Filter both to common classes
    adata_train = adata_train[adata_train.obs['mapped_label'].isin(common_classes)].copy()
    adata_test = adata_test[adata_test.obs['final_label'].isin(common_classes)].copy()
    
    # Normalize & Log
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    
    # Feature Intersection
    common_genes = list(set(adata_train.var_names).intersection(adata_test.var_names))
    print(f"Common Genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found! Check gene ID formats.")
    
    adata_train = adata_train[:, common_genes].copy()
    adata_test = adata_test[:, common_genes].copy()
    
    # Select HVGs on Train (or just take all common if small?)
    # Test has only ~3000 genes usually.
    # Let's rely on Train's HVG selection within the common set.
    sc.pp.highly_variable_genes(adata_train, n_top_genes=min(n_hvgs, len(common_genes)), subset=True)
    hvgs = adata_train.var_names
    adata_test = adata_test[:, hvgs].copy()
    
    print(f"Final Train: {adata_train.shape}")
    print(f"Final Test: {adata_test.shape}")
    
    # Extract
    X_train = adata_train.X.toarray().astype('float32') if hasattr(adata_train.X, 'toarray') else adata_train.X.astype('float32')
    y_train = adata_train.obs['mapped_label'].values
    
    X_test = adata_test.X.toarray().astype('float32') if hasattr(adata_test.X, 'toarray') else adata_test.X.astype('float32')
    y_test = adata_test.obs['final_label'].values
    
    # Encode
    le = LabelEncoder()
    le.fit(common_classes) # Fit on defined common classes
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)
    
    enc = OneHotEncoder(sparse_output=False)
    L_train = enc.fit_transform(y_train_int.reshape(-1, 1))
    # L_test not needed for training, but good for simple verification
    
    return X_train, L_train, y_train_int, X_test, y_test_int, le.classes_

def plot_distributions(model, X_test, y_test, y_pred, le_classes, model_name):
    # Select indices
    correct_indices = np.where(y_pred == y_test)[0]
    incorrect_indices = np.where(y_pred != y_test)[0]
    
    indices = []
    if len(correct_indices) >= 5: indices.extend(np.random.choice(correct_indices, 5, replace=False))
    if len(incorrect_indices) >= 5: indices.extend(np.random.choice(incorrect_indices, 5, replace=False))
    
    pred_dist = model.predict(X_test)
    if pred_dist.shape[1] > len(le_classes):
        beliefs = pred_dist[:, :-1]
        beliefs_sum = np.sum(beliefs, axis=1, keepdims=True) + 1e-10
        probs = beliefs / beliefs_sum
    else:
        probs = pred_dist
        
    n_cols = 5
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        true_label = le_classes[y_test[idx]]
        pred_label = le_classes[y_pred[idx]]
        status = "CORRECT" if y_test[idx] == y_pred[idx] else "WRONG"
        color = 'green' if status == "CORRECT" else 'red'
        
        # Plot Top 5 probs
        p = probs[idx]
        top_k = np.argsort(p)[-5:]
        
        ax.barh(np.arange(5), p[top_k], color=color, alpha=0.6)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels([le_classes[j] for j in top_k])
        ax.set_title(f"{status}\nTrue: {true_label}\nPred: {pred_label}", fontsize=9)
        ax.set_xlim(0, 1.0)
        
    plt.suptitle(f"{model_name} Example Predictions (Distributions)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"tonsil_dist_{model_name.lower().replace(' ', '_')}.png")
    print(f"Saved distro plot for {model_name}")

def main():
    train_path = 'data/scrna_tonsil.h5ad'
    test_path = 'data/adata.h5ad'
    test_csv = 'data/tonsil_cell_types.csv'
    
    X_train, L_train, y_train, X_test, y_test, class_names = preprocess_and_harmonize(train_path, test_path, test_csv)
    
    n_features = X_train.shape[1]
    n_outputs = len(class_names)
    
    common_params = {
        "n_features": n_features,
        "n_outputs": n_outputs,
        "n_hidden": 512,
        "epochs": 20,
        "batch_size": 256,
        "device": device,
        "encoder_type": 'mlp'
    }
    
    models = [
        ("LIBLE", LIBLE(**common_params)),
        ("ConcentrationLE", ConcentrationLE(**common_params)),
        ("HybridLEVI", HybridLEVI(**{**common_params, "n_latent": 512, "gamma": 10.0, "alpha": 0.01}))
    ]
    
    results = []
    
    for name, model in models:
        print(f"\n--- {name} ---")
        start = time.time()
        model.fit(X_train, L_train)
        
        # Predict
        D_pred = model.predict(X_test)
        if D_pred.shape[1] > n_outputs:
            y_pred = np.argmax(D_pred[:, :-1], axis=1)
        else:
            y_pred = np.argmax(D_pred, axis=1)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} Accuracy: {acc*100:.2f}%")
        
        plot_distributions(model, X_test, y_test, y_pred, class_names, name)
        
        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "F1": f1
        })
        
    pd.DataFrame(results).to_csv("tonsil_cross_results.csv", index=False)

if __name__ == "__main__":
    main()

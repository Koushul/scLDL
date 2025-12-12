
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import scanpy as sc
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import models
from scLDL.label_enhancer import HybridLEVI

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

def load_tonsil_data(h5ad_path, csv_path):
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Original shape: {adata.shape}")
    
    # Load New Labels
    print(f"Loading labels from {csv_path}...")
    df_labels = pd.read_csv(csv_path, index_col=0) # Index is Barcode
    
    # Merge
    # Ensure barcodes match
    common_barcodes = adata.obs_names.intersection(df_labels.index)
    print(f"Found {len(common_barcodes)} common cells out of {adata.n_obs} original.")
    
    adata = adata[common_barcodes].copy()
    adata.obs['cell_type_2'] = df_labels.loc[common_barcodes, 'cell_type_2']
    
    # Preprocessing
    print("Preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print(f"Selecting top 3000 HVGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    
    # Extract X
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    X = X.astype('float32')
    
    # Extract Labels
    y_labels = adata.obs['cell_type_2'].values
    print(f"Label Counts:\n{pd.Series(y_labels).value_counts()}")
    
    le = LabelEncoder()
    y_int = le.fit_transform(y_labels)
    enc = OneHotEncoder(sparse_output=False)
    L = enc.fit_transform(y_int.reshape(-1, 1))
    class_names = le.classes_
    
    print(f"Final Data shape: {X.shape}")
    print(f"Labels shape: {L.shape}")
    
    return X, L, y_int, class_names

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_per_class_accuracy(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort
    indices = np.argsort(per_class_acc)[::-1]
    sorted_acc = per_class_acc[indices]
    sorted_names = [class_names[i] for i in indices]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_names, y=sorted_acc, palette='viridis')
    plt.title(title)
    plt.xlabel('Cell Type')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    h5ad_path = 'data/adata.h5ad'
    csv_path = 'data/tonsil_cell_types.csv'
    
    X, L, y_int, class_names = load_tonsil_data(h5ad_path, csv_path)
    
    # Split
    X_train, X_test, L_train, L_test, y_train, y_test = train_test_split(
        X, L, y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    
    n_features = X.shape[1]
    n_outputs = len(class_names)
    
    # Configure HybridLEVI (Re-engineered Settings)
    hybrid_params = {
        "n_features": n_features,
        "n_outputs": n_outputs,
        "n_hidden": 512,        
        "n_latent": 512,        # Matched
        "epochs": 30,           # Dataset is small, train longer?
        "batch_size": 128,      # Small dataset
        "lr": 1e-3,
        "device": device,
        "encoder_type": 'mlp',
        "gamma": 10.0,
        "alpha": 0.01
    }
    
    print("\n--- Training HybridLEVI ---")
    model = HybridLEVI(**hybrid_params)
    start = time.time()
    try:
        model.fit(X_train, L_train)
    except Exception as e:
        print(f"Training failed: {e}")
        return
        
    print(f"Training Time: {time.time() - start:.2f}s")
    
    print("Predicting...")
    try:
        D_pred = model.predict(X_test)
        if D_pred.shape[1] > n_outputs:
            y_pred = np.argmax(D_pred[:, :-1], axis=1)
        else:
            y_pred = np.argmax(D_pred, axis=1)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"HybridLEVI Accuracy: {acc*100:.2f}%")
        print(f"HybridLEVI F1 Score: {f1:.4f}")
        
        plot_confusion_matrix(y_test, y_pred, class_names, "Tonsil: HybridLEVI Confusion Matrix", "cm_tonsil_hybridlevi.png")
        plot_per_class_accuracy(y_test, y_pred, class_names, "Tonsil: HybridLEVI Per-Class Acc", "acc_tonsil_hybridlevi.png")
        
        # Save results
        res = pd.DataFrame([{
            'Algorithm': 'HybridLEVI',
            'Accuracy': acc,
            'F1': f1,
            'HVGs': 3000,
            'Latent': 512
        }])
        res.to_csv("tonsil_results.csv", index=False)
        
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()

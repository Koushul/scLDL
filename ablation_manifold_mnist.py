import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import models
from scLDL.label_enhancer import ImprovedLEVI

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

def load_mnist_data(n_samples=None):
    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Normalize to [0, 1]
    X /= 255.0
    
    # Reshape to (N, 1, 28, 28) for ResNet
    # Although ImprovedLEVI handles reshaping if input_shape is passed, 
    # let's keep X flat (N, 784) and pass input_shape to model.
    
    if n_samples is not None:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    L = enc.fit_transform(y.reshape(-1, 1))
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {L.shape}")
    
    return X, L, y

def evaluate_model(model, name, X_train, L_train, X_test, L_test, y_test):
    print(f"--- Evaluating {name} ---")
    start_time = time.time()
    
    print("Fitting on Train data...")
    model.fit(X_train, L_train)
    
    print("Predicting on Test data...")
    D_pred = model.predict(X_test, L_test)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Accuracy
    y_pred = np.argmax(D_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    
    res = {
        "Algorithm": name,
        "Accuracy": acc,
        "Time (s)": elapsed_time
    }
    
    # Return history if available
    history = model.history if hasattr(model, 'history') else {}
    
    return res, history

def main():
    # Load Data (Subset for speed, but large enough for stable results)
    # Using 10k samples
    X, L, y = load_mnist_data(n_samples=10000)
    
    # Split
    X_train, X_test, L_train, L_test, y_train, y_test = train_test_split(
        X, L, y, test_size=0.2, random_state=42, stratify=y
    )
    
    n_features = X.shape[1]
    n_outputs = L.shape[1]
    input_shape = (1, 28, 28)
    
    params = {
        "n_features": n_features,
        "n_outputs": n_outputs,
        "n_hidden": 128,
        "epochs": 20, 
        "batch_size": 128,
        "device": device,
        "encoder_type": "resnet",
        "input_shape": input_shape,
        "use_mixup": True,
        "mixup_alpha": 1.0
    }
    
    results = []
    
    # 1. Without Manifold Regularization (with Mixup)
    print("\n1. Training ImprovedLEVI (Mixup, No Manifold)...")
    base_params = params.copy()
    base_params['manifold_reg'] = 0.0
    model_base = ImprovedLEVI(**base_params)
    res_base, hist_base = evaluate_model(model_base, "Mixup + No Manifold", X_train, L_train, X_test, L_test, y_test)
    results.append(res_base)
    
    # 2. With Manifold Regularization (with Mixup)
    print("\n2. Training ImprovedLEVI (Mixup + Manifold)...")
    reg_params = params.copy()
    reg_params['manifold_reg'] = 1.0 # Standard strength
    model_reg = ImprovedLEVI(**reg_params)
    res_reg, hist_reg = evaluate_model(model_reg, "Mixup + Manifold", X_train, L_train, X_test, L_test, y_test)
    results.append(res_reg)
    
    # Results
    df = pd.DataFrame(results)
    print("\n=== Ablation Results ===")
    print(df)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot Loss (Total)
    plt.subplot(1, 2, 1)
    if 'loss' in hist_base:
        plt.plot(hist_base['loss'], label='No Manifold (Total Loss)', linestyle='--')
    if 'loss' in hist_reg:
        plt.plot(hist_reg['loss'], label='With Manifold (Total Loss)')
    plt.title("Total Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot Manifold Loss (verify it exists or is 0)
    plt.subplot(1, 2, 2)
    if 'manifold_loss' in hist_reg:
        plt.plot(hist_reg['manifold_loss'], label='Manifold Loss Term (Reg Model)', color='orange')
    plt.title("Manifold Loss Component")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("ablation_mnist_manifold.png")
    print("Plot saved to ablation_mnist_manifold.png")

if __name__ == "__main__":
    main()

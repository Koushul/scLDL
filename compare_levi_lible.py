import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from scLDL.label_enhancer import LIBLE, LEVI, HybridLE
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Flatten images and create one-hot labels
    X_train = train_dataset.data.float().view(-1, 28*28) / 255.0
    y_train = train_dataset.targets
    L_train = F.one_hot(y_train, num_classes=10).float()
    
    X_test = test_dataset.data.float().view(-1, 28*28) / 255.0
    y_test = test_dataset.targets
    L_test = F.one_hot(y_test, num_classes=10).float()
    
    return X_train.numpy(), L_train.numpy(), X_test.numpy(), L_test.numpy(), y_test.numpy()

def evaluate_model(model, name, X_train, L_train, X_test, L_test, y_test):
    print(f"--- Evaluating {name} ---")
    start_time = time.time()
    
    print("Fitting on Train data...")
    model.fit(X_train, L_train)
    
    print("Predicting on Test data...")
    if name == "LEVI":
        # LEVI requires L for prediction in this implementation
        D_pred = model.predict(X_test, L_test)
    else: # LIBLE and HybridLE
        D_pred = model.predict(X_test)
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Evaluate accuracy (max prob vs true label)
    y_pred = np.argmax(D_pred, axis=1)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"{name} Accuracy: {accuracy:.2%}")
    print(f"{name} Time: {duration:.2f}s")
    
    return {"Algorithm": name, "Accuracy": accuracy, "Time (s)": duration}, D_pred, y_pred

def plot_confusion_matrices(y_true, predictions, names):
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    for i, (y_pred, name) in enumerate(zip(predictions, names)):
        cm = confusion_matrix(y_true, y_pred)
        ax = axes[i] if n_models > 1 else axes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    print("Confusion matrices saved to confusion_matrices.png")

def main():
    print("Loading MNIST data...")
    X_train, L_train, X_test, L_test, y_test = get_mnist_data()
    
    results_data = []
    predictions = []
    pred_labels = []
    model_names = []
    
    # Common params
    params = dict(n_features=784, n_outputs=10, epochs=10, batch_size=64, encoder_type='cnn', input_shape=(1, 28, 28))
    
    # LIBLE
    lible = LIBLE(**params)
    res, D_lible, y_lible = evaluate_model(lible, "LIBLE", X_train, L_train, X_test, L_test, y_test)
    results_data.append(res)
    predictions.append(D_lible)
    pred_labels.append(y_lible)
    model_names.append("LIBLE")
    
    # LEVI
    levi = LEVI(**params)
    res, D_levi, y_levi = evaluate_model(levi, "LEVI", X_train, L_train, X_test, L_test, y_test)
    results_data.append(res)
    predictions.append(D_levi)
    pred_labels.append(y_levi)
    model_names.append("LEVI")
    
    # HybridLE
    hybrid = HybridLE(**params)
    res, D_hybrid, y_hybrid = evaluate_model(hybrid, "HybridLE", X_train, L_train, X_test, L_test, y_test)
    results_data.append(res)
    predictions.append(D_hybrid)
    pred_labels.append(y_hybrid)
    model_names.append("HybridLE")
    
    # Create table
    df = pd.DataFrame(results_data)
    print("\n=== Comparison Results ===")
    print(df)
    
    # Confusion Matrices
    print("Generating confusion matrices...")
    plot_confusion_matrices(y_test, pred_labels, model_names)
    
    # Visualization of Distributions
    print("Generating distribution comparison plots...")
    
    # Identify categories
    # We want to find interesting cases.
    # 1. All Wrong
    # 2. Hybrid Right, Others Wrong
    # 3. All Right
    
    all_wrong = np.where((y_lible != y_test) & (y_levi != y_test) & (y_hybrid != y_test))[0]
    hybrid_right_others_wrong = np.where((y_hybrid == y_test) & (y_lible != y_test) & (y_levi != y_test))[0]
    all_right = np.where((y_lible == y_test) & (y_levi == y_test) & (y_hybrid == y_test))[0]
    
    categories = [
        ("All Models Wrong", all_wrong),
        ("Hybrid Right, Others Wrong", hybrid_right_others_wrong),
        ("All Models Right", all_right)
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, (cat_name, indices) in enumerate(categories):
        if len(indices) < 3:
            print(f"Not enough samples for category '{cat_name}'. Found {len(indices)}.")
            # If empty, skip or take what we have
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, 3, replace=False)
            
        for j, idx in enumerate(selected_indices):
            if j >= 3: break
            
            ax = axes[i, j]
            
            # Plot distributions
            classes = np.arange(10)
            width = 0.25
            ax.bar(classes - width, D_lible[idx], width=width, label='LIBLE', alpha=0.7)
            ax.bar(classes, D_levi[idx], width=width, label='LEVI', alpha=0.7)
            ax.bar(classes + width, D_hybrid[idx], width=width, label='HybridLE', alpha=0.7)
            
            # Mark true label
            true_label = y_test[idx]
            ax.axvline(x=true_label, color='red', linestyle='--', label='True Label')
            
            ax.set_title(f"{cat_name}\nTrue: {true_label}\nL:{y_lible[idx]} V:{y_levi[idx]} H:{y_hybrid[idx]}")
            ax.set_xticks(classes)
            if i == 0 and j == 0:
                ax.legend()
                
    plt.suptitle("LIBLE vs LEVI vs HybridLE Distribution Comparison")
    plt.savefig("distribution_comparison.png")
    print("Plot saved to distribution_comparison.png")

if __name__ == "__main__":
    main()


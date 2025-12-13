import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scLDL.label_enhancer import DiffLEVI, ImprovedLEVI, ConcentrationLE

def get_mnist_subset(n_train=2000, n_test=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download if needed
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Subset
    np.random.seed(42)
    train_idx = np.random.choice(len(train_data), n_train, replace=False)
    test_idx = np.random.choice(len(test_data), n_test, replace=False)
    
    # Extract tensors
    X_train = train_data.data[train_idx].float().unsqueeze(1) / 255.0
    y_train = train_data.targets[train_idx]
    
    X_test = test_data.data[test_idx].float().unsqueeze(1) / 255.0
    y_test = test_data.targets[test_idx]
    
    return X_train, y_train, X_test, y_test

def corrupt_labels(targets, n_classes=10, noise_rate=0.4):
    """
    Flip noise_rate fraction of labels to random other class.
    Returns: 
        y_noisy (one-hot), y_true (one-hot)
    """
    n_samples = len(targets)
    y_true = F.one_hot(targets, num_classes=n_classes).float()
    
    # Generate noise
    n_noise = int(n_samples * noise_rate)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    
    y_noisy_indices = targets.clone()
    # Random swap
    random_labels = torch.randint(0, n_classes, (n_noise,))
    y_noisy_indices[noise_idx] = random_labels
    
    y_noisy = F.one_hot(y_noisy_indices, num_classes=n_classes).float()
    
    return y_noisy, y_true

def plot_distribution_comparison(X, y_true_label, predictions, model_names, save_path="mnist_dist_comparison.png"):
    n_models = len(predictions)
    fig, axs = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 3))
    
    # Plot Image
    axs[0].imshow(X.squeeze(), cmap='gray')
    axs[0].set_title(f"True Label: {y_true_label}")
    axs[0].axis('off')
    
    # Plot Predictions
    classes = np.arange(10)
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        axs[i+1].bar(classes, pred)
        axs[i+1].set_title(f"{name}\nMax: {np.argmax(pred)}")
        axs[i+1].set_ylim(0, 1)
        axs[i+1].set_xticks(classes)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

def evaluate_model(model_cls, model_name, X_train, L_train_noisy, X_test, y_test_true, L_train_true, **kwargs):
    print(f"\n--- Training {model_name} ---")
    
    n_features = 784 # ignored for resnet usually but required by init
    n_classes = 10
    input_shape = (1, 28, 28)
    
    model = model_cls(
        n_features=n_features,
        n_outputs=n_classes,
        input_shape=input_shape,
        **kwargs
    )
    
    # Fit on noisy data
    model.fit(X_train, L_train_noisy)
    
    # Predict on Test (Accuracy check)
    if "Concentration" in model_name:
         # ConcentrationLE predict output: [Beliefs, Uncertainty]
         # We just want beliefs for accuracy
         probs = model.predict(X_test)[:, :n_classes]
    elif "ImprovedLEVI" in model_name:
         # ImprovedLEVI requires L. For test, we don't have L.
         # We'll use uniform/dummy L.
         dummy_L = torch.zeros(len(X_test), n_classes).float()
         probs = model.predict(X_test, dummy_L)
    else:
         probs = model.predict(X_test)
         
    # Handle ImprovedLEVI predict returning tuple if I recall correctly on fit, but predict usually returns numpy
    
    y_pred = np.argmax(probs, axis=1)
    acc = (y_pred == y_test_true.numpy()).mean()
    
    # Recover Training Labels (Check if it fixed the noise)
    subset_size = 500
    if "Concentration" in model_name:
         probs_train = model.predict(X_train[:subset_size])[:, :n_classes]
    elif "ImprovedLEVI" in model_name:
         # For recovery, we pass the noisy labels we want to enhance!
         probs_train = model.predict(X_train[:subset_size], L_train_noisy[:subset_size])
    else:
         probs_train = model.predict(X_train[:subset_size])
         
    # KL Divergence from True Soft Labels (Here True is One-Hot, so KL reduces to -log(p_true_class))
    # We use Clean True labels for evaluation
    y_train_true_subset = L_train_true[:subset_size].numpy()
    
    eps = 1e-8
    kl = np.sum(y_train_true_subset * np.log((y_train_true_subset + eps) / (probs_train + eps)), axis=1).mean()
    
    print(f"{model_name} Results:")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Train Recovery KL: {kl:.4f}")
    
    return probs_train, acc, kl

def main():
    X_train, y_train, X_test, y_test = get_mnist_subset()
    
    # Corrupt Training Labels
    L_train_noisy, L_train_true = corrupt_labels(y_train, noise_rate=0.4)
    
    print(f"Data Loaded. Train: {len(X_train)}, Test: {len(X_test)}")
    print("Training labels corrupted with 40% noise.")
    
    results = {}
    
    # 1. ImprovedLEVI
    # Uses ResNet18 by default if type='resnet'
    res_levi = evaluate_model(
        ImprovedLEVI, "ImprovedLEVI", X_train, L_train_noisy, X_test, y_test, L_train_true,
        encoder_type='resnet', 
        epochs=10, 
        batch_size=32,
        lr=1e-3,
        alpha=0.1
    )
    results['ImprovedLEVI'] = res_levi
    
    # 2. DiffLEVI
    res_diff = evaluate_model(
        DiffLEVI, "DiffLEVI", X_train, L_train_noisy, X_test, y_test, L_train_true,
        encoder_type='resnet',
        epochs=10,
        batch_size=32,
        timesteps=50,
        lr=2e-4 # Diffusion usually simpler lower LR
    )
    results['DiffLEVI'] = res_diff
    
    # 3. ConcentrationLE
    res_conc = evaluate_model(
        ConcentrationLE, "ConcentrationLE", X_train, L_train_noisy, X_test, y_test, L_train_true,
        encoder_type='resnet',
        epochs=10,
        batch_size=32,
        lr=1e-3
    )
    results['ConcentrationLE'] = res_conc
    
    print("\n--- Final Summary ---")
    print(f"{'Model':<20} | {'Test Acc':<10} | {'Recovery KL':<12}")
    print("-" * 50)
    for name, (_, acc, kl) in results.items():
        print(f"{name:<20} | {acc*100:.2f}%     | {kl:.4f}")
        
    # Visualize one sample
    idx = 0
    predictions = [results[name][0][idx] for name in results]
    plot_distribution_comparison(X_train[idx], np.argmax(L_train_true[idx]), predictions, list(results.keys()))

if __name__ == "__main__":
    main()

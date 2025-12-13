
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.label_enhancer import DiffLEVI, ImprovedLEVI, ConcentrationLE, HybridLEVI

def get_mnist_subset(n_train=60000, n_test=10000):
    transform = transforms.Compose([
        transforms.ToTensor(), # [0,1]
    ])
    
    # Download MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Select subset
    if n_train < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:n_train]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
    if n_test < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:n_test]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    # Load all into memory
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    
    # Create One-Hot Labels (Clean)
    L_train = F.one_hot(y_train, num_classes=10).float()
    
    return X_train, L_train, y_train, X_test, y_test

def predict_proba_concentration(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(model.device)
        # Check signature of model
        # ConcentrationLE forward -> evidence, alpha
        # HybridLEVI forward -> mean, logvar, z, x_hat, evidence. Then alpha=evidence+1
        
        if isinstance(model, ConcentrationLE):
            evidence, alpha = model(X_tensor)
        elif isinstance(model, HybridLEVI):
             _, _, _, _, evidence = model(X_tensor)
             alpha = evidence + 1
             
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        return probs.cpu().numpy()

def main():
    # 1. Load Data
    print("Loading Full MNIST...")
    X_train, L_train, y_train, X_test, y_test = get_mnist_subset() # Defaults to full
    
    input_shape = (1, 28, 28)
    n_features = 784
    n_outputs = 10
    epochs = 10
    
    models = {}
    
    # 2. Initialize Models
    print("\n--- Initializing Models ---")
    
    # DiffLEVI
    models['DiffLEVI'] = DiffLEVI(
        n_features=n_features,
        n_outputs=n_outputs,
        encoder_type='resnet',
        input_shape=input_shape,
        epochs=epochs,
        batch_size=64,
        timesteps=1000,
        lr=2e-4
    )
    
    # ImprovedLEVI
    models['ImprovedLEVI'] = ImprovedLEVI(
        n_features=n_features,
        n_outputs=n_outputs,
        encoder_type='resnet',
        input_shape=input_shape,
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        gamma=1.0,
        alpha=1.0
    )
    
    # ConcentrationLE
    models['ConcentrationLE'] = ConcentrationLE(
        n_features=n_features,
        n_outputs=n_outputs,
        encoder_type='resnet',
        input_shape=input_shape,
        epochs=epochs,
        batch_size=64,
        lr=1e-3
    )

    # HybridLEVI
    models['HybridLEVI'] = HybridLEVI(
        n_features=n_features,
        n_outputs=n_outputs,
        encoder_type='cnn', # HybridLEVI only supports 'cnn' in its __init__ logic shown in snippet?
        # Re-checking snippet: if encoder_type == 'cnn': ... else: mlp.
        # It doesn't seem to have explicit 'resnet' logic like others? 
        # Wait, line 935: "if encoder_type == 'cnn': ... else: self.encoder_fc = ..."
        # It doesn't look like updated HybridLEVI has ResNet support yet?
        # Let's check the snippet again.
        # Lines 935-956: if 'cnn' -> conv layers. else -> linear.
        # So passing 'resnet' would fall into 'else' and try to make linear layer with n_features=784.
        # MNIST is image. We should use 'cnn' for HybridLEVI for now to be safe, or update HybridLEVI?
        # "train and save HybridLEVI as well".
        # I'll use 'cnn' for HybridLEVI to ensure it works.
        input_shape=input_shape,
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        alpha=1.0,
        gamma=1.0
    )

    
    results = {}
    
    # 3. Train and Evaluate
    for name, model in models.items():
        print(f"\nProcessing {name} on {model.device}...")
        
        save_path = f"models/{name}_mnist.pth"
        
        if os.path.exists(save_path):
            print(f"Loading existing model from {save_path}...")
            model.load_state_dict(torch.load(save_path, map_location=model.device))
        else:
            print(f"Training {name}...")
            try:
                model.fit(X_train, L_train)
                # Save Model
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved {name} model to {save_path}")
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
            
        print(f"Evaluating {name}...")
        try:
            if name == 'ConcentrationLE' or name == 'HybridLEVI':
                probs = predict_proba_concentration(model, X_test)
            elif name == 'DiffLEVI':
                 # Use smaller n_samples for speed comparison if needed, but 5 is standard
                 probs = model.predict(X_test, n_samples=5)
            else:
                dummy_L = torch.ones(len(X_test), n_outputs) / n_outputs
                probs = model.predict(X_test, dummy_L) 
        except Exception as e:

            print(f"Prediction failed for {name}: {e}")
            probs = np.zeros((len(X_test), n_outputs))
            
        y_pred = np.argmax(probs, axis=1)
        acc = (y_pred == y_test.numpy()).mean()
        print(f"{name} Test Accuracy: {acc*100:.2f}%")
        results[name] = {'probs': probs, 'acc': acc}
        
        
        # Save Model block moved to training loop

        
    
    # 4. Comparative Distributions Plot
    print("\nGenerating comparative plots...")
    
    # Select 10 diverse samples (one for each digit)
    indices = []
    seen_digits = set()
    for idx, label in enumerate(y_test):
         label = int(label)
         if label not in seen_digits:
             indices.append(idx)
             seen_digits.add(label)
         if len(seen_digits) == 10:
             break
    indices.sort(key=lambda i: int(y_test[i]))
    
    # Plot
    # Rows: 10 samples
    # Cols: 1 (Image) + N_Models (Distributions)
    n_models = len(models)
    fig, axes = plt.subplots(10, n_models + 1, figsize=(4 * (n_models + 1), 20))
    
    for row, idx in enumerate(indices):
        # Image
        ax_img = axes[row, 0]
        ax_img.imshow(X_test[idx].squeeze(), cmap='gray')
        ax_img.set_title(f"True: {y_test[idx]}")
        ax_img.axis('off')
        
        # Distributions
        for col, (name, res) in enumerate(results.items()):
            ax = axes[row, col + 1]
            p = res['probs'][idx]
            
            # Bar plot
            bars = ax.bar(range(10), p, color='skyblue')
            
            # Highlight max
            pred_label = np.argmax(p)
            true_label = int(y_test[idx])
            
            if pred_label == true_label:
                bars[pred_label].set_color('green')
            else:
                bars[pred_label].set_color('red')
                bars[true_label].set_color('blue') # True label in blue if missed
                
            ax.set_ylim(0, 1)
            ax.set_title(f"{name}\nPred: {pred_label}")
            if row == 9:
                ax.set_xlabel("Class")
            else:
                ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig("comparison_distributions.png")
    print("Saved comparison plot to comparison_distributions.png")
    
    # Also save a simple accuracy bar chart
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accs = [res['acc'] for res in results.values()]
    plt.bar(names, accs, color=['purple', 'orange', 'cyan'])
    plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy Comparison on MNIST (10 Epochs)")
    plt.ylim(0, 1)
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')
    plt.savefig("comparison_accuracy.png")
    print("Saved accuracy comparison to comparison_accuracy.png")

if __name__ == "__main__":
    main()

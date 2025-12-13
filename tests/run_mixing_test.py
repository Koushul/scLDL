
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.label_enhancer import DiffLEVI, ImprovedLEVI, ConcentrationLE, HybridLEVI

# Consistent Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return test_dataset

class MixingDataset(Dataset):
    def __init__(self, mnist_dataset, n_samples=50000, n_digits=3):
        self.mnist_dataset = mnist_dataset
        self.n_samples = n_samples
        self.n_digits = n_digits
        self.samples = []
        
        print(f"Generating {n_samples} samples mixing {n_digits} digits...")
        
        # Pre-generate indices by label
        self.indices_by_label = [[] for _ in range(10)]
        for idx in range(len(mnist_dataset)):
            label = int(mnist_dataset[idx][1])
            self.indices_by_label[label].append(idx)
            
        for _ in tqdm(range(n_samples)):
            # Pick N distinct labels
            labels = np.random.choice(10, n_digits, replace=False)
            
            # Pick 1 random image for each label
            indices = [np.random.choice(self.indices_by_label[l]) for l in labels]
            
            # Pick weights
            if n_digits == 2:
                # Uniform mix for 2 digits
                w = np.random.uniform(0.2, 0.8)
                weights = [w, 1-w]
            else:
                # Dirichlet for >2 digits
                weights = np.random.dirichlet(np.ones(n_digits))
                
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
        
        # Mix
        img_mix = torch.zeros_like(self.mnist_dataset[indices[0]][0])
        for i, idx in enumerate(indices):
             img, _ = self.mnist_dataset[idx]
             img_mix += weights[i] * img
             
        # Return mixed image and the set of labels
        return img_mix, torch.tensor(s['labels'])

# Baseline Wrapper
class ResNetBaseline(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = models.resnet18(pretrained=False) # Weights loaded manually
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model.to(device)
        self.device = device
        
        # Transforms for inference (manual)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    def eval(self):
        self.model.eval()
        
    def predict(self, X, L=None):
        # X is numpy (N, 1, 28, 28) usually
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X).float().to(self.device)
            else:
                X_tensor = X
            
            # 1. Expand to 3 channels (grayscale -> RGB)
            if X_tensor.shape[1] == 1:
                X_tensor = X_tensor.repeat(1, 3, 1, 1)
                
            # 2. Normalize
            X_tensor = (X_tensor - self.norm_mean) / self.norm_std
            
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

def load_models(exclude=[]):
    input_shape = (1, 28, 28)
    n_features = 784
    n_outputs = 10
    
    models_dict = {}
    
    # Define model loaders
    def load_difflevi():
        m = DiffLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet', 
                     input_shape=input_shape, epochs=10, batch_size=64, timesteps=1000, lr=2e-4, device=DEVICE)
        m.load_state_dict(torch.load('models/DiffLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        return m

    def load_improvedlevi():
        m = ImprovedLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                         input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, gamma=1.0, alpha=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/ImprovedLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        return m
        
    def load_concentrationle():
        m = ConcentrationLE(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, device=DEVICE)
        m.load_state_dict(torch.load('models/ConcentrationLE_mnist.pth', map_location=DEVICE))
        m.eval()
        return m

    def load_hybridlevi():
        m = HybridLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='cnn',
                       input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, alpha=1.0, gamma=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/HybridLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        return m

    def load_resnet():
        m = ResNetBaseline(DEVICE)
        m.load_state_dict(torch.load('models/ResNet18_baseline.pth', map_location=DEVICE))
        m.eval()
        return m

    def load_resnet_hybrid():
        # Definition must match training script or be imported. 
        # For simplicity, let's skip re-defining class and assume user wants main models.
        # Efficient path: Just skip it for the general report unless explicitly asked.
        # But wait, the report should be comprehensive.
        # I'll stick to the 4 main ones for now: Improved, Conc, Hybrid, ResNetBase.
        pass

    loaders = {
        'DiffLEVI': load_difflevi,
        'ImprovedLEVI': load_improvedlevi,
        'ConcentrationLE': load_concentrationle,
        'HybridLEVI': load_hybridlevi,
        'ResNetBaseline': load_resnet
    }

    for name, loader in loaders.items():
        if name in exclude:
            continue
        try:
            print(f"Loading {name}...")
            models_dict[name] = loader()
        except FileNotFoundError:
            print(f"{name} not found.")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    return models_dict

def predict_batch(model, current_X):
    # Wrapper to handle different model signatures
    model.eval()
    with torch.no_grad():
        name = type(model).__name__
        if name in ['ConcentrationLE', 'HybridLEVI']:
            if name == 'ConcentrationLE':
                _, alpha = model(current_X)
            else:
                _, _, _, _, evidence = model(current_X)
                alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            return (alpha / S).cpu().numpy()
            
        elif name == 'ResNetBaseline':
            return model.predict(current_X)

        elif name == 'DiffLEVI':
            # Use n_samples=1 for efficiency. 
            return model.predict(current_X.cpu().numpy(), n_samples=1) 
            
        elif name == 'ImprovedLEVI':
            L_dummy = torch.ones(current_X.size(0), 10).to(DEVICE) / 10
            return model.predict(current_X.cpu().numpy(), L_dummy.cpu().numpy())
            
        else:
             return np.zeros((current_X.size(0), 10))

def main():
    parser = argparse.ArgumentParser(description='Run Systematic Mixing Test')
    parser.add_argument('--digits', type=int, default=3, help='Number of digits to mix (2 or 3)')
    parser.add_argument('--samples', type=int, default=50000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--exclude', nargs='*', default=[], help='List of models to exclude')
    
    args = parser.parse_args()
    
    mnist = load_mnist()
    dataset = MixingDataset(mnist, n_samples=args.samples, n_digits=args.digits)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    models = load_models(exclude=args.exclude)
    
    # Metric: All Correct in Top-3
    results = {name: {'exact_top3_match': 0} for name in models.keys()}
    
    print(f"\nStarting Evaluation ({args.digits}-Digit Mix)...")
    print(f"Total Samples: {len(dataset)}")
    print(f"Models: {list(models.keys())}")
    
    for batch_idx, (images, true_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(DEVICE)
        
        for name, model in models.items():
            probs = predict_batch(model, images)
            
            # probs: (B, 10)
            # true_labels: (B, N_DIGITS)
            
            top3_preds = np.argsort(probs, axis=1)[:, -3:] # indices of top 3
            
            for i in range(len(images)):
                pred_set = set(top3_preds[i])
                true_set = set(true_labels[i].numpy())
                
                # Check if true_set is subset of pred_set
                # If N_DIGITS=2, we check if both are in Top-3.
                # If N_DIGITS=3, we check if all 3 are in Top-3 (Exact Match).
                if true_set.issubset(pred_set):
                    results[name]['exact_top3_match'] += 1
        
        # Plotting (First batch only)
        if batch_idx == 0:
            os.makedirs('reports/plots', exist_ok=True)
            n_plot = min(5, len(images))
            fig, axes = plt.subplots(n_plot, len(models) + 1, figsize=(3 * (len(models) + 1), 3 * n_plot))
            
            # Row 0 labels
            cols = ["Input Image"] + list(models.keys())
            for ax, col in zip(axes[0], cols):
                ax.set_title(col)

            for i in range(n_plot):
                # Mixed Image
                img = images[i].cpu().squeeze().numpy()
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].axis('off')
                true_lbls = true_labels[i].numpy()
                axes[i, 0].text(0, 0, f"True: {true_lbls}", color='white', backgroundcolor='black')
                
                # Predictions
                for j, (name, model) in enumerate(models.items()):
                     # Get single prediction again to be sure (or cache it, but re-running is cheap for 5)
                     p = predict_batch(model, images[i:i+1])[0]
                     ax = axes[i, j+1]
                     ax.bar(range(10), p, color='skyblue')
                     ax.set_ylim(0, 1)
                     # Highlight Top
                     top = np.argsort(p)[::-1]
                     ax.set_xticks(range(10))
                     ax.tick_params(labelsize=6)
            
            plt.tight_layout()
            plt.savefig(f'reports/plots/mnist_mixing_{args.digits}way.png')
            plt.close()
            
    # Summary
    print(f"\n=== SYSTEMATIC MIXING TEST RESULTS ({args.digits}-Digit Mix, {args.samples} Samples) ===")
    print(f"{'Model':<20} | {'All In Top-3 %':<20}")
    print("-" * 45)
    
    for name, res in results.items():
        acc = (res['exact_top3_match'] / len(dataset)) * 100
        print(f"{name:<20} | {acc:<19.2f}%")

if __name__ == "__main__":
    main()

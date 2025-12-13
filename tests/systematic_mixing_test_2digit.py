
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
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

class MixedMNIST2Dataset(Dataset):
    def __init__(self, mnist_dataset, n_samples=50000):
        self.mnist_dataset = mnist_dataset
        self.n_samples = n_samples
        self.samples = []
        
        print(f"Generating {n_samples} 2-digit mixed samples...")
        
        # Pre-generate indices by label
        self.indices_by_label = [[] for _ in range(10)]
        for idx in range(len(mnist_dataset)):
            label = int(mnist_dataset[idx][1])
            self.indices_by_label[label].append(idx)
            
        for _ in tqdm(range(n_samples)):
            # Pick 2 distinct labels
            labels = np.random.choice(10, 2, replace=False)
            
            # Pick 1 random image for each label
            idx_a = np.random.choice(self.indices_by_label[labels[0]])
            idx_b = np.random.choice(self.indices_by_label[labels[1]])
            
            # Pick weights (Uniform)
            # w for A, (1-w) for B. 
            # Avoid 0 or 1 to ensure it's actually mixed? 
            # Or just standard uniform [0, 1].
            # Let's use uniform [0.2, 0.8] to ensure significant mixing, 
            # otherwise it's just a single digit test.
            w = np.random.uniform(0.2, 0.8)
            
            self.samples.append({
                'indices': (idx_a, idx_b),
                'labels': labels,
                'weight': w
            })
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        idx_a, idx_b = s['indices']
        w = s['weight']
        
        img_a, _ = self.mnist_dataset[idx_a]
        img_b, _ = self.mnist_dataset[idx_b]
        
        # Mix
        img_mix = w * img_a + (1 - w) * img_b
        
        # Return mixed image and the set of 2 labels
        return img_mix, torch.tensor(s['labels'])

from torchvision import models
import torch.nn as nn

# Baseline Wrapper (Copied from streamlit_app.py for consistency)
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

def load_models():
    input_shape = (1, 28, 28)
    n_features = 784
    n_outputs = 10
    
    models = {}
    
    try:
        print("Loading DiffLEVI...")
        m = DiffLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet', 
                     input_shape=input_shape, epochs=10, batch_size=64, timesteps=1000, lr=2e-4, device=DEVICE)
        m.load_state_dict(torch.load('models/DiffLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['DiffLEVI'] = m
    except FileNotFoundError:
        print("DiffLEVI not found.")

    try:
        print("Loading ImprovedLEVI...")
        m = ImprovedLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                         input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, gamma=1.0, alpha=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/ImprovedLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['ImprovedLEVI'] = m
    except FileNotFoundError:
        print("ImprovedLEVI not found.")
        
    try:
        print("Loading ConcentrationLE...")
        m = ConcentrationLE(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, device=DEVICE)
        m.load_state_dict(torch.load('models/ConcentrationLE_mnist.pth', map_location=DEVICE))
        m.eval()
        models['ConcentrationLE'] = m
    except FileNotFoundError:
        print("ConcentrationLE not found.")

    try:
        print("Loading HybridLEVI...")
        m = HybridLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='cnn',
                       input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, alpha=1.0, gamma=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/HybridLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['HybridLEVI'] = m
    except FileNotFoundError:
        print("HybridLEVI not found.")

    try:
        print("Loading ResNetBaseline...")
        m = ResNetBaseline(DEVICE)
        m.load_state_dict(torch.load('models/ResNet18_baseline.pth', map_location=DEVICE))
        m.eval()
        models['ResNetBaseline'] = m
    except FileNotFoundError:
        print("ResNetBaseline not found.")
        
    return models

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
    mnist = load_mnist()
    dataset = MixedMNIST2Dataset(mnist, n_samples=50000)
    
    # Batch size 256 for speed
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0) 
    
    models = load_models()
    
    # Metric: Both Correct in Top-3
    results = {name: {'both_in_top3': 0} for name in models.keys()}
    
    print("\nStarting Evaluation (2-Digit Mix)...")
    print(f"Total Samples: {len(dataset)}")
    
    for batch_idx, (images, true_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(DEVICE)
        
        for name, model in models.items():
            probs = predict_batch(model, images)
            
            # probs: (B, 10)
            # true_labels: (B, 2)
            
            top3_preds = np.argsort(probs, axis=1)[:, -3:] # indices of top 3
            
            for i in range(len(images)):
                pred_set = set(top3_preds[i])
                true_set = set(true_labels[i].numpy())
                
                # Check if true_set is subset of pred_set
                if true_set.issubset(pred_set):
                    results[name]['both_in_top3'] += 1
                
    # Summary
    print("\n=== SYSTEMATIC MIXING TEST RESULTS (2-Digit Mix, 50k Samples) ===")
    print(f"{'Model':<20} | {'Both In Top-3 %':<20}")
    print("-" * 45)
    
    for name, res in results.items():
        acc = (res['both_in_top3'] / len(dataset)) * 100
        print(f"{name:<20} | {acc:<19.2f}%")

if __name__ == "__main__":
    main()

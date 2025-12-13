import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.ldl_models import ConcentrationLDL, mse_loss_evidence, kl_divergence_dirichlet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class LDLDataset(Dataset):
    def __init__(self, images, soft_labels, transform=None):
        self.images = images
        self.soft_labels = soft_labels # Soft distributions
        self.transform = transform
        
    def __len__(self):
        return len(self.soft_labels)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.soft_labels[idx]).float()
        return img, label

def load_data():
    # 1. Load Distilled Labels
    with open('data/mnist_ldl_distilled.pkl', 'rb') as f:
        data = pickle.load(f)
        
    # 2. Load Raw Images (Standard MNIST)
    # We need to make sure order matches. Standard PyTorch MNIST without shuffle should be deterministic.
    # Note: Our distilled script ran without shuffle on the same loaders.
    
    # We want 1-channel images for our simple CNN encoder in ConcentrationLDL
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization? Or 0-1? The simple CNN usually expects something.
        # Let's use 0-1 for now (ToTensor does this).
    ])
    
    train_ds_raw = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds_raw = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Check consistency
    assert len(train_ds_raw) == len(data['train_probs'])
    assert len(test_ds_raw) == len(data['test_probs'])
    
    # Wrap in custom dataset
    # Note: datasets.MNIST is indexable. 
    # But for speed, let's pre-load into memory if possible or just wrap.
    # Wrapping is safer for memory.
    
    class PairedDataset(Dataset):
        def __init__(self, mnist_ds, soft_targets):
            self.mnist_ds = mnist_ds
            self.soft_targets = soft_targets
        def __len__(self):
            return len(self.mnist_ds)
        def __getitem__(self, idx):
            img, _ = self.mnist_ds[idx] # Ignore hard label from dataset
            target = torch.tensor(self.soft_targets[idx]).float()
            return img, target
            
    train_dataset = PairedDataset(train_ds_raw, data['train_probs'])
    test_dataset = PairedDataset(test_ds_raw, data['test_probs'])
    
    return train_dataset, test_dataset

def train():
    print(f"Using device: {DEVICE}")
    
    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Init Model
    # 'cnn' encoder expects 1 channel images
    model = ConcentrationLDL(n_features=None, n_classes=10, encoder_type='cnn', device=DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20
    print(f"Training ConcentrationLDL for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        mse_sum = 0
        kl_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Loss computation inside loop to handle batches
            # KL Annealing: Ramp from 0 to 0.01 over first 10 epochs
            # lambda_kl=1.0 proved too strong (forced uniform distribution)
            kl_coeff = min(0.01, (epoch / 10.0) * 0.01)
            
            loss, alpha, loss_mse, loss_kl = model.compute_loss(imgs, targets, global_step=epoch, lambda_kl=kl_coeff)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            mse_sum += loss_mse
            kl_sum += loss_kl
            
            pbar.set_postfix(mse=loss_mse, kl=loss_kl, lambda_kl=kl_coeff)
            
        # Validation (Metric: Cosine Similarity with Truth)
        model.eval()
        cos_sims = []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.numpy()
                
                probs = model.predict_distribution(imgs).cpu().numpy()
                
                # Cosine sim per sample
                # Dot product / norms
                norm_p = np.linalg.norm(probs, axis=1)
                norm_t = np.linalg.norm(targets, axis=1)
                dot = np.sum(probs * targets, axis=1)
                sim = dot / (norm_p * norm_t + 1e-8)
                cos_sims.append(sim)
                
        avg_sim = np.concatenate(cos_sims).mean()
        print(f"Epoch {epoch+1} Avg Cosine Sim: {avg_sim:.4f}")
        
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ConcentrationLDL_mnist.pth')
    print("Saved ConcentrationLDL model.")
    
    # Plotting Comparison
    plot_comparisons(model, test_dataset)

def plot_comparisons(model, dataset, n=5):
    model.eval()
    indices = np.random.choice(len(dataset), n, replace=False)
    
    fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
    
    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        img_dev = img.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_probs = model.predict_distribution(img_dev)[0].cpu().numpy()
            
        # 1. Image
        ax = axes[i, 0]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title("Input Image")
        ax.axis('off')
        
        # 2. Target Distribution (ResNet Soft)
        ax = axes[i, 1]
        ax.bar(range(10), target.numpy(), color='green', alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title("Target Dist (ResNet)")
        
        # 3. Predicted Distribution (LDL)
        ax = axes[i, 2]
        ax.bar(range(10), pred_probs, color='blue', alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title("Predicted Dist (LDL)")
        
    plt.tight_layout()
    plt.savefig('reports/ldl_mnist_comparison.png')
    print("Saved comparison plot to reports/ldl_mnist_comparison.png")

if __name__ == "__main__":
    train()

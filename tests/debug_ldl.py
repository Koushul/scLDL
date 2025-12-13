import pickle
import torch
import numpy as np
import os
import sys
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.ldl_models import ConcentrationLDL

DEVICE = "cpu" # Debug on CPU

def load_data():
    with open('data/mnist_ldl_distilled.pkl', 'rb') as f:
        data = pickle.load(f)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds_raw = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    class PairedDataset(Dataset):
        def __init__(self, mnist_ds, soft_targets):
            self.mnist_ds = mnist_ds
            self.soft_targets = soft_targets
        def __len__(self):
            return len(self.mnist_ds)
        def __getitem__(self, idx):
            img, _ = self.mnist_ds[idx]
            target = torch.tensor(self.soft_targets[idx]).float()
            return img, target
            
    train_dataset = PairedDataset(train_ds_raw, data['train_probs'])
    return train_dataset, None

def debug():
    # 1. Check Distilled Data Stats
    with open('data/mnist_ldl_distilled.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("Train Probs stats:")
    print("Mean:", np.mean(data['train_probs']))
    print("Max:", np.max(data['train_probs']))
    print("Min:", np.min(data['train_probs']))
    print("Sample[0]:", data['train_probs'][0])
    
    # Check if they are uniform?
    is_uniform = np.allclose(data['train_probs'][0], 0.1, atol=0.01)
    print(f"Is first sample uniform? {is_uniform}")
    
    # 2. Check Init Loss Balance
    train_ds, _ = load_data()
    imgs, targets = train_ds[0]
    imgs = imgs.unsqueeze(0).to(DEVICE)
    targets = targets.unsqueeze(0).to(DEVICE)
    
    print(f"\nTarget shape: {targets.shape}")
    print(f"Target vals: {targets}")
    
    model = ConcentrationLDL(n_features=None, n_classes=10, encoder_type='cnn', device=DEVICE)
    
    # Check init output
    with torch.no_grad():
        alpha_init = model(imgs)
        print(f"Init Alpha: {alpha_init}")
        print(f"Init Probs: {alpha_init / alpha_init.sum()}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    loss, alpha, loss_mse, loss_kl = model.compute_loss(imgs, targets, global_step=0, lambda_kl=1.0)
    
    print(f"\nInit Loss: Total={loss.item():.4f}, MSE={loss_mse:.4f}, KL={loss_kl:.4f}")
    
    loss.backward()
    
    print("\nGradients check (Encoder First Layer Weight Mean/Max):")
    g = model.encoder[0].weight.grad
    if g is not None:
        print(f"Mean Abs Grad: {torch.mean(torch.abs(g)).item()}")
        print(f"Max Abs Grad: {torch.max(torch.abs(g)).item()}")
    else:
        print("Gradient is None!")


if __name__ == "__main__":
    debug()

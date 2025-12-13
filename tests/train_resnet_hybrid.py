
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Model Definition ---

class ResNetHybridLEVI(nn.Module):
    def __init__(self, n_outputs=10, latent_dim=64, alpha_kl=1.0, alpha_evidence=1.0, gamma=1.0, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_outputs = n_outputs
        self.latent_dim = latent_dim
        self.alpha_kl = alpha_kl
        self.alpha_evidence = alpha_evidence
        self.gamma = gamma # Reconstruction weight
        
        # 1. Encoder (ResNet18 Backbone)
        resnet = models.resnet18(pretrained=True)
        # Remove fc layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        
        # 2. Latent Projection
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # 3. Decoder
        # Start from 7x7 feature map to easily reach 28x28
        self.decoder_input = nn.Linear(latent_dim, 128 * 7 * 7) 
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 14 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1), # 28 -> 28 (Refinement)
            nn.Sigmoid() 
        )
        
        # 4. Evidence Head
        self.evidence_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
            nn.Softplus() # alpha = output + 1
        )
        
        # Transforms for input (1ch -> 3ch + Norm)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (B, 1, 28, 28) usually
        
        # Preprocess for ResNet
        if x.shape[1] == 1:
            x_in = x.repeat(1, 3, 1, 1)
        else:
            x_in = x
        x_in = (x_in - self.norm_mean) / self.norm_std
        
        # Encode
        features = self.feature_extractor(x_in) # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)
        
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(self.decoder_input(z))
        
        # Evidence
        evidence = self.evidence_head(z)
        alpha = evidence + 1
        
        return recon_x, mu, logvar, alpha

    def loss_function(self, recon_x, x, mu, logvar, alpha, y_hot, epoch=0):
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # 2. KL Divergence
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        
        # 3. Evidence Loss (Type II ML)
        S = torch.sum(alpha, dim=1, keepdim=True)
        # expected risk
        A = torch.sum(y_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        evidence_loss = torch.mean(A)
        
        # KL term for evidence (regularization)
        alp = alpha * (1 - y_hot) + y_hot
        kl_evidence = 0.0 # Simplified for now, usually proportional to t * KL[Dir(alpha)||Dir(1)]
        # Use standard MSE for now if evidence loss is unstable? 
        # Revert to standard CrossEntropy if needed for stability? 
        # Implementing basic Eq 4 from Evidential DL paper:
        # L = sum( (y_i - alpha_i/S_i)^2 ) + ...
        # But let's stick to the LogLikelihood version if possible.
        # Actually, for stability, let's use the sum-of-squares error on probabilities as a proxy 
        # or the robust cross-entropy.
        # Let's use the implementation from prior code or standard:
        # L = sum (y_ij * (log S_i - log alpha_ij))
        
        # Using the formulation similar to HybridLEVI:
        loss = self.gamma * recon_loss + self.alpha_kl * kl_loss + self.alpha_evidence * evidence_loss
        
        return loss, recon_loss.item(), kl_loss.item(), evidence_loss.item()
        
    def predict(self, X):
        self.eval()
        with torch.no_grad():
             X_tensor = torch.tensor(X).float().to(self.device)
             _, _, _, alpha = self.forward(X_tensor)
             S = torch.sum(alpha, dim=1, keepdim=True)
             probs = alpha / S
             return probs.cpu().numpy()

# --- 2. Training Loop ---

def train_resnet_hybrid():
    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Init Model
    model = ResNetHybridLEVI(device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10 
    print(f"Training ResNetHybridLEVI for {epochs} epochs...")
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hot = F.one_hot(y, 10).float()
            
            optimizer.zero_grad()
            recon_x, mu, logvar, alpha = model(x)
            
            loss, rec, kl, ev = model.loss_function(recon_x, x, mu, logvar, alpha, y_hot, epoch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), rec=rec, kl=kl, ev=ev)
            
        # Eval Clean
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, _, _, alpha = model(x)
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
                pred = torch.argmax(probs, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")
        
    # Save
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/ResNetHybridLEVI_mnist.pth')
    print("Model saved to models/ResNetHybridLEVI_mnist.pth")
    return model

# --- 3. Benchmark ---

class ResNetBaseline(nn.Module):
    # Definition for loading
    def __init__(self, device):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model.to(device)
        self.device = device
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
             X_tensor = torch.tensor(X).float().to(self.device)
             if X_tensor.shape[1] == 1: X_tensor = X_tensor.repeat(1, 3, 1, 1)
             X_tensor = (X_tensor - self.norm_mean) / self.norm_std
             logits = self.model(X_tensor)
             return F.softmax(logits, dim=1).cpu().numpy()

def run_benchmark(hybrid_model):
    print("\n--- Running Benchmark: ResNetBaseline vs ResNetHybridLEVI ---")
    
    # Load Baseline
    try:
        baseline = ResNetBaseline(DEVICE)
        baseline.model.load_state_dict(torch.load('models/ResNet18_baseline.pth', map_location=DEVICE))
        print("Loaded Baseline.")
    except Exception as e:
        print(f"Failed to load baseline: {e}")
        return

    # Dataset for Mixing
    from torchvision import datasets, transforms
    ds = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    
    # Generate 10k mixed samples (2-way)
    n_samples = 10000
    print(f"Generating {n_samples} 2-Way Mixed Samples...")
    
    indices_by_label = [[] for _ in range(10)]
    for idx in range(len(ds)):
        indices_by_label[ds[idx][1]].append(idx)
        
    correct_base = 0
    correct_hyb = 0
    
    # Batch processing
    for _ in tqdm(range(n_samples // 100)): # 100 batches of 100
        batch_imgs = []
        batch_labels = []
        
        for _ in range(100):
            lbls = np.random.choice(10, 2, replace=False)
            idx_a = np.random.choice(indices_by_label[lbls[0]])
            idx_b = np.random.choice(indices_by_label[lbls[1]])
            w = np.random.uniform(0.2, 0.8)
            img = w * ds[idx_a][0] + (1-w) * ds[idx_b][0]
            batch_imgs.append(img)
            batch_labels.append(set(lbls))
            
        batch_X = torch.stack(batch_imgs).numpy()
        
        # Predict
        p_base = baseline.predict(batch_X)
        p_hyb = hybrid_model.predict(batch_X)
        
        # Check Top-3
        top3_base = np.argsort(p_base, axis=1)[:, -3:]
        top3_hyb = np.argsort(p_hyb, axis=1)[:, -3:]
        
        for i in range(100):
            if batch_labels[i].issubset(set(top3_base[i])):
                correct_base += 1
            if batch_labels[i].issubset(set(top3_hyb[i])):
                correct_hyb += 1
                
    acc_base = 100 * correct_base / n_samples
    acc_hyb = 100 * correct_hyb / n_samples
    
    print("\n=== RESULTS (2-Way Mixing) ===")
    print(f"ResNetBaseline:   {acc_base:.2f}%")
    print(f"ResNetHybridLEVI: {acc_hyb:.2f}%")
    
    if acc_hyb > acc_base:
        print("\nSUCCESS: Hybrid model improved robustness!")
    else:
        print("\nNOTE: Baseline still stronger. VAE regularization might need tuning.")

if __name__ == "__main__":
    model = train_resnet_hybrid()
    run_benchmark(model)

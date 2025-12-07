import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from .label_enhancer import LabelEnhancer

class LabelEnhancerTrainer:
    def __init__(self, model: LabelEnhancer, lr: float = 1e-3, pretrain_lr: float = 5e-3, beta: float = 0.001, lambda_gap: float = 1.0, lambda_spatial: float = 0.0):
        """
        Args:
            model: LabelEnhancer model instance.
            lr: Learning rate for the main training phase.
            pretrain_lr: Learning rate for the pretraining phase.
            beta: Weight for the Information Bottleneck term (I(Z;X)).
            lambda_gap: Weight for the Gap Estimation loss.
            lambda_spatial: Weight for the Spatial Regularization loss.
        """
        self.model = model
        self.lr = lr
        self.pretrain_lr = pretrain_lr
        self.beta = beta
        self.lambda_gap = lambda_gap
        self.lambda_spatial = lambda_spatial
        self.device = model.device
        
        # Loss history
        self.loss_history = {
            'total': [],
            'rec': [],
            'kl': [],
            'gap': [],
            'spatial': []
        }

    def train(self, train_loader: DataLoader, epochs: int = 100, pretrain_epochs: int = 50, log_interval: int = 10):
        self.model.train()
        
        # --- Pretraining Phase ---
        if pretrain_epochs > 0:
            print(f"Starting Pretraining for {pretrain_epochs} epochs...")
            optimizer_pre = optim.Adam(
                list(self.model.encoder.parameters()) + list(self.model.logical_decoder.parameters()),
                lr=self.pretrain_lr
            )
            
            for epoch in range(pretrain_epochs):
                epoch_loss = 0
                epoch_rec = 0
                epoch_kl = 0
                
                pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}", unit="batch", disable=True)
                for batch_idx, (x, l_onehot, s, idx) in enumerate(pbar):
                    x = x.to(self.device)
                    l_onehot = l_onehot.to(self.device)
                    
                    optimizer_pre.zero_grad()
                    
                    # Forward pass for VAE only
                    mu, log_var = self.model.encoder(x)
                    z = self.model.reparameterize(mu, log_var)
                    l_logits = self.model.logical_decoder(z)
                    
                    # Reconstruction Loss (I(Z; L))
                    loss_rec = F.cross_entropy(l_logits, l_onehot.argmax(dim=1))
                    
                    # KL Divergence (I(Z; X))
                    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                    
                    loss = loss_rec + self.beta * kl_div
                    
                    loss.backward()
                    optimizer_pre.step()
                    
                    epoch_loss += loss.item()
                    epoch_rec += loss_rec.item()
                    epoch_kl += kl_div.item()
                
                if (epoch + 1) % log_interval == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"Pretrain Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # --- Main Training Phase ---
        print(f"Starting Main Training for {epochs} epochs...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Scheduler as in artifact
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_rec = 0
            epoch_kl = 0
            epoch_gap = 0
            epoch_spatial = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", disable=True)
            for batch_idx, (x, l_onehot, s, idx) in enumerate(pbar):
                x = x.to(self.device)
                l_onehot = l_onehot.to(self.device)
                s = s.to(self.device)
                
                optimizer.zero_grad()
                
                mu, log_var, l_logits, d_logits, gap_sigma = self.model(x)
                
                # 1. Reconstruction Loss (I(Z; L)) - Cross Entropy
                loss_rec = F.cross_entropy(l_logits, l_onehot.argmax(dim=1))
                
                # 2. Information Bottleneck Loss (I(Z; X)) - KL Divergence
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                
                # 3. Gap Loss (I(Z; G))
                d_pred = F.softmax(d_logits, dim=1)
                # Gap squared error per sample (sum over classes)
                gap_sq = (l_onehot - d_pred).pow(2).sum(dim=1, keepdim=True)
                
                # Artifact Loss: mean_over_classes( (y-y_hat)^2 / (2*sigma^2) + log(sigma) )
                # Our gap_sq is sum_over_classes((y-y_hat)^2).
                # So we need to divide by d_dim to get mean squared error.
                # loss_gap = ( (gap_sq / self.model.d_dim) / (2 * gap_sigma.pow(2)) + torch.log(gap_sigma) ).mean()
                
                # Let's match the artifact exactly:
                # lost_obj = (log_l - d_pre)**2
                # lost_obj = torch.mul(lost_obj, (0.5*torch.pow(gap, -2))) + torch.log(torch.abs(gap))
                # lost_obj = lost_obj.mean(1, keepdim=True)
                
                # In our terms:
                # term1 = (l_onehot - d_pred)**2 * (0.5 * gap_sigma**-2)
                # term2 = torch.log(gap_sigma)
                # loss_gap_per_sample = (term1 + term2).mean(dim=1)
                
                # Since gap_sigma is scalar per sample (shape [B, 1]), it broadcasts.
                # term1 sum over dim 1 is gap_sq * 0.5 * gap_sigma**-2
                # term1 mean over dim 1 is (gap_sq / d_dim) * 0.5 * gap_sigma**-2
                # term2 mean over dim 1 is log(gap_sigma)
                
                loss_gap = ( (gap_sq / self.model.d_dim) / (2 * gap_sigma.pow(2)) + torch.log(gap_sigma) ).mean()
                
                # 4. Spatial Regularization (Optional)
                loss_spatial = torch.tensor(0.0, device=self.device)
                if self.lambda_spatial > 0:
                    s_dist = torch.cdist(s, s)
                    weights = torch.exp(-s_dist)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                    d_dist = torch.cdist(d_pred, d_pred).pow(2)
                    loss_spatial = (weights * d_dist).sum() / x.size(0)

                # Total Loss
                loss = loss_rec + self.beta * kl_div + self.lambda_gap * loss_gap + self.lambda_spatial * loss_spatial
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_rec += loss_rec.item()
                epoch_kl += kl_div.item()
                epoch_gap += loss_gap.item()
                epoch_spatial += loss_spatial.item()
                
                pbar.set_postfix({'Loss': loss.item(), 'Rec': loss_rec.item(), 'KL': kl_div.item()})
            
            scheduler.step()
            
            # Average losses for the epoch
            num_batches = len(train_loader)
            self.loss_history['total'].append(epoch_loss / num_batches)
            self.loss_history['rec'].append(epoch_rec / num_batches)
            self.loss_history['kl'].append(epoch_kl / num_batches)
            self.loss_history['gap'].append(epoch_gap / num_batches)
            self.loss_history['spatial'].append(epoch_spatial / num_batches)

            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}: Avg Loss = {self.loss_history['total'][-1]:.4f}")

    def predict(self, data_loader: DataLoader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x, _, _, _ in data_loader:
                x = x.to(self.device)
                d_pred = self.model.get_label_distribution(x)
                predictions.append(d_pred.cpu().numpy())
        return np.concatenate(predictions, axis=0)

    def plot_losses(self):
        """
        Plots all training losses (Total, Reconstruction, KL Divergence, Gap, Spatial)
        in a single row of subplots.
        """
        import matplotlib.pyplot as plt

        losses = {
            "Total Loss": self.loss_history["total"],
            "Reconstruction Loss": self.loss_history["rec"],
            "KL Divergence": self.loss_history["kl"],
            "Gap Loss": self.loss_history["gap"],
            "Spatial Loss": self.loss_history["spatial"],
        }

        epochs = range(1, len(self.loss_history['total']) + 1)

        fig, axes = plt.subplots(1, len(losses), figsize=(20, 4))

        for ax, (title, values) in zip(axes, losses.items()):
            ax.plot(epochs, values)
            ax.set_title(title)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")

        plt.tight_layout()
        plt.show()

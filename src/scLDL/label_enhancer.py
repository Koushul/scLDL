import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LIBLE(nn.Module):
    def __init__(self, n_features, n_outputs, n_hidden=64, n_latent=64, alpha=1e-3, beta=1e-3, lr=1e-3, epochs=100, batch_size=32, device=None):
        super(LIBLE, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        self.encoder_hidden = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh()
        )
        self.encoder_mean = nn.Linear(n_hidden, n_latent)
        self.encoder_logvar = nn.Linear(n_hidden, n_latent) # Output log variance directly

        # Decoder L (Label reconstruction)
        self.decoder_L = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs)
        )

        # Decoder D (Label distribution reconstruction)
        self.decoder_D = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs)
        )

        # Decoder g (Uncertainty/Scale)
        self.decoder_g = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)

    def encode(self, x):
        h = self.encoder_hidden(x)
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, transform=False):
        mean, logvar = self.encode(x)
        if transform:
            # For prediction, we might use just the mean to generate D
            return self.decoder_D(mean)
        
        z = self.reparameterize(mean, logvar)
        
        L_hat = self.decoder_L(z)
        D_hat = self.decoder_D(z)
        g = self.decoder_g(z)
        
        return mean, logvar, L_hat, D_hat, g

    def get_z(self, x):
        x_tensor = torch.FloatTensor(x).to(self.device)
        mean, logvar = self.encode(x_tensor)
        z = self.reparameterize(mean, logvar)
        return z

    def fit(self, X, L):
        X_tensor = torch.FloatTensor(X).to(self.device)
        L_tensor = torch.FloatTensor(L).to(self.device)
        
        dataset = TensorDataset(X_tensor, L_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_L in dataloader:
                optimizer.zero_grad()
                
                mean, logvar, L_hat, D_hat, g = self.forward(batch_X)
                
                # KL Divergence
                # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                
                # Reconstruction Loss L
                rec_L_loss = torch.sum((batch_L - L_hat)**2, dim=1).mean()
                
                # Reconstruction Loss D (Gaussian NLL-like)
                # Original: g**(-2) * (L - D_hat)**2 + log(g**2)
                # Note: g is sigmoid output (0, 1). g**2 might be small.
                # Adding epsilon for stability if needed, but following formula:
                g_sq = g.pow(2) + 1e-6
                rec_D_loss = torch.sum((batch_L - D_hat)**2 / g_sq + torch.log(g_sq), dim=1).mean()
                
                loss = rec_L_loss + self.alpha * kl_loss + self.beta * rec_D_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")
                
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.forward(X_tensor, transform=True)
            return F.softmax(logits, dim=1).cpu().numpy()

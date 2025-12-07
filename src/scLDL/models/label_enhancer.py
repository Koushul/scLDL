import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class LabelEncoder(nn.Module):
    """
    Encoder for LabelEnhancer. Maps gene expression to latent space.
    """
    def __init__(self, x_dim: int, h_dim: int = 64, hidden_layers: Optional[List[int]] = None, dropout: float = 0.1):
        super(LabelEncoder, self).__init__()
        self.h_dim = h_dim
        
        if hidden_layers is None:
            hidden_layers = [256, 256]
            
        layers = []
        in_dim = x_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_dim = h
            
        self.encode = nn.Sequential(*layers)
        
        # Output layers for mu and log_var (std)
        self.to_latent = nn.Linear(hidden_layers[-1], 2 * h_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.encode(x)
        stats = self.to_latent(stats)
        mu = stats[:, :self.h_dim]
        # Predict log_variance for numerical stability
        log_var = stats[:, self.h_dim:] 
        return mu, log_var

class LabelLogicalDecoder(nn.Module):
    """
    Decodes latent space back to Logical Labels (observed hard labels).
    """
    def __init__(self, h_dim: int, d_dim: int, hidden_layers: Optional[List[int]] = None):
        super(LabelLogicalDecoder, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = []
            
        layers = []
        in_dim = h_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h
            
        layers.append(nn.Linear(in_dim, d_dim))
        self.decode = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Returns logits for logical labels
        return self.decode(z)

class LabelDistributionDecoder(nn.Module):
    """
    Decodes latent space to the target Label Distribution (soft labels).
    """
    def __init__(self, h_dim: int, d_dim: int, hidden_layers: Optional[List[int]] = None):
        super(LabelDistributionDecoder, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 128]
            
        layers = []
        in_dim = h_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h
            
        layers.append(nn.Linear(in_dim, d_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Returns logits
        return self.net(z)

class LabelGapDecoder(nn.Module):
    """
    Estimates the Label Gap variance/uncertainty.
    """
    def __init__(self, h_dim: int, hidden_layers: Optional[List[int]] = None):
        super(LabelGapDecoder, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 128]
            
        layers = []
        in_dim = h_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Returns sigma (must be positive)
        return F.softplus(self.net(z)) + 1e-6

class LabelEnhancer(nn.Module):
    """
    LabelEnhancer Module.
    Uses the Label Information Bottleneck principle to recover 
    soft label distributions from hard logical labels.
    """
    def __init__(self, x_dim: int, d_dim: int, h_dim: int = 64, device: str = 'cpu'):
        super(LabelEnhancer, self).__init__()
        self.device = device
        self.x_dim = x_dim
        self.d_dim = d_dim
        self.h_dim = h_dim
        
        self.encoder = LabelEncoder(x_dim, h_dim)
        self.logical_decoder = LabelLogicalDecoder(h_dim, d_dim)
        self.distribution_decoder = LabelDistributionDecoder(h_dim, d_dim)
        self.gap_decoder = LabelGapDecoder(h_dim)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # 1. Reconstruct Logical Labels
        l_logits = self.logical_decoder(z)
        
        # 2. Predict Label Distribution
        d_logits = self.distribution_decoder(z)
        
        # 3. Estimate Gap
        gap_sigma = self.gap_decoder(z)
        
        return mu, log_var, l_logits, d_logits, gap_sigma

    def get_label_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Inference method to get the enhanced label distribution."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x.to(self.device))
            # Use mean of latent for prediction
            d_logits = self.distribution_decoder(mu)
            return F.softmax(d_logits, dim=1)

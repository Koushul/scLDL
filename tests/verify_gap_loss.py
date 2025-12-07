import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scLDL.models.label_enhancer import LabelEnhancer
from scLDL.models.trainer import LabelEnhancerTrainer
from torch.utils.data import DataLoader, TensorDataset

def verify_gap_loss_convergence():
    """
    Verify that the Gap Loss minimizes to the correct sigma.
    For Gaussian NLL, sigma should converge to RMSE.
    """
    print("Verifying Gap Loss Convergence...")
    
    # Setup a dummy problem
    # We want to minimize: NLL(y, pred, sigma)
    # Let's fix y and pred, and optimize sigma.
    
    d_dim = 10
    batch_size = 100
    
    # True error (RMSE) we want to recover
    true_rmse = 0.5
    
    # Create dummy data where (y - pred)^2 sum is known
    # gap_sq = sum((y - pred)^2)
    # We want mean(gap_sq) / d_dim = true_rmse^2
    
    # Let's just simulate the loss function directly since we modified trainer.py
    # But we need to use the Trainer's loss logic.
    
    # We will create a mock model that just returns a learnable sigma
    class MockModel(nn.Module):
        def __init__(self, d_dim):
            super().__init__()
            self.d_dim = d_dim
            self.log_sigma = nn.Parameter(torch.tensor([0.0])) # Start at sigma=1
            self.device = 'cpu'
            
        def forward(self, x):
            # Return dummy values except for gap_sigma
            batch_size = x.shape[0]
            sigma = torch.exp(self.log_sigma).expand(batch_size, 1)
            return None, None, None, None, sigma
            
        def train(self): pass
        def eval(self): pass
    
    model = MockModel(d_dim)
    
    # We need to manually run the optimization loop using the loss formula from trainer.py
    # loss_gap = (gap_sq / (2 * gap_sigma.pow(2)) + self.model.d_dim * torch.log(gap_sigma)).mean()
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Fixed gap_sq (SSE)
    # Let's say per sample SSE is constant C
    # We want sigma^2 = C / d_dim => sigma = sqrt(C/d_dim)
    
    # Let C = d_dim * true_rmse^2 = 10 * 0.25 = 2.5
    C = d_dim * (true_rmse ** 2)
    gap_sq = torch.tensor([C] * batch_size).unsqueeze(1) # (B, 1)
    
    print(f"Target Sigma: {true_rmse}")
    
    for i in range(2000):
        optimizer.zero_grad()
        
        # Get sigma
        _, _, _, _, gap_sigma = model(torch.randn(batch_size, 1))
        
        # Loss calculation from trainer.py
        # loss_gap = (gap_sq / (2 * gap_sigma.pow(2)) + self.model.d_dim * torch.log(gap_sigma)).mean()
        loss = (gap_sq / (2 * gap_sigma.pow(2)) + model.d_dim * torch.log(gap_sigma)).mean()
        
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            current_sigma = gap_sigma[0].item()
            print(f"Iter {i}: Sigma = {current_sigma:.4f}, Loss = {loss.item():.4f}")
            
    final_sigma = gap_sigma[0].item()
    print(f"Final Sigma: {final_sigma:.4f}")
    
    if abs(final_sigma - true_rmse) < 1e-2:
        print("SUCCESS: Sigma converged to RMSE.")
    else:
        print(f"FAILURE: Sigma {final_sigma} did not converge to RMSE {true_rmse}.")
        # Check what it converged to
        # Previous incorrect formula: sigma = sqrt(SSE) = sqrt(C) = sqrt(2.5) = 1.58
        print(f"Note: If incorrect, it might be {np.sqrt(C):.4f}")

if __name__ == "__main__":
    verify_gap_loss_convergence()

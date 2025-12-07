import torch
from torch.utils.data import DataLoader, TensorDataset
from scLDL.models.label_enhancer import LabelEnhancer
from scLDL.models.trainer import LabelEnhancerTrainer
import matplotlib.pyplot as plt

# Mock data
x = torch.randn(100, 50)
l_onehot = torch.zeros(100, 5)
l_onehot.scatter_(1, torch.randint(0, 5, (100, 1)), 1)
s = torch.rand(100, 2)
idx = torch.arange(100)

dataset = TensorDataset(x, l_onehot, s, idx)
dataloader = DataLoader(dataset, batch_size=10)

# Model
model = LabelEnhancer(input_dim=50, label_dim=5, z_dim=10)

# Trainer
trainer = LabelEnhancerTrainer(model, lambda_spatial=0.1)

# Train
print("Training...")
trainer.train(dataloader, epochs=5, log_interval=1)

# Check history
print("\nLoss History Keys:", trainer.loss_history.keys())
print("Total Loss History:", trainer.loss_history['total'])

# Plot
print("\nPlotting losses...")
try:
    trainer.plot_losses()
    print("Plotting successful (window might not open in headless env, but function ran).")
except Exception as e:
    print(f"Plotting failed: {e}")

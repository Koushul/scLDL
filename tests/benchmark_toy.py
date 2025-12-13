import torch
import torch.nn.functional as F # Re-added for LEVI class if needed, checking dependencies
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scLDL.label_enhancer import DiffLEVI, LEVI

def generate_toy_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    # 1. Latent Z: Uniform(-2, 2)
    z = np.random.uniform(-2, 2, size=(n_samples, 1)).astype(np.float32)
    
    # 2. Input X: Non-linear mapping z^3 + noise
    x = (z**3) + np.random.normal(0, 0.1, size=(n_samples, 1)).astype(np.float32)
    
    # Normalize X for easier learning
    x = (x - x.mean()) / x.std()
    
    # 3. True Labels: Softmax([z, -z, z^2])
    # Class 0: proportional to exp(z)
    # Class 1: proportional to exp(-z)
    # Class 2: proportional to exp(z^2)
    logits = np.concatenate([z, -z, z**2], axis=1)
    y_true_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    
    # 4. Observed Labels: Sample hard labels from true probabilities
    y_hard_indices = np.array([np.random.choice(3, p=probs) for probs in y_true_probs])
    y_obs = np.zeros_like(y_true_probs)
    y_obs[np.arange(n_samples), y_hard_indices] = 1.0
    
    return x, y_obs.astype(np.float32), y_true_probs.astype(np.float32), z

def evaluate_model(model_cls, model_name, X, Y_obs, Y_true, **kwargs):
    print(f"\n--- Training {model_name} ---")
    n_features = X.shape[1]
    n_classes = Y_obs.shape[1]
    
    model = model_cls(
        n_features=n_features,
        n_outputs=n_classes,
        **kwargs
    )
    
    model.fit(X, Y_obs)
    
    # Predict
    if model_name == "DiffLEVI":
        Y_pred = model.predict(X, n_samples=20) # Average multiple samples for better estimation
    else:
        # LEVI predict requires L for forward pass in its signature usually, 
        # but let's check implementation. 
        # Standard LEVI predict might need dummy L or just X.
        # Checking predict signature: predict(self, X, L) -> returns transformed mean.
        # Wait, the predict method in LEVI returns Softmax(mean).
        # It takes X and L. But at inference we don't know L?
        # Standard VAE prediction usually uses just X -> mean -> output.
        # Let's check the LEVI.predict implementation...
        # It calls self.forward(X, L, transform=True).
        # We should pass dummy L.
        dummy_L = np.zeros_like(Y_obs)
        Y_pred = model.predict(X, dummy_L)

    # Metrics
    # MSE
    mse = np.mean((Y_pred - Y_true)**2)
    
    # KL Divergence: sum(p * log(p/q))
    # Add epsilon to avoid log(0)
    eps = 1e-8
    kl = np.sum(Y_true * np.log((Y_true + eps) / (Y_pred + eps)), axis=1).mean()
    
    print(f"{model_name} Results:")
    print(f"MSE: {mse:.6f}")
    print(f"KL Divergence: {kl:.6f}")
    
    return Y_pred, mse, kl

def main():
    X, Y_obs, Y_true, Z = generate_toy_data()
    
    print("Data shapes:", X.shape, Y_obs.shape, Y_true.shape)
    
    # Train LEVI
    # Using MLP encoder for 1D data
    y_pred_levi, mse_levi, kl_levi = evaluate_model(
        LEVI, "LEVI", max(X, Y_obs, Y_true, key=len) if 0 else X, Y_obs, Y_true,
        n_hidden=64,
        alpha=0.01, # KL weight
        epochs=100,
        batch_size=32,
        encoder_type='mlp' 
    )
    
    # Train DiffLEVI
    y_pred_diff, mse_diff, kl_diff = evaluate_model(
        DiffLEVI, "DiffLEVI", X, Y_obs, Y_true,
        n_hidden=64,
        n_latent=16, # internal diffusion dim
        timesteps=50,
        epochs=100,
        batch_size=32,
        encoder_type='mlp'
    )
    
    # Compare
    print("\n--- Summary ---")
    print(f"LEVI     - MSE: {mse_levi:.6f}, KL: {kl_levi:.6f}")
    print(f"DiffLEVI - MSE: {mse_diff:.6f}, KL: {kl_diff:.6f}")
    
    if kl_diff < kl_levi:
        print("DiffLEVI outperformed LEVI in distribution recovery!")
    else:
        print("LEVI outperformed DiffLEVI.")

if __name__ == "__main__":
    main()

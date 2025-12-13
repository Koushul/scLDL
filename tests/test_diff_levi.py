import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scLDL.label_enhancer import DiffLEVI

def test_diff_levi_toy():
    print("Testing DiffLEVI on Toy Data...")
    
    # 1. Generate synthetic data (Gaussian mixture)
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 200
    n_features = 10
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Simple logic: class depends on first feature
    y_labels = np.zeros(n_samples, dtype=int)
    y_labels[X[:, 0] < -0.5] = 0
    y_labels[(X[:, 0] >= -0.5) & (X[:, 0] < 0.5)] = 1
    y_labels[X[:, 0] >= 0.5] = 2
    
    # Create soft labels dependent on feature distance
    L = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        sigma = 0.5
        probs = np.exp(-0.5 * ((np.arange(n_classes) - y_labels[i]) / sigma) ** 2)
        L[i] = probs / probs.sum()
        
    # 2. Initialize DiffLEVI
    # Use small timesteps for quick test
    model = DiffLEVI(
        n_features=n_features,
        n_outputs=n_classes,
        n_hidden=32,
        n_latent=16,
        timesteps=50, 
        epochs=5,
        batch_size=32,
        encoder_type='mlp'
    )
    
    # 3. Train
    print("Training...")
    model.fit(X, L)
    
    # Check loss history
    print("Loss history:", model.history['loss'])
    assert model.history['loss'][-1] < model.history['loss'][0], "Loss should decrease"
    
    # 4. Predict
    print("Predicting...")
    preds = model.predict(X[:5])
    print("Predictions shape:", preds.shape)
    print("Sample prediction:", preds[0])
    
    # Check probability properties
    assert preds.shape == (5, n_classes)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5), "Predictions must sum to 1"
    
    print("\nDiffLEVI Test Passed!")

if __name__ == "__main__":
    test_diff_levi_toy()

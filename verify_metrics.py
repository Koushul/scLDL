import numpy as np
from scLDL.utils.metrics import compute_metrics

def test_compute_metrics():
    # Test case 1: 2D y_true (Distribution)
    y_true_2d = np.array([[0.8, 0.2], [0.1, 0.9]])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8]])
    
    print("Testing 2D y_true...")
    metrics_2d = compute_metrics(y_true_2d, y_pred)
    print("2D Metrics:", metrics_2d)
    
    # Test case 2: 1D y_true (Hard labels)
    y_true_1d = np.array([0, 1])
    # Corresponding one-hot: [[1.0, 0.0], [0.0, 1.0]]
    
    print("\nTesting 1D y_true...")
    metrics_1d = compute_metrics(y_true_1d, y_pred)
    print("1D Metrics:", metrics_1d)
    
    # Check if 1D handling produces expected results (should be different from 2D because 2D was soft)
    # Let's verify with explicit one-hot
    y_true_onehot = np.array([[1.0, 0.0], [0.0, 1.0]])
    metrics_onehot = compute_metrics(y_true_onehot, y_pred)
    
    # Assert 1D handling matches explicit one-hot
    for k in metrics_1d:
        assert np.isclose(metrics_1d[k], metrics_onehot[k]), f"Mismatch in {k}"
        
    print("\nVerification Successful: 1D y_true handled correctly.")

if __name__ == "__main__":
    test_compute_metrics()

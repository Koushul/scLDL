import numpy as np
from typing import Dict

def chebyshev(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Chebyshev distance: max absolute difference."""
    return np.mean(np.max(np.abs(y_true - y_pred), axis=1))

def clark(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clark distance: squared difference normalized by squared sum."""
    # Avoid division by zero
    eps = 1e-10
    num = (y_true - y_pred) ** 2
    den = (y_true + y_pred) ** 2 + eps
    return np.mean(np.sqrt(np.sum(num / den, axis=1)))

def canberra(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Canberra metric: absolute difference normalized by sum."""
    eps = 1e-10
    num = np.abs(y_true - y_pred)
    den = y_true + y_pred + eps
    return np.mean(np.sum(num / den, axis=1))

def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kullback-Leibler divergence."""
    eps = 1e-10
    # y_true * log(y_true / y_pred)
    # If y_true is 0, the term is 0.
    # We clip y_pred to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1.0)
    # We also clip y_true to avoid log(0) in case we flip the calculation, 
    # but for standard KL(P||Q) where P is true, we only care about P > 0
    
    # Calculate term only where y_true > 0 to avoid 0 * log(0) issues
    # But standard numpy handling with masking is safer
    
    # Simple robust implementation:
    return np.mean(np.sum(y_true * np.log((y_true + eps) / (y_pred + eps)), axis=1))

def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cosine coefficient."""
    # Dot product / (norm * norm)
    dot = np.sum(y_true * y_pred, axis=1)
    norm_true = np.sqrt(np.sum(y_true ** 2, axis=1))
    norm_pred = np.sqrt(np.sum(y_pred ** 2, axis=1))
    eps = 1e-10
    return np.mean(dot / (norm_true * norm_pred + eps))

def intersection_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Intersection similarity: sum of minimums."""
    return np.mean(np.sum(np.minimum(y_true, y_pred), axis=1))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all LDL metrics.
    
    Args:
        y_true: Ground truth label distributions (N, C)
        y_pred: Predicted label distributions (N, C)
        
    Returns:
        Dictionary of metric names and values.
    """
    # Handle 1D y_true (hard labels)
    if y_true.ndim == 1:
        n_samples = y_true.shape[0]
        n_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((n_samples, n_classes))
        # Ensure integer labels
        y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
        y_true = y_true_onehot

    return {
        "Chebyshev ↓": chebyshev(y_true, y_pred),
        "Clark ↓": clark(y_true, y_pred),
        "Canberra ↓": canberra(y_true, y_pred),
        "KL Divergence ↓": kl_divergence(y_true, y_pred),
        "Cosine Similarity ↑": cosine_similarity(y_true, y_pred),
        "Intersection Similarity ↑": intersection_similarity(y_true, y_pred)
    }

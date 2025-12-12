import numpy as np
import time
from scipy.linalg import orth
# Use numpy for everything else to match Numba implementation
from numpy.linalg import svd, norm, inv
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from scLDL.lesc.lesc import LESC

# Pure Python implementation for comparison (the "Slow" version)
def solve_l2_slow(w, lambda_val):
    nw = norm(w)
    if nw > lambda_val:
        return (nw - lambda_val) * w / nw
    else:
        return np.zeros_like(w)

def solve_l1l2_slow(W, lambda_val):
    E = np.zeros_like(W)
    for i in range(W.shape[1]):
        E[:, i] = solve_l2_slow(W[:, i], lambda_val)
    return E

def lrra_slow(X, A, lambda_val):
    tol = 1e-6
    maxIter = 1000 # Match Numba implementation
    d, n = X.shape
    m = A.shape[1]
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    E = np.zeros((d, n))
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    
    # Use numpy.linalg.inv
    inv_a = inv(A.T @ A + np.eye(m))
    atx = A.T @ X
    
    for iter_idx in range(maxIter):
        temp = Z + Y2 / mu
        # Use numpy.linalg.svd
        U, s, Vt = svd(temp, full_matrices=False)
        
        svp = np.sum(s > 1/mu)
        if svp >= 1:
            s = s[:svp] - 1/mu
        else:
            svp = 1
            s = np.array([0])
        
        J = U[:, :svp] @ np.diag(s) @ Vt[:svp, :]
        
        Z = inv_a @ (atx - A.T @ E + J + (A.T @ Y1 - Y2) / mu)
        
        xmaz = X - A @ Z
        temp = xmaz + Y1 / mu
        E = solve_l1l2_slow(temp, lambda_val / mu)
        
        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        
        if stopC < tol:
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            mu = min(max_mu, mu * rho)
            
    return Z, E

def main():
    print("Generating synthetic data...")
    # Generate random data
    n_samples = 500
    n_features = 50
    X = np.random.rand(n_samples, n_features)
    
    # Prepare inputs for LRR
    X_t = X.T
    Q = orth(X_t.T)
    A = X_t @ Q
    lambda_val = 0.1
    
    print(f"Data shape: {X.shape}")
    
    # Test Slow Version
    print("\nRunning Slow Version (Python/Numpy)...")
    start_time = time.time()
    Z_slow, E_slow = lrra_slow(X_t, A, lambda_val)
    slow_time = time.time() - start_time
    print(f"Slow Time: {slow_time:.4f}s")
    
    # Test Fast Version (Numba)
    print("\nRunning Fast Version (Numba)...")
    lesc = LESC()
    # First run includes compilation time
    start_time = time.time()
    Z_fast, E_fast = lesc.solve_lrr(X, lambda_val), None # solve_lrr returns only Z, but calls _lrra_numba
    # To properly compare, we should call the internal function directly or trust solve_lrr
    # solve_lrr does: X_t -> orth -> A -> _lrra_numba -> Z -> Q*Z
    # So Z_fast here is the final Z. Z_slow above is the intermediate Z.
    # Let's verify by calling the internal numba function directly if possible, 
    # OR just compare the final Z from solve_lrr if we implemented solve_lrr_slow fully.
    # Let's just call the public solve_lrr which uses Numba.
    fast_time_1 = time.time() - start_time
    print(f"Fast Time (1st run + compile): {fast_time_1:.4f}s")
    
    # Second run (compiled)
    start_time = time.time()
    Z_fast_final = lesc.solve_lrr(X, lambda_val)
    fast_time_2 = time.time() - start_time
    print(f"Fast Time (2nd run): {fast_time_2:.4f}s")
    
    print(f"\nSpeedup (vs 2nd run): {slow_time / fast_time_2:.2f}x")
    
    # Verification
    # We need to compute Z_slow_final = Q @ Z_slow to compare with Z_fast_final
    Z_slow_final = Q @ Z_slow
    
    diff = np.linalg.norm(Z_fast_final - Z_slow_final)
    print(f"\nDifference norm: {diff:.6e}")
    
    if diff < 1e-4:
        print("SUCCESS: Results match!")
    else:
        print("WARNING: Results differ significantly.")

if __name__ == "__main__":
    main()

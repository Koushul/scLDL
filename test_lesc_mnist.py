import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from scLDL.lesc.lesc import LESC

def load_mnist_subset(n_samples=None):
    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Normalize
    X /= 255.0
    
    if n_samples is not None:
        # Subset
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    L = enc.fit_transform(y.reshape(-1, 1))
    
    return X, L, y

def main():
    # Load Data
    # Half of MNIST (35,000 samples). Kernel matrix will be ~5-10GB.
    X, L, y = load_mnist_subset(n_samples=35000)
    
    # Split
    X_train, X_test, L_train, L_test, y_train, y_test = train_test_split(
        X, L, y, test_size=0.2, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Initialize LESC
    # Parameters might need tuning, using defaults/heuristic
    # sigma for RBF: typically related to mean distance
    mean_dist = np.mean(np.linalg.norm(X_train - np.mean(X_train, axis=0), axis=1))
    sigma = mean_dist * 2
    
    lesc = LESC(lambda_param=0.1, beta=0.1, kernel='rbf', sigma=sigma)
    
    # Fit
    print("Training LESC...")
    lesc.fit(X_train, L_train)
    
    # Predict
    print("Predicting...")
    L_pred = lesc.predict(X_test)
    y_pred = np.argmax(L_pred, axis=1)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    
    # Check MSE
    mse = np.mean((L_pred - L_test)**2)
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    main()

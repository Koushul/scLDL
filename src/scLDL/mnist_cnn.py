import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from label_enhancer import LIBLE

def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Flatten images and create one-hot labels
    X_train = train_dataset.data.float().view(-1, 28*28) / 255.0
    y_train = train_dataset.targets
    L_train = F.one_hot(y_train, num_classes=10).float()
    
    X_test = test_dataset.data.float().view(-1, 28*28) / 255.0
    y_test = test_dataset.targets
    L_test = F.one_hot(y_test, num_classes=10).float()
    
    return X_train.numpy(), L_train.numpy(), X_test.numpy(), L_test.numpy()

def main():
    print("Loading MNIST data...")
    X_train, L_train, X_test, L_test = get_mnist_data()
    
    # Use a subset for faster testing if needed, but full set is fine for MNIST
    # X_train = X_train[:1000]
    # L_train = L_train[:1000]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {L_train.shape}")
    
    print("Initializing LIBLE...")
    model = LIBLE(
        n_features=784,
        n_outputs=10,
        n_hidden=128,
        n_latent=32,
        alpha=1e-3,
        beta=1e-3,
        lr=1e-3,
        epochs=5, # Short training for verification
        batch_size=64
    )
    
    print("Training LIBLE...")
    model.fit(X_train, L_train)
    
    print("Predicting...")
    D_pred = model.predict(X_test)
    
    # Evaluate
    # Since we don't have ground truth label distributions, we can check if the max probability matches the true label
    y_pred = np.argmax(D_pred, axis=1)
    y_true = np.argmax(L_test, axis=1)
    
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy (using predicted distribution max): {accuracy:.2%}")
    
    print("Sample prediction (first 5):")
    print("True:", y_true[:5])
    print("Pred:", y_pred[:5])
    print("Distributions (first 2):")
    print(D_pred[:2])

if __name__ == "__main__":
    main()

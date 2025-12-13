import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # ResNet expects 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 2. Load ResNet Baseline
    print("Loading ResNet18 Baseline...")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('models/ResNet18_baseline.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. Generate Soft Labels
    def generate_distributions(loader, desc):
        all_probs = []
        all_labels = []
        # We also need the original images, but MNIST dataset objects have them.
        # Collecting 60k images into RAM as tensors might be OK (Float32: 60000*3*28*28 * 4 bytes ~= 564 MB).
        # We will save them as arrays.
        
        all_images = []
        
        # Note: ResNet expects RGB normalized. For LDL model training later, 
        # we might want raw 1-channel images if we use a simple CNN, 
        # or we might want to consistent. 
        # Let's save the raw 1-channel images (un-normalized) from the source dataset for flexibility,
        # or just save indices? Saving raw pixels is safer.
        # Actually, let's load raw separately to save.
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images_dev = images.to(DEVICE)
                logits = model(images_dev)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # To save space, we won't replicate the huge tensor here unless needed.
                # We can reload images from standard MNIST dataset later using indices.
                # But to ensure alignment, let's just assume standard order.
                # Standard MNIST loaders without shuffle deterministically yield same order.
                
        return np.concatenate(all_probs), np.concatenate(all_labels)

    print("Generating Training Distributions...")
    train_probs, train_hard = generate_distributions(train_loader, "Train Stats")
    
    print("Generating Test Distributions...")
    test_probs, test_hard = generate_distributions(test_loader, "Test Stats")
    
    # 4. Save
    os.makedirs('data', exist_ok=True)
    data = {
        'train_probs': train_probs,
        'train_labels': train_hard,
        'test_probs': test_probs,
        'test_labels': test_hard
    }
    
    save_path = 'data/mnist_ldl_distilled.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved distilled distributions to {save_path}")
    print(f"Train Shape: {train_probs.shape}, Test Shape: {test_probs.shape}")

if __name__ == "__main__":
    main()

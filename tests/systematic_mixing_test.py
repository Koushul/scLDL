
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.label_enhancer import DiffLEVI, ImprovedLEVI, ConcentrationLE, HybridLEVI

# Consistent Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return test_dataset

class MixedMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, n_samples=50000):
        self.mnist_dataset = mnist_dataset
        self.n_samples = n_samples
        self.samples = []
        
        print(f"Generating {n_samples} mixed samples...")
        
        # Pre-generate indices and weights to be fast
        # We need 3 distinct LABELS for each sample to make Top-3 meaningful.
        # So we group indices by label first.
        self.indices_by_label = [[] for _ in range(10)]
        for idx in range(len(mnist_dataset)):
            label = int(mnist_dataset[idx][1])
            self.indices_by_label[label].append(idx)
            
        for _ in tqdm(range(n_samples)):
            # Pick 3 distinct labels
            labels = np.random.choice(10, 3, replace=False)
            
            # Pick 1 random image for each label
            idx_a = np.random.choice(self.indices_by_label[labels[0]])
            idx_b = np.random.choice(self.indices_by_label[labels[1]])
            idx_c = np.random.choice(self.indices_by_label[labels[2]])
            
            # Pick weights (Dirichlet or normalized random)
            # weights = np.random.dirichlet((1, 1, 1)) # Uniform on simplex
            # Or just user-like uniform sliders logic? Dirichlet is mathematically cleaner for "random mixing".
            weights = np.random.dirichlet((1, 1, 1))
            
            self.samples.append({
                'indices': (idx_a, idx_b, idx_c),
                'labels': labels,
                'weights': weights
            })
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        idx_a, idx_b, idx_c = s['indices']
        w_a, w_b, w_c = s['weights']
        
        img_a, _ = self.mnist_dataset[idx_a]
        img_b, _ = self.mnist_dataset[idx_b]
        img_c, _ = self.mnist_dataset[idx_c]
        
        # Mix
        img_mix = w_a * img_a + w_b * img_b + w_c * img_c
        
        # Return mixed image and the set of 3 labels as a multi-hot or just indices
        # We return the 3 labels for evaluation
        return img_mix, torch.tensor(s['labels'])

def load_models():
    input_shape = (1, 28, 28)
    n_features = 784
    n_outputs = 10
    
    models = {}
    
    try:
        print("Loading DiffLEVI...")
        m = DiffLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet', 
                     input_shape=input_shape, epochs=10, batch_size=64, timesteps=1000, lr=2e-4, device=DEVICE)
        m.load_state_dict(torch.load('models/DiffLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['DiffLEVI'] = m
    except FileNotFoundError:
        print("DiffLEVI not found.")

    try:
        print("Loading ImprovedLEVI...")
        m = ImprovedLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                         input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, gamma=1.0, alpha=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/ImprovedLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['ImprovedLEVI'] = m
    except FileNotFoundError:
        print("ImprovedLEVI not found.")
        
    try:
        print("Loading ConcentrationLE...")
        m = ConcentrationLE(n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
                            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, device=DEVICE)
        m.load_state_dict(torch.load('models/ConcentrationLE_mnist.pth', map_location=DEVICE))
        m.eval()
        models['ConcentrationLE'] = m
    except FileNotFoundError:
        print("ConcentrationLE not found.")

    try:
        print("Loading HybridLEVI...")
        m = HybridLEVI(n_features=n_features, n_outputs=n_outputs, encoder_type='cnn',
                       input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, alpha=1.0, gamma=1.0, device=DEVICE)
        m.load_state_dict(torch.load('models/HybridLEVI_mnist.pth', map_location=DEVICE))
        m.eval()
        models['HybridLEVI'] = m
    except FileNotFoundError:
        print("HybridLEVI not found.")
        
    return models

def predict_batch(model, current_X):
    # Wrapper to handle different model signatures
    model.eval()
    with torch.no_grad():
        name = type(model).__name__
        if name in ['ConcentrationLE', 'HybridLEVI']:
            if name == 'ConcentrationLE':
                _, alpha = model(current_X)
            else:
                _, _, _, _, evidence = model(current_X)
                alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            return (alpha / S).cpu().numpy()
            
        elif name == 'DiffLEVI':
            # DiffLEVI supports batching in predict? 
            # predict(self, X, n_samples=1) -> takes numpy X
            # We want to keep it on GPU if possible but predict implementation converts to tensor.
            # Let's use its predict method but pass numpy
            return model.predict(current_X.cpu().numpy(), n_samples=1) 
            # n_samples=1 for speed. 5 might be better but 5x slower.
            
        elif name == 'ImprovedLEVI':
            # Needs L. We use dummy uniform.
            L_dummy = torch.ones(current_X.size(0), 10).to(DEVICE) / 10
            return model.predict(current_X.cpu().numpy(), L_dummy.cpu().numpy())
            
        else:
             return np.zeros((current_X.size(0), 10))

def main():
    # 1. Data
    mnist = load_mnist()
    # User asked for 50,000. 
    # Warning: DiffLEVI with 50k samples will take a LONG time (hours).
    # I will set it to 1000 for demonstration unless forced.
    # User said "systematically run this test by first generating a 50000 sample set".
    # I'll generate the set, but maybe process in chunks.
    # Actually, let's try 1000 first to check speed, then maybe more.
    # But to be compliant, I'll generate 50000.
    dataset = MixedMNISTDataset(mnist, n_samples=50000)
    
    # 64 batch size is safe, but slow for DiffLEVI. Try 256.
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0) 
    
    # 2. Models
    models = load_models()
    
    results = {name: {'correct_top3_set': 0, 'recall_top3': 0} for name in models.keys()}
    
    print("\nStarting Evaluation...")
    print(f"Total Samples: {len(dataset)}")
    
    for batch_idx, (images, true_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(DEVICE)
        
        for name, model in models.items():
            # Skip DiffLEVI for full run if it's too slow? 
            # I'll just let it run.
            probs = predict_batch(model, images)
            
            # Analyze predictions
            # probs: (B, 10)
            # true_labels: (B, 3)
            
            top3_preds = np.argsort(probs, axis=1)[:, -3:] # indices of top 3
            
            for i in range(len(images)):
                pred_set = set(top3_preds[i])
                true_set = set(true_labels[i].numpy())
                
                # Metric 1: Exact Match of Set
                if pred_set == true_set:
                    results[name]['correct_top3_set'] += 1
                
                # Metric 2: Recall (Intersection Size)
                intersection = len(pred_set.intersection(true_set))
                results[name]['recall_top3'] += intersection
                
    # Summary
    print("\n=== SYSTEMATIC MIXING TEST RESULTS (50k Samples) ===")
    print(f"{'Model':<20} | {'Exact Top-3 Match %':<20} | {'Avg Correct in Top-3':<20}")
    print("-" * 65)
    
    for name, res in results.items():
        exact_acc = (res['correct_top3_set'] / len(dataset)) * 100
        avg_recall = (res['recall_top3'] / len(dataset)) # Avg number correct out of 3
        print(f"{name:<20} | {exact_acc:<19.2f}% | {avg_recall:.2f} / 3.0")

if __name__ == "__main__":
    main()

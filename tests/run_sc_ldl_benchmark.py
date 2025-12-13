import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scanpy as sc
import anndata
# import plotting_utils as pu # Removed unused
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.ldl_models import ConcentrationLDL
# from scLDL.label_enhancer import MLPBaseline # Removed, defining inline

class MLPBaseline(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_classes)
        )
        
    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def inject_noise(y, noise_rate=0.0):
    """
    Randomly flip labels for noise_rate fraction of samples.
    """
    if noise_rate <= 0:
        return y.copy()
        
    y_noisy = y.copy()
    n_samples = len(y)
    n_classes = len(np.unique(y))
    n_noise = int(n_samples * noise_rate)
    
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    for idx in noise_indices:
        # Flip to random other class
        current_label = y[idx]
        possible_labels = list(range(n_classes))
        possible_labels.remove(current_label)
        y_noisy[idx] = np.random.choice(possible_labels)
        
    return y_noisy

def train_teacher(X_train, y_train, n_classes, epochs=10):
    print(f"Training Teacher (MLPBaseline) on {len(X_train)} samples...")
    n_features = X_train.shape[1]
    
    model = MLPBaseline(n_features, n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def distill_soft_labels(teacher, X):
    teacher.eval()
    all_probs = []
    
    # Process in chunks to avoid memory issues if large
    batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).float().to(DEVICE)
            logits = teacher(batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs)

def train_student_ldl(X_train, soft_labels, n_classes, epochs=20):
    print(f"Training Student (ConcentrationLDL) on {len(X_train)} samples (Soft Labels)...")
    n_features = X_train.shape[1]
    
    model = ConcentrationLDL(n_features, n_classes, encoder_type='mlp', device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(soft_labels).float())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_mse = 0
        total_kl = 0
        
        # Annealing
        kl_coeff = min(0.01, (epoch / 10.0) * 0.01)
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            
            loss, alpha, loss_mse, loss_kl = model.compute_loss(batch_x, batch_y, global_step=epoch, lambda_kl=kl_coeff)
            
            loss.backward()
            optimizer.step()
            
            total_mse += loss_mse
            total_kl += loss_kl
            
    return model

def evaluate_acc(model, X_test, y_test):
    model.eval()
    batch_size = 1000
    preds = []
    
    with torch.no_grad():
         for i in range(0, len(X_test), batch_size):
             batch = torch.tensor(X_test[i:i+batch_size]).float().to(DEVICE)
             
             if isinstance(model, MLPBaseline):
                 logits = model(batch)
                 p = torch.argmax(logits, dim=1).cpu().numpy()
             else:
                 # LDL Model
                 probs = model.predict_distribution(batch).cpu()
                 p = torch.argmax(probs, dim=1).numpy()
                 
             preds.append(p)
             
    preds = np.concatenate(preds)
    acc = np.mean(preds == y_test)
    return acc

def main():
    # 1. Load Data
    print("Loading adata.h5ad...")
    adata = anndata.read_h5ad('data/adata.h5ad')
    
    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['cell_type'].values)
    n_classes = len(np.unique(y))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # --- Experiment 1: Clean Data ---
    print("\n=== Experiment 1: Clean Data ===")
    teacher_clean = train_teacher(X_train, y_train, n_classes)
    acc_teacher_clean = evaluate_acc(teacher_clean, X_test, y_test)
    print(f"Teacher (Clean) Acc: {acc_teacher_clean*100:.2f}%")
    
    # Distill
    soft_labels_clean = distill_soft_labels(teacher_clean, X_train)
    
    # Train Student
    student_clean = train_student_ldl(X_train, soft_labels_clean, n_classes)
    acc_student_clean = evaluate_acc(student_clean, X_test, y_test)
    print(f"Student LDL (Clean) Acc: {acc_student_clean*100:.2f}%")
    
    results['Clean'] = {'Teacher': acc_teacher_clean, 'Student': acc_student_clean}

    # --- Experiment 2: Noisy Data (20% Flip) ---
    print("\n=== Experiment 2: Noisy Data (20% Warped) ===")
    y_train_noisy = inject_noise(y_train, noise_rate=0.20)
    
    teacher_noisy = train_teacher(X_train, y_train_noisy, n_classes)
    acc_teacher_noisy = evaluate_acc(teacher_noisy, X_test, y_test) # Eval on CLEAN test
    print(f"Teacher (Noisy) Acc: {acc_teacher_noisy*100:.2f}%")
    
    # Distill (Teacher produces soft labels based on noisy training)
    # Does LDL help here? 
    # LDL learns the distribution the teacher outputs.
    # If teacher overfits to noise, LDL learns noise.
    # But if teacher outputs lower confidence for noisy samples (due to conflict), 
    # LDL evidential loss might handle it differently.
    
    soft_labels_noisy = distill_soft_labels(teacher_noisy, X_train)
    
    student_noisy = train_student_ldl(X_train, soft_labels_noisy, n_classes)
    acc_student_noisy = evaluate_acc(student_noisy, X_test, y_test)
    print(f"Student LDL (Noisy) Acc: {acc_student_noisy*100:.2f}%")
    
    results['Noisy'] = {'Teacher': acc_teacher_noisy, 'Student': acc_student_noisy}
    
    # Plotting
    labels = ['Clean Train', 'Noisy Train (20%)']
    teacher_accs = [results['Clean']['Teacher'], results['Noisy']['Teacher']]
    student_accs = [results['Clean']['Student'], results['Noisy']['Student']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, teacher_accs, width, label='Teacher (MLP Baseline)')
    rects2 = ax.bar(x + width/2, student_accs, width, label='Student (ConcentrationLDL)')
    
    ax.set_ylabel('Test Accuracy (Clean)')
    ax.set_title('Robustness Benchmark: MLP vs LDL')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('reports/sc_ldl_benchmark.png')
    print("\nSaved benchmark plot to reports/sc_ldl_benchmark.png")
    
if __name__ == "__main__":
    main()

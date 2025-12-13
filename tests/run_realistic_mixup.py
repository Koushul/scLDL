import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scLDL.ldl_models import ConcentrationLDL, kl_divergence_dirichlet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Inline Baseline ---
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
        kl_coeff = min(0.01, (epoch / 10.0) * 0.01)
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss, alpha, loss_mse, loss_kl = model.compute_loss(batch_x, batch_y, global_step=epoch, lambda_kl=kl_coeff)
            loss.backward()
            optimizer.step()
            
    return model

def generate_mixups(X_test, y_test_fine, y_test_coarse, n_mixes=5000):
    """
    Generate mixtures ONLY between cells of the SAME coarse type.
    """
    X_mix = []
    y_mix_dist = []
    unique_coarse = np.unique(y_test_coarse)
    n_classes = len(np.unique(y_test_fine)) # Max class index + 1
    # Ensure n_classes covers all labels
    n_classes = np.max(y_test_fine) + 1
    
    # Group indices by coarse type
    coarse_groups = {}
    for c in unique_coarse:
        coarse_groups[c] = np.where(y_test_coarse == c)[0]
        
    print(f"Generating {n_mixes} realistic mixes (within-coarse)...")
    
    for _ in range(n_mixes):
        # Pick a random coarse type
        c = np.random.choice(unique_coarse)
        indices = coarse_groups[c]
        
        if len(indices) < 2:
            continue
            
        # Pick 2 random cells from this coarse group
        idx1, idx2 = np.random.choice(indices, 2, replace=True)
        
        # Mix
        ratio = 0.5
        x_new = ratio * X_test[idx1] + (1 - ratio) * X_test[idx2]
        
        # Target Distribution
        # One hot encoded fine labels
        def one_hot(label, size):
            vec = np.zeros(size)
            vec[label] = 1.0
            return vec
            
        y1 = one_hot(y_test_fine[idx1], n_classes)
        y2 = one_hot(y_test_fine[idx2], n_classes)
        y_new = ratio * y1 + (1 - ratio) * y2
        
        X_mix.append(x_new)
        y_mix_dist.append(y_new)
        
    return np.array(X_mix), np.array(y_mix_dist)

def evaluate_distribution_metrics(model, X_mix, y_mix):
    model.eval()
    X_tensor = torch.tensor(X_mix).float().to(DEVICE)
    
    with torch.no_grad():
        if isinstance(model, MLPBaseline):
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probs = model.predict_distribution(X_tensor).cpu().numpy()
            
    # Metrics
    # 1. Cosine Similarity
    y_mix = np.array(y_mix)
    dot = np.sum(probs * y_mix, axis=1)
    norm_p = np.linalg.norm(probs, axis=1)
    norm_t = np.linalg.norm(y_mix, axis=1)
    cos_sim = dot / (norm_p * norm_t + 1e-8)
    
    # 2. KL Divergence (True || Pred)
    # Clip for stability
    probs_c = np.clip(probs, 1e-10, 1.0)
    y_mix_c = np.clip(y_mix, 1e-10, 1.0) 
    # Actually if y_mix has 0s, KL(y||p) is 0 * log(0/p) = 0.
    # We only care where y > 0.
    kl = np.sum(y_mix * np.log(y_mix_c / probs_c), axis=1)
    
    return np.mean(cos_sim), np.mean(kl)

def main():
    # 1. Load Data
    print("Loading Data...")
    adata = anndata.read_h5ad('data/adata.h5ad')
    df = pd.read_csv('data/tonsil_cell_types.csv')
    
    # Merge on index
    # adata.obs index is barcode. df['NAME'] is barcode.
    # Set index for df
    df = df.set_index('NAME')
    
    # Join
    # Keep only cells present in both
    common_cells = adata.obs.index.intersection(df.index)
    adata = adata[common_cells].copy()
    df = df.loc[common_cells]
    
    # Add Fine Label to adata
    adata.obs['cell_type_fine'] = df['cell_type_2']
    
    print(f"Merged Data: {adata.shape[0]} cells")
    print("Coarse Types:", adata.obs['cell_type'].unique())
    print("Fine Types (Sample):", adata.obs['cell_type_fine'].unique()[:5])
    
    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    # Encoders
    le_fine = LabelEncoder()
    y_fine = le_fine.fit_transform(adata.obs['cell_type_fine'].values)
    n_classes_fine = len(le_fine.classes_)
    
    y_coarse = adata.obs['cell_type'].values # Keep as string for grouping
    
    # Split
    # Stratify by Fine Type
    X_train, X_test, y_fine_train, y_fine_test, y_coarse_train, y_coarse_test = train_test_split(
        X, y_fine, y_coarse, test_size=0.2, random_state=42, stratify=y_fine
    )
    
    # 2. Train Teacher (Fine Labels)
    print("\n--- Training Teacher ---")
    teacher = train_teacher(X_train, y_fine_train, n_classes_fine)
    
    # 3. Distill
    print("\n--- Distilling ---")
    soft_labels = distill_soft_labels(teacher, X_train)
    
    # 4. Train Student (LDL)
    print("\n--- Training Student LDL ---")
    student_ldl = train_student_ldl(X_train, soft_labels, n_classes_fine)
    
    # 5. Generate Realistic Mixes (Test Set)
    print("\n--- Generating Mixes ---")
    X_mix, y_mix = generate_mixups(X_test, y_fine_test, y_coarse_test, n_mixes=10000)
    
    # 6. Evaluate
    print("\n--- Evaluating ---")
    cos_teacher, kl_teacher = evaluate_distribution_metrics(teacher, X_mix, y_mix)
    cos_student, kl_student = evaluate_distribution_metrics(student_ldl, X_mix, y_mix)
    
    print("\n=== RESULTS (Realistic Within-Cluster Mixing) ===")
    print(f"Teacher (Baseline) | CosSim: {cos_teacher:.4f} | KL: {kl_teacher:.4f}")
    print(f"Student (LDL)      | CosSim: {cos_student:.4f} | KL: {kl_student:.4f}")
    
    # Plotting
    metrics = ['Cosine Similarity', 'KL Divergence (Lower better)']
    teacher_scores = [cos_teacher, kl_teacher]
    student_scores = [cos_student, kl_student]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, teacher_scores, width, label='Teacher (MLP)')
    rects2 = ax.bar(x + width/2, student_scores, width, label='Student (LDL)')
    
    ax.set_title('Realistic Mixup Recovery (Within-Coarse Lineage)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    print("Saved plot to reports/realistic_mixup_plot.png")

    # --- Additional Visualizations ---
    print("\n--- Generating Additional Plots ---")
    
    # 1. Confusion Matrices (Clean Test Set)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    def plot_confusion_matrix(model, X, y_true, title, filename):
        model.eval()
        X_tensor = torch.tensor(X).float().to(DEVICE)
        with torch.no_grad():
            if isinstance(model, MLPBaseline):
                logits = model(X_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                probs = model.predict_distribution(X_tensor).cpu().numpy()
                preds = torch.argmax(torch.tensor(probs), dim=1).numpy()
        
        cm = confusion_matrix(y_true, preds)
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # We need class names for better interpretation, but indices are fine for overview
    # Plotting usually requires re-importing seaborn if not at top, but we added import above.
    
    plot_confusion_matrix(teacher, X_test, y_fine_test, 
                         'Teacher Confusion Matrix (Clean Test)', 
                         'reports/realistic_mixup_cm_teacher.png')
    
    plot_confusion_matrix(student_ldl, X_test, y_fine_test, 
                         'Student LDL Confusion Matrix (Clean Test)', 
                         'reports/realistic_mixup_cm_student.png')
                         
    print("Saved confusion matrices to reports/")

    # 2. Distribution Examples (Mixed Set)
    def plot_distribution_examples(model_t, model_s, X_mix, y_mix, n_examples=3, filename='reports/realistic_mixup_dist_examples.png'):
        indices = np.random.choice(len(X_mix), n_examples, replace=False)
        
        X_sample = torch.tensor(X_mix[indices]).float().to(DEVICE)
        y_sample = y_mix[indices]
        
        model_t.eval()
        model_s.eval()
        
        with torch.no_grad():
            # Teacher
            logits_t = model_t(X_sample)
            probs_t = torch.softmax(logits_t, dim=1).cpu().numpy()
            
            # Student
            probs_s = model_s.predict_distribution(X_sample).cpu().numpy()
            
        fig, axes = plt.subplots(n_examples, 1, figsize=(10, 4*n_examples))
        if n_examples == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            # Only plot top K classes for readability
            # Find classes with > 0 probability in GT
            gt_classes = np.where(y_sample[i] > 0.01)[0]
            # Add top predicted classes to view
            top_t = np.argsort(probs_t[i])[-3:]
            top_s = np.argsort(probs_s[i])[-3:]
            
            interesting_classes = np.unique(np.concatenate([gt_classes, top_t, top_s]))
            
            x = np.arange(len(interesting_classes))
            width = 0.25
            
            val_gt = y_sample[i][interesting_classes]
            val_t = probs_t[i][interesting_classes]
            val_s = probs_s[i][interesting_classes]
            
            ax.bar(x - width, val_gt, width, label='Ground Truth', color='black', alpha=0.6)
            ax.bar(x, val_t, width, label='Teacher', color='blue', alpha=0.6)
            ax.bar(x + width, val_s, width, label='Student LDL', color='green', alpha=0.6)
            
            ax.set_xticks(x)
            ax.set_xticklabels(interesting_classes)
            ax.set_title(f"Example {i+1}: True Mixed Classes {gt_classes}")
            ax.set_ylabel('Probability')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_distribution_examples(teacher, student_ldl, X_mix, y_mix)
    print("Saved distribution examples to reports/realistic_mixup_dist_examples.png")


if __name__ == "__main__":
    main()

import streamlit as st
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
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scLDL.ldl_models import ConcentrationLDL

# --- Configuration ---
st.set_page_config(page_title="scLDL Cross-Dataset Simulator", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Models ---
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

# --- Helper Functions ---
def load_and_preprocess(file_path):
    adata = anndata.read_h5ad(file_path)
    # Heuristic for raw counts
    if adata.n_vars > 5000:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # Don't HVG yet, we need to match genes later
    
    # Identify Label Column
    label_col = None
    for col in ['cell_type', 'cell_type_fine', 'cell_type_2', 'Cluster', 'Subcluster']:
        if col in adata.obs.columns:
            label_col = col
            break
    if label_col is None:
        cats = adata.obs.select_dtypes(include=['category', 'object']).columns
        if len(cats) > 0:
            label_col = cats[0]
            
    return adata, label_col

def try_match_genes(source_adata, target_names):
    """
    Tries to find a column in source_adata.var that overlaps with target_names.
    Returns: best_col_name, overlap_count
    """
    best_col = None
    best_overlap = 0
    
    # Check index first
    overlap = len(source_adata.var_names.intersection(target_names))
    if overlap > 500: # Strong overlap
        return 'index', overlap
        
    for col in source_adata.var.columns:
        if source_adata.var[col].dtype == object or source_adata.var[col].dtype.name == 'category':
            # Try matching this column to target names
            vals = set(source_adata.var[col].astype(str))
            current = len(vals.intersection(target_names))
            if current > best_overlap:
                best_overlap = current
                best_col = col
                
    return best_col, best_overlap

def align_datasets(adata_train, adata_test):
    # 1. Try Direct
    common = adata_train.var_names.intersection(adata_test.var_names)
    
    # 2. Heuristic Fix if overlap is poor (< 10% of smaller dataset or < 500 genes)
    min_genes = min(adata_train.n_vars, adata_test.n_vars)
    if len(common) < min(500, min_genes * 0.1):
        st.warning(f"Low gene overlap ({len(common)}). Attempting to auto-fix Gene IDs (ENSG vs Symbol)...")
        
        # Try to match Train columns to Test Index
        col_train, n_train = try_match_genes(adata_train, set(adata_test.var_names))
        
        # Try to match Test columns to Train Index
        col_test, n_test = try_match_genes(adata_test, set(adata_train.var_names))
        
        # Decision
        if col_train != 'index' and n_train > 500:
            st.success(f"Found gene match in Train data column: '{col_train}'. Swapping IDs.")
            adata_train.var_names = adata_train.var[col_train].astype(str).values
            # Re-index to ensure uniqueness if needed (scanpy warns typically)
            adata_train.var_names_make_unique()
            
        elif col_test != 'index' and n_test > 500:
            st.success(f"Found gene match in Test data column: '{col_test}'. Swapping IDs.")
            adata_test.var_names = adata_test.var[col_test].astype(str).values
            adata_test.var_names_make_unique()
            
        # Re-calc common
        common = adata_train.var_names.intersection(adata_test.var_names)
    
    if len(common) == 0:
        st.error("❌ No common genes found even after auto-fix attempt. Please check gene IDs manually.")
        st.stop()
    
    st.info(f"Aligned on {len(common)} common genes.")
    
    # If too few, maybe HVG on train first?
    if len(common) > 2000:
        # Calculate HVG on Train restricted to common
        temp_adata = adata_train[:, common].copy()
        sc.pp.highly_variable_genes(temp_adata, n_top_genes=2000, subset=True)
        final_genes = temp_adata.var_names
    else:
        final_genes = common
        
    return adata_train[:, final_genes].copy(), adata_test[:, final_genes].copy()

def train_models(adata_train, label_col):
    # Prepare Data
    X = adata_train.X
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    y_raw = adata_train.obs[label_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    
    # Split for Validation (internal to training process)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Teacher
    teacher = MLPBaseline(X.shape[1], n_classes).to(DEVICE)
    opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    ds_t = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    dl_t = DataLoader(ds_t, batch_size=64, shuffle=True)
    
    desc_t = st.empty()
    bar_t = st.progress(0)
    
    for epoch in range(10): 
        teacher.train()
        for bx, by in dl_t:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_t.zero_grad()
            out = teacher(bx)
            loss = crit(out, by)
            loss.backward()
            opt_t.step()
        bar_t.progress((epoch+1)/10)
        desc_t.text(f"Training Teacher: Epoch {epoch+1}/10")
        
    # Distill
    teacher.eval()
    with torch.no_grad():
        logits = teacher(torch.tensor(X_train).float().to(DEVICE))
        soft_labels = torch.softmax(logits, dim=1).cpu().numpy()
        
    # Student
    student = ConcentrationLDL(X.shape[1], n_classes, encoder_type='mlp', device=DEVICE).to(DEVICE)
    opt_s = optim.Adam(student.parameters(), lr=1e-3)
    
    ds_s = TensorDataset(torch.tensor(X_train).float(), torch.tensor(soft_labels).float())
    dl_s = DataLoader(ds_s, batch_size=64, shuffle=True)
    
    desc_s = st.empty()
    bar_s = st.progress(0)
    
    for epoch in range(20):
        student.train()
        kl_coeff = min(0.01, (epoch/10.0)*0.01)
        for bx, by in dl_s:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt_s.zero_grad()
            loss, _, _, _ = student.compute_loss(bx, by, global_step=epoch, lambda_kl=kl_coeff)
            loss.backward()
            opt_s.step()
        bar_s.progress((epoch+1)/20)
        desc_s.text(f"Training Student: Epoch {epoch+1}/20")
            
    desc_t.empty(); bar_t.empty()
    desc_s.empty(); bar_s.empty()
    
    return teacher, student, le

# --- Sidebar ---
st.sidebar.title("Configuration")
data_dir = "data"
files = [f for f in os.listdir(data_dir) if f.endswith(".h5ad")]

st.sidebar.subheader("1. Training Data")
file_train = st.sidebar.selectbox("Train Dataset", files, index=0)

st.sidebar.subheader("2. Testing Data (Simulator)")
file_test = st.sidebar.selectbox("Test Dataset", files, index=0) # Default to same

if st.sidebar.button("Load & Train"):
    with st.spinner("Loading and Aligning Datasets..."):
        ad_train, label_col_train = load_and_preprocess(os.path.join(data_dir, file_train))
        ad_test, label_col_test = load_and_preprocess(os.path.join(data_dir, file_test))
        
        ad_train, ad_test = align_datasets(ad_train, ad_test)
        
        st.session_state['ad_train'] = ad_train
        st.session_state['ad_test'] = ad_test
        st.session_state['label_col_train'] = label_col_train
        st.session_state['label_col_test'] = label_col_test
        
    with st.spinner("Training Models..."):
        teacher, student, le = train_models(ad_train, label_col_train)
        st.session_state['teacher'] = teacher
        st.session_state['student'] = student
        st.session_state['le_train'] = le
        st.sidebar.success("Ready!")

# --- Main ---
st.title("🧬 scLDL Cross-Dataset Simulator")

if 'teacher' in st.session_state:
    teacher = st.session_state['teacher']
    student = st.session_state['student']
    le_train = st.session_state['le_train']
    ad_test = st.session_state['ad_test']
    
    # Prepare Test Data
    X_test = ad_test.X
    if hasattr(X_test, "toarray"): X_test = X_test.toarray()
    n_test_cells = X_test.shape[0]
    
    # Test Labels (Ground Truth strings)
    y_test_labels = ad_test.obs[st.session_state['label_col_test']].values
    test_classes = np.unique(y_test_labels)
    
    train_classes = le_train.classes_
    
    st.info(f"**Training Classes**: {len(train_classes)} | **Test Classes**: {len(test_classes)} ({st.session_state['label_col_test']})")
    
    # --- Controls ---
    st.markdown("### 🎛️ Configure Mixture")
    
    col_n, col_x = st.columns([1, 3])
    with col_n:
        n_cells = st.number_input("Mix N Cells", 2, 5, 2)
        
    cols = st.columns(n_cells)
    
    # Ensure weights exist
    if 'weights' not in st.session_state or len(st.session_state.weights) != n_cells:
        st.session_state.weights = [1.0/n_cells] * n_cells
        
    selected_indices = []
    
    # Global Re-roll button
    if st.button("🎲 Re-roll Samples (Keep Types)"):
        # Clear specific indices so they get re-sampled below
        for key in list(st.session_state.keys()):
            if key.startswith("idx_"):
                del st.session_state[key]
    
    for i in range(n_cells):
        with cols[i]:
            st.markdown(f"**Component {i+1}**")
            
            # 1. Select Type
            # Default to random different types if not set
            if f"type_{i}" not in st.session_state:
                rand_type = np.random.choice(test_classes)
                st.session_state[f"type_{i}"] = rand_type
                
            c_type = st.selectbox(f"Cell Type", test_classes, key=f"type_{i}")
            
            # 2. Auto-Sample Index for this Type
            avail_indices = np.where(y_test_labels == c_type)[0]
            
            if len(avail_indices) == 0:
                st.error("No cells!")
                continue
                
            # If no index selected OR selected index doesn't match current type (user changed type), re-roll
            current_idx_key = f"idx_{i}"
            need_reroll = False
            
            if current_idx_key not in st.session_state:
                need_reroll = True
            else:
                # Check consistency
                curr = st.session_state[current_idx_key]
                if curr >= n_test_cells or y_test_labels[curr] != c_type:
                    need_reroll = True
            
            if need_reroll:
                st.session_state[current_idx_key] = int(np.random.choice(avail_indices))
            
            # Store final index
            final_idx = st.session_state[current_idx_key]
            selected_indices.append(final_idx)
            
            st.caption(f"Sample ID: `{final_idx}`")
            
            # 3. Weight
            w = st.slider(f"Weight", 0.0, 1.0, st.session_state.weights[i], key=f"w_{i}")
            st.session_state.weights[i] = w

    # Mix
    st.divider()
    weights = np.array([st.session_state.weights[i] for i in range(n_cells)])
    if weights.sum() == 0: weights = np.ones(n_cells)
    weights = weights / weights.sum()
    
    # Gather True Ratios (Aggregated by Type)
    true_composition = {}
    for i, idx in enumerate(selected_indices):
        lbl = y_test_labels[idx]
        true_composition[lbl] = true_composition.get(lbl, 0.0) + weights[i]
        
    if len(selected_indices) == n_cells:
        x_mix = np.zeros_like(X_test[0])
        for i, idx in enumerate(selected_indices):
            x_mix += weights[i] * X_test[idx]
            
        x_mix_tensor = torch.tensor(x_mix).float().unsqueeze(0).to(DEVICE)
        
        # Predict
        teacher.eval(); student.eval()
        with torch.no_grad():
            p_teacher = torch.softmax(teacher(x_mix_tensor), dim=1).cpu().numpy()[0]
            beliefs, uncertainty = student.predict_evidence(x_mix_tensor)
            p_student = beliefs.cpu().numpy()[0]
            u_student = uncertainty.cpu().item()
            
        # --- Comparison Table ---
        st.markdown("### 📊 Composition Analysis")
        
        # Build Table Data
        comp_data = []
        
        # Get all involved types (True + Preds)
        top_t = np.argsort(p_teacher)[-3:]
        top_s = np.argsort(p_student)[-3:]
        pred_types = train_classes[np.unique(np.concatenate([top_t, top_s]))]
        
        all_types = sorted(list(set(list(true_composition.keys()) + list(pred_types))))
        
        for t in all_types:
            # True Ratio
            true_val = true_composition.get(t, 0.0)
            
            # Predicted Values
            # Be careful: Test Labels might not match Train Labels exactly if datasets differ?
            # We align assumption: Train Classes cover Test Classes (or share names)
            if t in train_classes:
                idx = np.where(train_classes == t)[0][0]
                t_val = p_teacher[idx]
                s_val = p_student[idx]
            else:
                t_val = 0.0; s_val = 0.0 # Input type not in trained model
                
            if true_val > 0.01 or t_val > 0.05 or s_val > 0.05:
                comp_data.append({
                    "Cell Type": t,
                    "True Ratio": f"{true_val:.1%}",
                    "Teacher (P)": f"{t_val:.1%}",
                    "Student (B)": f"{s_val:.1%}",
                    # Numeric for sorting if needed
                    "_true": true_val
                })
        
        # Add Uncertainty row
        comp_data.append({
            "Cell Type": "Uncertainty (Background)",
            "True Ratio": "-",
            "Teacher (P)": "-", 
            "Student (B)": f"{u_student:.1%}",
            "_true": -1
        })
        
        df_comp = pd.DataFrame(comp_data) #.sort_values("_true", ascending=False)
        st.table(df_comp[["Cell Type", "True Ratio", "Teacher (P)", "Student (B)"]])

        # --- Plots ---
        st.markdown("### 📉 Distribution")

        # Plot Logic (Same as before but filtered)
        # Use Matplotlib
        import matplotlib.pyplot as plt
        
        # Use the union of types we found relevant above
        relevant_classes_idx = []
        for t in all_types:
            if t in train_classes:
               relevant_classes_idx.append(np.where(train_classes == t)[0][0])
               
        # If empty (unexpected), fallback
        if not relevant_classes_idx:
            relevant_classes_idx = top_t
            
        relevant_classes_idx = np.array(relevant_classes_idx)
        relevant_labels = train_classes[relevant_classes_idx]
        
        p_col1, p_col2 = st.columns(2)
        
        # Teacher
        with p_col1:
            fig_t, ax_t = plt.subplots(figsize=(5, 3))
            y_vals = p_teacher[relevant_classes_idx]
            x_pos = np.arange(len(relevant_labels))
            
            bars = ax_t.bar(x_pos, y_vals, color='#4c78a8', alpha=0.9)
            ax_t.set_title("Teacher")
            ax_t.set_xticks(x_pos)
            ax_t.set_xticklabels(relevant_labels, rotation=45, ha='right')
            ax_t.set_ylim(0, 1.1)
             # Add labels
            for bar in bars:
                height = bar.get_height()
                ax_t.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig_t)
            
        # Student
        with p_col2:
            fig_s, ax_s = plt.subplots(figsize=(5, 3))
            
            s_labels_plot = list(relevant_labels) + ['Uncertainty']
            s_vals_plot = list(p_student[relevant_classes_idx]) + [u_student]
            
            x_pos_s = np.arange(len(s_labels_plot))
            colors = ['#55a868'] * len(relevant_labels) + ['gray']
            
            bars_s = ax_s.bar(x_pos_s, s_vals_plot, color=colors, alpha=0.9)
            ax_s.set_title("Student")
            ax_s.set_xticks(x_pos_s)
            ax_s.set_xticklabels(s_labels_plot, rotation=45, ha='right')
            ax_s.set_ylim(0, 1.1)
             # Add labels
            for bar in bars_s:
                height = bar.get_height()
                ax_s.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig_s)

else:
    st.info("👈 Select Train/Test Data and Click 'Load & Train'")

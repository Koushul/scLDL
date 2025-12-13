
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scLDL.label_enhancer import DiffLEVI, ImprovedLEVI, ConcentrationLE, HybridLEVI
from torchvision import models
import torch.nn as nn

# Baseline Wrapper
class ResNetBaseline(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = models.resnet18(pretrained=False) # Weights loaded manually
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model.to(device)
        self.device = device
        
        # Transforms for inference (manual)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    def eval(self):
        self.model.eval()
        
    def predict(self, X, L=None):
        # X is numpy (N, 1, 28, 28) usually
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X).float().to(self.device)
            else:
                X_tensor = X
            
            # 1. Expand to 3 channels (grayscale -> RGB)
            if X_tensor.shape[1] == 1:
                X_tensor = X_tensor.repeat(1, 3, 1, 1)
                
            # 2. Normalize
            X_tensor = (X_tensor - self.norm_mean) / self.norm_std
            
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

st.set_page_config(page_title="LDL Mixing Experiment", layout="wide")

@st.cache_resource
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return test_dataset

@st.cache_resource
def load_models(device):
    input_shape = (1, 28, 28)
    n_features = 784
    n_outputs = 10
    
    models = {}
    
    # DiffLEVI
    try:
        difflevi = DiffLEVI(
            n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
            input_shape=input_shape, epochs=10, batch_size=64, timesteps=1000, lr=2e-4, device=device
        )
        difflevi.load_state_dict(torch.load('models/DiffLEVI_mnist.pth', map_location=device))
        difflevi.eval()
        models['DiffLEVI'] = difflevi
    except FileNotFoundError:
        pass 

    # ImprovedLEVI
    try:
        improved = ImprovedLEVI(
            n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, gamma=1.0, alpha=1.0, device=device
        )
        improved.load_state_dict(torch.load('models/ImprovedLEVI_mnist.pth', map_location=device))
        improved.eval()
        models['ImprovedLEVI'] = improved
    except FileNotFoundError:
        pass
        
    # ConcentrationLE
    try:
        concentration = ConcentrationLE(
            n_features=n_features, n_outputs=n_outputs, encoder_type='resnet',
            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, device=device
        )
        concentration.load_state_dict(torch.load('models/ConcentrationLE_mnist.pth', map_location=device))
        concentration.eval()
        models['ConcentrationLE'] = concentration
    except FileNotFoundError:
        pass

    # HybridLEVI
    try:
        hybrid = HybridLEVI(
            n_features=n_features, n_outputs=n_outputs, encoder_type='cnn',
            input_shape=input_shape, epochs=10, batch_size=64, lr=1e-3, alpha=1.0, gamma=1.0, device=device
        )
        hybrid.load_state_dict(torch.load('models/HybridLEVI_mnist.pth', map_location=device))
        hybrid.eval()
        models['HybridLEVI'] = hybrid
    except FileNotFoundError:
        pass
        
    # ResNet Baseline
    try:
        resnet = ResNetBaseline(device)
        resnet.load_state_dict(torch.load('models/ResNet18_baseline.pth', map_location=device))
        resnet.eval()
        models['Classes (ResNet)'] = resnet
    except FileNotFoundError:
        pass
        
    return models

def predict_proba_concentration(model, X_tensor):
    model.eval()
    with torch.no_grad():
        # Clean robust check
        model_name = type(model).__name__
        
        if model_name == 'ConcentrationLE':
             evidence, alpha = model(X_tensor)
        elif model_name == 'HybridLEVI':
             _, _, _, _, evidence = model(X_tensor)
             alpha = evidence + 1
        else:
             # Fallback if isinstance fails and name check fails (unlikely)
             # Try assuming it returns evidence if it has 'evidence' in signature? No.
             # Just raise clear error
             raise ValueError(f"Unknown model type for concentration logic: {type(model)}")
             
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        return probs.cpu().numpy()

def main():
    st.title("Digit Mixing Experiment")
    st.write("Linearly interpolate between two digits and observe model predictions.")
    
    device = torch.device("cpu") # Use CPU for inference
    
    try:
        test_dataset = load_data()
        models = load_models(device)
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return
    
    # --- Sidebar ---
    st.sidebar.header("Mixing Controls")
    
    # Session State for Indices
    if 'idx_a' not in st.session_state:
        st.session_state.idx_a = 0
    if 'idx_b' not in st.session_state:
        st.session_state.idx_b = 1
    if 'idx_c' not in st.session_state:
        st.session_state.idx_c = 2
        
    if st.sidebar.button("Pick 3 Random Different Digits"):
        # Sample until we get 3 distinct labels
        found = False
        while not found:
            idxs = np.random.choice(len(test_dataset), 3, replace=False)
            l1 = test_dataset[idxs[0]][1]
            l2 = test_dataset[idxs[1]][1]
            l3 = test_dataset[idxs[2]][1]
            
            if l1 != l2 and l1 != l3 and l2 != l3:
                st.session_state.idx_a = idxs[0]
                st.session_state.idx_b = idxs[1]
                st.session_state.idx_c = idxs[2]
                found = True
                
    # Use indices from session state
    idx_a = st.session_state.idx_a
    idx_b = st.session_state.idx_b
    idx_c = st.session_state.idx_c
    
    st.sidebar.text(f"Selected: {test_dataset[idx_a][1]}, {test_dataset[idx_b][1]}, {test_dataset[idx_c][1]}")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Weights")
    
    w_a = st.sidebar.slider("Weight A", 0.0, 1.0, 0.33, 0.01)
    w_b = st.sidebar.slider("Weight B", 0.0, 1.0, 0.33, 0.01)
    w_c = st.sidebar.slider("Weight C", 0.0, 1.0, 0.33, 0.01)
    
    # Normalize
    total_w = w_a + w_b + w_c
    if total_w == 0:
        w_a, w_b, w_c = 0.33, 0.33, 0.33
        total_w = 1.0
        
    norm_a = w_a / total_w
    norm_b = w_b / total_w
    norm_c = w_c / total_w
    
    st.sidebar.info(f"Normalized: A={norm_a:.2f}, B={norm_b:.2f}, C={norm_c:.2f}")
    
    # Data
    img_a, label_a = test_dataset[idx_a]
    img_b, label_b = test_dataset[idx_b]
    img_c, label_c = test_dataset[idx_c]
    
    # Mixing
    img_mix = norm_a * img_a + norm_b * img_b + norm_c * img_c
    
    # --- Main Display ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(img_a.squeeze().numpy(), caption=f"A: {label_a}", width=120)
    with col2:
        st.image(img_b.squeeze().numpy(), caption=f"B: {label_b}", width=120)
    with col3:
        st.image(img_c.squeeze().numpy(), caption=f"C: {label_c}", width=120)
    with col4:
        st.image(img_mix.squeeze().numpy(), caption="Mixed", width=120)
         
    st.markdown("---")
    st.subheader("Predicted Distributions on Mixed Image")
    
    X_in = img_mix.unsqueeze(0).to(device)
    L_dummy = torch.ones(1, 10).to(device) / 10
    
    preds_df = pd.DataFrame({'Class': range(10)})
    
    # Inference
    for name, model in models.items():
        try:
            model_name = type(model).__name__
            if model_name in ['ConcentrationLE', 'HybridLEVI']:
                probs = predict_proba_concentration(model, X_in)[0]
            elif model_name == 'DiffLEVI':
                probs = model.predict(X_in.cpu().numpy(), n_samples=5)[0]
            else:
                 probs = model.predict(X_in.cpu().numpy(), L_dummy.cpu().numpy())[0]
            
            preds_df[name] = probs
        except Exception as e:
            st.error(f"Prediction error for {name}: {e}")
            preds_df[name] = np.zeros(10)

    # Plotting
    plot_cols = st.columns(len(models))
    for idx, (name, model) in enumerate(models.items()):
        with plot_cols[idx]:
            st.markdown(f"**{name}**")
            probs = preds_df[name]
            
            # Simple Matplotlib Bar Plot for full control over colors/labels
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.bar(range(10), probs, color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_xticks(range(10))
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Highlight top class
            pred_lbl = np.argmax(probs)
            ax.set_title(f"Top: {pred_lbl} ({probs[pred_lbl]:.2f})", fontsize=10)
            
            st.pyplot(fig)
            
            # Show Top 3 Predictions
            st.caption("Top 3 Predictions:")
            top3_indices = np.argsort(probs)[::-1][:3]
            for i in top3_indices:
                st.write(f"**{i}**: {probs[i]:.4f}")

if __name__ == "__main__":
    main()

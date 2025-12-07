import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import anndata

class scDataset(Dataset):
    """
    PyTorch Dataset wrapper for AnnData objects.
    """
    def __init__(self, adata, label_key: str, layer: str = None, spatial_key: str = 'spatial'):
        """
        Args:
            adata: AnnData object.
            label_key: Key in adata.obs containing the logical labels (e.g., 'leiden').
            layer: Key in adata.layers to use as expression data. If None, uses adata.X.
            spatial_key: Key in adata.obsm containing spatial coordinates.
        """
        self.adata = adata
        self.label_key = label_key
        self.spatial_key = spatial_key
        
        # Extract Expression Data
        if layer:
            self.X = adata.layers[layer]
        else:
            self.X = adata.X
            
        # Handle sparse matrices
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray()
            
        # Extract Logical Labels
        if label_key not in adata.obs:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs")
            
        self.labels = adata.obs[label_key].values
        
        # Encode labels to integers if they are strings/categories
        if self.labels.dtype == 'O' or isinstance(self.labels.dtype, pd.CategoricalDtype):
            self.label_map = {k: v for v, k in enumerate(np.unique(self.labels))}
            self.labels = np.array([self.label_map[l] for l in self.labels])
            self.num_classes = len(self.label_map)
        else:
            self.label_map = None
            self.num_classes = len(np.unique(self.labels))
            
        # Extract Spatial Coordinates
        if spatial_key in adata.obsm:
            self.spatial = adata.obsm[spatial_key]
        else:
            self.spatial = np.zeros((self.adata.n_obs, 2)) # Dummy spatial if not present

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        l = torch.tensor(self.labels[index]).long()
        s = torch.from_numpy(self.spatial[index]).float()
        
        # One-hot encode logical label for the model
        l_onehot = torch.zeros(self.num_classes)
        l_onehot[l] = 1.0
        
        return x, l_onehot, s, index

    def __len__(self):
        return self.adata.n_obs

    def get_num_classes(self):
        return self.num_classes

    def get_input_dim(self):
        return self.X.shape[1]



def split_adata(adata, train_size=0.8, label_key='cell_type', random_state=42):
    """
    Splits an AnnData object into train and test sets, preserving the distribution of labels.

    Args:
        adata: AnnData object.
        train_size: Proportion of the dataset to include in the train split.
        label_key: Key in adata.obs containing the labels for stratification.
        random_state: Random seed for reproducibility.

    Returns:
        train_adata, test_adata
    """
    from sklearn.model_selection import train_test_split
    
    if label_key not in adata.obs:
        raise ValueError(f"Label key '{label_key}' not found in adata.obs")

    labels = adata.obs[label_key].values
    indices = np.arange(adata.n_obs)
    
    train_idx, test_idx = train_test_split(
        indices, 
        train_size=train_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    
    return train_adata, test_adata

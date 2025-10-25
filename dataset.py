# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FlowDataset(Dataset):
    def __init__(self, X_path, y_path, indices=None):
        X = np.load(X_path)
        y = np.load(y_path)
        if indices is not None:
            X = X[indices]; y = y[indices]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loaders(X_path, y_path, batch_size=128, val_split=0.2, seed=42):
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.load(X_path); y = np.load(y_path)
    idx_train, idx_val = train_test_split(range(len(y)), test_size=val_split, stratify=y, random_state=seed)
    train_ds = FlowDataset(X_path, y_path, indices=idx_train)
    val_ds = FlowDataset(X_path, y_path, indices=idx_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

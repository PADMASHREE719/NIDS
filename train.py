# src/train.py
import os, joblib, time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from dataset import make_loaders
from capsnet import CapsBiLSTM
from utils import read_config, set_seed, ensure_dir

cfg = read_config()
set_seed(cfg['training']['seed'])
ARTIFACTS = cfg['artifacts_dir']
MODELS = cfg['models_dir']; ensure_dir(MODELS)

def train():
    Xp = os.path.join(ARTIFACTS,"X_all.npy")
    yp = os.path.join(ARTIFACTS,"y_all.npy")
    
    # Check if preprocessed data exists
    if not os.path.exists(Xp) or not os.path.exists(yp):
        print("ERROR: Preprocessed data not found!")
        print(f"Expected files:")
        print(f"  - {Xp}")
        print(f"  - {yp}")
        print("\nPlease run preprocessing first:")
        print("  python src/preprocess.py")
        exit(1)
    
    # Check if label encoder exists
    le_path = os.path.join(ARTIFACTS,"label_encoder.pkl")
    if not os.path.exists(le_path):
        print(f"ERROR: Label encoder not found at {le_path}")
        print("Please run preprocessing first: python src/preprocess.py")
        exit(1)
    
    print("Loading data...")
    
    # Option: Sample data for faster training (use 20% of data)
    USE_SAMPLE = True  # Set to False to use all data
    SAMPLE_FRACTION = 0.2  # Use 20% of data
    
    if USE_SAMPLE:
        # Load and sample data
        X_full = np.load(Xp)
        y_full = np.load(yp)
        
        from sklearn.model_selection import train_test_split
        # Stratified sampling to keep class distribution
        _, X_sample, _, y_sample = train_test_split(
            X_full, y_full, 
            test_size=SAMPLE_FRACTION,
            stratify=y_full,
            random_state=cfg['training']['seed']
        )
        
        # Save sampled data temporarily
        np.save(os.path.join(ARTIFACTS, "X_sample.npy"), X_sample)
        np.save(os.path.join(ARTIFACTS, "y_sample.npy"), y_sample)
        
        Xp = os.path.join(ARTIFACTS, "X_sample.npy")
        yp = os.path.join(ARTIFACTS, "y_sample.npy")
        
        print(f"✓ Using {SAMPLE_FRACTION*100:.0f}% sample: {len(y_sample):,} samples")
    
    train_loader, val_loader = make_loaders(Xp, yp, 
                                            batch_size=cfg['training']['batch_size'], 
                                            val_split=0.2, 
                                            seed=cfg['training']['seed'])
    
    n_classes = len(joblib.load(le_path).classes_)
    sample = np.load(Xp)
    feature_dim = sample.shape[1]
    
    print(f"Dataset info:")
    print(f"  Features: {feature_dim}")
    print(f"  Classes: {n_classes}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = CapsBiLSTM(feature_dim, n_classes,
                       primary_caps=cfg['model']['primary_caps'],
                       primary_dim=cfg['model']['primary_dim'],
                       lstm_hidden=cfg['model']['lstm_hidden'],
                       lstm_layers=cfg['model']['lstm_layers'],
                       dropout=cfg['model']['dropout']).to(device)
    
    # Weighted loss for imbalanced classes
    y_all = np.load(yp)
    classes = np.unique(y_all)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=classes, y=y_all)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'], 
                    weight_decay=cfg['training']['weight_decay'])
    
    best_val_f1 = 0.0
    print(f"\nStarting training for {cfg['training']['epochs']} epochs...")
    print("="*60)
    
    for epoch in range(cfg['training']['epochs']):
        # Training
        model.train()
        losses = []
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Validation
        model.eval()
        ys, preds_all = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                logits = model(Xv)
                pr = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.extend(pr.tolist())
                ys.extend(yv.numpy().tolist())
        
        val_f1 = f1_score(ys, preds_all, average='macro')
        avg_loss = np.mean(losses)
        
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}", end="")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            path = os.path.join(MODELS, "best.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, path)
            print(" ✅ [SAVED]")
        else:
            print()
    
    print("="*60)
    print(f"Training complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {MODELS}/best.pt")

if __name__ == "__main__":
    train()
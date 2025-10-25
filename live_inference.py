# src/live_inference.py
import os, time, joblib, numpy as np, pandas as pd
import torch
from capsnet import CapsBiLSTM
from utils import read_config

cfg = read_config()
ART = cfg['artifacts_dir']
MOD = cfg['models_dir']
LIVE_CSV = os.path.join(cfg['data']['flows_dir'], "live_flows.csv")

# Load artifacts
scaler = joblib.load(os.path.join(ART, "scaler.pkl"))
le = joblib.load(os.path.join(ART, "label_encoder.pkl"))
master = joblib.load(os.path.join(ART, "master_features.pkl"))

ckpt = torch.load(os.path.join(MOD, "best.pt"), map_location='cpu')
cfg_model = ckpt['cfg']
n_classes = len(le.classes_)
feature_dim = len(scaler.mean_)

# Instantiate model
model = CapsBiLSTM(feature_dim, n_classes,
                   primary_caps=cfg_model['model']['primary_caps'],
                   primary_dim=cfg_model['model']['primary_dim'],
                   lstm_hidden=cfg_model['model']['lstm_hidden'])
model.load_state_dict(ckpt['model_state'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

print("="*70)
print("LIVE NETWORK INTRUSION DETECTION")
print("="*70)
print(f"Watching: {LIVE_CSV}")
print(f"Model classes: {list(le.classes_)}")
print(f"Expected features: {feature_dim}")
print("="*70)
print("Starting live inference...\n")

last_pos = 0
processed_flows = 0

while True:
    if not os.path.exists(LIVE_CSV):
        time.sleep(1)
        continue
    
    try:
        df = pd.read_csv(LIVE_CSV)
        if df.shape[0] == 0:
            time.sleep(1)
            continue
        
        # Take last 256 flows (batch)
        new = df.tail(256).copy()
        
        # IMPORTANT: Keep only master features that exist in the CSV
        cols = [c for c in master if c in new.columns]
        
        if len(cols) == 0:
            print("⚠ Warning: No matching features found in CSV")
            time.sleep(5)
            continue
        
        # Extract features and ensure correct order
        X = new[cols].select_dtypes(include=[float, int]).fillna(0).values
        
        # Check feature count
        if X.shape[1] != feature_dim:
            print(f"⚠ Warning: Feature mismatch - CSV has {X.shape[1]} features, expected {feature_dim}")
            print(f"   Available: {len(cols)}/{len(master)} master features")
            
            # Pad with zeros if fewer features
            if X.shape[1] < feature_dim:
                padding = np.zeros((X.shape[0], feature_dim - X.shape[1]))
                X = np.hstack([X, padding])
                print(f"   → Padded to {feature_dim} features")
            # Truncate if more features
            elif X.shape[1] > feature_dim:
                X = X[:, :feature_dim]
                print(f"   → Truncated to {feature_dim} features")
        
        # Scale and predict
        Xs = scaler.transform(X.astype(float))
        Xt = torch.tensor(Xs, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        
        # Show predictions
        attack_counts = {}
        for i, p in enumerate(preds):
            label = le.inverse_transform([p])[0]
            prob = probs[i, p]
            
            # Count attacks
            attack_counts[label] = attack_counts.get(label, 0) + 1
            
            # Show first few predictions
            if processed_flows < 20 or label != 'normal':
                ts = pd.Timestamp.now()
                status = "🚨 ALERT" if label != 'normal' else "✓ Normal"
                print(f"[{ts.strftime('%H:%M:%S')}] {status} | Flow #{processed_flows+i+1} → {label:15s} (conf: {prob:.3f})")
            
            processed_flows += 1
        
        # Show summary every batch
        print(f"\n--- Batch Summary (processed {processed_flows} flows) ---")
        for label, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
            pct = count / len(preds) * 100
            print(f"  {label:15s}: {count:4d} flows ({pct:5.1f}%)")
        print()
        
    except Exception as e:
        print(f"Error in live loop: {e}")
        import traceback
        traceback.print_exc()
    
    time.sleep(2)
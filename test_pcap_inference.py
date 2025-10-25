# src/test_pcap_inference.py
"""
Test inference on PCAP-extracted flows
"""
import os, joblib, numpy as np, pandas as pd
import torch
from capsnet import CapsBiLSTM
from utils import read_config
from collections import Counter

cfg = read_config()
ART = cfg['artifacts_dir']
MOD = cfg['models_dir']

# Load model and preprocessors
print("Loading model...")
scaler = joblib.load(os.path.join(ART, "scaler.pkl"))
le = joblib.load(os.path.join(ART, "label_encoder.pkl"))
master = joblib.load(os.path.join(ART, "master_features.pkl"))

ckpt = torch.load(os.path.join(MOD, "best.pt"), map_location='cpu')
cfg_model = ckpt['cfg']
model = CapsBiLSTM(len(scaler.mean_), len(le.classes_),
                   primary_caps=cfg_model['model']['primary_caps'],
                   primary_dim=cfg_model['model']['primary_dim'],
                   lstm_hidden=cfg_model['model']['lstm_hidden'])
model.load_state_dict(ckpt['model_state'])
model.eval()

def predict_from_csv(csv_path, show_details=True):
    """Predict attacks from extracted flow CSV"""
    print(f"\n{'='*70}")
    print(f"Testing: {csv_path}")
    print(f"{'='*70}")
    
    # Load flows
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} flows")
    
    # Extract features
    cols = [c for c in master if c in df.columns]
    X = df[cols].select_dtypes(include=[float, int]).fillna(0).values
    print(f"✓ Extracted {X.shape[1]} features")
    
    # Scale and predict
    Xs = scaler.transform(X.astype(float))
    Xt = torch.tensor(Xs, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    
    # Get labels
    labels = [le.inverse_transform([p])[0] for p in preds]
    
    # Show results
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY:")
    print(f"{'='*70}")
    
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100
        status = "🚨" if label != 'normal' else "✓"
        print(f"  {status} {label:20s}: {count:6d} flows ({pct:5.2f}%)")
    
    # Show sample alerts
    if show_details:
        print(f"\n{'='*70}")
        print("SAMPLE ALERTS (first 10 attacks):")
        print(f"{'='*70}")
        
        alert_count = 0
        for i, (label, prob) in enumerate(zip(labels, probs[range(len(preds)), preds])):
            if label != 'normal':
                print(f"  Flow {i+1:5d}: {label:20s} (confidence: {prob:.3f})")
                alert_count += 1
                if alert_count >= 10:
                    break
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default: look for flows in data/flows/
        csv_path = "data/flows/monday_flows.csv"
    
    if os.path.exists(csv_path):
        predict_from_csv(csv_path)
    else:
        print(f"✗ File not found: {csv_path}")
        print("\nUsage: python src/test_pcap_inference.py <path/to/flows.csv>")
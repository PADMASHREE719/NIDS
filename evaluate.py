# src/evaluate.py
import os, joblib, numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from dataset import FlowDataset
from capsnet import CapsBiLSTM
from utils import read_config
cfg = read_config()
ART = cfg['artifacts_dir']; MOD = cfg['models_dir']

def evaluate():
    Xp = os.path.join(ART, "X_all.npy")
    yp = os.path.join(ART, "y_all.npy")
    X = np.load(Xp); y = np.load(yp)
    # load best model
    ckpt = torch.load(os.path.join(MOD, "best.pt"))
    cfg_model = ckpt['cfg']
    n_classes = len(joblib.load(os.path.join(ART,"label_encoder.pkl")).classes_)
    feature_dim = X.shape[1]
    model = CapsBiLSTM(feature_dim, n_classes,
                       primary_caps=cfg_model['model']['primary_caps'],
                       primary_dim=cfg_model['model']['primary_dim'],
                       lstm_hidden=cfg_model['model']['lstm_hidden']).eval()
    model.load_state_dict(ckpt['model_state'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = FlowDataset(Xp, yp)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    ys, preds = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            pr = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pr.tolist()); ys.extend(yb.numpy().tolist())
    print(classification_report(ys, preds, digits=4))
    cm = confusion_matrix(ys, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion matrix")
    plt.savefig("results/confusion_matrix.png")
    print("Saved confusion matrix to results/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()

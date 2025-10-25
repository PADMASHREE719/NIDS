# src/server.py
from fastapi import FastAPI
import joblib, torch, numpy as np
from pydantic import BaseModel
from capsnet import CapsBiLSTM
from utils import read_config
cfg = read_config()
ART = cfg['artifacts_dir']; MOD = cfg['models_dir']
scaler = joblib.load(ART+"/scaler.pkl")
le = joblib.load(ART+"/label_encoder.pkl")
ckpt = torch.load(MOD+"/best.pt", map_location='cpu')
model = CapsBiLSTM(len(scaler.mean_), len(le.classes_))
model.load_state_dict(ckpt['model_state'])
model.eval()
app = FastAPI()

class FlowIn(BaseModel):
    features: list

@app.post("/predict")
def predict(flow: FlowIn):
    x = np.array(flow.features, dtype=float).reshape(1,-1)
    x = scaler.transform(x)
    import torch
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    p = probs.argmax()
    return {"label": le.inverse_transform([p])[0], "prob": float(probs[p])}

# run: uvicorn src.server:app --reload --port 8000

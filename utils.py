# src/utils.py
import os, json, math, random
import numpy as np
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Only import torch when this function is called
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not needed for preprocessing

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_config(path="src/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)

def pad_to_square_and_reshape(arr):
    # arr: 1D numpy vector (n_features,)
    n = arr.shape[0]
    side = math.ceil(math.sqrt(n))
    target = side * side
    pad = target - n
    a = np.pad(arr, (0,pad), mode='constant', constant_values=0)
    return a.reshape(1, side, side).astype(np.float32)
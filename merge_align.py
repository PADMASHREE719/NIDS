# src/merge_align.py
import glob, pandas as pd, os, joblib
from utils import ensure_dir, read_config
cfg = read_config()

def get_headers(path, nrows=5):
    return pd.read_csv(path, nrows=nrows).columns.tolist()

def compute_master_feature_list(raw_dirs, mode='intersection'):
    # raw_dirs: list of folders containing CSVs
    sets = []
    for d in raw_dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            sets.append(set(get_headers(f)))
    if mode == 'intersection':
        master = set.intersection(*sets) if sets else set()

    else:
        master = set.union(*sets)
    master = sorted(list(master))
    return master

if __name__ == "__main__":
    raw = [os.path.join(cfg['data']['raw_dir'], "CICIDS2017"),
           os.path.join(cfg['data']['raw_dir'], "UNSW_NB15")]
    master = compute_master_feature_list(raw, mode=cfg['preprocess']['feature_mode'])
    print(f"Master features ({len(master)}): {master[:20]} ...")
    ensure_dir(cfg['artifacts_dir'])
    joblib.dump(master, os.path.join(cfg['artifacts_dir'], "master_features.pkl"))
    print("Saved master feature list.")

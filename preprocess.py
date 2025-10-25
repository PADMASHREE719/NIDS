import os, glob, joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import read_config, ensure_dir
cfg = read_config()

RAW_DIR = os.path.join(cfg['data']['raw_dir'], "CICIDS2017")
ARTIFACTS = cfg['artifacts_dir']
ensure_dir(ARTIFACTS)

# CICIDS2017 label mapping
LABEL_MAP = {
    'BENIGN': 'normal',
    'DoS Hulk': 'dos',
    'DoS GoldenEye': 'dos',
    'DoS slowloris': 'dos',
    'DoS Slowhttptest': 'dos',
    'DDoS': 'ddos',
    'PortScan': 'portscan',
    'Bot': 'botnet',
    'FTP-Patator': 'bruteforce',
    'SSH-Patator': 'bruteforce',
    'Web Attack - Brute Force': 'webattack',
    'Web Attack - XSS': 'webattack',
    'Web Attack - Sql Injection': 'webattack',
    'Infiltration': 'infiltration',
    'Heartbleed': 'heartbleed',
}

def map_label(x):
    label = str(x).strip()
    return LABEL_MAP.get(label, label.lower())

def process_file(file, master_features, chunk_size=200000):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(file)}")
    chunks_processed = 0
    rows_total = 0
    
    try:
        # First, peek at the file to find label column
        peek = pd.read_csv(file, nrows=5)
        print(f"  Available columns (first 10): {list(peek.columns[:10])}")
        
        # Find label column
        label_col = None
        for col in peek.columns:
            col_lower = col.strip().lower()
            if 'label' in col_lower:
                label_col = col
                print(f"  Found label column: '{label_col}'")
                break
        
        if label_col is None:
            print(f"  WARNING: No label column found! Skipping file.")
            return
        
        # Process in chunks
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False, encoding='utf-8', encoding_errors='ignore'):
            chunks_processed += 1
            chunk = chunk.reset_index(drop=True)
            
            # Keep only master features that exist
            cols = [c for c in master_features if c in chunk.columns]
            df = chunk[cols + [label_col]].copy()
            
            # Drop identifier columns
            id_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Src IP', 'Dst IP',
                       ' Source IP', ' Destination IP', ' Src IP', ' Dst IP',
                       'Timestamp', ' Timestamp', 'Flow Bytes/s', 'Flow Packets/s',
                       ' Flow Bytes/s', ' Flow Packets/s']
            for idcol in id_cols:
                if idcol in df.columns:
                    df.drop(columns=[idcol], inplace=True, errors='ignore')
            
            # Drop any non-numeric columns (except label)
            for col in df.columns:
                if col != label_col and df[col].dtype == 'object':
                    df.drop(columns=[col], inplace=True, errors='ignore')
            
            # Replace infinite values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill missing values ONLY in numeric columns
            num_cols = [c for c in df.columns if c != label_col and df[c].dtype in [np.number, 'float64', 'float32', 'int64', 'int32']]
            if num_cols:
                df[num_cols] = df[num_cols].fillna(0)
            
            # Map labels
            label_values = df[label_col].values
            labels_mapped = []
            for x in label_values:
                if isinstance(x, (list, np.ndarray)):
                    x = x[0] if len(x) > 0 else 'unknown'
                labels_mapped.append(map_label(x))
            df['label_norm'] = labels_mapped
            
            rows_total += len(df)
            yield df
        
        print(f"  Processed {chunks_processed} chunks, {rows_total} rows")
    
    except Exception as e:
        print(f"  ERROR: {e}")

def build_dataset():
    master_path = os.path.join(ARTIFACTS, "master_features.pkl")
    if not os.path.exists(master_path):
        print("ERROR: master_features.pkl not found!")
        print("Please run: python src/merge_align.py first")
        exit(1)
    
    master = joblib.load(master_path)
    print(f"\n{'='*60}")
    print(f"Loaded {len(master)} master features")
    print(f"{'='*60}")
    
    all_X = []
    all_y = []
    
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    
    if not files:
        print(f"\nERROR: No CSV files found in {RAW_DIR}")
        exit(1)
    
    print(f"\nFound {len(files)} CSV files")
    print(f"{'='*60}")
    print("Starting data processing...")
    print(f"{'='*60}")
    
    files_processed = 0
    for f in files:
        try:
            chunks_from_file = 0
            for df in process_file(f, master, chunk_size=cfg['preprocess']['chunk_size']):
                # Extract labels
                y = df['label_norm'].values
                
                # Extract features - EXCLUDE label columns and ensure numeric only
                feature_cols = [c for c in master if c in df.columns]
                # Remove label-related columns
                feature_cols = [c for c in feature_cols if 'label' not in c.lower()]
                # Ensure numeric only
                feature_cols = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.number, 'float64', 'float32', 'int64', 'int32', 'int16', 'int8']]
                
                if len(feature_cols) == 0:
                    print(f"  WARNING: No numeric features found, skipping chunk")
                    continue
                
                X = df[feature_cols].values
                
                all_X.append(X)
                all_y.append(y)
                chunks_from_file += 1
            
            if chunks_from_file > 0:
                files_processed += 1
                print(f"  Added {chunks_from_file} chunks from this file")
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(f)}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Files successfully processed: {files_processed}/{len(files)}")
    
    if not all_X:
        print("\nERROR: No data was processed!")
        exit(1)
    
    # Combine all data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Total samples: {X.shape[0]:,}")
    print(f"  Total features: {X.shape[1]}")
    print(f"\n  Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label:20s}: {count:8,} ({count/len(y)*100:5.2f}%)")
    
    # Remove 'nan' labels
    valid_mask = y != 'nan'
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"\n  After removing 'nan' labels: {X.shape[0]:,} samples")
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n  Encoded classes: {list(le.classes_)}")
    
    # Scale features
    print(f"\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    
    # Save artifacts
    print(f"\nSaving artifacts...")
    joblib.dump(scaler, os.path.join(ARTIFACTS, "scaler.pkl"))
    joblib.dump(le, os.path.join(ARTIFACTS, "label_encoder.pkl"))
    np.save(os.path.join(ARTIFACTS, "X_all.npy"), X_scaled)
    np.save(os.path.join(ARTIFACTS, "y_all.npy"), y_enc)
    
    print(f"{'='*60}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Saved files:")
    print(f"  X_all.npy ({X_scaled.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  y_all.npy ({y_enc.nbytes / 1024:.1f} KB)")
    print(f"  scaler.pkl")
    print(f"  label_encoder.pkl")
    print(f"{'='*60}")

if __name__ == "__main__":
    build_dataset()
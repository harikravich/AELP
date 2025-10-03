#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
from scipy import sparse
import joblib


HF_BASE = "https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main"
ZIP_URL = f"{HF_BASE}/Avazu_x1.zip?download=true"
FILES = {
    'train': f"{HF_BASE}/train.csv",
    'valid': f"{HF_BASE}/valid.csv",
    'test':  f"{HF_BASE}/test.csv",
}


def ensure_data(dirpath: Path):
    import subprocess
    dirpath.mkdir(parents=True, exist_ok=True)
    token = os.getenv('HUGGINGFACE_TOKEN')
    # If CSVs missing, download zip once and extract
    needed = [split for split in FILES if not (dirpath / f"{split}.csv").exists()]
    if needed:
        zip_path = dirpath / 'Avazu_x1.zip'
        if not zip_path.exists():
            cmd = ["curl", "-sSfL", "-o", str(zip_path), ZIP_URL]
            if token:
                cmd = ["curl", "-sSfL", "-H", "Authorization: Bearer %s" % token, "-o", str(zip_path), ZIP_URL]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0:
                raise RuntimeError(f"Failed to download zip from {ZIP_URL}: {r.stderr.decode('utf-8')}")
            print(f"Downloaded {zip_path}")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dirpath)
        print(f"Extracted to {dirpath}")


CAT_COLS = [f"C{i}" for i in range(1, 23)]


def iter_batches(csv_path: Path, batch_size: int = 1_000_000, n_rows: int | None = None) -> Iterator[pd.DataFrame]:
    # Detect schema on first chunk
    first = True
    used_cols = None
    read_rows = 0
    for df in pd.read_csv(csv_path, chunksize=batch_size):
        if first:
            cols = set(df.columns)
            if 'label' in cols and 'feat_1' in cols:
                used_cols = ['label'] + [f'feat_{i}' for i in range(1,23)]
            elif 'click' in cols and 'C1' in cols:
                used_cols = ['click'] + [f'C{i}' for i in range(1,23)]
            else:
                cols_preview = df.columns.tolist()[:10]
                raise ValueError(f"Unknown column schema in {csv_path}: {cols_preview} ...")
            first = False
        df = df[used_cols]
        read_rows += len(df)
        yield df
        if n_rows and read_rows >= n_rows:
            break


def to_hashed_features(df: pd.DataFrame, hasher: FeatureHasher) -> Tuple[sparse.csr_matrix, np.ndarray]:
    if 'label' in df.columns:
        y = df['label'].astype(int).values
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        feats = [[f"{c}={str(v)}" for c, v in row.items()] for row in df[feat_cols].astype(str).to_dict(orient='records')]
    else:
        y = df['click'].astype(int).values
        feat_cols = [c for c in df.columns if c.startswith('C')]
        feats = [[f"{c}={str(v)}" for c, v in row.items()] for row in df[feat_cols].astype(str).to_dict(orient='records')]
    X = hasher.transform(feats)
    return X, y


def train(data_dir: Path, model_out: Path, hash_bits: int = 2**20, max_rows: int | None = None) -> dict:
    hasher = FeatureHasher(n_features=hash_bits, input_type='string')
    clf = SGDClassifier(loss='log_loss', alpha=1e-5, learning_rate='optimal')
    classes = np.array([0,1])
    seen = 0
    for batch in iter_batches(data_dir / 'train.csv', n_rows=max_rows):
        X, y = to_hashed_features(batch, hasher)
        if seen == 0:
            clf.partial_fit(X, y, classes=classes)
        else:
            clf.partial_fit(X, y)
        seen += len(y)
        print(f"trained on {seen} rows")
    joblib.dump({'clf': clf, 'hash_bits': hash_bits}, model_out)
    return {'trained_rows': seen}


def evaluate(data_dir: Path, model_path: Path, split: str = 'test', max_rows: int | None = None) -> dict:
    obj = joblib.load(model_path)
    clf: SGDClassifier = obj['clf']
    hasher = FeatureHasher(n_features=obj['hash_bits'], input_type='string')
    ys, ps = [], []
    seen = 0
    for batch in iter_batches(data_dir / f"{split}.csv", n_rows=max_rows):
        X, y = to_hashed_features(batch, hasher)
        p = clf.predict_proba(X)[:,1]
        ys.append(y); ps.append(p)
        seen += len(y)
        print(f"evaluated {seen} rows")
    y = np.concatenate(ys); p = np.concatenate(ps)
    metrics = {
        'rows': int(len(y)),
        'auc': float(roc_auc_score(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0,1])),
        'brier': float(np.mean((p - y)**2)),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='artifacts/avazu')
    ap.add_argument('--model-out', default='artifacts/models/avazu_hash_sgd.joblib')
    ap.add_argument('--train-rows', type=int, default=5_000_000, help='rows for training (None for all)')
    ap.add_argument('--test-rows', type=int, default=2_000_000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ensure_data(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    train_stats = train(data_dir, Path(args.model_out), max_rows=args.train_rows)
    test_metrics = evaluate(data_dir, Path(args.model_out), 'test', max_rows=args.test_rows)
    print({'train': train_stats, 'test': test_metrics})


if __name__ == '__main__':
    main()

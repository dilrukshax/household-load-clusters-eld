
import os
import math
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# -------------------- Reproducibility --------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Data Loading & Preprocessing --------------------
def load_eld_daily(
    file_path: str,
    resample_rule: str = "1h",
    normalize: str = "zscore",  # "zscore" | "minmax" | "sum1" | None
    drop_zeros: bool = True,
    sample_size: int = 2000,     # set None to use all
    random_state: int = 42,
    client_subset: int | None = None,  # set an int to randomly choose this many clients
):
    """
    Loads the UCI ElectricityLoadDiagrams2011-2014 dataset and returns daily profiles.
    Each sample is one (client, day) 24-dim vector (hourly mean).

    Returns:
        X: np.ndarray of shape (N, 24)
        meta: dict with columns, dates, and index mapping
    """
    rng = np.random.default_rng(random_state)
    # Read semicolon CSV; index is timestamp
    df = pd.read_csv(
        file_path,
        sep=";",
        index_col=0,
        low_memory=False,
        parse_dates=True,
        dayfirst=False,
    )
    # Fix potential extra empty column at the end
    if "" in df.columns:
        df = df.drop(columns=[""])
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optionally pick a subset of clients to keep things light
    if client_subset is not None:
        keep_cols = rng.choice(df.columns.to_numpy(), size=min(client_subset, len(df.columns)), replace=False)
        df = df.loc[:, keep_cols]

    # Resample to hourly means
    df_hour = df.resample(resample_rule).mean()

    # Group by date and create daily 24-d profiles per client
    # Create MultiIndex: (date, hour)
    df_hour["date"] = df_hour.index.date
    df_hour["hour"] = df_hour.index.hour

    # Build daily matrices per client and stack
    X_list = []
    idx_map = []  # tuples of (client, date)
    for col in df_hour.columns:
        if col in ("date", "hour"):
            continue
        pivot = df_hour.pivot_table(values=col, index="date", columns="hour", aggfunc="mean")
        # Some days may be missing hours due to DST; reindex to 0..23
        pivot = pivot.reindex(columns=list(range(24)), fill_value=np.nan)
        # Drop days with too many NaNs (shouldn't happen according to dataset doc)
        pivot = pivot.dropna(thresh=20, axis=0)  # keep days with >=20 hours
        # Optionally drop all-zero days
        if drop_zeros:
            mask_nonzero = ~(pivot.abs().sum(axis=1) == 0)
            pivot = pivot[mask_nonzero]
        # Normalize per-day
        if normalize == "zscore":
            mu = pivot.mean(axis=1)
            sigma = pivot.std(axis=1).replace(0, 1.0)
            pivot = (pivot.sub(mu, axis=0)).div(sigma, axis=0)
        elif normalize == "minmax":
            mn = pivot.min(axis=1)
            mx = pivot.max(axis=1)
            denom = (mx - mn).replace(0, 1.0)
            pivot = (pivot.sub(mn, axis=0)).div(denom, axis=0)
        elif normalize == "sum1":
            s = pivot.sum(axis=1).replace(0, 1.0)
            pivot = pivot.div(s, axis=0)
        elif normalize is None:
            pass
        else:
            raise ValueError("Unknown normalize mode")

        X_list.append(pivot.values.astype(np.float32))
        idx_map += [(col, pd.to_datetime(str(d)).date()) for d in pivot.index]

    if len(X_list) == 0:
        raise RuntimeError("No data loaded. Check your file path or preprocessing filters.")
    X = np.vstack(X_list)
    
    # Handle any remaining NaN values by forward/backward fill then zero fill
    # This can happen for missing hours in some days
    mask_nan = np.isnan(X)
    if mask_nan.any():
        # Forward fill along each row (day)
        for i in range(X.shape[0]):
            row = X[i]
            mask = np.isnan(row)
            if mask.any():
                # Forward fill
                idx = np.where(~mask, np.arange(len(row)), 0)
                np.maximum.accumulate(idx, out=idx)
                row[mask] = row[idx[mask]]
                # If still NaN (all values were NaN), fill with 0
                row[np.isnan(row)] = 0.0
                X[i] = row

    # Optional sampling for speed
    if (sample_size is not None) and (X.shape[0] > sample_size):
        sel = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[sel]
        idx_map = [idx_map[i] for i in sel]

    meta = {
        "index_map": idx_map,       # list of tuples (client, date)
        "n_samples": len(idx_map),
        "n_features": X.shape[1],
    }
    return X, meta

# -------------------- Evaluation and Visualization --------------------
def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    metrics = {}
    # Some metrics need at least 2 clusters with more than 1 sample
    if len(set(labels)) > 1 and (np.bincount(labels).min() > 1):
        metrics["silhouette"] = float(silhouette_score(X, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    else:
        metrics["silhouette"] = float("nan")
        metrics["davies_bouldin"] = float("nan")
    return metrics

def plot_tsne(X: np.ndarray, labels: np.ndarray | None = None, title: str = "t-SNE", savepath: str | None = None):
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30, random_state=42)
    X2d = tsne.fit_transform(X)
    plt.figure(figsize=(7,5))
    if labels is None:
        plt.scatter(X2d[:,0], X2d[:,1], s=14)
    else:
        for lab in sorted(set(labels)):
            m = (labels == lab)
            plt.scatter(X2d[m,0], X2d[m,1], s=14, label=f"Cluster {lab}")
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=140)
    plt.show()

def save_outputs(out_dir: str, name: str, labels: np.ndarray, embeddings: np.ndarray | None, metrics: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), labels)
    if embeddings is not None:
        np.save(os.path.join(out_dir, f"{name}_embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

#!/usr/bin/env python3
"""Train Table III wildfire ignition models for local or dstack runs.

The script mirrors the sequence construction in replication.ipynb:
75 timesteps x 15 GRIDMET variables, with a binary label indicating whether
any wildfire occurs inside the 75-day block.

Classical baselines flatten sequences to 1125 features. Neural baselines use
the 3D sequence tensor. The Chronos section freezes a pretrained Chronos
encoder, mean-pools one univariate embedding per feature, concatenates those
embeddings, and then trains the same baseline family on those embeddings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import time
import importlib.util
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


FEATURES = [
    "pr",
    "rmax",
    "rmin",
    "sph",
    "srad",
    "tmmn",
    "tmmx",
    "vs",
    "bi",
    "fm100",
    "fm1000",
    "erc",
    "etr",
    "pet",
    "vpd",
]

TABLE_ORDER = [
    "CNN-LSTM (ours)",
    "XGBoost*",
    "Gradient Boosting*",
    "Random Forest*",
    "K-Nearest Neighbors*",
    "Simple-MLP*",
    "Decision Tree*",
    "Two-Layer-LSTM",
    "LightTS-Inspired",
    "Logistic Regression*",
    "Naive Bayes*",
    "Chronos Embeddings + XGBoost*",
    "Chronos Embeddings + Gradient Boosting*",
    "Chronos Embeddings + Random Forest*",
    "Chronos Embeddings + K-Nearest Neighbors*",
    "Chronos Embeddings + Simple-MLP*",
    "Chronos Embeddings + Decision Tree*",
    "Chronos Embeddings + Logistic Regression*",
    "Chronos Embeddings + Naive Bayes*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="Wildfire_Dataset.csv")
    parser.add_argument("--output-dir", default="artifacts/table_iii_cloud")
    parser.add_argument("--seq-len", type=int, default=75)
    parser.add_argument("--fill-value", type=float, default=32767.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--time-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-04-30")
    parser.add_argument("--reference-index", type=int, default=60)
    parser.add_argument(
        "--sequence-mode",
        choices=["event_order", "sorted_chunks"],
        default="event_order",
        help="event_order preserves the Kaggle CSV's 75-row event samples; sorted_chunks matches the earlier notebook prototype.",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mlp-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--xgb-trees", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-mini")
    parser.add_argument("--chronos-batch-size", type=int, default=256)
    parser.add_argument("--chronos-pca-components", type=int, default=256)
    parser.add_argument(
        "--chronos-features",
        default="all",
        help="Comma-separated feature names for Chronos, or 'all'.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_data(data_path: Path, download: bool) -> Path:
    if data_path.exists():
        return data_path
    if not download:
        raise FileNotFoundError(f"{data_path} does not exist and --no-download was set")

    print(f"{data_path} not found; downloading firecastrl/us-wildfire-dataset with kagglehub")
    import kagglehub

    cache_path = Path(kagglehub.dataset_download("firecastrl/us-wildfire-dataset"))
    candidates = sorted(cache_path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in Kaggle download: {cache_path}")
    source = next((p for p in candidates if p.name == data_path.name), candidates[0])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, data_path)
    print(f"Copied {source} to {data_path}")
    return data_path


def build_sequences(
    data_path: Path,
    seq_len: int,
    fill_value: float,
    sequence_mode: str,
    reference_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading {data_path}")
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    print(f"Rows: {len(df):,}")

    if sequence_mode == "sorted_chunks":
        print("Sequence mode: sorted_chunks")
        df = df.loc[~(df == fill_value).any(axis=1)].copy()
        df = df.sort_values(["latitude", "longitude", "datetime"]).reset_index(drop=True)
    else:
        print("Sequence mode: event_order")
        # The Kaggle CSV is already expanded as consecutive 75-row event samples.
        # Preserve that order so rows from one event never mix with another.
        df = df.reset_index(drop=True)

    n_seq = len(df) // seq_len
    usable_rows = n_seq * seq_len
    if usable_rows != len(df):
        print(f"Dropping {len(df) - usable_rows:,} trailing rows that do not form a full sequence")
        df = df.iloc[:usable_rows].copy()

    feature_values = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    wildfire_values = (df["Wildfire"].to_numpy() == "Yes")
    date_values = df["datetime"].to_numpy()

    seqs = feature_values.reshape(n_seq, seq_len, len(FEATURES))
    labels = wildfire_values.reshape(n_seq, seq_len).any(axis=1).astype(np.int64)
    anchor_dates = date_values.reshape(n_seq, seq_len)[:, reference_index]

    event_has_fill = (seqs == fill_value).any(axis=(1, 2))
    if event_has_fill.any():
        print(f"Dropping {int(event_has_fill.sum()):,} full events containing fill values")
        keep = ~event_has_fill
        seqs, labels, anchor_dates = seqs[keep], labels[keep], anchor_dates[keep]

    print(f"Sequences: {seqs.shape}")
    print(f"Labels: positives {labels.sum():,}, negatives {len(labels) - labels.sum():,}")
    print(f"Positive label share: {labels.mean():.4f}")
    print(f"Flattened baseline dimension: {seqs.shape[1] * seqs.shape[2]}")
    return seqs, labels, anchor_dates


def make_split(
    seqs: np.ndarray,
    labels: np.ndarray,
    anchor_dates: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.max_sequences and args.max_sequences < len(seqs):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(seqs))[: args.max_sequences]
        seqs, labels, anchor_dates = seqs[idx], labels[idx], anchor_dates[idx]
        print(f"Subsampled to {len(seqs):,} sequences")

    if args.time_split:
        test_start = pd.Timestamp(args.test_start)
        test_end = pd.Timestamp(args.test_end)
        test_mask = (anchor_dates >= test_start) & (anchor_dates <= test_end)
        train_mask = ~test_mask
        X_pretrain, X_test = seqs[train_mask], seqs[test_mask]
        y_pretrain, y_test = labels[train_mask], labels[test_mask]
        X_train, X_val, y_train, y_val = train_test_split(
            X_pretrain,
            y_pretrain,
            test_size=args.val_size,
            stratify=y_pretrain,
            random_state=args.seed,
        )
        split_name = f"time split with test={test_start.date()}..{test_end.date()}, validation from pre-test data"
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            seqs,
            labels,
            test_size=args.test_size,
            stratify=labels,
            random_state=args.seed,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=args.val_size,
            stratify=y_trainval,
            random_state=args.seed,
        )
        split_name = f"random stratified split with test_size={args.test_size}"

    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)
    X_test = np.ascontiguousarray(X_test)
    y_train = np.ascontiguousarray(y_train)
    y_val = np.ascontiguousarray(y_val)
    y_test = np.ascontiguousarray(y_test)

    print(f"Split: {split_name}")
    print(f"Train: {X_train.shape}, positives {y_train.sum():,}/{len(y_train):,}")
    print(f"Val: {X_val.shape}, positives {y_val.sum():,}/{len(y_val):,}")
    print(f"Test: {X_test.shape}, positives {y_test.sum():,}/{len(y_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(X.reshape(X.shape[0], -1))


def metric_row(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float | str]:
    row = {
        "Model": model_name,
        "Accuracy [%]": 100 * accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if threshold is not None:
        row["Threshold"] = threshold
    if y_score is not None and len(np.unique(y_true)) == 2:
        row["ROC-AUC"] = roc_auc_score(y_true, y_score)
        row["PR-AUC"] = average_precision_score(y_true, y_score)
    return row


def positive_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return estimator.predict(X).astype(float)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.nanargmax(f1s))])


def write_results(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["Model"] = pd.Categorical(table["Model"], categories=TABLE_ORDER, ordered=True)
    table = table.sort_values("Model").reset_index(drop=True)
    table["Model"] = table["Model"].astype(str)
    rounded = table.round({
        "Accuracy [%]": 1,
        "Precision": 2,
        "Recall": 2,
        "F1": 2,
        "Threshold": 3,
        "ROC-AUC": 3,
        "PR-AUC": 3,
    })
    rounded.to_csv(output_dir / "table_iii_model_comparison.csv", index=False)
    with (output_dir / "table_iii_model_comparison.json").open("w") as f:
        json.dump(rounded.to_dict(orient="records"), f, indent=2)
    return rounded


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoLayerLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.30)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class LightTSInspired(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class CNNLSTM(nn.Module):
    def __init__(self, input_dim: int, conv_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_dim),
            nn.Dropout(0.20),
        )
        self.lstm = nn.LSTM(conv_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        out = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(out)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_torch_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    focal_gamma: float,
) -> nn.Module:
    model = model.to(device)
    criterion = FocalLoss(weight=class_weights(y_train, device), gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1:02d}/{epochs}, loss={total_loss / len(train_loader):.4f}")
    return model


def predict_torch_scores(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for Xb, _ in loader:
            logits = model(Xb.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(scores)


def safe_artifact_name(name: str) -> str:
    return (
        name.replace("*", "")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def classical_baseline_models(
    args: argparse.Namespace,
    scaled,
    unscaled,
    prefix: str = "",
) -> list[tuple[str, Pipeline]]:
    return [
        (
            f"{prefix}XGBoost*",
            unscaled(
                XGBClassifier(
                    n_estimators=args.xgb_trees,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                )
            ),
        ),
        (f"{prefix}Gradient Boosting*", unscaled(GradientBoostingClassifier(random_state=args.seed))),
        (
            f"{prefix}Random Forest*",
            unscaled(
                RandomForestClassifier(
                    n_estimators=args.rf_trees,
                    class_weight="balanced",
                    n_jobs=args.n_jobs,
                    random_state=args.seed,
                )
            ),
        ),
        (f"{prefix}K-Nearest Neighbors*", scaled(KNeighborsClassifier(n_neighbors=5))),
        (f"{prefix}Decision Tree*", unscaled(DecisionTreeClassifier(class_weight="balanced", random_state=args.seed))),
        (
            f"{prefix}Logistic Regression*",
            scaled(LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=args.n_jobs, random_state=args.seed)),
        ),
        (f"{prefix}Naive Bayes*", unscaled(GaussianNB())),
    ]


def fit_classical_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    scaled = lambda estimator: Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_sequences, validate=False)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    unscaled = lambda estimator: Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_sequences, validate=False)),
            ("model", estimator),
        ]
    )

    models = classical_baseline_models(args, scaled, unscaled)

    rows = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in models:
        print(f"Fitting {name}")
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        val_score = positive_scores(estimator, X_val)
        threshold = best_f1_threshold(y_val, val_score)
        test_score = positive_scores(estimator, X_test)
        y_pred = (test_score >= threshold).astype(np.int64)
        row = metric_row(name, y_test, y_pred, y_score=test_score, threshold=threshold)
        row["Seconds"] = time.perf_counter() - start
        rows.append(row)
        safe_name = safe_artifact_name(name)
        joblib.dump(estimator, model_dir / f"{safe_name}.joblib")
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
            output_dir / f"predictions_{safe_name}.csv", index=False
        )
        print(f"  done in {row['Seconds']:.1f}s: F1={row['F1']:.3f}")
    return rows


def fit_neural_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    rows = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    flat_scaler = StandardScaler()
    X_train_flat = flat_scaler.fit_transform(flatten_sequences(X_train))
    X_val_flat = flat_scaler.transform(flatten_sequences(X_val))
    X_test_flat = flat_scaler.transform(flatten_sequences(X_test))

    seq_scaler = StandardScaler()
    X_train_seq = seq_scaler.fit_transform(X_train.reshape(-1, len(FEATURES))).reshape(X_train.shape)
    X_val_seq = seq_scaler.transform(X_val.reshape(-1, len(FEATURES))).reshape(X_val.shape)
    X_test_seq = seq_scaler.transform(X_test.reshape(-1, len(FEATURES))).reshape(X_test.shape)

    neural_specs = [
        ("Simple-MLP*", MLPClassifierTorch(X_train_flat.shape[1]), X_train_flat, X_val_flat, X_test_flat, args.mlp_epochs),
        ("Two-Layer-LSTM", TwoLayerLSTM(len(FEATURES)), X_train_seq, X_val_seq, X_test_seq, args.epochs),
        ("LightTS-Inspired", LightTSInspired(len(FEATURES)), X_train_seq, X_val_seq, X_test_seq, args.epochs),
        ("CNN-LSTM (ours)", CNNBiLSTM(len(FEATURES)), X_train_seq, X_val_seq, X_test_seq, args.epochs),
    ]

    for name, model, Xtr, Xv, Xte, epochs in neural_specs:
        print(f"Fitting {name} on {device}")
        start = time.perf_counter()
        loader = make_loader(Xtr, y_train, batch_size=args.batch_size, shuffle=True)
        model = train_torch_classifier(
            model, loader, y_train, device, epochs=epochs, lr=args.lr, focal_gamma=args.focal_gamma
        )
        val_score = predict_torch_scores(model, Xv, device, batch_size=args.batch_size * 2)
        threshold = best_f1_threshold(y_val, val_score)
        test_score = predict_torch_scores(model, Xte, device, batch_size=args.batch_size * 2)
        y_pred = (test_score >= threshold).astype(np.int64)
        row = metric_row(name, y_test, y_pred, y_score=test_score, threshold=threshold)
        row["Seconds"] = time.perf_counter() - start
        rows.append(row)
        safe_name = safe_artifact_name(name)
        torch.save(model.state_dict(), model_dir / f"{safe_name}.pt")
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
            output_dir / f"predictions_{safe_name}.csv", index=False
        )
        print(f"  done in {row['Seconds']:.1f}s: F1={row['F1']:.3f}")
    return rows


def parse_chronos_features(feature_arg: str) -> tuple[list[str], list[int]]:
    if feature_arg.strip().lower() == "all":
        return FEATURES, list(range(len(FEATURES)))
    names = [part.strip() for part in feature_arg.split(",") if part.strip()]
    missing = [name for name in names if name not in FEATURES]
    if missing:
        raise ValueError(f"Unknown Chronos feature names: {missing}")
    return names, [FEATURES.index(name) for name in names]


@torch.inference_mode()
def chronos_embeddings(
    seq_array: np.ndarray,
    pipeline,
    batch_size: int,
    feature_names: Iterable[str],
) -> np.ndarray:
    names = list(feature_names)
    blocks = []
    for f, name in enumerate(names):
        slices = []
        for start in range(0, len(seq_array), batch_size):
            ctx = torch.tensor(seq_array[start : start + batch_size, :, f], dtype=torch.float32)
            out = pipeline.embed(ctx)
            emb = out[0] if isinstance(out, tuple) else out
            slices.append(emb.mean(dim=1).cpu().numpy())
        block = np.concatenate(slices, axis=0)
        blocks.append(block)
        print(f"  Chronos encoded {f + 1:02d}/{len(names)} {name}: {block.shape}")
    return np.concatenate(blocks, axis=1)


def extract_chronos_embedding_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading Chronos model: {args.chronos_model}")
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" and importlib.util.find_spec("hf_transfer") is None:
        print("HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is unavailable; disabling fast transfer.")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from chronos import ChronosPipeline

    feature_names, feature_idx = parse_chronos_features(args.chronos_features)
    X_train_ch = X_train[:, :, feature_idx]
    X_val_ch = X_val[:, :, feature_idx]
    X_test_ch = X_test[:, :, feature_idx]

    model_tag = args.chronos_model.replace("/", "_")
    split_tag = f"n{len(X_train)}_{len(X_val)}_{len(X_test)}_{args.chronos_features.replace(',', '-')}"
    emb_dir = output_dir / "chronos_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    train_path = emb_dir / f"{model_tag}_{split_tag}_train.npy"
    val_path = emb_dir / f"{model_tag}_{split_tag}_val.npy"
    test_path = emb_dir / f"{model_tag}_{split_tag}_test.npy"

    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Loading cached Chronos embeddings")
        X_train_emb = np.load(train_path)
        X_val_emb = np.load(val_path)
        X_test_emb = np.load(test_path)
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipeline = ChronosPipeline.from_pretrained(args.chronos_model, device_map=str(device), torch_dtype=dtype)
        print("Encoding train set with frozen Chronos")
        X_train_emb = chronos_embeddings(X_train_ch, pipeline, args.chronos_batch_size, feature_names)
        print("Encoding validation set with frozen Chronos")
        X_val_emb = chronos_embeddings(X_val_ch, pipeline, args.chronos_batch_size, feature_names)
        print("Encoding test set with frozen Chronos")
        X_test_emb = chronos_embeddings(X_test_ch, pipeline, args.chronos_batch_size, feature_names)
        np.save(train_path, X_train_emb)
        np.save(val_path, X_val_emb)
        np.save(test_path, X_test_emb)

    pca_n = min(args.chronos_pca_components, X_train_emb.shape[1], X_train_emb.shape[0])
    pca = PCA(n_components=pca_n, random_state=args.seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(pca.fit_transform(X_train_emb))
    X_val_s = scaler.transform(pca.transform(X_val_emb))
    X_test_s = scaler.transform(pca.transform(X_test_emb))
    print(f"Chronos embedding shape: {X_train_emb.shape}; PCA components: {pca_n}")

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "scaler": scaler}, model_dir / "chronos_embedding_preprocessor.joblib")
    np.save(emb_dir / f"{model_tag}_{split_tag}_train_pca_scaled.npy", X_train_s)
    np.save(emb_dir / f"{model_tag}_{split_tag}_val_pca_scaled.npy", X_val_s)
    np.save(emb_dir / f"{model_tag}_{split_tag}_test_pca_scaled.npy", X_test_s)
    return X_train_s, X_val_s, X_test_s


def fit_embedding_baseline_models(
    X_train_emb: np.ndarray,
    X_val_emb: np.ndarray,
    X_test_emb: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    identity = lambda estimator: Pipeline([("model", estimator)])
    scaled_identity = lambda estimator: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )

    rows = []
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    models = classical_baseline_models(
        args,
        scaled=scaled_identity,
        unscaled=identity,
        prefix="Chronos Embeddings + ",
    )

    for name, estimator in models:
        print(f"Fitting {name}")
        start = time.perf_counter()
        estimator.fit(X_train_emb, y_train)
        val_score = positive_scores(estimator, X_val_emb)
        threshold = best_f1_threshold(y_val, val_score)
        test_score = positive_scores(estimator, X_test_emb)
        y_pred = (test_score >= threshold).astype(np.int64)
        row = metric_row(name, y_test, y_pred, y_score=test_score, threshold=threshold)
        row["Seconds"] = time.perf_counter() - start
        rows.append(row)
        safe_name = safe_artifact_name(name)
        joblib.dump(estimator, model_dir / f"{safe_name}.joblib")
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
            output_dir / f"predictions_{safe_name}.csv", index=False
        )
        print(f"  done in {row['Seconds']:.1f}s: F1={row['F1']:.3f}")

    start = time.perf_counter()
    name = "Chronos Embeddings + Simple-MLP*"
    print(f"Fitting {name} on {device}")
    loader = make_loader(X_train_emb, y_train, batch_size=args.batch_size, shuffle=True)
    model = train_torch_classifier(
        MLPClassifierTorch(X_train_emb.shape[1]),
        loader,
        y_train,
        device,
        epochs=args.mlp_epochs,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
    )
    val_score = predict_torch_scores(model, X_val_emb, device, batch_size=args.batch_size * 2)
    threshold = best_f1_threshold(y_val, val_score)
    test_score = predict_torch_scores(model, X_test_emb, device, batch_size=args.batch_size * 2)
    y_pred = (test_score >= threshold).astype(np.int64)
    row = metric_row(name, y_test, y_pred, y_score=test_score, threshold=threshold)
    row["Seconds"] = time.perf_counter() - start
    rows.append(row)
    safe_name = safe_artifact_name(name)
    torch.save(model.state_dict(), model_dir / f"{safe_name}.pt")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        output_dir / f"predictions_{safe_name}.csv", index=False
    )
    print(f"  done in {row['Seconds']:.1f}s: F1={row['F1']:.3f}")
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Device: {device}")
    data_path = ensure_data(Path(args.data_path), download=not args.no_download)
    seqs, labels, anchor_dates = build_sequences(
        data_path,
        args.seq_len,
        args.fill_value,
        sequence_mode=args.sequence_mode,
        reference_index=args.reference_index,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(seqs, labels, anchor_dates, args)
    del seqs, labels, anchor_dates
    gc.collect()

    rows = []
    if not args.skip_classical:
        rows.extend(fit_classical_models(X_train, X_val, X_test, y_train, y_val, y_test, args, output_dir))
        print(write_results(rows, output_dir))

    if not args.skip_neural:
        rows.extend(fit_neural_models(X_train, X_val, X_test, y_train, y_val, y_test, args, output_dir, device))
        print(write_results(rows, output_dir))

    if not args.skip_chronos:
        X_train_emb, X_val_emb, X_test_emb = extract_chronos_embedding_features(
            X_train, X_val, X_test, args, output_dir, device
        )
        rows.extend(
            fit_embedding_baseline_models(
                X_train_emb, X_val_emb, X_test_emb, y_train, y_val, y_test, args, output_dir, device
            )
        )

    table = write_results(rows, output_dir)
    print("\nTABLE III: Wildfire ignition model comparison")
    print(table.to_string(index=False))
    print(
        "\n* Baselines not inherently time-series-aware; feature vectors flattened to "
        "1125 dimensions (15 variables x 75 timesteps). Chronos Embeddings rows use "
        "frozen Chronos embeddings plus PCA/scaling before the listed baseline model."
    )
    print(f"\nWrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()

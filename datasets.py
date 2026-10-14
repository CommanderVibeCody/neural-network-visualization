"""Public dataset loading (via scikit-learn into pandas) with automatic task detection."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# name -> (loader, task hint)
DATASETS = {
    "Iris": skd.load_iris,
    "Wine": skd.load_wine,
    "Breast Cancer": skd.load_breast_cancer,
    "Digits": skd.load_digits,
    "Diabetes": skd.load_diabetes,
    "California Housing": skd.fetch_california_housing,  # downloaded once, then cached
}


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    features: list
    target_col: str
    task: str                       # "binary", "multiclass" or "regression"
    class_names: list = field(default_factory=list)
    # prepared splits (filled by prepare())
    X_tr: np.ndarray = None
    y_tr: np.ndarray = None
    X_te: np.ndarray = None
    y_te: np.ndarray = None
    X_te_raw: np.ndarray = None
    y_mean: float = 0.0
    y_std: float = 1.0

    @property
    def n_features(self):
        return len(self.features)

    @property
    def n_classes(self):
        return len(self.class_names) if self.task != "regression" else 0

    def summary_lines(self):
        y = self.frame[self.target_col]
        lines = [f"Samples: {len(self.frame)}    Input features: {self.n_features}",
                 f"Task: {self.task}" + (f"    Classes: {self.n_classes}" if self.n_classes else "")]
        if self.task == "regression":
            lines.append(f"Target '{self.target_col}': min {y.min():.2f}, mean {y.mean():.2f}, max {y.max():.2f}")
        else:
            counts = y.value_counts().sort_index()
            lines.append("Targets: " + ", ".join(f"{self.class_names[int(k)]}={v}" for k, v in counts.items()))
        return lines

    def prepare(self, max_samples=4000, seed=42):
        """Split, standardize X (and y for regression). Large datasets are subsampled for live speed."""
        df = self.frame
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=seed)
        X = df[self.features].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy()
        strat = y if self.task != "regression" else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
        sc = StandardScaler().fit(X_tr)
        self.X_tr, self.X_te = sc.transform(X_tr).astype(np.float32), sc.transform(X_te).astype(np.float32)
        self.X_te_raw = X_te
        if self.task == "regression":
            self.y_mean, self.y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
            self.y_tr = ((y_tr - self.y_mean) / self.y_std).astype(np.float32)
            self.y_te = ((y_te - self.y_mean) / self.y_std).astype(np.float32)
        else:
            self.y_tr, self.y_te = y_tr.astype(np.int64), y_te.astype(np.int64)


def load_dataset(name):
    bunch = DATASETS[name](as_frame=True)
    df = bunch.frame.copy()
    features = list(bunch.feature_names)
    target_col = [c for c in df.columns if c not in features][0]
    y = df[target_col]
    # Task detection: integer-like targets with few unique values -> classification
    is_class = (pd.api.types.is_integer_dtype(y) or np.allclose(y, np.round(y))) and y.nunique() <= 20 \
        and name not in ("Diabetes", "California Housing")
    if is_class:
        k = int(y.nunique())
        names = getattr(bunch, "target_names", None)
        class_names = [str(n) for n in names] if names is not None and len(names) == k else [str(i) for i in range(k)]
        task = "binary" if k == 2 else "multiclass"
    else:
        class_names, task = [], "regression"
    return Dataset(name, df, features, target_col, task, class_names)

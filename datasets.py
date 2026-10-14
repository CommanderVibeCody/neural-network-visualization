"""Public dataset loading (sklearn -> pandas) with hold-out and K-fold splits.

The held-out test set is never used for training or model selection.
Scaler and (for regression) target normalization are fit on train data only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "Iris": skd.load_iris,
    "Wine": skd.load_wine,
    "Breast Cancer": skd.load_breast_cancer,
    "Digits": skd.load_digits,
    "Diabetes": skd.load_diabetes,
    "California Housing": skd.fetch_california_housing,
}


@dataclass
class SplitConfig:
    mode: str = "holdout"          # "holdout" or "kfold"
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15
    k: int = 5
    test_pct_kfold: int = 20
    stratified: bool = True
    seed: int = 42

    def validate(self, task):
        errs = []
        if self.mode == "holdout":
            if self.train_pct + self.val_pct + self.test_pct != 100:
                errs.append("Train + Val + Test must sum to 100%")
            if self.test_pct <= 0:
                errs.append("Test % must be > 0 (never train on the test set)")
            if self.train_pct <= 0 or self.val_pct <= 0:
                errs.append("Train and Val % must be > 0")
        else:
            if self.k < 2:
                errs.append("K must be at least 2")
            if not (5 <= self.test_pct_kfold <= 40):
                errs.append("Held-out test % for K-fold should be between 5 and 40")
            if self.stratified and task == "regression":
                errs.append("Stratified CV is only for classification")
        return errs


@dataclass
class PreparedData:
    """Arrays after split + scaling. Indices for K-fold refer into X_pool / y_pool."""
    X_tr: np.ndarray = None
    y_tr: np.ndarray = None
    X_va: np.ndarray = None
    y_va: np.ndarray = None
    X_te: np.ndarray = None
    y_te: np.ndarray = None
    X_te_raw: np.ndarray = None
    X_va_raw: np.ndarray = None
    # K-fold pool (train+val before folding); folds = list of (tr_idx, va_idx)
    X_pool: np.ndarray = None
    y_pool: np.ndarray = None
    X_pool_raw: np.ndarray = None
    folds: list = field(default_factory=list)
    y_mean: float = 0.0
    y_std: float = 1.0
    n_samples: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    features: list
    target_col: str
    task: str
    class_names: list = field(default_factory=list)
    prepared: PreparedData = None

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
            lines.append("Targets: " + ", ".join(
                f"{self.class_names[int(k)]}={v}" for k, v in counts.items()))
        return lines

    def _raw_xy(self, max_samples, seed):
        df = self.frame
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=seed)
        X = df[self.features].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy()
        return X, y

    def _scale_y(self, y_fit, *ys):
        if self.task != "regression":
            return [yy.astype(np.int64) for yy in ys]
        mean, std = float(y_fit.mean()), float(y_fit.std() + 1e-8)
        out = [((yy - mean) / std).astype(np.float32) for yy in ys]
        return mean, std, out

    def prepare(self, cfg: SplitConfig, max_samples=4000) -> PreparedData:
        """Build train / val / test (and K-fold indices). Test is never used for fitting."""
        X, y = self._raw_xy(max_samples, cfg.seed)
        is_cls = self.task != "regression"
        strat_all = y if is_cls and cfg.stratified else None
        p = PreparedData(n_samples=len(X))

        if cfg.mode == "holdout":
            test_frac = cfg.test_pct / 100.0
            X_tmp, X_te, y_tmp, y_te = train_test_split(
                X, y, test_size=test_frac, random_state=cfg.seed, stratify=strat_all)
            # Remaining is train+val; split val from that.
            val_of_tmp = cfg.val_pct / (cfg.train_pct + cfg.val_pct)
            strat_tmp = y_tmp if is_cls and cfg.stratified else None
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tmp, y_tmp, test_size=val_of_tmp, random_state=cfg.seed, stratify=strat_tmp)
            sc = StandardScaler().fit(X_tr)
            p.X_tr = sc.transform(X_tr).astype(np.float32)
            p.X_va = sc.transform(X_va).astype(np.float32)
            p.X_te = sc.transform(X_te).astype(np.float32)
            p.X_te_raw, p.X_va_raw = X_te, X_va
            if self.task == "regression":
                p.y_mean, p.y_std, (p.y_tr, p.y_va, p.y_te) = self._scale_y(y_tr, y_tr, y_va, y_te)
            else:
                p.y_tr, p.y_va, p.y_te = y_tr.astype(np.int64), y_va.astype(np.int64), y_te.astype(np.int64)
            p.n_train, p.n_val, p.n_test = len(X_tr), len(X_va), len(X_te)
        else:
            test_frac = cfg.test_pct_kfold / 100.0
            X_pool, X_te, y_pool, y_te = train_test_split(
                X, y, test_size=test_frac, random_state=cfg.seed, stratify=strat_all)
            # Scaler fit on the whole training pool (test never seen). Per-fold training
            # still only updates weights on the fold's train indices.
            sc = StandardScaler().fit(X_pool)
            p.X_pool = sc.transform(X_pool).astype(np.float32)
            p.X_te = sc.transform(X_te).astype(np.float32)
            p.X_pool_raw, p.X_te_raw = X_pool, X_te
            if self.task == "regression":
                p.y_mean, p.y_std, (p.y_pool, p.y_te) = self._scale_y(y_pool, y_pool, y_te)
            else:
                p.y_pool, p.y_te = y_pool.astype(np.int64), y_te.astype(np.int64)
            use_strat = is_cls and cfg.stratified
            splitter = (StratifiedKFold if use_strat else KFold)(
                n_splits=cfg.k, shuffle=True, random_state=cfg.seed)
            y_for_split = y_pool if use_strat else p.y_pool
            p.folds = list(splitter.split(p.X_pool, y_for_split))
            p.n_test = len(X_te)
            p.n_train = len(X_pool) - len(X_pool) // cfg.k
            p.n_val = len(X_pool) // cfg.k

        self.prepared = p
        return p


def load_dataset(name):
    bunch = DATASETS[name](as_frame=True)
    df = bunch.frame.copy()
    features = list(bunch.feature_names)
    target_col = [c for c in df.columns if c not in features][0]
    y = df[target_col]
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

"""LightGBM projection model: a mean model plus two quantile models that give a
floor (15th pct) and ceiling (85th pct) for start/sit tie-breaks."""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import feature_columns

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

BASE_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, verbose=-1,
)
QUANTILES = {"floor": 0.15, "ceiling": 0.85}


def _fit(X, y, objective, n_rounds, **extra):
    m = lgb.LGBMRegressor(objective=objective, n_estimators=n_rounds, **BASE_PARAMS, **extra)
    m.fit(X, y, categorical_feature=["pos_code"])
    return m


def pick_rounds(train: pd.DataFrame, max_rounds: int = 3000) -> int:
    """Early-stop on the last season inside the training window."""
    cols = feature_columns()
    last = train["season"].max()
    tr, va = train[train["season"] < last], train[train["season"] == last]
    m = lgb.LGBMRegressor(objective="regression", n_estimators=max_rounds, **BASE_PARAMS)
    m.fit(tr[cols], tr["pts"], eval_X=(va[cols],), eval_y=(va["pts"],),
          categorical_feature=["pos_code"], callbacks=[lgb.early_stopping(100, verbose=False)])
    return max(100, m.best_iteration_)


def train(train_df: pd.DataFrame, n_rounds: int | None = None) -> dict:
    train_df = train_df[train_df["pts"].notna()]
    cols = feature_columns()
    n = n_rounds or pick_rounds(train_df)
    X, y = train_df[cols], train_df["pts"]
    models = {"mean": _fit(X, y, "regression", n), "n_rounds": n}
    for name, a in QUANTILES.items():
        models[name] = _fit(X, y, "quantile", n, alpha=a)
    return models


def train_per_position(train_df: pd.DataFrame) -> dict:
    """Ablation: an independent model (own early-stopped round count) per position."""
    return {"per_position": {pos: train(g) for pos, g in train_df.groupby("position")}}


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    if "per_position" in models:
        parts = [predict(m, df[df["position"] == pos]) for pos, m in models["per_position"].items()]
        return pd.concat(parts).loc[df.index]
    cols = feature_columns()
    out = df.copy()
    out["proj"] = models["mean"].predict(df[cols])
    for name in QUANTILES:
        out[name] = models[name].predict(df[cols])
    # quantile models are fit independently; keep them ordered around the mean
    out["floor"] = np.minimum(out["floor"], out["proj"])
    out["ceiling"] = np.maximum(out["ceiling"], out["proj"])
    return out


def save(models: dict, name: str = "current") -> Path:
    p = MODEL_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(models, f)
    return p


def load(name: str = "current") -> dict:
    with open(MODEL_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

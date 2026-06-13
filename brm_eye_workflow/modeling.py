"""Minimal supervised modeling helpers for the workflow."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, PoissonRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_elasticnet(random_state: int = 13) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000)),
    ])


def make_poisson() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
    ])


def fit_evaluate_by_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    split_col: str,
    model_name: str = "elasticnet",
) -> pd.DataFrame:
    """Fit one classical baseline per target and evaluate held-out partitions."""
    records = []
    train = df[df[split_col] == "train"]
    test_parts = [x for x in ["val", "test"] if x in set(df[split_col])]
    for target in target_cols:
        if model_name == "poisson":
            model = make_poisson()
            y_train = np.clip(train[target].astype(float), 0, None)
        else:
            model = make_elasticnet()
            y_train = train[target].astype(float)
        model.fit(train[list(feature_cols)], y_train)
        for part in test_parts:
            cur = df[df[split_col] == part]
            pred = model.predict(cur[list(feature_cols)])
            records.append({"target": target, "partition": part, "model": model_name, "R2": r2_score(cur[target].astype(float), pred), "n": len(cur)})
    return pd.DataFrame(records)

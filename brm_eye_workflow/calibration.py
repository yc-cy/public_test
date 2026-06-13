"""Residual small-sample reader calibration utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def residual_reader_calibration(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    reader_col: str,
    k_ratio: float = 0.01,
    shrinkage: float = 10.0,
    seed: int = 13,
) -> pd.DataFrame:
    """Apply reader-specific residual-bias correction from a small calibration set.

    For each reader, a subset of rows is sampled as calibration data. The mean
    residual is shrunk toward zero and added to all predictions for that reader.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["is_calibration"] = False
    out["calibrated_pred"] = out[pred_col].astype(float)
    for reader, idx in out.groupby(reader_col).groups.items():
        idx = np.array(list(idx))
        k = max(1, int(round(len(idx) * k_ratio))) if k_ratio > 0 else 0
        if k == 0:
            continue
        calib_idx = rng.choice(idx, size=min(k, len(idx)), replace=False)
        residual = (out.loc[calib_idx, y_col].astype(float) - out.loc[calib_idx, pred_col].astype(float)).mean()
        weight = k / (k + shrinkage)
        correction = weight * residual
        out.loc[idx, "calibrated_pred"] = out.loc[idx, pred_col].astype(float) + correction
        out.loc[calib_idx, "is_calibration"] = True
    return out


def calibration_curve(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    reader_col: str,
    ratios: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1),
    repeats: int = 20,
    shrinkage: float = 10.0,
    seed: int = 13,
) -> pd.DataFrame:
    """Estimate R2 gains across calibration budgets."""
    records = []
    base = r2_score(df[y_col], df[pred_col])
    for r in ratios:
        if r == 0:
            records.append({"k_ratio": r, "repeat": 0, "R2": base, "delta_R2": 0.0})
            continue
        for rep in range(repeats):
            calibrated = residual_reader_calibration(df, y_col, pred_col, reader_col, r, shrinkage, seed + rep)
            score = r2_score(calibrated[y_col], calibrated["calibrated_pred"])
            records.append({"k_ratio": r, "repeat": rep, "R2": score, "delta_R2": score - base})
    return pd.DataFrame(records)

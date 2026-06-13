"""Parse and evaluate LLM-generated eye-movement predictions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .io_utils import read_table, safe_literal_list, write_table

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_llm_numeric_list(output: Any) -> List[float]:
    """Extract numeric values from LLM output.

    Supports lists like ["词-0.23", "词-1.2"] and free text containing numbers.
    """
    items = safe_literal_list(output)
    if items:
        vals = []
        for item in items:
            nums = NUMBER_RE.findall(str(item))
            vals.append(float(nums[-1]) if nums else np.nan)
        return vals
    nums = NUMBER_RE.findall(str(output))
    return [float(x) for x in nums]


def records_from_prediction_table(
    df: pd.DataFrame,
    target_col: str,
    output_col: str = "OUTPUT",
    token_col: str = "IA_LABEL",
    extra_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert row-level sequence outputs into token-level records.

    Returns token records and an error log.
    """
    records: List[dict] = []
    errors: List[dict] = []
    for row_idx, row in df.iterrows():
        tokens = safe_literal_list(row.get(token_col))
        true_vals = safe_literal_list(row.get(target_col))
        pred_vals = parse_llm_numeric_list(row.get(output_col))
        n = min(len(tokens), len(true_vals), len(pred_vals))
        if n == 0 or len(tokens) != len(true_vals) or len(tokens) != len(pred_vals):
            errors.append({
                "row_index": row_idx,
                "n_tokens": len(tokens),
                "n_true": len(true_vals),
                "n_pred": len(pred_vals),
                "reason": "empty_or_length_mismatch",
            })
        base = {col: row.get(col) for col in extra_cols}
        for i in range(n):
            rec = dict(base)
            rec.update({"row_index": row_idx, "position": i, "token": tokens[i], "y_true": true_vals[i], "y_pred": pred_vals[i]})
            records.append(rec)
    return pd.DataFrame(records), pd.DataFrame(errors)


def token_metrics(tokens: pd.DataFrame) -> Dict[str, float]:
    """Compute token-level metrics from columns y_true and y_pred."""
    y_true = pd.to_numeric(tokens["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(tokens["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n_tokens": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Pearson": np.nan, "Spearman": np.nan, "Kendall": np.nan, "Wasserstein": np.nan}
    return {
        "n_tokens": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else np.nan,
        "Kendall": float(kendalltau(y_true, y_pred)[0]) if len(y_true) > 2 else np.nan,
        "Wasserstein": float(wasserstein_distance(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def group_diagnostics(tokens: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Compute group-order recovery and worst-group bias for each grouping axis."""
    records = []
    for group_col in group_cols:
        if group_col not in tokens.columns:
            continue
        means = tokens.groupby(group_col, dropna=False)[["y_true", "y_pred"]].mean(numeric_only=True).dropna()
        if len(means) == 0:
            continue
        tau = kendalltau(means["y_true"], means["y_pred"])[0] if len(means) > 2 else np.nan
        bias = (means["y_pred"] - means["y_true"]).abs().max()
        records.append({"grouping": group_col, "kendall_tau_b": tau, "worst_group_abs_bias": bias, "n_groups": len(means)})
    return pd.DataFrame(records)


def evaluate_prediction_file(
    input_file: str | Path,
    output_prefix: str | Path,
    target_col: str,
    output_col: str = "OUTPUT",
    token_col: str = "IA_LABEL",
    group_cols: Sequence[str] = ("country", "AoA_bin", "LoE_bin", "HSK_bin"),
) -> dict:
    """Parse one prediction file and save token-level metrics and diagnostics."""
    df = read_table(input_file)
    tokens, errors = records_from_prediction_table(df, target_col, output_col, token_col, extra_cols=group_cols)
    metrics = pd.DataFrame([token_metrics(tokens)])
    groups = group_diagnostics(tokens, group_cols)
    prefix = Path(output_prefix)
    write_table(tokens, prefix.with_suffix(".tokens.csv"))
    write_table(errors, prefix.with_suffix(".errors.csv"))
    write_table(metrics, prefix.with_suffix(".token_metrics.csv"))
    write_table(groups, prefix.with_suffix(".group_metrics.csv"))
    return {"token_metrics": metrics, "group_metrics": groups, "errors": errors}

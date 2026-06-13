"""I/O and safe parsing helpers.

These utilities replace project-internal uses of ``eval`` with ``ast.literal_eval``
and expose path arguments instead of hard-coded local files.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd


def safe_literal_list(value: Any) -> List[Any]:
    """Parse a Python-list-like cell safely.

    Parameters
    ----------
    value:
        A list, a string representation of a list, or a missing value.

    Returns
    -------
    list
        Parsed list; returns an empty list if parsing fails.
    """
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV/TSV/XLSX based on file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write CSV/TSV/XLSX based on file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False, **kwargs)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False, **kwargs)
    else:
        df.to_csv(path, index=False, **kwargs)


def format_token_value_pairs(tokens: Sequence[Any], values: Sequence[Any] | None = None) -> str:
    """Return token-value pairs as a parseable Python list of strings."""
    if values is None:
        pairs = [f"{token}-value{i}" for i, token in enumerate(tokens, start=1)]
    else:
        pairs = [f"{token}-{value}" for token, value in zip(tokens, values)]
    return json.dumps(pairs, ensure_ascii=False)


def ensure_numeric_series(values: Iterable[Any]) -> pd.Series:
    """Convert iterable to numeric pandas Series with invalid values as NaN."""
    return pd.to_numeric(pd.Series(list(values)), errors="coerce")

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


NULL_LIKE_VALUES = {"", "null", "none", "nan", "nat", "na"}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(column_name) for column_name in df.columns]
    return df



def normalize_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )



def normalize_text_value(value):
    if value is None:
        return pd.NA
    text = str(value).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return pd.NA
    return text



def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column_name in columns:
        if column_name not in df.columns:
            df[column_name] = pd.NA
    return df



def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()



def build_canonical_id(*parts: object) -> str:
    clean_parts = [str(part).strip() for part in parts if part is not None and str(part).strip()]
    return ":".join(clean_parts)

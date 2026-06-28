from __future__ import annotations

import pandas as pd

MARKDOWN_COLS: tuple[str, ...] = (
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
)


def fill_markdowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing markdown values with zero.

    Missing markdowns represent weeks with no promotional markdown activity.
    """
    df = df.copy()

    missing = set(MARKDOWN_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"Missing markdown columns: {missing}")

    df.loc[:, MARKDOWN_COLS] = df.loc[:, MARKDOWN_COLS].fillna(0)

    return df


def encode_store_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode store type.
    """
    if "Type" not in df.columns:
        raise KeyError("'Type' column not found.")

    df = pd.get_dummies(
        df,
        columns=["Type"],
        drop_first=True,
        dtype=int,
    )

    for col in ("Type_B", "Type_C"):
        if col not in df.columns:
            df[col] = 0

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the preprocessing pipeline.
    """
    df = fill_markdowns(df)
    df = encode_store_type(df)

    return df
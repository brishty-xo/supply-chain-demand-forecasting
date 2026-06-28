"""Feature engineering: temporal, lag, and rolling window features."""

from __future__ import annotations

import pandas as pd

# Feature columns used for model training
FEATURE_COLS: tuple[str, ...] = (
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Size",
    "Year",
    "Month",
    "Week",
    "Quarter",
    "Lag_1",
    "Lag_4",
    "Lag_8",
    "Lag_12",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_Mean_12",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "Type_B",
    "Type_C",
)

# Constants for feature generation
LAG_PERIODS: tuple[int, ...] = (1, 4, 8, 12)
ROLLING_WINDOWS: tuple[int, ...] = (4, 8, 12)

# Columns whose missing values are expected after lag generation
LAG_COLUMNS: tuple[str, ...] = (
    "Lag_1",
    "Lag_4",
    "Lag_8",
    "Lag_12",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_Mean_12",
)


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    """Validate that required columns exist."""
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from Date.
    """
    _validate_columns(df, {"Date"})

    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features grouped by Store and Department.

    Uses only historical observations (shift),
    preventing target leakage.
    """
    _validate_columns(df, {"Store", "Dept", "Weekly_Sales"})

    df = df.copy()

    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in LAG_PERIODS:
        df[f"Lag_{lag}"] = grp.shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling mean features grouped by Store and Department.

    shift(1) ensures only historical observations are used.
    """
    _validate_columns(df, {"Store", "Dept", "Weekly_Sales"})

    df = df.copy()

    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for window in ROLLING_WINDOWS:
        df[f"Rolling_Mean_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.

    Steps
    -----
    1. Temporal features
    2. Lag features
    3. Rolling mean features
    4. Remove rows without sufficient historical context
    """

    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = df.dropna(subset=LAG_COLUMNS).reset_index(drop=True)

    return df
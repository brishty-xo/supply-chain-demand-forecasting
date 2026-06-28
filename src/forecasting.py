"""Demand forecasting: temporal split, XGBoost training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from xgboost import XGBRegressor

from .feature_engineering import FEATURE_COLS

SPLIT_DATE = "2012-01-01"

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def temporal_split(
    df: pd.DataFrame,
    split_date: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into chronological train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
    split_date : str

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if "Date" not in df.columns:
        raise KeyError("'Date' column not found.")

    train = df[df["Date"] < split_date].copy()
    test = df[df["Date"] >= split_date].copy()

    return train, test


def train_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    xgb_params: dict | None = None,
) -> XGBRegressor:
    """
    Train an XGBoost regressor.

    Parameters
    ----------
    train_df : pd.DataFrame
    feature_cols : Sequence[str] | None
    xgb_params : dict | None

    Returns
    -------
    XGBRegressor
    """
    feature_cols = feature_cols or FEATURE_COLS
    params = xgb_params or XGB_PARAMS

    missing = set(feature_cols) - set(train_df.columns)
    if missing:
        raise KeyError(
            f"Missing feature columns: {sorted(missing)}"
        )

    X = train_df[list(feature_cols)]
    y = train_df["Weekly_Sales"]

    model = XGBRegressor(**params)
    model.fit(X, y)

    return model


def predict(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Generate predictions.
    """
    feature_cols = feature_cols or FEATURE_COLS

    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing feature columns: {sorted(missing)}"
        )

    return model.predict(df[list(feature_cols)])


def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics.

    Returns
    -------
    dict
        MAE
        RMSE
        R2
        WMAPE
    """

    denominator = np.abs(y_true).sum()

    wmape = (
        np.nan
        if denominator == 0
        else float(np.abs(y_true - y_pred).sum() / denominator)
    )

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "WMAPE": round(wmape * 100, 4)
        if not np.isnan(wmape)
        else np.nan,
    }


def feature_importance(
    model: XGBRegressor,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importance as a sorted DataFrame.
    """
    feature_cols = feature_cols or FEATURE_COLS

    importance = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }
    )

    return (
        importance.sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )
"""Data loading and merging utilities."""

from pathlib import Path

import pandas as pd


def load_raw_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, features, and stores datasets.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing train.csv, features.csv and stores.csv.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, features, stores)
    """
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    features = pd.read_csv(data_dir / "features.csv")
    stores = pd.read_csv(data_dir / "stores.csv")

    return train, features, stores


def merge_data(
    train: pd.DataFrame,
    features: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Walmart datasets into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame sorted chronologically by Store, Department and Date.
    """

    df = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        stores,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values(["Store", "Dept", "Date"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Merged dataframe is empty.")

    return df


def load_and_merge(data_dir: str | Path) -> pd.DataFrame:
    """
    Convenience wrapper to load and merge all Walmart datasets.
    """
    train, features, stores = load_raw_data(data_dir)
    return merge_data(train, features, stores)
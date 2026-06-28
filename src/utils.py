"""Shared utilities: logging, timing, serialization."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import joblib
import pandas as pd


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager that prints elapsed time for a block.

    Example
    -------
    >>> with timer("Training"):
    ...     model.fit(X_train, y_train)
    """
    start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - start

    print(f"[{label}] {elapsed:.2f}s")


def save_model(
    model: Any,
    path: str | Path,
) -> None:
    """
    Serialize a trained model to disk.

    Parameters
    ----------
    model : Any
        Trained sklearn-compatible estimator.
    path : str | Path
        Destination path.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, path)

    print(f"Model saved → {path}")


def load_model(
    path: str | Path,
) -> Any:
    """
    Load a serialized model.
    """
    return joblib.load(Path(path))


def save_kpis(
    kpis: dict[str, float],
    path: str | Path,
) -> None:
    """
    Save KPI dictionary as JSON.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            kpis,
            f,
            indent=2,
        )

    print(f"KPIs saved → {path}")


def print_kpis(
    kpis: dict[str, float],
) -> None:
    """
    Pretty-print KPI dictionary.
    """
    print("\n" + "=" * 40)
    print("BUSINESS KPIs")
    print("=" * 40)

    for key, value in kpis.items():
        print(f"{key:<30} {value}")

    print("=" * 40 + "\n")


def display_head(
    df: pd.DataFrame,
    n: int = 5,
    label: str = "",
) -> None:
    """
    Display the first few rows of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
    label : str
    """
    if label:
        print(f"\n--- {label} ---")

    print(df.head(n).to_string())

    print(f"\nShape: {df.shape}\n")
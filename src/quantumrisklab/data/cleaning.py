"""Data cleaning for price series.

Even synthetic data goes through the same cleaning path a real feed would, so the
pipeline is representative:
  * drop fully-empty columns,
  * forward-fill then back-fill short gaps,
  * drop any remaining rows with missing values,
  * enforce strictly positive prices (log/simple returns require this).
"""

from __future__ import annotations

import pandas as pd


def clean_prices(prices: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Return a cleaned copy of a (dates x tickers) price frame."""
    if prices.empty:
        return prices.copy()

    cleaned = prices.copy()
    cleaned = cleaned.sort_index()

    # Drop columns that are entirely missing.
    cleaned = cleaned.dropna(axis=1, how="all")

    # Forward/back fill short gaps only.
    cleaned = cleaned.ffill(limit=max_ffill).bfill(limit=max_ffill)

    # Drop any rows that still have missing values.
    cleaned = cleaned.dropna(axis=0, how="any")

    # Enforce positive prices.
    non_positive = (cleaned <= 0).any()
    bad_cols = non_positive[non_positive].index.tolist()
    if bad_cols:
        cleaned = cleaned.drop(columns=bad_cols)

    return cleaned


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute simple and log daily returns from adjusted close prices.

    Returns ``(simple_returns, log_returns)``, each aligned to prices minus the
    first row (which has no prior observation).
    """
    import numpy as np

    if prices.empty:
        return prices.copy(), prices.copy()
    simple = prices.pct_change().dropna(how="any")
    log = np.log(prices / prices.shift(1)).dropna(how="any")
    return simple, log

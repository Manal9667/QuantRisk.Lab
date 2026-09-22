"""Risk segmentation: assign borrowers to PD-based risk bands.

The bands (Low / Moderate / High / Very High) use fixed PD thresholds defined
in :data:`config.RISK_BANDS`. These thresholds are transparent modelling
choices for this prototype - they are NOT an industry or regulatory standard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def assign_risk_band(pd_value: float) -> str:
    """Map a single PD to its risk-band label."""
    if not (0 <= pd_value <= 1):
        raise ValueError("PD must be within [0, 1].")
    for label, low, high in config.RISK_BANDS:
        if low <= pd_value < high:
            return label
    return config.RISK_BANDS[-1][0]  # PD == 1.0 edge case


def assign_risk_bands(pd_values) -> pd.Categorical:
    """Vectorised risk-band assignment returning an ordered categorical."""
    edges = [config.RISK_BANDS[0][1]] + [b[2] for b in config.RISK_BANDS]
    labels = config.RISK_BAND_ORDER
    bands = pd.cut(
        np.clip(np.asarray(pd_values, dtype=float), 0.0, 1.0),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return bands.astype(pd.CategoricalDtype(categories=labels, ordered=True))


def add_risk_bands(loss_table: pd.DataFrame, pd_column: str = "pd") -> pd.DataFrame:
    """Append a ``risk_band`` column to a borrower-level table."""
    out = loss_table.copy()
    out["risk_band"] = assign_risk_bands(out[pd_column].values)
    return out


def segment_summary(loss_table: pd.DataFrame) -> pd.DataFrame:
    """Summarise the portfolio by risk band.

    Columns: number of borrowers, total EAD, average PD, expected loss and the
    share of total portfolio exposure per band.
    """
    if "risk_band" not in loss_table.columns:
        loss_table = add_risk_bands(loss_table)

    grouped = (
        loss_table.groupby("risk_band", observed=False)
        .agg(
            n_borrowers=("pd", "size"),
            total_ead=("ead", "sum"),
            average_pd=("pd", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .reindex(config.RISK_BAND_ORDER)
        .reset_index()
    )
    total_ead = loss_table["ead"].sum()
    grouped["exposure_pct"] = grouped["total_ead"] / total_ead if total_ead else 0.0
    grouped["expected_loss_rate"] = np.where(
        grouped["total_ead"] > 0, grouped["expected_loss"] / grouped["total_ead"], 0.0
    )
    return grouped

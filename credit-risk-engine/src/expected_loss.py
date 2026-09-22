"""Expected Credit Loss (ECL) engine.

    Expected Loss (EL) = PD x LGD x EAD

where
    PD  = model-predicted Probability of Default (from :mod:`pd_model`)
    LGD = Loss Given Default (assumption, default 45% - configurable)
    EAD = Exposure At Default

EAD note
--------
This dataset has no explicit EAD field. We use the loan amount
(``loan_amnt``) as the EAD proxy. This assumes the full drawn principal is
outstanding at the moment of default, which is a transparent, slightly
conservative simplification for an amortising installment loan.

IMPORTANT: Expected Loss is a *forward-looking statistical expectation*, not a
realised/incurred loss. It is what you would expect to lose on average given
the modelled default probability - not what any individual borrower actually
lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class PortfolioMetrics:
    """Portfolio-level Expected Loss aggregates (model outputs + assumptions)."""

    n_borrowers: int
    total_ead: float
    average_pd: float          # exposure-weighted average PD
    total_expected_loss: float
    expected_loss_rate: float  # EL / total EAD
    high_risk_exposure: float
    high_risk_exposure_pct: float
    lgd: float

    def as_dict(self) -> dict:
        return {
            "n_borrowers": self.n_borrowers,
            "total_ead": self.total_ead,
            "average_pd": self.average_pd,
            "total_expected_loss": self.total_expected_loss,
            "expected_loss_rate": self.expected_loss_rate,
            "high_risk_exposure": self.high_risk_exposure,
            "high_risk_exposure_pct": self.high_risk_exposure_pct,
            "lgd": self.lgd,
        }


def expected_loss(pd_value, lgd: float, ead) -> np.ndarray | float:
    """Core formula: EL = PD x LGD x EAD.

    Accepts scalars or array-likes; returns the same shape. Inputs are
    validated to lie in sensible ranges.
    """
    pd_arr = np.asarray(pd_value, dtype=float)
    ead_arr = np.asarray(ead, dtype=float)

    if np.any((pd_arr < 0) | (pd_arr > 1)):
        raise ValueError("PD must be within [0, 1].")
    if not (0 <= lgd <= 1):
        raise ValueError("LGD must be within [0, 1].")
    if np.any(ead_arr < 0):
        raise ValueError("EAD must be non-negative.")

    result = pd_arr * lgd * ead_arr
    # Preserve scalar-in / scalar-out ergonomics.
    if result.ndim == 0:
        return float(result)
    return result


def build_loss_table(
    df: pd.DataFrame,
    pd_values,
    lgd: float = config.DEFAULT_LGD,
    ead_column: str = config.EAD_COLUMN,
) -> pd.DataFrame:
    """Return a borrower-level table with PD, LGD, EAD and Expected Loss.

    The input ``df`` is not mutated; a new frame is returned with the risk
    columns appended.
    """
    out = df.copy()
    out["pd"] = np.clip(np.asarray(pd_values, dtype=float), 0.0, 1.0)
    out["lgd"] = lgd
    out["ead"] = out[ead_column].astype(float)
    out["expected_loss"] = expected_loss(out["pd"].values, lgd, out["ead"].values)
    return out


def portfolio_metrics(loss_table: pd.DataFrame) -> PortfolioMetrics:
    """Aggregate a borrower-level loss table into portfolio metrics.

    ``loss_table`` must already contain ``pd``, ``ead`` and ``expected_loss``
    columns (from :func:`build_loss_table`) and, for high-risk exposure, a
    ``risk_band`` column (from :mod:`risk_segmentation`).
    """
    total_ead = float(loss_table["ead"].sum())
    total_el = float(loss_table["expected_loss"].sum())
    lgd = float(loss_table["lgd"].iloc[0]) if "lgd" in loss_table else config.DEFAULT_LGD

    # Exposure-weighted average PD is the economically meaningful summary.
    avg_pd = float(np.average(loss_table["pd"], weights=loss_table["ead"])) if total_ead else 0.0

    if "risk_band" in loss_table.columns:
        high_mask = loss_table["risk_band"].isin(config.HIGH_RISK_BANDS)
        high_risk_exposure = float(loss_table.loc[high_mask, "ead"].sum())
    else:
        high_risk_exposure = float("nan")

    return PortfolioMetrics(
        n_borrowers=int(len(loss_table)),
        total_ead=total_ead,
        average_pd=avg_pd,
        total_expected_loss=total_el,
        expected_loss_rate=(total_el / total_ead) if total_ead else 0.0,
        high_risk_exposure=high_risk_exposure,
        high_risk_exposure_pct=(high_risk_exposure / total_ead) if total_ead else 0.0,
        lgd=lgd,
    )


def expected_loss_by(loss_table: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate expected loss and exposure by a categorical dimension.

    Useful for risk-concentration analysis (e.g. by ``loan_grade``,
    ``loan_intent``, ``risk_band``).
    """
    grouped = (
        loss_table.groupby(dimension, observed=True)
        .agg(
            n_borrowers=("expected_loss", "size"),
            total_ead=("ead", "sum"),
            average_pd=("pd", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
    )
    total_ead = loss_table["ead"].sum()
    grouped["exposure_pct"] = grouped["total_ead"] / total_ead if total_ead else 0.0
    grouped["expected_loss_rate"] = np.where(
        grouped["total_ead"] > 0, grouped["expected_loss"] / grouped["total_ead"], 0.0
    )
    return grouped.sort_values("expected_loss", ascending=False).reset_index(drop=True)

"""Stress-testing engine - the centrepiece of the application.

We define hypothetical macro-deterioration scenarios and measure how the
portfolio's Probability of Default and Expected Loss respond. For each
scenario we:

    1. transform borrower inputs (income, leverage/DTI, interest rate, tenure);
    2. re-score PD with the trained model;
    3. recompute Expected Loss (PD x LGD x EAD);
    4. aggregate portfolio results and compare against the base case.

IMPORTANT
---------
These scenarios are **hypothetical sensitivity analyses**, NOT economic
forecasts. The shock magnitudes are illustrative analyst assumptions and are
fully configurable. Nothing here predicts what the economy will do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .expected_loss import build_loss_table, portfolio_metrics
from .risk_segmentation import add_risk_bands, segment_summary


@dataclass
class StressScenario:
    """A hypothetical deterioration applied to borrower characteristics.

    Parameters
    ----------
    income_shock:
        Multiplicative factor applied to ``person_income``
        (0.90 => a 10% income decline).
    rate_shock_pp:
        Additive shock (percentage points) to ``loan_int_rate``
        (2.0 => +200bps).
    dti_multiplier:
        Extra multiplicative stress on ``loan_percent_income`` on top of the
        mechanical rise caused by lower income (1.10 => +10%).
    emp_length_shock_years:
        Years subtracted from ``person_emp_length`` (job-tenure erosion).
    """

    name: str
    description: str
    income_shock: float = 1.0
    rate_shock_pp: float = 0.0
    dti_multiplier: float = 1.0
    emp_length_shock_years: float = 0.0

    def is_base(self) -> bool:
        return (
            self.income_shock == 1.0
            and self.rate_shock_pp == 0.0
            and self.dti_multiplier == 1.0
            and self.emp_length_shock_years == 0.0
        )


# Default scenario library (illustrative, configurable assumptions).
DEFAULT_SCENARIOS: dict[str, StressScenario] = {
    "Base": StressScenario(
        name="Base",
        description="No stress applied - current book as observed.",
    ),
    "Moderate": StressScenario(
        name="Moderate",
        description=(
            "Mild downturn: -10% income, +200bps rates, +10% leverage/DTI, "
            "-1yr job tenure."
        ),
        income_shock=0.90,
        rate_shock_pp=2.0,
        dti_multiplier=1.10,
        emp_length_shock_years=1.0,
    ),
    "Severe": StressScenario(
        name="Severe",
        description=(
            "Severe recession: -25% income, +500bps rates, +25% leverage/DTI, "
            "-3yr job tenure."
        ),
        income_shock=0.75,
        rate_shock_pp=5.0,
        dti_multiplier=1.25,
        emp_length_shock_years=3.0,
    ),
}


def apply_scenario(df: pd.DataFrame, scenario: StressScenario) -> pd.DataFrame:
    """Return a copy of ``df`` with the scenario's shocks applied.

    Exposure (``loan_amnt``) is deliberately held constant - we are stressing
    borrower creditworthiness, not re-underwriting loan sizes.
    """
    out = df.copy()
    if scenario.is_base():
        return out

    # 1) Income shock.
    out["person_income"] = out["person_income"] * scenario.income_shock

    # 2) Leverage / DTI. Lower income mechanically raises loan-to-income; we
    #    recompute it from the shocked income and then apply the extra stress
    #    multiplier. Bounded to a sane range.
    mechanical_dti = out["loan_amnt"] / out["person_income"].replace(0, np.nan)
    stressed_dti = mechanical_dti * scenario.dti_multiplier
    out["loan_percent_income"] = stressed_dti.clip(lower=0, upper=2.0).fillna(
        out["loan_percent_income"]
    )

    # 3) Interest-rate repricing (worsening credit conditions).
    out["loan_int_rate"] = out["loan_int_rate"] + scenario.rate_shock_pp

    # 4) Employment tenure erosion.
    out["person_emp_length"] = (
        out["person_emp_length"] - scenario.emp_length_shock_years
    ).clip(lower=0)

    return out


@dataclass
class ScenarioResult:
    """Portfolio outcome under one scenario, plus deltas vs. the base case."""

    name: str
    description: str
    metrics: dict
    loss_table: pd.DataFrame = field(repr=False)
    segment: pd.DataFrame = field(repr=False)
    el_increase_vs_base: float = 0.0
    pct_increase_vs_base: float = 0.0


def run_scenario(
    df: pd.DataFrame,
    scenario: StressScenario,
    pd_model,
    lgd: float = config.DEFAULT_LGD,
) -> ScenarioResult:
    """Apply a scenario, re-score PD, and compute portfolio-level results."""
    stressed = apply_scenario(df, scenario)
    pd_hat = pd_model.predict_pd(stressed)

    loss_table = build_loss_table(stressed, pd_hat, lgd=lgd)
    loss_table = add_risk_bands(loss_table)
    metrics = portfolio_metrics(loss_table)
    segment = segment_summary(loss_table)

    return ScenarioResult(
        name=scenario.name,
        description=scenario.description,
        metrics=metrics.as_dict(),
        loss_table=loss_table,
        segment=segment,
    )


def run_all_scenarios(
    df: pd.DataFrame,
    pd_model,
    lgd: float = config.DEFAULT_LGD,
    scenarios: dict[str, StressScenario] | None = None,
) -> dict[str, ScenarioResult]:
    """Run every scenario and attach change-vs-base statistics.

    Returns a dict keyed by scenario name. The "Base" scenario is used as the
    reference point for the increase calculations:

        el_increase   = stress_EL - base_EL
        pct_increase  = (stress_EL / base_EL - 1) * 100
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    results = {
        name: run_scenario(df, sc, pd_model, lgd=lgd)
        for name, sc in scenarios.items()
    }

    base_el = results["Base"].metrics["total_expected_loss"] if "Base" in results else None
    if base_el:
        for res in results.values():
            res.el_increase_vs_base = res.metrics["total_expected_loss"] - base_el
            res.pct_increase_vs_base = (
                res.metrics["total_expected_loss"] / base_el - 1.0
            ) * 100.0
    return results


def scenario_comparison_table(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """Flatten scenario results into a comparison DataFrame for display."""
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append(
            {
                "scenario": name,
                "average_pd": m["average_pd"],
                "total_ead": m["total_ead"],
                "expected_loss": m["total_expected_loss"],
                "expected_loss_rate": m["expected_loss_rate"],
                "high_risk_exposure": m["high_risk_exposure"],
                "el_increase_vs_base": res.el_increase_vs_base,
                "pct_increase_vs_base": res.pct_increase_vs_base,
            }
        )
    return pd.DataFrame(rows)


def segment_stress_comparison(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """Expected loss by risk band across scenarios (wide format).

    Answers "which segments see the largest increase in expected loss?".
    """
    frames = []
    for name, res in results.items():
        seg = res.segment[["risk_band", "expected_loss"]].copy()
        seg = seg.rename(columns={"expected_loss": name})
        frames.append(seg.set_index("risk_band"))
    wide = pd.concat(frames, axis=1).reset_index()
    if "Base" in wide.columns and "Severe" in wide.columns:
        wide["change_base_to_severe"] = wide["Severe"] - wide["Base"]
    return wide

"""Credit Risk Stress Testing Engine - Streamlit dashboard.

Run with:

    streamlit run dashboard/app.py

The dashboard ties together the PD model, Expected Loss engine, risk
segmentation, stress testing and borrower-level explainability into a single
credit-risk analytics view.

Reminders shown throughout the UI:
* Expected Loss is a forward-looking statistical expectation, not a realised loss.
* Stress scenarios are hypothetical sensitivity analyses, not economic forecasts.
* Risk-band thresholds and the LGD assumption are transparent modelling choices,
  not regulatory standards.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when launched via ``streamlit run dashboard/app.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import load_raw_data, profile_data
from src.preprocessing import clean_data
from src.pd_model import PDModel, load_report
from src.expected_loss import build_loss_table, portfolio_metrics, expected_loss_by
from src.risk_segmentation import add_risk_bands, segment_summary
from src.stress_testing import (
    DEFAULT_SCENARIOS,
    StressScenario,
    run_all_scenarios,
    scenario_comparison_table,
    segment_stress_comparison,
)
from src.explainability import explain_borrower, global_coefficients, contributions_frame

# --------------------------------------------------------------------------- #
# Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Credit Risk Stress Testing Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BAND_COLORS = {
    "Low Risk": "#2ca02c",
    "Moderate Risk": "#ffbf00",
    "High Risk": "#ff7f0e",
    "Very High Risk": "#d62728",
}
SCENARIO_COLORS = {"Base": "#1f77b4", "Moderate": "#ff7f0e", "Severe": "#d62728"}


# --------------------------------------------------------------------------- #
# Cached loaders / compute
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = clean_data(load_raw_data())
    df.insert(0, "borrower_id", [f"BRW-{i:05d}" for i in range(len(df))])
    return df


@st.cache_resource(show_spinner=False)
def get_model() -> PDModel:
    return PDModel.load()


@st.cache_data(show_spinner=False)
def get_report() -> dict:
    try:
        return load_report()
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner=False)
def score_base(filter_key: tuple) -> pd.DataFrame:
    """Score PD for the (optionally filtered) portfolio - PD is LGD-independent."""
    df = _apply_filters(get_data(), filter_key)
    model = get_model()
    df = df.copy()
    df["pd"] = model.predict_pd(df)
    return df


def _apply_filters(df: pd.DataFrame, filter_key: tuple) -> pd.DataFrame:
    grades, intents, ownership = filter_key
    out = df
    if grades:
        out = out[out["loan_grade"].isin(grades)]
    if intents:
        out = out[out["loan_intent"].isin(intents)]
    if ownership:
        out = out[out["person_home_ownership"].isin(ownership)]
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_scenarios(filter_key: tuple, lgd: float) -> dict:
    """Run all stress scenarios; return a plain-dict payload (cache-friendly)."""
    df = _apply_filters(get_data(), filter_key)
    model = get_model()
    results = run_all_scenarios(df, model, lgd=lgd)
    payload = {
        "comparison": scenario_comparison_table(results),
        "segment_stress": segment_stress_comparison(results),
        "descriptions": {n: r.description for n, r in results.items()},
    }
    return payload


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    return f"${x:,.0f}"


def kpi_row(pm) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Exposure (EAD)", money(pm.total_ead))
    c2.metric("Average PD (exp-wtd)", f"{pm.average_pd:.2%}")
    c3.metric("Expected Loss", money(pm.total_expected_loss))
    c4.metric("Expected Loss Rate", f"{pm.expected_loss_rate:.2%}")
    c5.metric("High-Risk Exposure", money(pm.high_risk_exposure),
              f"{pm.high_risk_exposure_pct:.1%} of book")


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
def sidebar():
    st.sidebar.title("🏦 Risk Controls")
    st.sidebar.caption(
        "A prototype credit-risk analytics engine. Assumptions are explicit and "
        "configurable - nothing here is a regulatory standard or a forecast."
    )

    st.sidebar.subheader("Loss assumption")
    lgd = st.sidebar.slider(
        "Loss Given Default (LGD)",
        min_value=0.10, max_value=0.90, value=config.DEFAULT_LGD, step=0.05,
        help="Assumed fraction of exposure lost if a borrower defaults. "
             "Default 45% is a common senior-unsecured placeholder, not an estimate.",
    )

    st.sidebar.subheader("Stress scenario")
    scenario_name = st.sidebar.radio(
        "Highlighted scenario",
        options=list(DEFAULT_SCENARIOS.keys()),
        index=0,
        help="Hypothetical sensitivity analysis - not an economic forecast.",
    )

    st.sidebar.subheader("Portfolio filters")
    data = get_data()
    grades = st.sidebar.multiselect("Loan grade", sorted(data["loan_grade"].unique()))
    intents = st.sidebar.multiselect("Loan purpose", sorted(data["loan_intent"].unique()))
    ownership = st.sidebar.multiselect(
        "Home ownership", sorted(data["person_home_ownership"].unique())
    )

    filter_key = (tuple(grades), tuple(intents), tuple(ownership))
    return lgd, scenario_name, filter_key


# --------------------------------------------------------------------------- #
# Tab: Portfolio Overview
# --------------------------------------------------------------------------- #
def tab_overview(loss_table: pd.DataFrame, lgd: float):
    st.header("Portfolio Overview")
    pm = portfolio_metrics(loss_table)
    kpi_row(pm)

    st.info(
        "**Expected Loss** = PD × LGD × EAD, aggregated across borrowers. "
        "It is a forward-looking *statistical expectation* of loss - not a realised "
        f"or incurred loss. LGD is fixed at **{lgd:.0%}** (assumption) and EAD uses "
        "the loan amount as a transparent proxy.",
        icon="ℹ️",
    )

    left, right = st.columns([1.2, 1])
    with left:
        seg = segment_summary(loss_table)
        fig = px.bar(
            seg, x="risk_band", y="expected_loss", color="risk_band",
            color_discrete_map=BAND_COLORS,
            title="Expected Loss by Risk Band",
            labels={"expected_loss": "Expected Loss ($)", "risk_band": ""},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        prof = profile_data(get_data())
        st.subheader("Dataset (observed)")
        st.markdown(
            f"- **Borrowers scored:** {pm.n_borrowers:,}\n"
            f"- **Full-dataset default rate:** {prof.default_rate:.2%}\n"
            f"- **Numeric features:** {len(prof.numeric_features)}\n"
            f"- **Categorical features:** {len(prof.categorical_features)}\n"
            f"- **EAD proxy:** `{config.EAD_COLUMN}` (loan amount)"
        )
        st.caption(
            "Default rate is observed in the data. PD, Expected Loss and bands "
            "are model outputs given the LGD assumption."
        )


# --------------------------------------------------------------------------- #
# Tab: Scenario Analysis
# --------------------------------------------------------------------------- #
def tab_scenarios(filter_key: tuple, lgd: float, highlighted: str):
    st.header("Scenario Analysis (Stress Testing)")
    st.warning(
        "Scenarios below are **hypothetical sensitivity analyses**, not economic "
        "forecasts. Shock magnitudes are illustrative analyst assumptions.",
        icon="⚠️",
    )

    payload = get_scenarios(filter_key, lgd)
    comp = payload["comparison"]
    for name, desc in payload["descriptions"].items():
        st.markdown(f"- **{name}** — {desc}")

    # KPI-style scenario comparison.
    st.subheader("Portfolio impact by scenario")
    show = comp.copy()
    show["Expected Loss"] = show["expected_loss"].map(money)
    show["Avg PD"] = show["average_pd"].map(lambda v: f"{v:.2%}")
    show["EL Rate"] = show["expected_loss_rate"].map(lambda v: f"{v:.2%}")
    show["High-Risk Exposure"] = show["high_risk_exposure"].map(money)
    show["Δ EL vs Base"] = show["el_increase_vs_base"].map(money)
    show["% Increase vs Base"] = show["pct_increase_vs_base"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(
        show[["scenario", "Avg PD", "Expected Loss", "EL Rate",
              "High-Risk Exposure", "Δ EL vs Base", "% Increase vs Base"]],
        hide_index=True, width="stretch",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(comp, x="scenario", y="expected_loss", color="scenario",
                     color_discrete_map=SCENARIO_COLORS, title="Expected Loss")
        fig.update_layout(showlegend=False, height=340,
                          yaxis_title="Expected Loss ($)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(comp, x="scenario", y="average_pd", color="scenario",
                     color_discrete_map=SCENARIO_COLORS, title="Average PD")
        fig.update_layout(showlegend=False, height=340,
                          yaxis_title="Average PD", xaxis_title="")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with c3:
        fig = px.bar(comp, x="scenario", y="high_risk_exposure", color="scenario",
                     color_discrete_map=SCENARIO_COLORS, title="High-Risk Exposure")
        fig.update_layout(showlegend=False, height=340,
                          yaxis_title="High-Risk Exposure ($)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    # Highlighted scenario callout.
    row = comp[comp["scenario"] == highlighted].iloc[0]
    if highlighted != "Base":
        st.success(
            f"**{highlighted} scenario:** Expected Loss rises by "
            f"**{money(row['el_increase_vs_base'])}** "
            f"(**{row['pct_increase_vs_base']:+.1f}%**) versus the base case.",
            icon="📈",
        )

    # Which segments move the most.
    st.subheader("Where does the stress land? Expected Loss by risk band")
    seg = payload["segment_stress"]
    melt = seg.melt(id_vars="risk_band",
                    value_vars=[c for c in ["Base", "Moderate", "Severe"] if c in seg],
                    var_name="scenario", value_name="expected_loss")
    fig = px.bar(melt, x="risk_band", y="expected_loss", color="scenario",
                 barmode="group", color_discrete_map=SCENARIO_COLORS,
                 title="Expected Loss by Risk Band across Scenarios",
                 category_orders={"risk_band": config.RISK_BAND_ORDER})
    fig.update_layout(height=380, yaxis_title="Expected Loss ($)", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    disp = seg.copy()
    for col in [c for c in ["Base", "Moderate", "Severe", "change_base_to_severe"] if c in disp]:
        disp[col] = disp[col].map(money)
    st.dataframe(disp, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Tab: Risk Distribution
# --------------------------------------------------------------------------- #
def tab_distribution(loss_table: pd.DataFrame):
    st.header("Risk Distribution")
    st.caption(
        "Risk bands use fixed PD thresholds (Low <5%, Moderate 5-15%, "
        "High 15-30%, Very High ≥30%). These are modelling choices, not "
        "regulatory categories."
    )
    seg = segment_summary(loss_table).fillna(0)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(seg, x="risk_band", y="n_borrowers", color="risk_band",
                     color_discrete_map=BAND_COLORS, title="Borrowers per Risk Band")
        fig.update_layout(showlegend=False, height=360,
                          yaxis_title="Borrowers", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.pie(seg, names="risk_band", values="total_ead", hole=0.45,
                     color="risk_band", color_discrete_map=BAND_COLORS,
                     title="Exposure Share by Risk Band")
        fig.update_layout(height=360)
        st.plotly_chart(fig, width="stretch")

    disp = seg.copy()
    disp["total_ead"] = disp["total_ead"].map(money)
    disp["average_pd"] = disp["average_pd"].map(lambda v: f"{v:.2%}")
    disp["expected_loss"] = disp["expected_loss"].map(money)
    disp["exposure_pct"] = disp["exposure_pct"].map(lambda v: f"{v:.1%}")
    disp["expected_loss_rate"] = disp["expected_loss_rate"].map(lambda v: f"{v:.2%}")
    disp = disp.rename(columns={
        "risk_band": "Risk Band", "n_borrowers": "Borrowers", "total_ead": "Total EAD",
        "average_pd": "Avg PD", "expected_loss": "Expected Loss",
        "exposure_pct": "% Exposure", "expected_loss_rate": "EL Rate",
    })
    st.dataframe(disp, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Tab: Risk Concentration
# --------------------------------------------------------------------------- #
def tab_concentration(loss_table: pd.DataFrame):
    st.header("Risk Concentration")
    st.caption("Where is the portfolio's credit risk concentrated?")

    dimensions = {
        "Loan grade": "loan_grade",
        "Loan purpose": "loan_intent",
        "Home ownership": "person_home_ownership",
        "Prior default on file": "cb_person_default_on_file",
        "Risk band": "risk_band",
    }
    label = st.selectbox("Break down Expected Loss by", list(dimensions.keys()))
    dim = dimensions[label]

    agg = expected_loss_by(loss_table, dim)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.bar(agg, x=dim, y="expected_loss", color="expected_loss_rate",
                     color_continuous_scale="Reds",
                     title=f"Expected Loss by {label}",
                     labels={"expected_loss": "Expected Loss ($)",
                             "expected_loss_rate": "EL rate"})
        fig.update_layout(height=400, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        top = agg.iloc[0]
        st.metric(f"Top contributor ({label})", str(top[dim]),
                  f"{money(top['expected_loss'])} EL")
        st.caption(
            "Bars ranked by total Expected Loss; colour shows the EL rate "
            "(EL / EAD) so you can separate *size* of exposure from *riskiness*."
        )

    disp = agg.copy()
    disp["total_ead"] = disp["total_ead"].map(money)
    disp["average_pd"] = disp["average_pd"].map(lambda v: f"{v:.2%}")
    disp["expected_loss"] = disp["expected_loss"].map(money)
    disp["exposure_pct"] = disp["exposure_pct"].map(lambda v: f"{v:.1%}")
    disp["expected_loss_rate"] = disp["expected_loss_rate"].map(lambda v: f"{v:.2%}")
    st.dataframe(disp, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Tab: Borrower Risk
# --------------------------------------------------------------------------- #
def tab_borrower(loss_table: pd.DataFrame, lgd: float):
    st.header("Borrower Risk View")
    ids = loss_table["borrower_id"].tolist()
    if not ids:
        st.warning("No borrowers match the current filters.")
        return

    selected = st.selectbox("Select a borrower", ids, index=0)
    row = loss_table[loss_table["borrower_id"] == selected].iloc[[0]]
    r = row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Default", f"{r['pd']:.2%}")
    c2.metric("Exposure (EAD)", money(r["ead"]))
    c3.metric("Expected Loss", money(r["expected_loss"]))
    c4.metric("Risk Band", str(r["risk_band"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("LGD (assumption)", f"{lgd:.0%}")
    c2.metric("Loan grade", str(r["loan_grade"]))
    c3.metric("Loan purpose", str(r["loan_intent"]))

    st.subheader("Borrower risk drivers")
    st.caption(
        "Signed log-odds contributions from the logistic PD model. Language is "
        "*associational*: a driver is 'associated with higher/lower predicted PD', "
        "not a cause of default."
    )

    model = get_model()
    try:
        higher, lower = explain_borrower(model, row)
        cf = contributions_frame(model, row)
        fig = px.bar(
            cf, x="contribution", y="feature", orientation="h", color="direction",
            color_discrete_map={"higher risk": "#d62728", "lower risk": "#2ca02c"},
            title="Feature contributions to this borrower's log-odds of default",
        )
        fig.update_layout(height=420, xaxis_title="← lower PD    contribution    higher PD →",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Associated with higher predicted PD**")
            for fc in higher:
                st.markdown(f"- {fc.feature}  (`+{fc.contribution:.2f}`)")
        with cc2:
            st.markdown("**Associated with lower predicted PD**")
            for fc in lower:
                st.markdown(f"- {fc.feature}  (`{fc.contribution:.2f}`)")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Could not compute contributions for this model: {exc}")


# --------------------------------------------------------------------------- #
# Tab: Model Diagnostics
# --------------------------------------------------------------------------- #
def tab_diagnostics():
    st.header("Model Diagnostics")
    report = get_report()
    if not report:
        st.warning("No metrics found. Train the model first: `python -m src.pd_model`.")
        return

    st.caption(
        "All figures below are computed on a held-out, stratified test fold. "
        "This is a prototype - not a validated, production credit model."
    )

    ds = report.get("dataset", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Train size", f"{ds.get('n_train', 0):,}")
    c2.metric("Test size", f"{ds.get('n_test', 0):,}")
    c3.metric("Test default rate", f"{ds.get('test_default_rate', 0):.2%}")

    models = report.get("models", {})
    rows = []
    for key, m in models.items():
        rows.append({
            "Model": m["model_name"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
            "Brier": m["brier_score"],
        })
    comp = pd.DataFrame(rows)
    st.subheader("Model comparison")
    st.dataframe(
        comp.style.format({
            "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1": "{:.3f}", "Brier": "{:.4f}",
        }),
        hide_index=True, width="stretch",
    )
    st.info(
        f"**Deployed PD model:** {report.get('chosen_model', 'n/a')}. "
        "Logistic regression is used as the PD scorer for transparency and "
        "calibration. Gradient boosting ranks slightly higher on ROC/PR-AUC but "
        "is kept only as a benchmark for this prototype.",
        icon="🧭",
    )

    # Pick the chosen (calibrated if available) logistic model for curves.
    pick = models.get("logistic_calibrated") or models.get("logistic_uncalibrated")
    if pick is None:
        return

    c1, c2 = st.columns(2)
    with c1:
        roc = pick["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"ROC (AUC={pick['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"), name="Chance"))
        fig.update_layout(title="ROC Curve", height=360,
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width="stretch")
        st.caption("ROC-AUC = probability the model ranks a random defaulter above "
                   "a random non-defaulter. Higher is better (0.5 = chance).")
    with c2:
        pr = pick["pr_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                 name=f"PR (AP={pick['pr_auc']:.3f})"))
        fig.update_layout(title="Precision-Recall Curve", height=360,
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, width="stretch")
        st.caption("PR-AUC is informative under class imbalance (defaults are the "
                   "minority class) where accuracy would be misleading.")

    c1, c2 = st.columns(2)
    with c1:
        cal = pick["calibration_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"],
                                 mode="lines+markers", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="grey"),
                                 name="Perfectly calibrated"))
        fig.update_layout(title="Calibration Curve (predicted PD vs observed default rate)",
                          height=380, xaxis_title="Mean predicted PD",
                          yaxis_title="Observed default frequency")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Calibration matters because PD feeds Expected Loss directly. "
            f"Brier score (lower is better): **{pick['brier_score']:.4f}**. "
            "Points on the diagonal mean predicted PDs match observed frequencies."
        )
    with c2:
        cm = np.array(pick["confusion"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: No default", "Pred: Default"],
                        y=["Actual: No default", "Actual: Default"],
                        title=f"Confusion Matrix (threshold={pick['threshold']:.2f})")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("A 0.50 cut-off is shown for illustration; the operating "
                   "threshold should be chosen from the cost of misses vs false alarms.")

    st.subheader("Feature coefficients (log-odds)")
    try:
        coefs = global_coefficients(get_model()).head(15)
        fig = px.bar(coefs.sort_values("coefficient"), x="coefficient", y="feature",
                     orientation="h", color="coefficient",
                     color_continuous_scale="RdBu_r",
                     title="Top logistic-regression coefficients")
        fig.update_layout(height=460, xaxis_title="Coefficient (log-odds)", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Positive coefficient => feature associated with higher predicted "
                   "PD. Coefficients act on standardised inputs, so magnitudes are "
                   "broadly comparable across features.")
    except Exception as exc:  # pragma: no cover
        st.info(f"Coefficient view unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("🏦 Credit Risk Stress Testing Engine")
    st.markdown(
        "PD modelling → Expected Credit Loss (PD × LGD × EAD) → risk segmentation "
        "→ **stress testing** → explainability, on a public consumer-loan dataset."
    )

    try:
        get_model()
    except FileNotFoundError:
        st.error(
            "Trained model not found. Run `python -m src.pd_model` from the "
            "project root to train and persist it, then reload."
        )
        st.stop()

    lgd, scenario_name, filter_key = sidebar()

    scored = score_base(filter_key)
    if scored.empty:
        st.warning("No borrowers match the current filters. Clear some filters.")
        st.stop()
    loss_table = add_risk_bands(build_loss_table(scored, scored["pd"].values, lgd=lgd))

    tabs = st.tabs([
        "📊 Portfolio Overview", "🌩️ Scenario Analysis", "📈 Risk Distribution",
        "🎯 Risk Concentration", "👤 Borrower Risk", "🔬 Model Diagnostics",
    ])
    with tabs[0]:
        tab_overview(loss_table, lgd)
    with tabs[1]:
        tab_scenarios(filter_key, lgd, scenario_name)
    with tabs[2]:
        tab_distribution(loss_table)
    with tabs[3]:
        tab_concentration(loss_table)
    with tabs[4]:
        tab_borrower(loss_table, lgd)
    with tabs[5]:
        tab_diagnostics()

    st.divider()
    st.caption(
        "Prototype for demonstration only. Not a validated or production credit "
        "model; makes no regulatory-compliance claim. Expected Loss is a statistical "
        "expectation, not a realised loss; stress scenarios are hypothetical."
    )


if __name__ == "__main__":
    main()

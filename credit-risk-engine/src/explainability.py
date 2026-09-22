"""Borrower-level explainability for the PD model.

For a logistic-regression PD model the cleanest, exact attribution is the
*log-odds contribution* of each (preprocessed) feature:

    logit(PD) = intercept + sum_i (coef_i * x_i)

where ``x_i`` is the standardised / one-hot-encoded feature value. Each term
``coef_i * x_i`` is that feature's push on the borrower's log-odds of default.
Positive terms are associated with *higher* predicted PD; negative terms with
*lower* predicted PD.

We deliberately use associational language ("associated with higher predicted
PD"), never causal claims ("causes default").

This approach also works when the logistic pipeline is wrapped in
``CalibratedClassifierCV`` (we average the coefficients of the underlying
fold models). If SHAP is installed it can be layered on, but coefficient
contributions are exact for a linear model and require no extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


@dataclass
class FeatureContribution:
    feature: str
    contribution: float  # signed log-odds contribution
    direction: str       # "higher risk" | "lower risk"


def _extract_linear_pieces(estimator):
    """Return (preprocessor, coef_vector, intercept) from a fitted PD model.

    Handles both a bare logistic ``Pipeline`` and a ``CalibratedClassifierCV``
    wrapping such pipelines (coefficients are averaged across CV folds).
    """
    if isinstance(estimator, Pipeline):
        clf = estimator.named_steps.get("clf")
        if not isinstance(clf, LogisticRegression):
            raise TypeError("Coefficient explainer requires a LogisticRegression pipeline.")
        return estimator.named_steps["preprocess"], clf.coef_[0], float(clf.intercept_[0])

    if isinstance(estimator, CalibratedClassifierCV):
        coefs, intercepts, preproc = [], [], None
        for cal in estimator.calibrated_classifiers_:
            pipe = cal.estimator
            clf = pipe.named_steps["clf"]
            coefs.append(clf.coef_[0])
            intercepts.append(float(clf.intercept_[0]))
            preproc = pipe.named_steps["preprocess"]
        return preproc, np.mean(coefs, axis=0), float(np.mean(intercepts))

    raise TypeError(f"Unsupported estimator type for explanation: {type(estimator)}")


def _pretty_feature_name(raw: str) -> str:
    """Turn ColumnTransformer output names into readable labels."""
    name = raw.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ")


def global_coefficients(pd_model) -> pd.DataFrame:
    """Return model-wide coefficients (log-odds) sorted by magnitude.

    Used on the diagnostics page. Positive coefficient => the feature is
    associated with higher predicted PD.
    """
    preproc, coef, _ = _extract_linear_pieces(pd_model.estimator)
    names = [_pretty_feature_name(n) for n in preproc.get_feature_names_out()]
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def explain_borrower(pd_model, borrower_row: pd.DataFrame, top_n: int = 6):
    """Explain a single borrower's PD via signed log-odds contributions.

    Parameters
    ----------
    borrower_row:
        A one-row DataFrame containing the raw feature columns.

    Returns
    -------
    (higher_risk, lower_risk):
        Two lists of :class:`FeatureContribution`, sorted by magnitude - the
        features pushing the borrower's PD up and down respectively.
    """
    if len(borrower_row) != 1:
        raise ValueError("explain_borrower expects exactly one borrower row.")

    preproc, coef, _ = _extract_linear_pieces(pd_model.estimator)
    x = preproc.transform(borrower_row[config.FEATURES])
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()

    contributions = coef * x
    names = [_pretty_feature_name(n) for n in preproc.get_feature_names_out()]

    items = [
        FeatureContribution(
            feature=n,
            contribution=float(c),
            direction="higher risk" if c > 0 else "lower risk",
        )
        for n, c in zip(names, contributions)
        if abs(c) > 1e-9
    ]
    items.sort(key=lambda fc: abs(fc.contribution), reverse=True)

    higher = [fc for fc in items if fc.contribution > 0][:top_n]
    lower = [fc for fc in items if fc.contribution < 0][:top_n]
    return higher, lower


def contributions_frame(pd_model, borrower_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Return the top borrower contributions as a tidy DataFrame (for charts)."""
    higher, lower = explain_borrower(pd_model, borrower_row, top_n=top_n)
    rows = [
        {"feature": fc.feature, "contribution": fc.contribution, "direction": fc.direction}
        for fc in higher + lower
    ]
    return pd.DataFrame(rows).sort_values("contribution")

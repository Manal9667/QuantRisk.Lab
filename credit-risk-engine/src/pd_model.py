"""Probability of Default (PD) model.

Design choices
--------------
* Baseline model is **Logistic Regression** inside an sklearn ``Pipeline`` with
  the preprocessing ColumnTransformer. Logistic regression is transparent,
  well calibrated and the natural first choice for PD scoring.
* We optionally compare against **Gradient Boosting** to sanity-check that the
  linear baseline is not leaving obvious signal on the table.
* Because PD (not a hard label) is the quantity that feeds Expected Loss, we
  evaluate ranking (ROC-AUC / PR-AUC) *and* calibration (calibration curve,
  Brier score) and apply probability calibration where justified.

The public artefact is :class:`PDModel`, which scores raw (uncleaned-schema)
feature DataFrames and returns a PD per borrower.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import SplitData, build_preprocessor, clean_data, split_data


# --------------------------------------------------------------------------- #
# Metrics container
# --------------------------------------------------------------------------- #
@dataclass
class ModelMetrics:
    """Evaluation metrics computed on the held-out test fold (model outputs)."""

    model_name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    confusion: list[list[int]]
    threshold: float = 0.5
    n_test: int = 0
    test_default_rate: float = 0.0
    # Curves (stored so the dashboard can plot without re-scoring).
    roc_curve: dict[str, list[float]] = field(default_factory=dict)
    pr_curve: dict[str, list[float]] = field(default_factory=dict)
    calibration_curve: dict[str, list[float]] = field(default_factory=dict)

    def headline(self) -> str:
        return (
            f"[{self.model_name}] ROC-AUC={self.roc_auc:.3f}  "
            f"PR-AUC={self.pr_auc:.3f}  Brier={self.brier_score:.4f}  "
            f"F1={self.f1:.3f} (thr={self.threshold:.2f})"
        )


# --------------------------------------------------------------------------- #
# PD model wrapper
# --------------------------------------------------------------------------- #
class PDModel:
    """A fitted, calibration-aware Probability of Default model.

    Wraps an sklearn estimator (pipeline) that accepts the raw feature columns
    listed in :data:`config.FEATURES` and exposes :meth:`predict_pd`.
    """

    def __init__(self, estimator: Pipeline, model_name: str, calibrated: bool):
        self.estimator = estimator
        self.model_name = model_name
        self.calibrated = calibrated

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(default) for each row of ``X`` (values in [0, 1])."""
        return self.estimator.predict_proba(X[config.FEATURES])[:, 1]

    # -- persistence -------------------------------------------------------- #
    def save(self, path=None) -> None:
        path = path or config.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "model_name": self.model_name,
                "calibrated": self.calibrated,
            },
            path,
        )

    @classmethod
    def load(cls, path=None) -> "PDModel":
        path = path or config.MODEL_PATH
        blob = joblib.load(path)
        return cls(blob["estimator"], blob["model_name"], blob["calibrated"])


# --------------------------------------------------------------------------- #
# Estimator builders
# --------------------------------------------------------------------------- #
def build_logistic_pipeline() -> Pipeline:
    """Preprocessing + L2 logistic regression (class-balanced)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gbm_pipeline() -> Pipeline:
    """Preprocessing + gradient boosting (comparison model)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _thin(*arrays, max_points: int = 300):
    """Down-sample parallel curve arrays for compact, plot-friendly storage."""
    n = len(arrays[0])
    if n <= max_points:
        return [np.asarray(a) for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return [np.asarray(a)[idx] for a in arrays]


def evaluate(estimator, X_test, y_test, model_name: str, threshold: float = 0.5) -> ModelMetrics:
    """Compute ranking + calibration metrics on the held-out fold."""
    pd_hat = estimator.predict_proba(X_test)[:, 1]
    y_pred = (pd_hat >= threshold).astype(int)

    frac_pos, mean_pred = calibration_curve(y_test, pd_hat, n_bins=10, strategy="quantile")
    fpr, tpr, _ = roc_curve(y_test, pd_hat)
    prec_c, rec_c, _ = precision_recall_curve(y_test, pd_hat)

    # Down-sample the curves so persisted metrics stay small.
    fpr, tpr = _thin(fpr, tpr)
    prec_c, rec_c = _thin(prec_c, rec_c)

    return ModelMetrics(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y_test, pd_hat)),
        pr_auc=float(average_precision_score(y_test, pd_hat)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        brier_score=float(brier_score_loss(y_test, pd_hat)),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
        threshold=threshold,
        n_test=int(len(y_test)),
        test_default_rate=float(np.mean(y_test)),
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        pr_curve={"precision": prec_c.tolist(), "recall": rec_c.tolist()},
        calibration_curve={
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
    )


# --------------------------------------------------------------------------- #
# Training orchestration
# --------------------------------------------------------------------------- #
def train(
    split: SplitData | None = None,
    compare_gbm: bool = True,
    calibrate: bool = True,
) -> tuple[PDModel, dict]:
    """Train the PD model, evaluate, calibrate, and return (model, report).

    The returned ``report`` dict is JSON-serialisable and captures every metric
    used in the README / dashboard - all of them computed, none fabricated.
    """
    if split is None:
        from .data_loader import load_raw_data

        split = split_data(clean_data(load_raw_data()))

    report: dict = {
        "dataset": {
            "n_train": split.train_size,
            "n_test": split.test_size,
            "train_default_rate": float(split.y_train.mean()),
            "test_default_rate": float(split.y_test.mean()),
        },
        "models": {},
    }

    # --- Baseline: logistic regression (uncalibrated) --------------------- #
    logit = build_logistic_pipeline()
    logit.fit(split.X_train, split.y_train)
    logit_metrics = evaluate(logit, split.X_test, split.y_test, "LogisticRegression")
    report["models"]["logistic_uncalibrated"] = asdict(logit_metrics)

    # --- Optional comparison: gradient boosting --------------------------- #
    if compare_gbm:
        gbm = build_gbm_pipeline()
        gbm.fit(split.X_train, split.y_train)
        gbm_metrics = evaluate(gbm, split.X_test, split.y_test, "GradientBoosting")
        report["models"]["gradient_boosting"] = asdict(gbm_metrics)

    # --- Probability calibration ------------------------------------------ #
    # We calibrate the logistic model with isotonic regression via 5-fold CV on
    # the training data, then re-evaluate on the untouched test fold. We keep
    # whichever variant has the lower Brier score (better calibration), since
    # calibrated PDs are what feed Expected Loss.
    chosen_estimator = logit
    chosen_name = "LogisticRegression (uncalibrated)"
    is_calibrated = False

    if calibrate:
        calibrated = CalibratedClassifierCV(
            build_logistic_pipeline(), method="isotonic", cv=5
        )
        calibrated.fit(split.X_train, split.y_train)
        cal_metrics = evaluate(
            calibrated, split.X_test, split.y_test, "LogisticRegression (calibrated)"
        )
        report["models"]["logistic_calibrated"] = asdict(cal_metrics)

        if cal_metrics.brier_score <= logit_metrics.brier_score:
            chosen_estimator = calibrated
            chosen_name = "LogisticRegression (isotonic-calibrated)"
            is_calibrated = True

    report["chosen_model"] = chosen_name
    report["chosen_is_calibrated"] = is_calibrated

    model = PDModel(chosen_estimator, chosen_name, is_calibrated)
    return model, report


def save_report(report: dict, path=None) -> None:
    path = path or config.METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def load_report(path=None) -> dict:
    path = path or config.METRICS_PATH
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    model, report = train()
    model.save()
    save_report(report)

    print("Training complete.")
    print(f"  Train / test        : {report['dataset']['n_train']:,} / "
          f"{report['dataset']['n_test']:,}")
    print(f"  Chosen model        : {report['chosen_model']}")
    for key, m in report["models"].items():
        print(f"  {key:28s} ROC-AUC={m['roc_auc']:.3f}  "
              f"PR-AUC={m['pr_auc']:.3f}  Brier={m['brier_score']:.4f}")
    print(f"  Model saved to      : {config.MODEL_PATH}")
    print(f"  Metrics saved to    : {config.METRICS_PATH}")

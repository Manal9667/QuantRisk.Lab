"""Cleaning, feature encoding and leakage-safe train/test splitting.

Pipeline:

    raw data
      -> validation (data_loader)
      -> row-level cleaning (drop implausible records)
      -> stratified train/test split
      -> ColumnTransformer (impute + scale numeric, impute + one-hot categorical)

The imputers/scalers are fit on the *training* fold only (inside an sklearn
Pipeline), which is what prevents information from the test fold leaking into
preprocessing statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class SplitData:
    """Container for a stratified train/test split (features + target)."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obvious data-entry errors and reset the index.

    We remove biologically/economically impossible records (e.g. age 144,
    123 years of employment). Missing values are *not* handled here - they are
    imputed inside the modelling pipeline so the statistics are learned only
    from training data.
    """
    cleaned = df.copy()

    before = len(cleaned)
    cleaned = cleaned[cleaned["person_age"] <= config.MAX_PLAUSIBLE_AGE]
    cleaned = cleaned[
        (cleaned["person_emp_length"] <= config.MAX_PLAUSIBLE_EMP_LENGTH)
        | (cleaned["person_emp_length"].isna())
    ]
    cleaned = cleaned.reset_index(drop=True)

    removed = before - len(cleaned)
    if removed:
        # Informational only; keep quiet in library use but handy when run directly.
        pass
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside the model pipeline.

    * Numeric  : median imputation + standardisation.
    * Categorical: most-frequent imputation + one-hot encoding.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_data(df: pd.DataFrame) -> SplitData:
    """Stratified train/test split on the (imbalanced) default target."""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # preserve default rate across folds
    )
    return SplitData(X_train, X_test, y_train, y_test)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names after fitting the preprocessor."""
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    from .data_loader import load_raw_data

    raw = load_raw_data()
    clean = clean_data(raw)
    split = split_data(clean)
    print(f"Rows after cleaning : {len(clean):,} (from {len(raw):,})")
    print(f"Train size          : {split.train_size:,}")
    print(f"Test size           : {split.test_size:,}")
    print(f"Train default rate  : {split.y_train.mean():.2%}")
    print(f"Test default rate   : {split.y_test.mean():.2%}")

"""Load and validate the raw credit-risk dataset.

Dataset: "Credit Risk Dataset" (public, Kaggle mirror). ~32.6k consumer loan
records with borrower, loan and credit-bureau attributes plus a binary
``loan_status`` default flag. See README.md for full provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config


@dataclass
class DataProfile:
    """A lightweight, printable summary of a dataset.

    Everything here is *observed data* - it is computed directly from the file
    and involves no modelling assumptions.
    """

    n_rows: int
    n_cols: int
    default_rate: float
    n_defaults: int
    missing_by_column: dict[str, int]
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Dataset profile (observed data)",
            "-------------------------------",
            f"Observations      : {self.n_rows:,}",
            f"Columns           : {self.n_cols}",
            f"Default events    : {self.n_defaults:,}",
            f"Default rate      : {self.default_rate:.2%}",
            f"Numeric features  : {len(self.numeric_features)}",
            f"Categorical feats : {len(self.categorical_features)}",
        ]
        missing = {k: v for k, v in self.missing_by_column.items() if v > 0}
        if missing:
            lines.append("Missing values    :")
            for col, n in missing.items():
                lines.append(f"    - {col}: {n:,}")
        else:
            lines.append("Missing values    : none")
        return "\n".join(lines)


# Expected schema: column -> whether nulls are tolerated at load time.
EXPECTED_COLUMNS = {
    "person_age": False,
    "person_income": False,
    "person_home_ownership": False,
    "person_emp_length": True,       # known to contain NaNs
    "loan_intent": False,
    "loan_grade": False,
    "loan_amnt": False,
    "loan_int_rate": True,           # known to contain NaNs
    "loan_status": False,
    "loan_percent_income": False,
    "cb_person_default_on_file": False,
    "cb_person_cred_hist_length": False,
}


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and run structural validation.

    Raises
    ------
    FileNotFoundError
        If the dataset is not present.
    ValueError
        If required columns are missing or the target is not binary.
    """
    path = Path(path) if path is not None else config.RAW_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "See README.md for download instructions."
        )

    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    target_values = set(df[config.TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target '{config.TARGET}' must be binary (0/1); found {sorted(target_values)}"
        )

    # Columns not tolerating nulls really should not have any.
    for col, nullable in EXPECTED_COLUMNS.items():
        if not nullable and df[col].isna().any():
            raise ValueError(f"Unexpected missing values in non-nullable column '{col}'")


def profile_data(df: pd.DataFrame) -> DataProfile:
    """Compute an observed-data summary of the dataset."""
    return DataProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        default_rate=float(df[config.TARGET].mean()),
        n_defaults=int(df[config.TARGET].sum()),
        missing_by_column=df.isna().sum().to_dict(),
        numeric_features=list(config.NUMERIC_FEATURES),
        categorical_features=list(config.CATEGORICAL_FEATURES),
    )


if __name__ == "__main__":
    frame = load_raw_data()
    print(profile_data(frame).summary())

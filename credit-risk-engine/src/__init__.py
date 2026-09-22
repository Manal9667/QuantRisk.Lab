"""Credit Risk Stress Testing Engine.

A compact, modular credit-risk analytics toolkit:

    data_loader        -> load + validate the raw dataset
    preprocessing      -> clean, encode, split (leakage-safe sklearn Pipeline)
    pd_model           -> Probability of Default model (+ calibration, metrics)
    expected_loss      -> Expected Credit Loss engine (EL = PD x LGD x EAD)
    risk_segmentation  -> assign borrowers to PD-based risk bands
    stress_testing     -> hypothetical economic stress scenarios
    explainability     -> borrower-level risk drivers

The Streamlit dashboard in ``dashboard/app.py`` ties everything together.
"""

__version__ = "0.1.0"

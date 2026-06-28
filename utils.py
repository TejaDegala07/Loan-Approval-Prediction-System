"""
utils.py
Shared utilities for Loan Approval Prediction.

IMPORTANT: LoanFeatureEngineer MUST live in this module (not in __main__).
Pickle stores the class as 'utils.LoanFeatureEngineer'.
When Flask loads loan_model.pkl via joblib, pickle automatically imports
this module and reconstructs the transformer — no changes needed in app.py.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless sklearn-compatible transformer that adds three engineered
    features to the 11-column input array.

    Expected input column order (matches Flask app.py exactly):
        0  Gender            (1=Male, 0=Female)
        1  Married           (1=Yes,  0=No)
        2  Dependents        (0 / 1 / 2 / 3)
        3  Education         (0=Graduate, 1=Not Graduate)
        4  Self_Employed     (1=Yes, 0=No)
        5  ApplicantIncome   (raw ₹)
        6  CoapplicantIncome (raw ₹)
        7  LoanAmount        (₹ thousands)
        8  Loan_Amount_Term  (months)
        9  Credit_History    (1=Good, 0=Bad/None)
        10 Property_Area     (0=Rural, 1=Semiurban, 2=Urban)

    Engineered features appended (columns 11–13):
        11 TotalIncome     — log1p(ApplicantIncome + CoapplicantIncome)
                             Compresses the heavy right-skew in income data.
        12 EMI             — LoanAmount / max(Loan_Amount_Term, 1)
                             Monthly instalment proxy; key affordability signal.
        13 Balance_Income  — TotalIncome − EMI
                             Surplus-income proxy; strong approval predictor.
    """

    def fit(self, X, y=None):          # Nothing to learn — purely stateless
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)

        # Combined income on log scale (reduces outlier sensitivity)
        total_income = np.log1p(X[:, 5] + X[:, 6])

        # Monthly EMI estimate
        term = np.where(X[:, 8] > 0, X[:, 8], 1.0)
        emi = X[:, 7] / term

        # Net income after EMI (creditworthiness proxy)
        balance_income = total_income - emi

        return np.column_stack([X, total_income, emi, balance_income])


def preprocess_input(data):
    """Legacy stub — kept for backward compatibility."""
    return data

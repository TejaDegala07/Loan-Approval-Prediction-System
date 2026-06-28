# -*- coding: utf-8 -*-
"""
train_model.py
==============
Loan Approval Prediction -- Complete ML Pipeline Training Script

Run from project root:
    python train_model.py

Outputs:
    model/loan_model.pkl  -- best sklearn Pipeline (imputer -> engineer -> scaler -> clf)

Feature order in the Pipeline matches Flask app.py exactly:
    Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount,
    Loan_Amount_Term, Credit_History, Property_Area
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# LoanFeatureEngineer lives in utils.py so that pickle stores
# the class as 'utils.LoanFeatureEngineer', making it importable
# when Flask loads the model (no changes needed in app.py).
from utils import LoanFeatureEngineer

# -- Constants ----------------------------------------------------------------
DATASET_PATH = "dataset/loan_data.csv"
MODEL_PATH   = "model/loan_model.pkl"
RANDOM_STATE = 42
N_SPLITS     = 5
TEST_SIZE    = 0.20
N_ITER       = 40          # RandomizedSearchCV iterations per model

# Feature order MUST match Flask app.py request.form extraction order exactly
FEATURE_COLS = [
    "Gender",           # 0  -- 1=Male, 0=Female
    "Married",          # 1  -- 1=Yes, 0=No
    "Dependents",       # 2  -- 0/1/2/3
    "Education",        # 3  -- 0=Graduate, 1=Not Graduate
    "Self_Employed",    # 4  -- 1=Yes, 0=No
    "ApplicantIncome",  # 5  -- raw Rs
    "CoapplicantIncome",# 6  -- raw Rs
    "LoanAmount",       # 7  -- Rs thousands
    "Loan_Amount_Term", # 8  -- months
    "Credit_History",   # 9  -- 1=Good, 0=Bad/None
    "Property_Area",    # 10 -- 0=Rural, 1=Semiurban, 2=Urban
]

# -- Helpers ------------------------------------------------------------------

def header(text: str) -> None:
    print(f"\n{'='*68}\n  {text}\n{'='*68}")

def subheader(text: str) -> None:
    print(f"\n  {'-'*64}\n  {text}\n  {'-'*64}")


def load_and_encode(path: str):
    """
    Load CSV and encode categoricals using the SAME mapping that the
    Flask form uses, so training features are identical to inference features.
    Returns X (DataFrame, 11 cols) and y (Series, binary).
    """
    df = pd.read_csv(path)
    df.drop(columns=["Loan_ID"], inplace=True)

    # -- Categorical encoding (mirrors the HTML <select> option values) ------
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    # NaN -> left as NaN -> imputed to median during pipeline fit

    df["Married"] = df["Married"].map({"Yes": 1, "No": 0})

    df["Dependents"] = df["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3})
    # "3+" -> 3  (HTML sends "3" for 3+ dependents option)

    df["Education"] = df["Education"].map({"Graduate": 0, "Not Graduate": 1})

    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})

    df["Property_Area"] = df["Property_Area"].map(
        {"Rural": 0, "Semiurban": 1, "Urban": 2}
    )

    # Target
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df[FEATURE_COLS]
    return X, y


def build_pipeline(classifier) -> Pipeline:
    """
    Wraps any classifier in the standard preprocessing Pipeline:
        1. SimpleImputer (median)      -- handles missing values in CSV
        2. LoanFeatureEngineer         -- adds TotalIncome, EMI, Balance_Income
        3. StandardScaler              -- normalises all 14 features
        4. classifier
    """
    return Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("engineer", LoanFeatureEngineer()),
        ("scaler",   StandardScaler()),
        ("model",    classifier),
    ])


# -- Model registry -----------------------------------------------------------

def get_candidates() -> dict:
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    # Optional heavy-hitters
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )
        print("  [OK] XGBoost found")
    except ImportError:
        print("  [!]  XGBoost not installed -- skipping")

    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        print("  [OK] LightGBM found")
    except ImportError:
        print("  [!]  LightGBM not installed -- skipping")

    return candidates


# -- Hyperparameter search spaces ---------------------------------------------

PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C":      [0.001, 0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "liblinear"],
        "model__penalty":["l2"],
    },
    "Decision Tree": {
        "model__max_depth":        [3, 5, 7, 10, None],
        "model__min_samples_split":[2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__criterion":        ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators":     [100, 200, 400],
        "model__max_depth":        [None, 5, 10, 15, 20],
        "model__min_samples_split":[2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features":     ["sqrt", "log2"],
        "model__bootstrap":        [True, False],
    },
    "Extra Trees": {
        "model__n_estimators":     [100, 200, 400],
        "model__max_depth":        [None, 5, 10, 15],
        "model__min_samples_split":[2, 5, 10],
        "model__max_features":     ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "model__n_estimators":     [100, 200, 400],
        "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":        [2, 3, 4, 5],
        "model__subsample":        [0.7, 0.8, 0.9, 1.0],
        "model__min_samples_split":[2, 5, 10],
    },
    "AdaBoost": {
        "model__n_estimators":  [50, 100, 200, 300],
        "model__learning_rate": [0.3, 0.5, 0.8, 1.0, 1.5],
    },
    "XGBoost": {
        "model__n_estimators":      [100, 200, 400],
        "model__learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":         [3, 4, 5, 6],
        "model__subsample":         [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "model__gamma":             [0, 0.1, 0.2],
        "model__reg_alpha":         [0, 0.1, 0.5],
    },
    "LightGBM": {
        "model__n_estimators":  [100, 200, 400],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth":     [-1, 4, 6, 8],
        "model__num_leaves":    [15, 31, 63, 127],
        "model__reg_alpha":     [0, 0.1, 0.5],
        "model__reg_lambda":    [0, 0.1, 0.5],
    },
}


# -- Main training routine -----------------------------------------------------

def main():
    header("Loan Approval Prediction -- ML Pipeline Training")

    # -- 1. Load & encode ------------------------------------------------------
    print(f"\n  Loading: {DATASET_PATH}")
    X, y = load_and_encode(DATASET_PATH)

    print(f"  Dataset  : {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Approved : {y.sum():>4} ({y.mean()*100:.1f}%)")
    print(f"  Rejected : {(1-y).sum():>4} ({(1-y.mean())*100:.1f}%)")
    print(f"\n  Missing values per feature:")
    for col, n in X.isnull().sum().items():
        if n: print(f"    {col:<22} {n:>3} missing")

    # -- 2. Train / Test split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)} rows   Test: {len(X_test)} rows")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # -- 3. Baseline cross-validation ------------------------------------------
    subheader(f"Phase 1 -- {N_SPLITS}-Fold Stratified Cross-Validation (all models)")
    candidates = get_candidates()

    print(f"  {'Model':<22}  {'ROC-AUC':>9}  {'Accuracy':>9}  {'F1':>7}  {'Std':>7}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}")

    cv_scores: dict[str, dict] = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        roc  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc",  n_jobs=-1)
        acc  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        f1   = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1",       n_jobs=-1)
        cv_scores[name] = {
            "roc": roc.mean(), "acc": acc.mean(), "f1": f1.mean(),
            "std": roc.std(),  "clf": clf,
        }
        print(f"  {name:<22}  {roc.mean():>9.4f}  {acc.mean():>9.4f}  {f1.mean():>7.4f}  {roc.std():>7.4f}")

    # -- 4. Select top 3 by ROC-AUC for tuning --------------------------------
    ranked = sorted(cv_scores.items(), key=lambda x: x[1]["roc"], reverse=True)
    top3   = ranked[:3]

    subheader(f"Phase 2 -- RandomizedSearchCV (n_iter={N_ITER}) on top 3")
    print(f"  Tuning: {[n for n, _ in top3]}\n")

    best_name  = None
    best_score = -np.inf
    best_pipe  = None

    for name, meta in top3:
        clf    = meta["clf"]
        pipe   = build_pipeline(clf)
        params = PARAM_GRIDS.get(name, {})

        if params:
            rscv = RandomizedSearchCV(
                pipe, params,
                n_iter=N_ITER, cv=skf, scoring="roc_auc",
                n_jobs=-1, random_state=RANDOM_STATE, refit=True,
            )
            rscv.fit(X_train, y_train)
            score       = rscv.best_score_
            fitted_pipe = rscv.best_estimator_
            best_params = rscv.best_params_
        else:
            pipe.fit(X_train, y_train)
            score = cross_val_score(
                pipe, X_train, y_train, cv=skf, scoring="roc_auc"
            ).mean()
            fitted_pipe = pipe
            best_params = {}

        indicator = "  * BEST" if score > best_score else ""
        print(f"  {name:<22}  tuned ROC-AUC: {score:.4f}{indicator}")
        if best_params:
            for k, v in sorted(best_params.items()):
                print(f"    {k.replace('model__',''):<26} = {v}")

        if score > best_score:
            best_score = score
            best_name  = name
            best_pipe  = fitted_pipe

    # -- 5. Final evaluation on held-out test set ------------------------------
    header(f"Best Model: {best_name}  (tuned CV ROC-AUC = {best_score:.4f})")

    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    print(f"\n  +-----------------------------------------------------+")
    print(f"  |           Test-Set Performance Summary              |")
    print(f"  +-----------------------------------------------------+")
    print(f"  |  Accuracy   : {accuracy_score(y_test, y_pred):.4f}                               |")
    print(f"  |  ROC AUC    : {roc_auc_score(y_test, y_proba):.4f}                               |")
    print(f"  |  F1 Score   : {f1_score(y_test, y_pred):.4f}                               |")
    print(f"  |  Precision  : {precision_score(y_test, y_pred):.4f}                               |")
    print(f"  |  Recall     : {recall_score(y_test, y_pred):.4f}                               |")
    print(f"  +-----------------------------------------------------+")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"  {'':20}  Predicted Rejected  Predicted Approved")
    print(f"  {'Actual Rejected':20}  {tn:18d}  {fp:18d}")
    print(f"  {'Actual Approved':20}  {fn:18d}  {tp:18d}")

    # -- 6. Why this model? ----------------------------------------------------
    reasons = {
        "Random Forest":       (
            "handles non-linear feature interactions and is robust to "
            "outliers in income data without aggressive scaling."
        ),
        "Extra Trees":         (
            "uses random splits which reduces overfitting on small datasets "
            "and is faster than Random Forest."
        ),
        "Gradient Boosting":   (
            "sequential boosting corrects residual errors, ideal for "
            "tabular data with mixed feature types."
        ),
        "XGBoost":             (
            "regularised gradient boosting with column subsampling gives "
            "best bias-variance trade-off on small tabular datasets."
        ),
        "LightGBM":            (
            "histogram-based gradient boosting is fastest and matches "
            "XGBoost accuracy with better handling of categorical-like features."
        ),
        "Logistic Regression": (
            "linear decision boundary is well-calibrated after StandardScaler "
            "and benefits from the engineered log-income feature."
        ),
        "Decision Tree":       (
            "single tree with tuned depth provides interpretable, fast "
            "predictions suitable for this dataset size."
        ),
        "AdaBoost":            (
            "ensemble of weak learners converges well on the class-imbalanced "
            "approval dataset."
        ),
    }
    reason = reasons.get(best_name, "it achieved the highest tuned cross-validation ROC-AUC.")
    print(f"\n  Why {best_name}?\n  -> It was selected because {reason}")

    print(f"\n  Key pipeline improvements over baseline Logistic Regression:")
    print(f"    * SimpleImputer(median)   -- no more NaN crashes at inference time")
    print(f"    * LoanFeatureEngineer     -- TotalIncome (log), EMI, Balance_Income")
    print(f"    * StandardScaler          -- income outliers no longer dominate")
    print(f"    * StratifiedKFold(n=5)    -- stable estimate, no lucky split variance")
    print(f"    * RandomizedSearchCV      -- tuned hyperparameters, not defaults")

    # -- 7. Save ---------------------------------------------------------------
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH, compress=3)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n  [OK] Saved -> {MODEL_PATH}  ({size_kb:.1f} KB)")
    print(f"  Pipeline: Imputer -> LoanFeatureEngineer -> Scaler -> {best_name}")
    print(f"\n  Restart Flask and the new model is live -- no app.py changes needed.")
    header("Training complete")


if __name__ == "__main__":
    main()

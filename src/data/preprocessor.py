"""
Feature engineering et pipeline de préparation sklearn.
Toutes les transformations sont fitées sur le train set uniquement
pour éviter le data leakage.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


TARGET = "churn"

DROP_COLS = ["customer_id", "city", "country"]

CATEGORICAL_COLS = [
    "gender", "customer_segment", "signup_channel",
    "contract_type", "payment_method",
    "complaint_type", "survey_response",
]

NUMERICAL_COLS = [
    "age", "tenure_months", "monthly_logins", "weekly_active_days",
    "avg_session_time", "features_used", "usage_growth_rate",
    "last_login_days_ago", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_resolution_time",
    "csat_score", "escalations", "email_open_rate",
    "marketing_click_rate", "nps_score", "referral_count",
    # features engineered (ajoutées plus bas)
    "tickets_per_month", "revenue_per_month",
    "engagement_score", "risk_score",
    "discount_applied_num", "price_increase_num",
]



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des variables métier dérivées."""
    df = df.copy()

    # Éviter division par zéro
    tenure_safe = df["tenure_months"].replace(0, 1)

    # Intensité du support
    df["tickets_per_month"] = df["support_tickets"] / tenure_safe

    # Densité de revenu
    df["revenue_per_month"] = df["total_revenue"] / tenure_safe

    # Score d'engagement composite
    df["engagement_score"] = (
        df["monthly_logins"] / df["monthly_logins"].max() * 0.4
        + df["weekly_active_days"] / df["weekly_active_days"].max() * 0.4
        + df["avg_session_time"] / df["avg_session_time"].max() * 0.2
    )

    # Score de risque métier (heuristique)
    df["risk_score"] = (
        df["payment_failures"].clip(0, 5) / 5 * 0.35
        + df["support_tickets"].clip(0, 10) / 10 * 0.25
        + (100 - df["nps_score"].clip(-100, 100)) / 200 * 0.20
        + df["last_login_days_ago"].clip(0, 90) / 90 * 0.20
    )

    # Encodage binaire
    df["discount_applied_num"] = (df["discount_applied"] == "Yes").astype(int)
    df["price_increase_num"] = (df["price_increase_last_3m"] == "Yes").astype(int)

    return df


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Construit le ColumnTransformer.
    ⚠️ À fitter uniquement sur le train set.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline complet :
    1. Feature engineering
    2. Split stratifié train/test
    3. Fit preprocessor sur train uniquement
    4. Transform train et test

    Retourne : X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    # 1. Feature engineering
    df = engineer_features(df)

    # 2. Séparer X et y
    cols_to_drop = DROP_COLS + [TARGET, "discount_applied", "price_increase_last_3m"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[TARGET]

    # 3. Colonnes effectives
    num_cols = [c for c in NUMERICAL_COLS if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    # 4. Split stratifié (préserve le ratio de churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Split : Train={X_train.shape[0]:,} | Test={X_test.shape[0]:,}")
    print(f"   Churn rate train : {y_train.mean():.3f} | test : {y_test.mean():.3f}")

    # 5. Fit sur train uniquement — jamais sur test !
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # 6. Noms des features pour SHAP / importance
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names

    print(f"   Features après encoding : {len(feature_names)}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    """Sauvegarde le preprocessor."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"💾 Preprocessor sauvegardé : {path}")


if __name__ == "__main__":
    from loader import load_data

    df = load_data("data/raw/customer_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    save_preprocessor(preprocessor)
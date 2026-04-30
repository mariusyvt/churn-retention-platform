"""
Script EDA textuel.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

from src.data.loader import audit_dataset, load_data


def print_target(df: pd.DataFrame) -> None:
    print("\n[1/5] Distribution de la cible")
    counts = df["churn"].value_counts().sort_index()
    total = len(df)
    for label, name in [(0, "No Churn"), (1, "Churn")]:
        value = int(counts.get(label, 0))
        pct = (value / total * 100) if total else 0
        print(f"- {name} ({label}): {value:,} ({pct:.1f}%)")


def print_numerical_summary(df: pd.DataFrame) -> None:
    print("\n[2/5] Resume des variables numeriques")
    num_cols = [
        "age",
        "tenure_months",
        "monthly_logins",
        "monthly_fee",
        "total_revenue",
        "payment_failures",
        "support_tickets",
        "nps_score",
        "csat_score",
        "last_login_days_ago",
    ]
    available = [c for c in num_cols if c in df.columns]
    grouped_mean = df.groupby("churn")[available].mean().T
    grouped_mean.columns = ["no_churn_mean", "churn_mean"]
    grouped_mean["delta"] = grouped_mean["churn_mean"] - grouped_mean["no_churn_mean"]
    print(grouped_mean.round(3).to_string())


def print_risk_factors(df: pd.DataFrame) -> None:
    print("\n[3/5] Facteurs de risque (moyennes par classe)")
    risk_cols = [
        "payment_failures",
        "support_tickets",
        "nps_score",
        "last_login_days_ago",
        "usage_growth_rate",
        "csat_score",
    ]
    available = [c for c in risk_cols if c in df.columns]
    table = df.groupby("churn")[available].mean().T
    table.columns = ["no_churn_mean", "churn_mean"]
    table["churn_no_churn_ratio"] = table["churn_mean"] / table["no_churn_mean"].replace(0, np.nan)
    print(table.round(3).to_string())


def print_categorical_churn(df: pd.DataFrame) -> None:
    print("\n[4/5] Taux de churn par variables categorielles")
    cat_cols = ["contract_type", "customer_segment", "payment_method", "gender", "signup_channel"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        print(f"\n- {col}:")
        churn_rate = (df.groupby(col)["churn"].mean() * 100).sort_values(ascending=False)
        print(churn_rate.round(2).to_string())


def print_top_correlations(df: pd.DataFrame) -> None:
    print("\n[5/5] Top correlations avec la cible")
    numeric_df = df.select_dtypes(include=np.number)
    if "churn" not in numeric_df.columns:
        print("- Colonne 'churn' absente des variables numeriques.")
        return
    corr = numeric_df.corr(numeric_only=True)["churn"].drop(labels=["churn"]).abs().sort_values(ascending=False)
    print(corr.head(10).round(3).to_string())


if __name__ == "__main__":
    print("=" * 50)
    print("EDA - Customer Churn Dataset")
    print("=" * 50)

    df = load_data("data/raw/customer_churn.csv")
    audit_dataset(df)
    print_target(df)
    print_numerical_summary(df)
    print_risk_factors(df)
    print_categorical_churn(df)
    print_top_correlations(df)

    print("\nEDA terminee.")

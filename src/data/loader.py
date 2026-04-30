"""
Chargement et audit du dataset customer_churn.csv
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV et retourne un DataFrame."""
    df = pd.read_csv(path)
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Retourne un audit complet du dataset."""

    print("\n" + "=" * 50)
    print("AUDIT DU DATASET")
    print("=" * 50)

    # Shape
    print(f"\n📐 Shape : {df.shape}")

    # Types
    print(f"\n📋 Types de colonnes :")
    print(df.dtypes)

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    print(f"\n❓ Valeurs manquantes :")
    print(missing[missing > 0] if missing.sum() > 0 else "  → Aucune valeur manquante ✅")

    # Doublons
    dupes = df.duplicated().sum()
    print(f"\n🔁 Doublons : {dupes}")

    # Distribution cible
    target_counts = df["churn"].value_counts()
    ratio = target_counts[0] / target_counts[1]
    print(f"\n🎯 Distribution cible :")
    print(f"  No Churn (0) : {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"  Churn    (1) : {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
    print(f"  Ratio déséquilibre : {ratio:.1f}:1")

    print("\n" + "=" * 50)

    return {
        "shape": df.shape,
        "missing_values": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "duplicates": dupes,
        "target_distribution": target_counts.to_dict(),
        "imbalance_ratio": round(ratio, 2),
    }


if __name__ == "__main__":
    df = load_data("data/raw/customer_churn.csv")
    audit = audit_dataset(df)
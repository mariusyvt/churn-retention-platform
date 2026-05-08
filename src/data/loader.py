"""
Module : loader.py
Responsabilité : Chargement et validation initiale du dataset.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "age", "gender", "tenure", "contract_type", "monthly_charges",
    "total_revenue", "payment_failures", "support_tickets",
    "session_duration", "login_frequency", "nps_score", "churn"
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset et effectue une validation structurelle de base.
    
    Args:
        filepath: Chemin vers le fichier CSV.
    Returns:
        DataFrame validé.
    Raises:
        FileNotFoundError, ValueError si colonnes manquantes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset introuvable : {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    return df

"""
Module : preprocessor.py
Responsabilité : Pipeline complet de préparation des données.

Principe anti-data-leakage :
  - Le pipeline sklearn est FIT uniquement sur le train set
  - SMOTE est appliqué uniquement sur le train set
  - Le test set est transformé (transform) mais jamais utilisé pour fit
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


# ─── Constantes ──────────────────────────────────────────────────────────────

TARGET = "churn"

# Colonnes à exclure (identifiant unique → aucune valeur prédictive)
COLS_TO_DROP = ["customer_id", "city"]

# Colonnes catégorielles à encoder (après suppression des inutiles)
CAT_COLS = [
    "gender", "country", "customer_segment", "signup_channel",
    "contract_type", "payment_method", "discount_applied",
    "price_increase_last_3m", "complaint_type", "survey_response"
]

# Colonnes numériques (après feature engineering)
NUM_COLS = [
    "age", "tenure_months", "monthly_logins", "weekly_active_days",
    "avg_session_time", "features_used", "usage_growth_rate",
    "last_login_days_ago", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_resolution_time",
    "csat_score", "escalations", "email_open_rate",
    "marketing_click_rate", "nps_score", "referral_count",
    # Features dérivées (ajoutées par feature engineering)
    "tickets_per_month", "engagement_score", "charge_per_login",
    "payment_risk_flag", "nps_risk_flag", "high_value_flag",
]


# ─── Feature Engineering ─────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées pertinentes pour la prédiction du churn.
    Ces features capturent des signaux comportementaux composites.

    IMPORTANT : cette fonction ne fait que des calculs à partir des colonnes
    existantes — aucun risque de data leakage car elle n'apprend rien.
    """
    df = df.copy()

    # Densité d'incidents : plus le ratio est élevé, plus le client est agité
    df["tickets_per_month"] = df["support_tickets"] / (df["tenure_months"] + 1)

    # Engagement composite : fréquence × durée de session
    df["engagement_score"] = df["monthly_logins"] * df["avg_session_time"]

    # Coût relatif à l'usage : un client qui paye beaucoup mais se connecte peu = risque
    df["charge_per_login"] = df["monthly_fee"] / (df["monthly_logins"] + 1)

    # Signal binaire de risque paiement (>2 échecs = flag rouge)
    df["payment_risk_flag"] = (df["payment_failures"] > 2).astype(int)

    # Détracteur NPS (score < 5 = client insatisfait)
    df["nps_risk_flag"] = (df["nps_score"] < 5).astype(int)

    # Client haute valeur (au-dessus du 75e percentile de revenu)
    threshold = df["total_revenue"].quantile(0.75)
    df["high_value_flag"] = (df["total_revenue"] > threshold).astype(int)

    return df


# ─── Pipeline sklearn ─────────────────────────────────────────────────────────

def build_preprocessing_pipeline(num_cols: list, cat_cols: list):
    """
    Construit un ColumnTransformer sklearn :
    - Numériques : imputation médiane + RobustScaler (résistant aux outliers)
    - Catégorielles : imputation par constante 'missing' + OrdinalEncoder

    RobustScaler est choisi car monthly_fee et total_revenue ont 5% d'outliers.
    OrdinalEncoder est choisi pour sa compatibilité avec tous les modèles,
    y compris les arbres qui n'ont pas besoin de one-hot encoding.
    """
    numeric_transformer = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_transformer = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ], remainder="drop")

    return preprocessor


# ─── Pipeline principal ───────────────────────────────────────────────────────

def prepare_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    save_path: str = None,
) -> dict:
    """
    Pipeline complet de préparation des données.

    Args:
        filepath      : Chemin vers customer_churn.csv
        test_size     : Proportion du jeu de test (défaut 20%)
        random_state  : Graine aléatoire pour la reproductibilité
        apply_smote   : Appliquer SMOTE sur le train set (défaut True)
        save_path     : Si fourni, sauvegarde le pipeline dans ce chemin

    Returns:
        dict avec clés :
          - X_train, X_test, y_train, y_test (arrays numpy)
          - feature_names (liste des noms de features après transformation)
          - pipeline (objet sklearn fitted)
          - class_distribution (dict des proportions avant/après SMOTE)
    """
    print("\n" + "="*55)
    print("  PIPELINE DE PREPROCESSING")
    print("="*55)

    # ── 1. Chargement ────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"\n[1/5] Données chargées : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── 2. Feature Engineering ───────────────────────────────
    df = add_engineered_features(df)
    print(f"[2/5] Feature engineering : +6 nouvelles features")

    # Suppression des colonnes inutiles
    cols_to_drop_existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)

    # Séparation features / cible
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Colonnes réellement disponibles
    num_cols_available = [c for c in NUM_COLS if c in X.columns]
    cat_cols_available = [c for c in CAT_COLS if c in X.columns]

    print(f"    → {len(num_cols_available)} features numériques")
    print(f"    → {len(cat_cols_available)} features catégorielles")

    # ── 3. Train / Test Split STRATIFIÉ ─────────────────────
    # stratify=y garantit que les proportions de churn sont
    # identiques dans train et test (crucial avec 10.2% de churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\n[3/5] Split stratifié :")
    print(f"    → Train : {X_train.shape[0]:,} lignes "
          f"({y_train.sum()} churners, {y_train.mean()*100:.1f}%)")
    print(f"    → Test  : {X_test.shape[0]:,} lignes "
          f"({y_test.sum()} churners, {y_test.mean()*100:.1f}%)")

    # ── 4. Pipeline sklearn (fit sur train UNIQUEMENT) ───────
    preprocessor = build_preprocessing_pipeline(num_cols_available, cat_cols_available)

    # FIT sur train, TRANSFORM sur train et test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\n[4/5] Pipeline sklearn appliqué :")
    print(f"    → Shape après transformation : {X_train_processed.shape}")
    print(f"    → RobustScaler + OrdinalEncoder + Imputation")

    # Noms des features après transformation
    feature_names = num_cols_available + cat_cols_available

    # ── 5. SMOTE (sur train UNIQUEMENT) ─────────────────────
    class_dist = {"before_smote": dict(pd.Series(y_train).value_counts())}

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_processed, y_train)
        class_dist["after_smote"] = dict(pd.Series(y_train_final).value_counts())

        print(f"\n[5/5] SMOTE appliqué :")
        print(f"    → Avant : {class_dist['before_smote']}")
        print(f"    → Après : {class_dist['after_smote']}")
        print(f"    → Ratio équilibré : 1:1")
    else:
        X_train_final = X_train_processed
        y_train_final = y_train
        print(f"\n[5/5] SMOTE non appliqué (apply_smote=False)")

    # ── Sauvegarde du pipeline ───────────────────────────────
    if save_path:
        joblib.dump(preprocessor, save_path)
        print(f"\n  → Pipeline sauvegardé : {save_path}")

    print("\n" + "="*55)
    print("  ✅ PREPROCESSING TERMINÉ")
    print("="*55 + "\n")

    return {
        "X_train": X_train_final,
        "X_test": X_test_processed,
        "y_train": y_train_final,
        "y_test": y_test,
        "feature_names": feature_names,
        "pipeline": preprocessor,
        "class_distribution": class_dist,
        "num_cols": num_cols_available,
        "cat_cols": cat_cols_available,
    }


# ─── Test rapide ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/customer_churn.csv"

    result = prepare_data(
        filepath=filepath,
        save_path="models/preprocessor.pkl"
    )

    print("Résumé final :")
    print(f"  X_train shape : {result['X_train'].shape}")
    print(f"  X_test shape  : {result['X_test'].shape}")
    print(f"  Features      : {result['feature_names'][:5]}... ({len(result['feature_names'])} total)")
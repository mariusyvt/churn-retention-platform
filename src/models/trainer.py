"""
Module : trainer.py
Responsabilité : Entraînement et comparaison de 4 modèles de classification.

Modèles comparés :
  1. Régression Logistique  — baseline interprétable
  2. Random Forest          — robuste, non-linéaire
  3. XGBoost                — gradient boosting, haute performance
  4. MLP (Deep Learning)    — réseau neuronal multicouches

Stratégie anti-déséquilibre :
  - SMOTE appliqué en amont (dans preprocessor.py)
  - class_weight='balanced' en complément sur les modèles qui le supportent
  - Validation croisée stratifiée 5-fold sur le train set
"""

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import prepare_data


# ─── Définition des modèles ───────────────────────────────────────────────────

def get_models() -> dict:
    """
    Retourne les 4 modèles avec leurs hyperparamètres de départ.

    Choix justifiés :
    - LogReg     : baseline linéaire, très interprétable, rapide
    - RandomForest: capture la non-linéarité, robuste au bruit
    - XGBoost    : souvent le meilleur sur données tabulaires
    - MLP        : deep learning, capture des interactions complexes
                   mais nécessite plus de données et de tuning
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # complément au SMOTE
            random_state=42,
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,                 # utilise tous les cœurs CPU
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,        # déjà rééquilibré par SMOTE
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "MLP (Deep Learning)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 couches cachées
            activation="relu",
            solver="adam",
            alpha=0.001,               # régularisation L2 (anti-overfitting)
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,       # arrêt si pas d'amélioration
            validation_fraction=0.1,
            random_state=42,
        ),
    }


# ─── Évaluation d'un modèle ──────────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    cv_folds: int = 5,
) -> dict:
    """
    Entraîne un modèle et calcule toutes les métriques nécessaires.

    Métriques calculées :
    - Recall    : priorité absolue (minimiser les faux négatifs = churners manqués)
    - F1-Score  : équilibre précision/rappel
    - ROC-AUC   : discrimination globale du modèle
    - Precision : éviter trop de faux positifs (coût des actions inutiles)
    - CV scores : robustesse via validation croisée stratifiée 5-fold
    """
    print(f"\n  → Entraînement : {model_name}...")
    start_time = time.time()

    # Entraînement sur le train set complet (avec SMOTE)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prédictions sur le test set (jamais vu)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Métriques sur le test set ────────────────────────────
    results = {
        "model_name": model_name,
        "train_time_sec": round(train_time, 2),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # ── Validation croisée stratifiée ───────────────────────
    # Effectuée sur le train set pour mesurer la stabilité du modèle
    print(f"     Cross-validation {cv_folds}-fold stratifiée...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["recall", "f1", "roc_auc"],
        n_jobs=-1,
    )

    results["cv_recall_mean"] = round(cv_scores["test_recall"].mean(), 4)
    results["cv_recall_std"] = round(cv_scores["test_recall"].std(), 4)
    results["cv_f1_mean"] = round(cv_scores["test_f1"].mean(), 4)
    results["cv_f1_std"] = round(cv_scores["test_f1"].std(), 4)
    results["cv_roc_auc_mean"] = round(cv_scores["test_roc_auc"].mean(), 4)
    results["cv_roc_auc_std"] = round(cv_scores["test_roc_auc"].std(), 4)

    # Affichage résumé
    print(f"     ✓ Recall={results['test_recall']:.3f} | "
          f"F1={results['test_f1']:.3f} | "
          f"ROC-AUC={results['test_roc_auc']:.3f} | "
          f"Temps={train_time:.1f}s")
    print(f"     ✓ CV Recall={results['cv_recall_mean']:.3f}±{results['cv_recall_std']:.3f}")

    return results


# ─── Pipeline d'entraînement complet ─────────────────────────────────────────

def train_all_models(
    data_filepath: str,
    output_dir: str = "models/",
) -> list:
    """
    Pipeline complet : preprocessing → entraînement → évaluation → sauvegarde.

    Args:
        data_filepath : Chemin vers customer_churn.csv
        output_dir    : Dossier de sauvegarde des modèles (.pkl)

    Returns:
        Liste de dicts de résultats pour chaque modèle
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═"*55)
    print("  ENTRAÎNEMENT MULTI-MODÈLES — CHURN PREDICTION")
    print("═"*55)

    # ── Preprocessing ────────────────────────────────────────
    data = prepare_data(
        filepath=data_filepath,
        save_path=os.path.join(output_dir, "preprocessor.pkl"),
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"\nDonnées prêtes :")
    print(f"  X_train : {X_train.shape} (après SMOTE)")
    print(f"  X_test  : {X_test.shape}")

    # ── Entraînement ─────────────────────────────────────────
    models = get_models()
    all_results = []

    print(f"\n{'─'*55}")
    print(f"  Entraînement de {len(models)} modèles...")
    print(f"{'─'*55}")

    for model_name, model in models.items():
        results = evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )
        all_results.append(results)

        # Sauvegarde du modèle entraîné
        model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"     → Modèle sauvegardé : {model_path}")

    # ── Tableau comparatif ───────────────────────────────────
    print_comparison_table(all_results)

    # Sauvegarde des résultats
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → Résultats sauvegardés : {results_path}")

    # Identification du meilleur modèle (par Recall, priorité métier)
    best = max(all_results, key=lambda x: x["test_recall"])
    print(f"\n  🏆 Meilleur modèle (Recall) : {best['model_name']}")
    print(f"     Recall={best['test_recall']} | F1={best['test_f1']} | ROC-AUC={best['test_roc_auc']}")

    return all_results


def print_comparison_table(results: list):
    """Affiche un tableau comparatif lisible dans le terminal."""
    print("\n" + "═"*75)
    print("  TABLEAU COMPARATIF DES MODÈLES")
    print("═"*75)
    header = f"{'Modèle':<28} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV-Recall':>10} {'Temps':>7}"
    print(header)
    print("─"*75)

    for r in results:
        line = (
            f"{r['model_name']:<28} "
            f"{r['test_recall']:>8.3f} "
            f"{r['test_f1']:>8.3f} "
            f"{r['test_roc_auc']:>9.3f} "
            f"{r['cv_recall_mean']:>7.3f}±{r['cv_recall_std']:.2f} "
            f"{r['train_time_sec']:>6.1f}s"
        )
        print(line)

    print("═"*75)
    print("  Note : Recall = métrique prioritaire (minimiser les churners manqués)")
    print("═"*75 + "\n")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/customer_churn.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models/"

    results = train_all_models(
        data_filepath=data_path,
        output_dir=output_dir,
    )
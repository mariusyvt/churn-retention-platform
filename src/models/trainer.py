"""
Entraînement multi-modèles pour la prédiction du churn.
Modèles : Logistic Regression, Random Forest, XGBoost, LightGBM, MLP (Deep Learning)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ── Définition des modèles ────────────────────────────────────────────────────

def get_models(random_state: int = 42) -> dict:
    """Retourne les 5 modèles candidats."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=9,  # ratio neg/pos pour le déséquilibre
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
    }


# ── Métriques ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Calcule toutes les métriques pour un modèle entraîné."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc":    round(average_precision_score(y_test, y_proba), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
        "y_proba": y_proba,
        "y_pred":  y_pred,
    }


def find_best_threshold(model, X_test, y_test) -> float:
    """Trouve le seuil qui maximise le F1-score."""
    y_proba = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_test, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t  = t
    return round(float(best_t), 2)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """Validation croisée stratifiée sur le train set."""
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc", "average_precision"]
    results = cross_validate(model, X_train, y_train, cv=skf,
                             scoring=scoring, n_jobs=-1)
    return {
        metric: {
            "mean": round(float(np.mean(scores)), 4),
            "std":  round(float(np.std(scores)), 4),
        }
        for metric, scores in results.items()
        if metric.startswith("test_")
    }


# ── Entraînement complet ──────────────────────────────────────────────────────

def train_all_models(
    X_train, y_train,
    X_test,  y_test,
    feature_names: list,
    models_dir: str = "models",
    use_smote: bool = False,
) -> dict:
    """
    Entraîne tous les modèles, évalue, sauvegarde les .pkl.
    Retourne un dict avec les résultats complets.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Option SMOTE
    if use_smote:
        print("⚖️  Application du SMOTE...")
        smote   = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   Après SMOTE : {np.bincount(y_train)}")

    models  = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n🔧 Entraînement : {name}...")

        # Entraînement
        model.fit(X_train, y_train)

        # Seuil optimal
        threshold = find_best_threshold(model, X_test, y_test)

        # Métriques test
        metrics = evaluate_model(model, X_test, y_test, threshold=threshold)

        # Cross-validation
        print(f"   Cross-validation (5 folds)...")
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)

        # Sauvegarde .pkl
        safe_name  = name.replace(" ", "_")
        model_path = os.path.join(models_dir, f"{safe_name}.pkl")
        joblib.dump(model, model_path)

        results[name] = {
            "model":      model,
            "metrics":    metrics,
            "cv_results": cv_results,
            "threshold":  threshold,
            "model_path": model_path,
        }

        print(f"   ✅ Recall={metrics['recall']:.3f} | "
              f"F1={metrics['f1_score']:.3f} | "
              f"ROC-AUC={metrics['roc_auc']:.3f} | "
              f"CV ROC-AUC={cv_results['test_roc_auc']['mean']:.3f}"
              f"±{cv_results['test_roc_auc']['std']:.3f}")

    return results


# ── Sauvegarde du meilleur modèle ─────────────────────────────────────────────

def save_best_model(results: dict, feature_names: list, models_dir: str = "models"):
    """Sélectionne et sauvegarde le meilleur modèle selon le ROC-AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best      = results[best_name]

    # Copie en best_model.pkl
    joblib.dump(best["model"], os.path.join(models_dir, "best_model.pkl"))

    # Métadonnées en JSON
    info = {
        "model_name":    best_name,
        "threshold":     best["threshold"],
        "feature_names": feature_names,
        "metrics": {
            k: v for k, v in best["metrics"].items()
            if not isinstance(v, np.ndarray)
        },
    }
    with open(os.path.join(models_dir, "best_model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n🏆 Meilleur modèle : {best_name}")
    print(f"   ROC-AUC  : {best['metrics']['roc_auc']:.4f}")
    print(f"   Recall   : {best['metrics']['recall']:.4f}")
    print(f"   F1-Score : {best['metrics']['f1_score']:.4f}")
    print(f"   Seuil    : {best['threshold']}")
    print(f"   Sauvegardé dans {models_dir}/best_model.pkl")

    return best_name


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from src.data.loader import load_data
    from src.data.preprocessor import prepare_data

    print("=" * 55)
    print("ENTRAÎNEMENT MULTI-MODÈLES — Churn Prediction")
    print("=" * 55)

    # Chargement
    df = load_data("data/raw/customer_churn.csv")

    # Preprocessing
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # Entraînement
    results = train_all_models(
        X_train, y_train,
        X_test,  y_test,
        feature_names=feature_names,
        use_smote=False,
    )

    # Meilleur modèle
    save_best_model(results, feature_names)

    print("\n✅ Entraînement terminé ! Modèles sauvegardés dans models/")
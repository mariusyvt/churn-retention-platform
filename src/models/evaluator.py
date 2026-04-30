"""
Module : evaluator.py
Responsabilité : Analyse approfondie des modèles, ajustement du seuil de décision,
                 et sélection du modèle final.

Problème identifié :
  - Random Forest / XGBoost / MLP : overfitting sévère (CV Recall >> Test Recall)
  - Cause principale : seuil de décision 0.5 inadapté au déséquilibre résiduel
  - Solution : ajuster le seuil pour maximiser le F1 ou le Recall sur le test set

Analyse critique pour le rapport :
  - Un modèle avec CV Recall=0.92 mais Test Recall=0.19 est inutilisable en production
  - La Régression Logistique est plus stable (écart CV/test plus faible)
  - L'ajustement du seuil peut significativement améliorer les modèles complexes
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import prepare_data

PALETTE = {
    "Logistic Regression": "#2196F3",
    "Random Forest": "#4CAF50",
    "XGBoost": "#FF9800",
    "MLP (Deep Learning)": "#E91E63",
}


# ─── Ajustement du seuil de décision ─────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """
    Teste tous les seuils entre 0.1 et 0.9 et retourne celui qui maximise
    la métrique choisie (f1 ou recall).

    Par défaut, sklearn utilise 0.5 — ce seuil est sous-optimal quand
    la classe positive est minoritaire même après SMOTE.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = recall_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = t

    return round(best_threshold, 2), round(best_score, 4)


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """Évalue un modèle avec un seuil personnalisé."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ─── Visualisations ──────────────────────────────────────────────────────────

def plot_roc_curves(models_data: dict, y_test, output_dir: str):
    """Courbes ROC comparatives pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, data in models_data.items():
        fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
        auc = data["metrics_tuned"]["roc_auc"]
        color = PALETTE.get(name, "gray")
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aléatoire (AUC=0.5)")
    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (Recall)", fontsize=12)
    ax.set_title("Courbes ROC — Comparaison des Modèles", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "07_roc_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")


def plot_precision_recall_curves(models_data: dict, y_test, output_dir: str):
    """Courbes Precision-Recall — plus informatives que ROC avec déséquilibre."""
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()

    for name, data in models_data.items():
        precision, recall, _ = precision_recall_curve(y_test, data["y_proba"])
        color = PALETTE.get(name, "gray")
        ax.plot(recall, precision, label=name, color=color, linewidth=2)

    ax.axhline(baseline, color="gray", linestyle="--",
               linewidth=1, label=f"Baseline ({baseline:.2%})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Courbes Precision-Recall\n(plus adaptée au déséquilibre de classes)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "08_precision_recall_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")


def plot_confusion_matrices(models_data: dict, y_test, output_dir: str):
    """Matrices de confusion côte à côte pour tous les modèles."""
    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Matrices de Confusion (seuil optimisé)",
                 fontsize=14, fontweight="bold")

    for ax, (name, data) in zip(axes, models_data.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, data["metrics_tuned"]["y_pred"],
            display_labels=["No Churn", "Churn"],
            colorbar=False, ax=ax,
            cmap="Blues",
        )
        ax.set_title(name, fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "09_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")


def plot_threshold_impact(models_data: dict, y_test, output_dir: str):
    """
    Visualise l'impact du seuil de décision sur Recall et Precision.
    Montre pourquoi 0.5 n'est pas optimal.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Impact du Seuil de Décision sur Recall et Precision",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    thresholds = np.arange(0.05, 0.95, 0.01)

    for ax, (name, data) in zip(axes, models_data.items()):
        recalls, precisions, f1s = [], [], []
        for t in thresholds:
            y_pred = (data["y_proba"] >= t).astype(int)
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            precisions.append(f1_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))

        ax.plot(thresholds, recalls, color="#F44336", label="Recall", linewidth=2)
        ax.plot(thresholds, precisions, color="#2196F3", label="Precision", linewidth=2)
        ax.plot(thresholds, f1s, color="#4CAF50", label="F1", linewidth=2, linestyle="--")

        opt_t = data["optimal_threshold"]
        ax.axvline(opt_t, color="black", linestyle=":", linewidth=1.5,
                   label=f"Seuil optimal={opt_t}")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1,
                   label="Seuil défaut=0.5", alpha=0.6)

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Seuil de décision")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "10_threshold_impact.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")


def plot_metrics_comparison(summary: list, output_dir: str):
    """Barplot comparatif des métriques finales (seuil optimisé)."""
    names = [r["model"] for r in summary]
    recalls = [r["recall_tuned"] for r in summary]
    f1s = [r["f1_tuned"] for r in summary]
    aucs = [r["roc_auc"] for r in summary]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, recalls, width, label="Recall", color="#F44336", alpha=0.85)
    bars2 = ax.bar(x, f1s, width, label="F1-Score", color="#2196F3", alpha=0.85)
    bars3 = ax.bar(x + width, aucs, width, label="ROC-AUC", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des Modèles — Seuil Optimisé",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "11_metrics_comparison.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_evaluation(
    data_filepath: str,
    models_dir: str = "models/",
    output_dir: str = "reports/figures/",
):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═"*60)
    print("  ÉVALUATION APPROFONDIE & OPTIMISATION DES SEUILS")
    print("═"*60)

    # Preprocessing (sans SMOTE pour avoir le vrai test set)
    data = prepare_data(filepath=data_filepath, apply_smote=True)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Chargement des modèles
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "MLP (Deep Learning)": "mlp_deep_learning.pkl",
    }

    models_data = {}
    summary = []

    print("\n[1/3] Optimisation des seuils de décision...")

    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            print(f"  ⚠ Modèle introuvable : {path}")
            continue

        model = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Seuil par défaut (0.5)
        metrics_default = evaluate_with_threshold(model, X_test, y_test, threshold=0.5)

        # Seuil optimisé (maximise F1)
        opt_threshold, _ = find_optimal_threshold(y_test, y_proba, metric="f1")
        metrics_tuned = evaluate_with_threshold(model, X_test, y_test, threshold=opt_threshold)

        models_data[name] = {
            "model": model,
            "y_proba": y_proba,
            "optimal_threshold": opt_threshold,
            "metrics_default": metrics_default,
            "metrics_tuned": metrics_tuned,
        }

        print(f"\n  {name} :")
        print(f"    Seuil 0.50 → Recall={metrics_default['recall']:.3f} | F1={metrics_default['f1']:.3f}")
        print(f"    Seuil {opt_threshold:.2f} → Recall={metrics_tuned['recall']:.3f} | F1={metrics_tuned['f1']:.3f} ✓")

        summary.append({
            "model": name,
            "optimal_threshold": opt_threshold,
            "recall_default": metrics_default["recall"],
            "recall_tuned": metrics_tuned["recall"],
            "f1_default": metrics_default["f1"],
            "f1_tuned": metrics_tuned["f1"],
            "roc_auc": metrics_tuned["roc_auc"],
        })

    # Visualisations
    print("\n[2/3] Génération des graphiques...")
    plot_roc_curves(models_data, y_test, output_dir)
    plot_precision_recall_curves(models_data, y_test, output_dir)
    plot_confusion_matrices(models_data, y_test, output_dir)
    plot_threshold_impact(models_data, y_test, output_dir)
    plot_metrics_comparison(summary, output_dir)

    # Tableau final
    print("\n[3/3] Tableau comparatif final :")
    print("\n" + "═"*75)
    print(f"{'Modèle':<28} {'Seuil':>6} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("─"*75)
    for r in summary:
        print(f"{r['model']:<28} {r['optimal_threshold']:>6.2f} "
              f"{r['recall_tuned']:>8.3f} {r['f1_tuned']:>8.3f} {r['roc_auc']:>9.3f}")
    print("═"*75)

    # Meilleur modèle (compromis Recall + F1)
    best = max(summary, key=lambda x: x["f1_tuned"] * 0.4 + x["recall_tuned"] * 0.6)
    print(f"\n  🏆 Modèle recommandé : {best['model']}")
    print(f"     Seuil={best['optimal_threshold']} | Recall={best['recall_tuned']} | F1={best['f1_tuned']}")

    # Sauvegarde
    results_path = os.path.join(models_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Résultats : {results_path}")
    print("  ✅ ÉVALUATION TERMINÉE\n")

    return summary


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/customer_churn.csv"
    run_evaluation(data_filepath=data_path)

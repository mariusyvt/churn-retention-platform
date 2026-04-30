"""
Module : shap_analysis.py
Responsabilité : Interprétabilité du modèle final (Random Forest) via SHAP.

Pourquoi SHAP ?
  - Feature importance native (Gini) peut être biaisée par les variables
    à haute cardinalité
  - SHAP donne une explication locale (pourquoi CE client est à risque)
    ET globale (quelles variables comptent le plus en général)
  - Indispensable pour convaincre un responsable CRM/marketing

Lectures :
  - Global : quelles features influencent le modèle en général
  - Local  : pourquoi un client précis est classé "Churn"
"""

import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import prepare_data


def run_shap_analysis(
    data_filepath: str,
    model_path: str = "models/random_forest.pkl",
    output_dir: str = "reports/figures/",
    n_samples: int = 500,
):
    """
    Analyse SHAP complète sur le modèle Random Forest.

    Args:
        data_filepath : Chemin vers customer_churn.csv
        model_path    : Chemin vers le modèle sauvegardé (.pkl)
        output_dir    : Dossier de sortie pour les graphiques
        n_samples     : Nombre d'échantillons pour SHAP (500 = bon compromis)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═"*55)
    print("  ANALYSE D'INTERPRÉTABILITÉ — SHAP")
    print("═"*55)

    # ── Chargement données et modèle ─────────────────────────
    data = prepare_data(filepath=data_filepath, apply_smote=False)
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    model = joblib.load(model_path)
    print(f"\n  Modèle chargé : {model_path}")
    print(f"  Features      : {len(feature_names)}")
    print(f"  Échantillons  : {n_samples} (sur {X_test.shape[0]} disponibles)")

    # Sous-échantillon pour accélérer SHAP
    idx = np.random.choice(X_test.shape[0], size=min(n_samples, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]
    y_sample = np.array(y_test)[idx]

    # ── Calcul SHAP ──────────────────────────────────────────
    print("\n[1/4] Calcul des valeurs SHAP (peut prendre 1-2 min)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Pour un classifieur binaire, shap_values est une liste [class0, class1]
    # On prend les valeurs pour la classe 1 (Churn)
    if isinstance(shap_values, list):
        shap_vals_churn = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals_churn = shap_values[:, :, 1]
    else:
        shap_vals_churn = shap_values

    print(f"  ✓ SHAP calculé sur {n_samples} échantillons")

    # ── Graphique 1 : Importance globale (bar plot) ──────────
    print("\n[2/4] Feature importance globale...")
    mean_shap = np.abs(shap_vals_churn).mean(axis=0)
    feature_importance = pd.Series(mean_shap, index=feature_names).sort_values(ascending=True)
    top_features = feature_importance.tail(15)  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#F44336" if v > top_features.median() else "#2196F3"
              for v in top_features.values]
    bars = ax.barh(top_features.index, top_features.values, color=colors, alpha=0.85)
    ax.set_xlabel("Importance SHAP moyenne (|valeur|)", fontsize=12)
    ax.set_title("Top 15 Features — Importance Globale (SHAP)\nRandom Forest — Prédiction Churn",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, top_features.values):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "12_shap_global_importance.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")

    # ── Graphique 2 : Beeswarm plot ──────────────────────────
    print("\n[3/4] Beeswarm plot (impact directionnel)...")
    top_indices = np.argsort(mean_shap)[-15:]
    top_names = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(11, 8))
    shap.summary_plot(
        shap_vals_churn[:, top_indices],
        X_sample[:, top_indices],
        feature_names=top_names,
        show=False,
        plot_type="dot",
        max_display=15,
        alpha=0.6,
    )
    plt.title("Impact Directionnel des Features sur le Churn (SHAP Beeswarm)\n"
              "Rouge = augmente le risque | Bleu = réduit le risque",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "13_shap_beeswarm.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  → {path}")

    # ── Graphique 3 : Explication locale d'un churner ────────
    print("\n[4/4] Explication locale (waterfall plot)...")

    # Trouver un vrai churner dans l'échantillon
    churner_indices = np.where(y_sample == 1)[0]
    if len(churner_indices) > 0:
        idx_churner = churner_indices[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap_single = shap_vals_churn[idx_churner]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        # Top features pour ce client
        top_k = 10
        top_idx = np.argsort(np.abs(shap_single))[-top_k:]
        top_shap = shap_single[top_idx]
        top_feat = [feature_names[i] for i in top_idx]
        top_vals = X_sample[idx_churner, top_idx]

        colors = ["#F44336" if v > 0 else "#2196F3" for v in top_shap]
        labels = [f"{n}\n(val={v:.2f})" for n, v in zip(top_feat, top_vals)]

        sorted_idx = np.argsort(top_shap)
        ax.barh(
            [labels[i] for i in sorted_idx],
            [top_shap[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx],
            alpha=0.85,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution SHAP (impact sur la prédiction)", fontsize=11)
        ax.set_title(
            f"Explication Locale — Pourquoi ce client est classé CHURN ?\n"
            f"(Base: {base_value:.3f} | Rouge = augmente risque | Bleu = réduit risque)",
            fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        path = os.path.join(output_dir, "14_shap_local_explanation.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  → {path}")
    else:
        print("  ⚠ Aucun churner trouvé dans l'échantillon — augmente n_samples")

    # ── Résumé textuel pour le rapport ──────────────────────
    print("\n" + "═"*55)
    print("  TOP 10 FEATURES (pour ton rapport)")
    print("═"*55)
    top10 = feature_importance.tail(10).sort_values(ascending=False)
    for i, (feat, val) in enumerate(top10.items(), 1):
        print(f"  {i:>2}. {feat:<35} SHAP={val:.4f}")

    print("\n  Interprétation métier :")
    print("  • Rouge  = la valeur élevée de cette feature AUGMENTE le risque de churn")
    print("  • Bleu   = la valeur élevée RÉDUIT le risque de churn")
    print("  • L'écart entre les points = variance de l'impact selon les clients")
    print("═"*55 + "\n")

    return feature_importance


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/customer_churn.csv"
    run_shap_analysis(data_filepath=data_path)

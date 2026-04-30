"""
Module : eda.py
Responsabilité : Audit complet du dataset (EDA).
Usage : python eda.py --input data/raw/customer_churn.csv --output reports/figures/

Méthodologie :
  1. Audit structurel (types, valeurs manquantes, doublons)
  2. Analyse de la variable cible (déséquilibre de classes)
  3. Analyse univariée (distributions numériques et catégorielles)
  4. Analyse bivariée (corrélations, relations features → churn)
  5. Détection des outliers (IQR method)
  6. Rapport synthétique (audit_report.json)
"""

import argparse
import json
import os
import warnings
import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Style global ────────────────────────────────────────────────────────────
PALETTE_CHURN = {0: "#2196F3", 1: "#F44336"}  # Bleu = No Churn, Rouge = Churn
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── 1. AUDIT STRUCTUREL ─────────────────────────────────────────────────────

def audit_structure(df: pd.DataFrame) -> dict:
    """
    Retourne un dictionnaire avec le bilan structurel du dataset :
    - shape, types, valeurs manquantes, doublons, cardinalité.
    """
    report = {}

    report["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
    report["duplicates"] = int(df.duplicated().sum())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    }

    report["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    # Cardinalité des colonnes catégorielles
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    report["categorical_cardinality"] = {
        col: int(df[col].nunique()) for col in cat_cols
    }

    # Statistiques descriptives rapides
    report["numeric_summary"] = df.describe().round(2).to_dict()

    logger.info("Audit structurel terminé.")
    logger.info(f"  → {report['shape']['rows']} lignes | {report['shape']['cols']} colonnes")
    logger.info(f"  → {report['duplicates']} doublons")
    logger.info(f"  → {len(report['missing_values'])} colonnes avec valeurs manquantes")

    return report


def print_audit_summary(report: dict):
    """Affiche un résumé lisible de l'audit dans le terminal."""
    print("\n" + "═" * 60)
    print("  RAPPORT D'AUDIT STRUCTUREL")
    print("═" * 60)
    print(f"  Dimensions   : {report['shape']['rows']:,} lignes × {report['shape']['cols']} colonnes")
    print(f"  Doublons     : {report['duplicates']}")

    if report["missing_values"]:
        print("\n  Valeurs manquantes :")
        for col, info in report["missing_values"].items():
            print(f"    • {col:<30} {info['count']:>5} ({info['pct']}%)")
    else:
        print("\n  ✓ Aucune valeur manquante détectée.")

    if report["categorical_cardinality"]:
        print("\n  Cardinalité des variables catégorielles :")
        for col, card in report["categorical_cardinality"].items():
            print(f"    • {col:<30} {card:>3} valeurs uniques")
    print("═" * 60 + "\n")


# ─── 2. ANALYSE DE LA VARIABLE CIBLE ─────────────────────────────────────────

def analyze_target(df: pd.DataFrame, target: str, output_dir: str) -> dict:
    """
    Analyse le déséquilibre de classes de la variable cible.
    Génère un graphique de distribution.
    """
    counts = df[target].value_counts()
    pcts = df[target].value_counts(normalize=True) * 100

    target_report = {
        "class_counts": counts.to_dict(),
        "class_pcts": pcts.round(2).to_dict(),
        "imbalance_ratio": round(counts.max() / counts.min(), 2),
    }

    is_imbalanced = target_report["imbalance_ratio"] > 1.5
    target_report["is_imbalanced"] = is_imbalanced

    logger.info(f"\nVariable cible '{target}' :")
    for cls, cnt in counts.items():
        label = "Churn" if cls == 1 else "No Churn"
        logger.info(f"  → Classe {cls} ({label}) : {cnt:,} ({pcts[cls]:.1f}%)")
    logger.info(f"  → Ratio de déséquilibre : {target_report['imbalance_ratio']:.2f}")
    if is_imbalanced:
        logger.warning("  ⚠ Déséquilibre détecté → Utiliser Recall/F1/ROC-AUC comme métriques principales.")
        logger.warning("  ⚠ Envisager SMOTE, class_weight='balanced', ou ajustement du seuil de décision.")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribution de la Variable Cible : Churn", fontsize=14, fontweight="bold")

    # Barplot
    bars = axes[0].bar(
        [f"No Churn\n({pcts[0]:.1f}%)", f"Churn\n({pcts[1]:.1f}%)"],
        [counts[0], counts[1]],
        color=[PALETTE_CHURN[0], PALETTE_CHURN[1]],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )
    axes[0].set_title("Distribution des classes")
    axes[0].set_ylabel("Nombre de clients")
    for bar, cnt in zip(bars, [counts[0], counts[1]]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{cnt:,}",
            ha="center", fontweight="bold"
        )

    # Pie
    axes[1].pie(
        [counts[0], counts[1]],
        labels=["No Churn", "Churn"],
        colors=[PALETTE_CHURN[0], PALETTE_CHURN[1]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Répartition proportionnelle")

    plt.tight_layout()
    path = os.path.join(output_dir, "01_target_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")

    return target_report


# ─── 3. ANALYSE DES OUTLIERS ─────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, num_cols: list) -> dict:
    """
    Détecte les outliers via la méthode IQR pour chaque colonne numérique.
    Returns: dict {colonne: {"n_outliers": int, "pct": float, "bounds": [low, high]}}
    """
    outlier_report = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = {
            "n_outliers": int(n_out),
            "pct": round(n_out / len(df) * 100, 2),
            "bounds": [round(lower, 2), round(upper, 2)],
        }
    return outlier_report


def plot_outliers(df: pd.DataFrame, num_cols: list, output_dir: str):
    """Génère des boxplots pour visualiser les outliers."""
    n = len(num_cols)
    cols_per_row = 4
    n_rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Détection des Outliers (méthode IQR)", fontsize=14, fontweight="bold")

    for i, col in enumerate(num_cols):
        for churn_val, color in PALETTE_CHURN.items():
            subset = df[df["churn"] == churn_val][col]
            axes[i].boxplot(
                subset,
                positions=[churn_val],
                widths=0.4,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor=color, markersize=3, alpha=0.4),
            )
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["No Churn", "Churn"], fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_outliers_boxplots.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")


# ─── 4. ANALYSE UNIVARIÉE ────────────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame, num_cols: list, output_dir: str):
    """Histogrammes + KDE par classe de churn pour chaque variable numérique."""
    n = len(num_cols)
    cols_per_row = 3
    n_rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    fig.suptitle("Distributions des Variables Numériques par Classe de Churn", fontsize=14, fontweight="bold")

    for i, col in enumerate(num_cols):
        for churn_val, color in PALETTE_CHURN.items():
            label = "Churn" if churn_val == 1 else "No Churn"
            subset = df[df["churn"] == churn_val][col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
            # KDE
            if len(subset) > 1:
                kde_x = np.linspace(subset.min(), subset.max(), 200)
                kde = stats.gaussian_kde(subset)
                axes[i].plot(kde_x, kde(kde_x), color=color, linewidth=2)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Densité", fontsize=9)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "03_numeric_distributions.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")


def plot_categorical_distributions(df: pd.DataFrame, cat_cols: list, output_dir: str):
    """Barplots du taux de churn par modalité pour chaque variable catégorielle."""
    if not cat_cols:
        return

    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Taux de Churn par Variable Catégorielle", fontsize=14, fontweight="bold")

    for i, col in enumerate(cat_cols):
        churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False)
        bars = axes[i].bar(
            churn_rate.index,
            churn_rate.values,
            color=[PALETTE_CHURN[1] if v > df["churn"].mean() else PALETTE_CHURN[0]
                   for v in churn_rate.values],
            edgecolor="white", linewidth=1.5,
        )
        axes[i].axhline(df["churn"].mean(), color="grey", linestyle="--", linewidth=1.5, label="Taux moyen")
        axes[i].set_title(f"Taux de Churn par {col}")
        axes[i].set_ylabel("Taux de Churn")
        axes[i].set_ylim(0, 1)
        axes[i].legend()
        for bar, val in zip(bars, churn_rate.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01, f"{val:.1%}",
                ha="center", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "04_categorical_churn_rates.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")


# ─── 5. ANALYSE BIVARIÉE ────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, num_cols: list, output_dir: str):
    """Heatmap de corrélation + corrélations avec la variable cible."""
    corr_matrix = df[num_cols + ["churn"]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Analyse des Corrélations", fontsize=14, fontweight="bold")

    # Heatmap complète
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=axes[0],
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 8}
    )
    axes[0].set_title("Matrice de Corrélation (Pearson)", fontsize=12)

    # Corrélations avec le churn uniquement
    churn_corr = corr_matrix["churn"].drop("churn").sort_values(ascending=True)
    colors = [PALETTE_CHURN[1] if v > 0 else PALETTE_CHURN[0] for v in churn_corr.values]
    axes[1].barh(churn_corr.index, churn_corr.values, color=colors, edgecolor="white", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Corrélation avec la Variable Cible (Churn)", fontsize=12)
    axes[1].set_xlabel("Coefficient de Pearson")
    for i, (val, label) in enumerate(zip(churn_corr.values, churn_corr.index)):
        axes[1].text(val + 0.005 if val >= 0 else val - 0.005, i,
                     f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_correlation_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")


def plot_key_relationships(df: pd.DataFrame, output_dir: str):
    """
    Visualisation des relations clés business :
    - payment_failures vs churn
    - nps_score vs churn  
    - tenure vs monthly_charges (coloré par churn)
    - support_tickets vs churn
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Relations Clés : Variables Comportementales → Churn", fontsize=14, fontweight="bold")

    # 1. payment_failures distribution
    ax1 = fig.add_subplot(gs[0, 0])
    for churn_val, color in PALETTE_CHURN.items():
        subset = df[df["churn"] == churn_val]["payment_failures"]
        label = "Churn" if churn_val == 1 else "No Churn"
        ax1.hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
    ax1.set_title("Échecs de Paiement → Churn", fontweight="bold")
    ax1.set_xlabel("payment_failures")
    ax1.set_ylabel("Densité")
    ax1.legend()

    # 2. nps_score distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for churn_val, color in PALETTE_CHURN.items():
        subset = df[df["churn"] == churn_val]["nps_score"]
        label = "Churn" if churn_val == 1 else "No Churn"
        ax2.hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
    ax2.set_title("Score NPS → Churn", fontweight="bold")
    ax2.set_xlabel("nps_score")
    ax2.set_ylabel("Densité")
    ax2.legend()

    # 3. Scatter tenure vs monthly_charges
    ax3 = fig.add_subplot(gs[1, 0])
    for churn_val, color in PALETTE_CHURN.items():
        subset = df[df["churn"] == churn_val]
        label = "Churn" if churn_val == 1 else "No Churn"
        ax3.scatter(subset["tenure_months"], subset["monthly_fee"],
                    c=color, alpha=0.3, s=10, label=label)
    ax3.set_title("Tenure × Monthly Charges (coloré par Churn)", fontweight="bold")
    ax3.set_xlabel("tenure (mois)")
    ax3.set_ylabel("monthly_charges (€)")
    ax3.legend()

    # 4. support_tickets vs churn (violin)
    ax4 = fig.add_subplot(gs[1, 1])
    parts = ax4.violinplot(
        [df[df["churn"] == 0]["support_tickets"].values,
         df[df["churn"] == 1]["support_tickets"].values],
        positions=[0, 1],
        showmedians=True,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE_CHURN[i])
        pc.set_alpha(0.7)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["No Churn", "Churn"])
    ax4.set_title("Tickets Support → Churn", fontweight="bold")
    ax4.set_ylabel("support_tickets")

    path = os.path.join(output_dir, "06_key_relationships.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  → Graphique sauvegardé : {path}")


# ─── 6. FEATURE ENGINEERING SUGGESTIONS ─────────────────────────────────────

def suggest_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Propose des features dérivées pertinentes sur la base des variables disponibles.
    NOTE : Ces features sont des SUGGESTIONS — à implémenter dans preprocessor.py.
    """
    suggestions = {
        "tickets_per_month": "support_tickets / (tenure + 1) — Densité d'incidents dans le temps",
        "charge_per_login": "monthly_charges / (login_frequency + 1) — Coût relatif à l'usage",
        "engagement_score": "login_frequency * session_duration — Indice d'engagement composite",
        "payment_risk_flag": "1 if payment_failures > 2 else 0 — Signal binaire de risque paiement",
        "high_value_flag": "1 if total_revenue > percentile(75) else 0 — Client haute valeur",
        "nps_risk_flag": "1 if nps_score < 5 else 0 — Détracteur NPS",
        "tenure_segment": "buckets: new (<6m), mid (6-24m), loyal (>24m) — Segmentation ancienneté",
    }
    logger.info(f"\n{len(suggestions)} features dérivées suggérées pour le feature engineering.")
    return suggestions


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_eda(input_path: str, output_dir: str) -> dict:
    """
    Pipeline EDA complet.
    Args:
        input_path: Chemin vers customer_churn.csv
        output_dir: Dossier de sortie pour les figures
    Returns:
        audit_report: Dictionnaire complet du rapport d'audit
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{'='*60}")
    logger.info("  DÉMARRAGE DE L'ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    logger.info(f"{'='*60}")

    # Chargement
    df = pd.read_csv(input_path)
    logger.info(f"Dataset chargé : {df.shape}")

    # Identification des types de colonnes
    target_col = "churn"
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Variables numériques ({len(num_cols)}) : {num_cols}")
    logger.info(f"Variables catégorielles ({len(cat_cols)}) : {cat_cols}")

    # ─── Étape 1 : Audit structurel
    logger.info("\n[1/6] Audit structurel...")
    struct_report = audit_structure(df)
    print_audit_summary(struct_report)

    # ─── Étape 2 : Analyse variable cible
    logger.info("[2/6] Analyse de la variable cible...")
    target_report = analyze_target(df, target_col, output_dir)

    # ─── Étape 3 : Détection des outliers
    logger.info("[3/6] Détection des outliers (IQR)...")
    outlier_report = detect_outliers_iqr(df, num_cols)
    plot_outliers(df, num_cols, output_dir)
    for col, info in outlier_report.items():
        if info["n_outliers"] > 0:
            logger.info(f"  → {col}: {info['n_outliers']} outliers ({info['pct']}%) | bornes [{info['bounds'][0]}, {info['bounds'][1]}]")

    # ─── Étape 4 : Distributions univariées
    logger.info("[4/6] Analyse univariée...")
    plot_numeric_distributions(df, num_cols, output_dir)
    if cat_cols:
        plot_categorical_distributions(df, cat_cols, output_dir)

    # ─── Étape 5 : Corrélations & relations bivariées
    logger.info("[5/6] Analyse bivariée et corrélations...")
    plot_correlation_matrix(df, num_cols, output_dir)
    plot_key_relationships(df, output_dir)

    # ─── Étape 6 : Feature engineering suggestions
    logger.info("[6/6] Suggestions de feature engineering...")
    fe_suggestions = suggest_feature_engineering(df)

    # ─── Rapport final
    full_report = {
        "structure": struct_report,
        "target": target_report,
        "outliers": outlier_report,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_engineering_suggestions": fe_suggestions,
    }

    report_path = os.path.join(output_dir, "audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅ EDA TERMINÉE — Rapport sauvegardé : {report_path}")
    logger.info(f"  📊 {len(os.listdir(output_dir)) - 1} graphiques générés dans : {output_dir}")
    logger.info(f"{'='*60}")

    # Affichage des insights clés
    _print_key_insights(target_report, outlier_report, fe_suggestions)

    return full_report


def _print_key_insights(target_report: dict, outlier_report: dict, fe_suggestions: dict):
    """Affiche les insights clés à retenir pour la modélisation."""
    print("\n" + "═" * 60)
    print("  INSIGHTS CLÉS POUR LA MODÉLISATION")
    print("═" * 60)

    # Déséquilibre
    ratio = target_report["imbalance_ratio"]
    if target_report["is_imbalanced"]:
        print(f"  ⚠  DÉSÉQUILIBRE DE CLASSES (ratio {ratio:.2f}:1)")
        print("     → Utiliser : Recall, F1-Score, ROC-AUC (pas Accuracy seule)")
        print("     → Stratégie : SMOTE ou class_weight='balanced'")
        print("     → Validation : Stratified K-Fold obligatoire")
    else:
        print(f"  ✓  Classes équilibrées (ratio {ratio:.2f}:1)")

    # Outliers significatifs
    significant_outliers = {k: v for k, v in outlier_report.items() if v["pct"] > 5}
    if significant_outliers:
        print(f"\n  ⚠  Variables avec outliers importants (>5%) :")
        for col, info in significant_outliers.items():
            print(f"     • {col}: {info['pct']}%")
        print("     → Envisager : RobustScaler ou Winsorisation")
    else:
        print("  ✓  Outliers modérés sur toutes les variables")

    print(f"\n  💡 {len(fe_suggestions)} features dérivées disponibles dans audit_report.json")
    print("═" * 60 + "\n")


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA - Customer Churn Dataset")
    parser.add_argument("--input", type=str, default="data/raw/customer_churn.csv",
                        help="Chemin vers le dataset CSV")
    parser.add_argument("--output", type=str, default="reports/figures/",
                        help="Dossier de sortie pour les figures et le rapport")
    args = parser.parse_args()

    run_eda(args.input, args.output)

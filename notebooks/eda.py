"""
Script EDA — génère tous les graphiques dans reports/figures/
Usage : python notebooks/eda.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'affichage — sauvegarde directe
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_data, audit_dataset

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig, name: str):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Sauvegardé : {path}")


# ── 1. Distribution cible ─────────────────────────────────────────────────────

def plot_target(df):
    print("\n[1/5] Distribution de la variable cible...")
    counts = df["churn"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["No Churn (0)", "Churn (1)"], counts.values,
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Distribution du Churn", fontweight="bold")
    axes[0].set_ylabel("Nombre de clients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")

    axes[1].pie(counts.values, labels=["No Churn", "Churn"],
                colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Ratio Churn", fontweight="bold")

    fig.suptitle("Déséquilibre des classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "01_target_distribution.png")


# ── 2. Distributions numériques ───────────────────────────────────────────────

def plot_numerical(df):
    print("[2/5] Distributions numériques...")
    num_cols = ["age", "tenure_months", "monthly_logins", "monthly_fee",
                "total_revenue", "payment_failures", "support_tickets",
                "nps_score", "csat_score", "last_login_days_ago"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for val, color, label in [(0, "#2ecc71", "No Churn"), (1, "#e74c3c", "Churn")]:
            axes[i].hist(df[df["churn"] == val][col], bins=30,
                         alpha=0.6, color=color, label=label, density=True)
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
        axes[i].legend(fontsize=8)

    fig.suptitle("Distributions numériques par statut Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_numerical_distributions.png")


# ── 3. Facteurs de risque ─────────────────────────────────────────────────────

def plot_risk_factors(df):
    print("[3/5] Facteurs de risque...")
    risk_cols = ["payment_failures", "support_tickets", "nps_score",
                 "last_login_days_ago", "usage_growth_rate", "csat_score"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(risk_cols):
        bp = axes[i].boxplot(
            [df[df["churn"] == 0][col], df[df["churn"] == 1][col]],
            labels=["No Churn", "Churn"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")

    fig.suptitle("Facteurs de risque clés", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_risk_factors.png")


# ── 4. Variables catégorielles ────────────────────────────────────────────────

def plot_categorical(df):
    print("[4/5] Variables catégorielles...")
    cat_cols = ["contract_type", "customer_segment", "payment_method",
                "gender", "signup_channel"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False)
        axes[i].bar(churn_rate.index, churn_rate.values * 100,
                    color=sns.color_palette("husl", len(churn_rate)))
        axes[i].set_title(f"Churn rate — {col.replace('_', ' ').title()}", fontweight="bold")
        axes[i].set_ylabel("Churn Rate (%)")
        axes[i].tick_params(axis="x", rotation=30)
        for j, v in enumerate(churn_rate.values):
            axes[i].text(j, v * 100 + 0.2, f"{v*100:.1f}%", ha="center", fontsize=9)

    axes[-1].axis("off")
    fig.suptitle("Taux de churn par variables catégorielles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_categorical_churn.png")


# ── 5. Matrice de corrélation ─────────────────────────────────────────────────

def plot_correlation(df):
    print("[5/5] Matrice de corrélation...")
    corr = df.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Matrice de corrélation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_correlation_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("EDA — Customer Churn Dataset")
    print("=" * 50)

    df = load_data("data/raw/customer_churn.csv")
    audit_dataset(df)

    plot_target(df)
    plot_numerical(df)
    plot_risk_factors(df)
    plot_categorical(df)
    plot_correlation(df)

    print("\n✅ EDA terminée ! Graphiques dans reports/figures/")
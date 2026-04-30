# 📊 Churn Retention Platform

> Système intelligent de prédiction du churn client — EFREI M1 Dev. Manager Full Stack · Data Science 2025-26

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikit-learn)](https://scikit-learn.org)

---

## 🎯 Objectif

Prédire la probabilité qu'un client résilie son abonnement (churn) à partir de données comportementales et financières, et exposer ces prédictions via une API REST et un dashboard décisionnel interactif.

**Tâche prédictive :** Classification binaire — `churn = 1` (résiliation) vs `churn = 0` (fidèle)

**Dataset :** [`customer_churn.csv`](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) — 10 000 clients, 32 variables, ~10.2% de churners

---

## 🏗️ Architecture du Projet

```
churn-retention-platform/
├── data/
│   ├── raw/                    # customer_churn.csv
│   └── processed/              # données transformées
├── models/                     # modèles sérialisés (.pkl) + métriques (.json)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   ├── mlp_deep_learning.pkl
│   ├── preprocessor.pkl
│   ├── comparison_results.json
│   └── evaluation_results.json
├── notebooks/
│   └── main.ipynb              # EDA + entraînement des modèles
├── src/
│   ├── data/
│   │   ├── preprocessor.py     # pipeline sklearn (imputation + encodage + scaling)
│   │   ├── loader.py           # chargement des données
│   │   └── eda.py              # analyse exploratoire
│   ├── models/
│   │   ├── trainer.py          # entraînement des 4 modèles
│   │   └── evaluator.py        # métriques + comparaison
│   ├── explainability/
│   │   └── shap_analysis.py    # analyse SHAP (importance globale + locale)
│   └── api/
│       └── main.py             # API REST FastAPI
├── dashboard/
│   └── app.py                  # dashboard Streamlit
├── reports/
│   └── figures/                # graphiques générés (SHAP, EDA, etc.)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Cloner le repo

```bash
git clone https://github.com/mariusyvt/churn-retention-platform.git
cd churn-retention-platform
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer le dataset

Télécharger `customer_churn.csv` depuis [Kaggle](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) et le placer dans :

```
data/raw/customer_churn.csv
```

---

## 🚀 Lancement

### Étape 1 — Entraîner les modèles

```bash
jupyter notebook notebooks/main.ipynb
```

Exécuter toutes les cellules. Cela génère les fichiers `.pkl` dans `models/`.

### Étape 2 — Analyse SHAP (interprétabilité)

```bash
python src/explainability/shap_analysis.py data/raw/customer_churn.csv
```

Génère 3 graphiques dans `reports/figures/` :
- `12_shap_global_importance.png` — importance globale des features
- `13_shap_beeswarm.png` — impact directionnel
- `14_shap_local_explanation.png` — explication d'un churner individuel

### Étape 3 — Lancer l'API REST

```bash
uvicorn src.api.main:app --reload --port 8000
```

L'API est disponible sur `http://localhost:8000`

Documentation Swagger interactive : `http://localhost:8000/docs`

### Étape 4 — Lancer le Dashboard

> ⚠️ L'API doit être lancée avant le dashboard (le simulateur appelle l'API)

```bash
streamlit run dashboard/app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`

---

## 🔌 API REST — Endpoints

### `GET /health`
Vérifie que le service est actif et que les modèles sont chargés.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "models_loaded": ["random_forest", "logistic_regression", "xgboost", "mlp_deep_learning"],
  "preprocessor_loaded": true
}
```

---

### `GET /model-info`
Informations sur les modèles disponibles.

```bash
curl http://localhost:8000/model-info
```

---

### `POST /predict`
Prédit la probabilité de churn pour un client.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "contract_type": "Monthly",
    "monthly_charges": 79.99,
    "total_revenue": 959.88,
    "payment_method": "Credit Card",
    "discount_applied": 10.0,
    "tenure_months": 12,
    "login_frequency": 15.0,
    "monthly_logins": 20.0,
    "session_duration": 30.0,
    "support_tickets": 2,
    "payment_failures": 0,
    "nps_score": 7.0,
    "csat_score": 4.0,
    "referral_count": 1,
    "model_name": "random_forest"
  }'
```

**Réponse :**

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.1234,
  "risk_level": "Low",
  "revenue_at_risk": 9.87,
  "model_used": "random_forest",
  "interpretation": "✅ Risque FAIBLE (probabilité churn : 12.3%). Client stable, aucune action urgente requise."
}
```

**Paramètre `model_name` accepté :**
- `random_forest` *(recommandé)*
- `logistic_regression`
- `xgboost`
- `mlp_deep_learning`

---

## 📊 Dashboard — 4 Pages

| Page | Description |
|------|-------------|
| 🏠 **Vue Globale** | KPIs (taux de churn, revenu à risque), distribution par contrat et ancienneté |
| 🎯 **Simulateur Client** | Formulaire interactif → appel API → jauge de risque en temps réel |
| 📈 **Comparaison Modèles** | Tableau des métriques, matrices de confusion, graphiques comparatifs |
| 🔍 **Interprétabilité** | Top 10 SHAP, interprétation métier, beeswarm plot, explication locale |

---

## 📈 Résultats des Modèles

| Modèle | Recall | Précision | F1 | ROC-AUC | CV ROC-AUC |
|--------|--------|-----------|-----|---------|------------|
| Logistic Regression | 0.642 | 0.194 | 0.297 | 0.735 | 0.766 |
| **Random Forest** ✅ | 0.186 | 0.266 | 0.219 | **0.785** | **0.983** |
| XGBoost | 0.059 | 0.429 | 0.103 | 0.780 | 0.982 |
| MLP Deep Learning | 0.245 | 0.221 | 0.233 | 0.653 | 0.970 |

**Modèle retenu : Random Forest** — meilleur compromis ROC-AUC (0.785 test / 0.983 CV), stabilité et interprétabilité.

---

## 🔍 Top Features SHAP

| Rang | Feature | SHAP Score | Interprétation |
|------|---------|------------|----------------|
| 1 | `csat_score` | 0.0964 | Score satisfaction — 1er signal de churn |
| 2 | `payment_failures` | 0.0432 | Échecs de paiement — signal critique |
| 3 | `tenure_months` | 0.0335 | Ancienneté élevée = client stable |
| 4 | `discount_applied` | 0.0240 | Remise appliquée |
| 5 | `total_revenue` | 0.0207 | Revenu total — client à fort enjeu |

---

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Preprocessing | scikit-learn Pipeline, RobustScaler, OrdinalEncoder, SMOTE |
| Modèles ML | scikit-learn, XGBoost |
| Deep Learning | scikit-learn MLPClassifier |
| Interprétabilité | SHAP, TreeExplainer |
| API REST | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Sérialisation | joblib |

---

## 📁 Pipeline de Preprocessing

```
customer_churn.csv
    ↓
1. Suppression colonnes inutiles (customer_id, city)
    ↓
2. Feature Engineering (+6 features : tickets_per_month, engagement_score, ...)
    ↓
3. Train / Test Split stratifié (80% / 20%, stratify=churn)
    ↓
4. Pipeline sklearn (Imputation → OrdinalEncoder → RobustScaler) — fit sur train uniquement
    ↓
5. SMOTE sur train uniquement (jamais sur le test)
    ↓
X_train_resampled, X_test, y_train_resampled, y_test
```

---

## 📄 Licence

Projet académique — EFREI Paris 2025-26

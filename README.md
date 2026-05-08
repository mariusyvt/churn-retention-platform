# Churn Retention Platform

Projet de Data Science pour prédire quels clients vont résilier leur abonnement (churn).  
On a entraîné plusieurs modèles de Machine Learning, créé un dashboard interactif et une API pour faire des prédictions.

---

## Ce que fait ce projet

À partir d'un fichier CSV contenant 10 000 clients, on essaie de prédire si un client va partir ou non.  
Pour ça, on a :
- analysé les données (EDA)
- entraîné 4 modèles différents (Logistic Regression, Random Forest, XGBoost, MLP)
- comparé leurs performances
- expliqué pourquoi le modèle prend ses décisions (SHAP)
- fait un dashboard pour visualiser tout ça
- créé une API pour faire des prédictions en temps réel

---

## Installation

```bash
# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Sur macOS : XGBoost a besoin de cette librairie système
brew install libomp
```

---

## Lancer le projet étape par étape

> Toutes les commandes sont à lancer depuis la racine du projet, avec le venv activé.

---

### 1. Analyser les données (EDA)

```bash
python3 src/data/eda.py --input data/raw/customer_churn.csv --output reports/figures/
```

Cette commande génère des graphiques qui permettent de comprendre les données :  
distribution du churn, corrélations entre variables, outliers, etc.  
Les graphiques sont sauvegardés dans `reports/figures/`.

---

### 2. Entraîner les modèles

```bash
python3 src/models/trainer.py data/raw/customer_churn.csv models/
```

Cette commande entraîne les 4 modèles et les sauvegarde dans `models/`.  
On peut voir dans le terminal les métriques de chaque modèle (Recall, F1, ROC-AUC).

---

### 3. Évaluer et optimiser les modèles

```bash
python3 src/models/evaluator.py data/raw/customer_churn.csv
```

Cette commande génère des graphiques de comparaison (courbes ROC, matrices de confusion...)  
et cherche le meilleur seuil de décision pour chaque modèle.

---

### 4. Analyser l'importance des variables (SHAP)

```bash
python3 src/explainability/shap_analysis.py data/raw/customer_churn.csv
```

Cette commande explique pourquoi le modèle prédit qu'un client va churner.  
Elle génère des graphiques qui montrent quelles variables ont le plus d'impact.

---

### 5. Lancer l'API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Lance l'API sur `http://localhost:8000`.  
On peut tester les endpoints sur `http://localhost:8000/docs` (Swagger).

> Si le port est déjà utilisé : `lsof -ti:8000 | xargs kill -9`

---

### 6. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

Lance le dashboard sur `http://localhost:8501`.  
Le dashboard permet de visualiser les données, simuler un client et voir les résultats des modèles.

> L'API doit tourner en parallèle pour que le simulateur client fonctionne.

---

## Résumé rapide

```bash
# 1. EDA
python3 src/data/eda.py --input data/raw/customer_churn.csv --output reports/figures/

# 2. Entraînement
python3 src/models/trainer.py data/raw/customer_churn.csv models/

# 3. Évaluation
python3 src/models/evaluator.py data/raw/customer_churn.csv

# 4. SHAP
python3 src/explainability/shap_analysis.py data/raw/customer_churn.csv

# Terminal 1 — API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

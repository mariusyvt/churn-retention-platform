"""
API REST — Churn Prediction Platform
Endpoints : POST /predict | GET /health | GET /model-info
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import uvicorn

# ── Chargement des modèles au démarrage ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")

AVAILABLE_MODELS = {
    "random_forest":       "random_forest.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "xgboost":             "xgboost.pkl",
    "mlp_deep_learning":   "mlp_deep_learning.pkl",
}

loaded_models = {}
preprocessor = None

def load_assets():
    global preprocessor, loaded_models
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise RuntimeError(f"Preprocessor introuvable : {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    for name, filename in AVAILABLE_MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)

load_assets()

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Retention Platform — API",
    description="Prédit la probabilité de churn d'un client. Modèles disponibles : random_forest, logistic_regression, xgboost, mlp_deep_learning.",
    version="1.0.0",
)

# ── Schéma d'entrée ───────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    # Données démographiques
    age: float = Field(..., ge=18, le=100, example=35)
    gender: str = Field(..., example="Male")
    
    # Contrat & facturation
    contract_type: str = Field(..., example="Monthly")
    monthly_charges: float = Field(..., ge=0, example=79.99)
    total_revenue: float = Field(..., ge=0, example=959.88)
    payment_method: str = Field(..., example="Credit Card")
    discount_applied: float = Field(..., ge=0, le=100, example=10.0)
    
    # Engagement
    tenure_months: float = Field(..., ge=0, example=12)
    login_frequency: float = Field(..., ge=0, example=15.0)
    monthly_logins: float = Field(..., ge=0, example=20.0)
    session_duration: float = Field(..., ge=0, example=30.0)
    
    # Support & satisfaction
    support_tickets: float = Field(..., ge=0, example=2)
    payment_failures: float = Field(..., ge=0, example=0)
    nps_score: float = Field(..., ge=0, le=10, example=7.0)
    csat_score: float = Field(..., ge=0, le=5, example=4.0)
    survey_response: Optional[str] = Field(None, example="Satisfied")
    
    # Autres
    referral_count: float = Field(..., ge=0, example=1)
    region: Optional[str] = Field(None, example="North")
    plan_type: Optional[str] = Field(None, example="Premium")
    churn_risk_score: Optional[float] = Field(None, ge=0, le=1, example=0.3)

    # Modèle à utiliser
    model_name: Literal[
        "random_forest", "logistic_regression", "xgboost", "mlp_deep_learning"
    ] = Field("random_forest", example="random_forest")

    @validator("gender")
    def gender_valid(cls, v):
        if v not in ("Male", "Female", "Other"):
            raise ValueError("gender doit être 'Male', 'Female' ou 'Other'")
        return v

    @validator("contract_type")
    def contract_valid(cls, v):
        valid = ("Monthly", "Quarterly", "Annual")
        if v not in valid:
            raise ValueError(f"contract_type doit être parmi {valid}")
        return v


# ── Schéma de sortie ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int          # 0 ou 1
    churn_probability: float       # probabilité classe 1
    risk_level: str                # Low / Medium / High
    revenue_at_risk: float         # monthly_charges * proba
    model_used: str
    interpretation: str            # message lisible métier


def get_risk_level(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    elif proba < 0.6:
        return "Medium"
    return "High"


def build_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convertit le schéma Pydantic en DataFrame attendu par le preprocessor."""
    data = customer.dict(exclude={"model_name"})
    return pd.DataFrame([data])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
def health():
    """Vérifie que l'API et les modèles sont opérationnels."""
    return {
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
        "preprocessor_loaded": preprocessor is not None,
    }


@app.get("/model-info", tags=["Infra"])
def model_info():
    """Retourne les informations sur les modèles disponibles."""
    return {
        "available_models": list(loaded_models.keys()),
        "default_model": "random_forest",
        "task": "binary_classification",
        "target": "churn (0 = No, 1 = Yes)",
        "features_expected": 20,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Prédit la probabilité de churn pour un client.

    - **churn_prediction** : 0 = restera, 1 = va churner
    - **churn_probability** : probabilité entre 0 et 1
    - **risk_level** : Low / Medium / High
    - **revenue_at_risk** : monthly_charges × probabilité de churn
    """
    # Vérifier que le modèle demandé est disponible
    if customer.model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{customer.model_name}' non disponible. Choisir parmi : {list(loaded_models.keys())}"
        )

    model = loaded_models[customer.model_name]

    # Construire le DataFrame et appliquer le preprocessor
    try:
        df = build_dataframe(customer)
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de preprocessing : {str(e)}")

    # Prédiction
    try:
        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    risk = get_risk_level(proba)
    revenue_at_risk = round(customer.monthly_charges * proba, 2)

    # Message métier
    if risk == "High":
        interpretation = (
            f"⚠️  Client à HAUT RISQUE (probabilité churn : {proba:.1%}). "
            f"Revenu mensuel à risque : {revenue_at_risk}€. Action immédiate recommandée."
        )
    elif risk == "Medium":
        interpretation = (
            f"⚡ Risque MODÉRÉ (probabilité churn : {proba:.1%}). "
            f"Surveiller l'engagement et envoyer une offre de fidélisation."
        )
    else:
        interpretation = (
            f"✅ Risque FAIBLE (probabilité churn : {proba:.1%}). "
            f"Client stable, aucune action urgente requise."
        )

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        risk_level=risk,
        revenue_at_risk=revenue_at_risk,
        model_used=customer.model_name,
        interpretation=interpretation,
    )


# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

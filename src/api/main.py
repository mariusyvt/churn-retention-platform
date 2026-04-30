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
    description="Prédit la probabilité de churn d'un client.",
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

    # Autres champs directs
    referral_count: float = Field(..., ge=0, example=1)
    region: Optional[str] = Field(None, example="North")
    plan_type: Optional[str] = Field(None, example="Premium")
    churn_risk_score: Optional[float] = Field(None, ge=0, le=1, example=0.3)

    # ── Champs dérivés (optionnels avec valeurs par défaut) ──────────────────
    # Le dashboard les calcule et les envoie ; sinon ils sont recalculés ici.
    complaint_type: Optional[str] = Field(None, example="none")
    email_open_rate: Optional[float] = Field(None, ge=0, le=1, example=0.3)
    tickets_per_month: Optional[float] = Field(None, ge=0, example=0.08)
    charge_per_login: Optional[float] = Field(None, ge=0, example=4.0)
    avg_session_time: Optional[float] = Field(None, ge=0, example=30.0)
    features_used: Optional[int] = Field(None, ge=0, example=3)
    engagement_score: Optional[float] = Field(None, ge=0, example=6.0)
    avg_resolution_time: Optional[float] = Field(None, ge=0, example=24.0)
    payment_risk_flag: Optional[int] = Field(None, ge=0, le=1, example=0)
    marketing_click_rate: Optional[float] = Field(None, ge=0, le=1, example=0.05)
    weekly_active_days: Optional[float] = Field(None, ge=0, le=7, example=3.5)
    monthly_fee: Optional[float] = Field(None, ge=0, example=79.99)
    country: Optional[str] = Field(None, example="FR")
    customer_segment: Optional[str] = Field(None, example="standard")
    last_login_days_ago: Optional[int] = Field(None, ge=0, example=5)
    usage_growth_rate: Optional[float] = Field(None, example=0.0)
    signup_channel: Optional[str] = Field(None, example="web")
    escalations: Optional[int] = Field(None, ge=0, example=0)
    nps_risk_flag: Optional[int] = Field(None, ge=0, le=1, example=0)
    price_increase_last_3m: Optional[int] = Field(None, ge=0, le=1, example=0)
    high_value_flag: Optional[int] = Field(None, ge=0, le=1, example=0)

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
    churn_prediction: int
    churn_probability: float
    risk_level: str
    revenue_at_risk: float
    model_used: str
    interpretation: str


def get_risk_level(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    elif proba < 0.6:
        return "Medium"
    return "High"


def build_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convertit le schéma Pydantic en DataFrame attendu par le preprocessor.
    Les champs dérivés sont recalculés si non fournis par le dashboard.
    """
    c = customer

    # Valeurs de base
    tenure      = max(float(c.tenure_months), 1)
    logins      = max(float(c.monthly_logins), 1)
    login_freq  = float(c.login_frequency)
    session_dur = float(c.session_duration)
    tickets     = float(c.support_tickets)
    monthly_ch  = float(c.monthly_charges)
    failures    = float(c.payment_failures)
    nps         = float(c.nps_score)
    revenue     = float(c.total_revenue)

    # Calcul des dérivés (utilisés si non fournis)
    tickets_per_month   = round(tickets / tenure, 4)
    charge_per_login    = round(monthly_ch / logins, 4)
    avg_session_time    = session_dur
    engagement_score    = round(min((logins * session_dur) / 100, 10), 4)
    payment_risk_flag   = 1 if failures > 2 else 0
    weekly_active_days  = round(min(login_freq / 4.33, 7), 2)
    last_login_days_ago = max(30 - int(login_freq), 0)
    nps_risk_flag       = 1 if nps < 5 else 0
    high_value_flag     = 1 if revenue > 5000 else 0

    data = {
        # Champs directs
        "age":                  float(c.age),
        "gender":               c.gender,
        "contract_type":        c.contract_type,
        "monthly_charges":      monthly_ch,
        "total_revenue":        revenue,
        "payment_method":       c.payment_method,
        "discount_applied":     float(c.discount_applied),
        "tenure_months":        float(c.tenure_months),
        "login_frequency":      login_freq,
        "monthly_logins":       float(c.monthly_logins),
        "session_duration":     session_dur,
        "support_tickets":      tickets,
        "payment_failures":     failures,
        "nps_score":            nps,
        "csat_score":           float(c.csat_score),
        "survey_response":      c.survey_response or "Satisfied",
        "referral_count":       float(c.referral_count),
        "region":               c.region or "Unknown",
        "plan_type":            c.plan_type or "Standard",
        "churn_risk_score":     float(c.churn_risk_score) if c.churn_risk_score is not None else 0.5,

        # Champs dérivés — valeur fournie par le dashboard OU recalculée
        "complaint_type":       c.complaint_type        if c.complaint_type        is not None else "none",
        "email_open_rate":      c.email_open_rate        if c.email_open_rate       is not None else 0.3,
        "tickets_per_month":    c.tickets_per_month      if c.tickets_per_month     is not None else tickets_per_month,
        "charge_per_login":     c.charge_per_login       if c.charge_per_login      is not None else charge_per_login,
        "avg_session_time":     c.avg_session_time       if c.avg_session_time      is not None else avg_session_time,
        "features_used":        c.features_used          if c.features_used         is not None else 3,
        "engagement_score":     c.engagement_score       if c.engagement_score      is not None else engagement_score,
        "avg_resolution_time":  c.avg_resolution_time    if c.avg_resolution_time   is not None else 24.0,
        "payment_risk_flag":    c.payment_risk_flag      if c.payment_risk_flag     is not None else payment_risk_flag,
        "marketing_click_rate": c.marketing_click_rate   if c.marketing_click_rate  is not None else 0.05,
        "weekly_active_days":   c.weekly_active_days     if c.weekly_active_days    is not None else weekly_active_days,
        "monthly_fee":          c.monthly_fee            if c.monthly_fee           is not None else monthly_ch,
        "country":              c.country                if c.country               is not None else "FR",
        "customer_segment":     c.customer_segment       if c.customer_segment      is not None else "standard",
        "last_login_days_ago":  c.last_login_days_ago    if c.last_login_days_ago   is not None else last_login_days_ago,
        "usage_growth_rate":    c.usage_growth_rate      if c.usage_growth_rate     is not None else 0.0,
        "signup_channel":       c.signup_channel         if c.signup_channel        is not None else "web",
        "escalations":          c.escalations            if c.escalations           is not None else 0,
        "nps_risk_flag":        c.nps_risk_flag          if c.nps_risk_flag         is not None else nps_risk_flag,
        "price_increase_last_3m": c.price_increase_last_3m if c.price_increase_last_3m is not None else 0,
        "high_value_flag":      c.high_value_flag        if c.high_value_flag       is not None else high_value_flag,
    }

    return pd.DataFrame([data])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
def health():
    return {
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
        "preprocessor_loaded": preprocessor is not None,
    }


@app.get("/model-info", tags=["Infra"])
def model_info():
    return {
        "available_models": list(loaded_models.keys()),
        "default_model": "random_forest",
        "task": "binary_classification",
        "target": "churn (0 = No, 1 = Yes)",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    if customer.model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{customer.model_name}' non disponible. Choisir parmi : {list(loaded_models.keys())}"
        )

    model = loaded_models[customer.model_name]

    try:
        df = build_dataframe(customer)
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de preprocessing : {str(e)}")

    try:
        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    risk = get_risk_level(proba)
    revenue_at_risk = round(customer.monthly_charges * proba, 2)

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
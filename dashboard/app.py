"""
Dashboard — Churn Retention Platform
4 pages : Vue Globale | Simulateur Client | Comparaison Modèles | Interprétabilité
Style : clean, minimaliste, sobre
"""

import json
import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Platform",
    page_icon="○",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "customer_churn.csv")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Reset background — dark theme */
.main, .block-container {
    background-color: #0a0a0a !important;
    padding-top: 1.5rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.875rem !important;
    color: #888 !important;
    padding: 6px 0;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.8rem;
    color: #555;
}

/* Headings */
h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #f0f0f0 !important; letter-spacing: -0.02em; }
h2 { font-size: 1rem !important; font-weight: 500 !important; color: #ccc !important; }
h3 { font-size: 0.875rem !important; font-weight: 500 !important; color: #888 !important; }

/* Body text */
p, span, div, label { color: #aaa; }

/* KPI cards */
.kpi-card {
    background: #141414;
    border-radius: 8px;
    padding: 18px 20px;
    border: 1px solid #1e1e1e;
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 600;
    line-height: 1;
    color: #f0f0f0;
}
.kpi-value.danger  { color: #e05c5c; }
.kpi-value.warn    { color: #d49a2a; }
.kpi-value.info    { color: #5b9cf6; }
.kpi-value.neutral { color: #f0f0f0; }

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #444;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 8px;
}

/* Risk badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-high   { background: rgba(224,92,92,0.12); color: #e05c5c; border: 1px solid rgba(224,92,92,0.25); }
.badge-medium { background: rgba(212,154,42,0.12); color: #d49a2a; border: 1px solid rgba(212,154,42,0.25); }
.badge-low    { background: rgba(52,168,83,0.12); color: #34a853; border: 1px solid rgba(52,168,83,0.25); }

/* Result card */
.result-card {
    background: #141414;
    border-radius: 8px;
    padding: 24px;
    border: 1px solid #1e1e1e;
}
.result-proba {
    font-size: 3rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.02em;
}

/* Table overrides */
.dataframe { font-size: 0.8rem !important; background: #141414 !important; color: #ccc !important; }

/* Button */
.stButton > button {
    background: #f0f0f0 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.stButton > button:hover {
    background: #d0d0d0 !important;
}

/* Inputs / sliders */
.stSlider, .stSelectbox, .stNumberInput { filter: brightness(0.85); }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #555 !important;
}
.stTabs [aria-selected="true"] {
    color: #f0f0f0 !important;
}

/* Divider */
hr { border: none; border-top: 1px solid #1e1e1e; margin: 20px 0; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return None

@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "comparison_results.json")
    with open(path) as f:
        return json.load(f)

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def plotly_base():
    """Renvoie un dict de layout commun propre — dark theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#141414",
        font=dict(family="Inter, sans-serif", color="#888", size=12),
        margin=dict(t=16, b=16, l=16, r=16),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size: 1rem; font-weight: 600; color: #f0f0f0; letter-spacing: -0.01em;'>
            Churn Platform
        </div>
        <div style='font-size: 0.75rem; color: #555; margin-top: 3px;'>Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    api_ok = check_api()
    dot_color = "#166534" if api_ok else "#c0392b"
    status_text = "API connectée" if api_ok else "API hors ligne"
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 7px; margin-bottom: 20px;
                font-size: 0.75rem; color: #999;'>
        <span style='color: {dot_color}; font-size: 0.6rem;'>●</span>
        {status_text}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:#444; margin-bottom:6px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Vue globale", "Simulateur client", "Comparaison modèles", "Interprétabilité"],
        label_visibility="collapsed"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE GLOBALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Vue globale":
    st.markdown("<h1>Vue globale</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.875rem; margin-bottom:24px;'>Indicateurs clés de rétention client</p>", unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("Impossible de charger les données.")
        st.stop()

    total      = len(df)
    churners   = int(df["churn"].sum())
    churn_rate = churners / total * 100
    rev_risk   = df[df["churn"] == 1]["monthly_charges"].sum() if "monthly_charges" in df.columns else 0
    avg_tenure = df["tenure_months"].mean() if "tenure_months" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Total clients</div>
            <div class='kpi-value neutral'>{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Taux de churn</div>
            <div class='kpi-value danger'>{churn_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Clients à risque</div>
            <div class='kpi-value warn'>{churners:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Revenu mensuel à risque</div>
            <div class='kpi-value danger'>{rev_risk:,.0f} €</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Distribution</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Churn par type de contrat</h3>", unsafe_allow_html=True)
        if "contract_type" in df.columns:
            cc = df.groupby("contract_type")["churn"].agg(["sum", "count"]).reset_index()
            cc["rate"] = cc["sum"] / cc["count"] * 100
            fig = go.Figure(go.Bar(
                x=cc["rate"],
                y=cc["contract_type"],
                orientation="h",
                marker_color=["#c0392b", "#b7770d", "#166534"],
                text=[f"{v:.1f}%" for v in cc["rate"]],
                textposition="outside",
                textfont=dict(size=12, color="#555"),
            ))
            layout = plotly_base()
            layout.update(dict(
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(gridcolor="#1e1e1e"),
                height=220,
            ))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h3>Fidèles vs Churners</h3>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Fidèles", "Churners"],
            values=[total - churners, churners],
            hole=0.6,
            marker_colors=["#2a2a2a", "#e05c5c"],
            textfont_size=12,
            showlegend=True,
        ))
        layout2 = plotly_base()
        layout2.update(dict(
            height=220,
            legend=dict(font=dict(color="#888", size=12)),
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b>",
                x=0.5, y=0.5,
                font=dict(size=18, color="#e05c5c"),
                showarrow=False
            )]
        ))
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-label'>Analyse comportementale</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<h3>Charges mensuelles</h3>", unsafe_allow_html=True)
        if "monthly_charges" in df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df[df["churn"]==0]["monthly_charges"], name="Fidèles",
                marker_color="#333", opacity=0.8, nbinsx=40))
            fig3.add_trace(go.Histogram(
                x=df[df["churn"]==1]["monthly_charges"], name="Churners",
                marker_color="#e05c5c", opacity=0.8, nbinsx=40))
            layout3 = plotly_base()
            layout3.update(dict(
                barmode="overlay",
                xaxis=dict(gridcolor="#1e1e1e", title="€/mois"),
                yaxis=dict(gridcolor="#1e1e1e"),
                height=240,
                legend=dict(font=dict(color="#888", size=11)),
            ))
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<h3>Churn par ancienneté</h3>", unsafe_allow_html=True)
        if "tenure_months" in df.columns:
            bins = pd.cut(df["tenure_months"], bins=[0,6,12,24,36,60,120],
                          labels=["0–6m","6–12m","12–24m","24–36m","36–60m","60m+"])
            tc = df.groupby(bins, observed=True)["churn"].mean() * 100
            fig4 = go.Figure(go.Bar(
                x=tc.index.astype(str), y=tc.values,
                marker_color="#f0f0f0",
                text=[f"{v:.1f}%" for v in tc.values],
                textposition="outside",
                textfont=dict(size=11, color="#888"),
            ))
            layout4 = plotly_base()
            layout4.update(dict(
                xaxis=dict(gridcolor="#1e1e1e"),
                yaxis=dict(gridcolor="#1e1e1e", title="%"),
                height=240,
            ))
            fig4.update_layout(**layout4)
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIMULATEUR CLIENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulateur client":
    st.markdown("<h1>Simulateur client</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.875rem; margin-bottom:24px;'>Prédisez le risque de churn d'un client en temps réel</p>", unsafe_allow_html=True)

    if not check_api():
        st.warning("API hors ligne — lancez `uvicorn src.api.main:app --port 8000` d'abord.")
        st.stop()

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("<div class='section-label'>Profil client</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age               = st.slider("Âge", 18, 80, 35)
            tenure            = st.slider("Ancienneté (mois)", 0, 120, 12)
            monthly_charges   = st.number_input("Charges mensuelles (€)", 0.0, 500.0, 79.99)
            total_revenue     = st.number_input("Revenu total (€)", 0.0, 50000.0, float(monthly_charges * tenure))
            login_frequency   = st.slider("Fréquence login (j/mois)", 0.0, 31.0, 15.0)
            monthly_logins    = st.slider("Connexions/mois", 0.0, 100.0, 20.0)
            session_duration  = st.slider("Durée session (min)", 0.0, 120.0, 30.0)
        with c2:
            gender            = st.selectbox("Genre", ["Male", "Female", "Other"])
            contract_type     = st.selectbox("Type de contrat", ["Monthly", "Quarterly", "Annual"])
            payment_method    = st.selectbox("Paiement", ["Credit Card", "Bank Transfer", "PayPal", "Check"])
            discount_applied  = st.slider("Remise appliquée (%)", 0.0, 50.0, 0.0)
            support_tickets   = st.slider("Tickets support", 0, 20, 1)
            payment_failures  = st.slider("Échecs de paiement", 0, 10, 0)
            nps_score         = st.slider("NPS", 0.0, 10.0, 7.0)
            csat_score        = st.slider("CSAT", 0.0, 5.0, 4.0)

        referral_count = st.slider("Parrainages", 0, 10, 0)
        model_name = st.selectbox("Modèle", ["random_forest", "logistic_regression", "xgboost", "mlp_deep_learning"])

        predict_btn = st.button("Analyser ce client", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-label'>Résultat</div>", unsafe_allow_html=True)

        if predict_btn:
            payload = {
                # ── Champs du formulaire ──────────────────────────────────
                "age": age,
                "gender": gender,
                "contract_type": contract_type,
                "monthly_charges": monthly_charges,
                "total_revenue": total_revenue,
                "payment_method": payment_method,
                "discount_applied": discount_applied,
                "tenure_months": tenure,
                "login_frequency": login_frequency,
                "monthly_logins": monthly_logins,
                "session_duration": session_duration,
                "support_tickets": support_tickets,
                "payment_failures": payment_failures,
                "nps_score": nps_score,
                "csat_score": csat_score,
                "referral_count": referral_count,
                "model_name": model_name,
                # ── Colonnes manquantes — valeurs neutres dérivées ────────
                "complaint_type": "none",
                "email_open_rate": 0.3,
                "tickets_per_month": round(support_tickets / max(tenure, 1), 4),
                "charge_per_login": round(monthly_charges / max(monthly_logins, 1), 4),
                "avg_session_time": session_duration,
                "features_used": 3,
                "engagement_score": round(min((monthly_logins * session_duration) / 100, 10), 4),
                "avg_resolution_time": 24.0,
                "payment_risk_flag": 1 if payment_failures > 2 else 0,
                "marketing_click_rate": 0.05,
                "weekly_active_days": round(min(login_frequency / 4.33, 7), 2),
                "monthly_fee": monthly_charges,
                "country": "FR",
                "customer_segment": "standard",
                "last_login_days_ago": max(30 - int(login_frequency), 0),
                "usage_growth_rate": 0.0,
                "signup_channel": "web",
                "escalations": 0,
                "nps_risk_flag": 1 if nps_score < 5 else 0,
                "price_increase_last_3m": 0,
                "high_value_flag": 1 if total_revenue > 5000 else 0,
            }
            with st.spinner("Analyse…"):
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    if r.status_code == 200:
                        result = r.json()
                        proba  = result["churn_probability"]
                        risk   = result["risk_level"]

                        badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[risk]
                        risk_fr     = {"High": "Élevé", "Medium": "Moyen", "Low": "Faible"}[risk]
                        proba_color = {"High": "#e05c5c", "Medium": "#d49a2a", "Low": "#34a853"}[risk]
                        decision    = "CHURN" if result["churn_prediction"] == 1 else "FIDÈLE"
                        dec_color   = "#e05c5c" if result["churn_prediction"] == 1 else "#34a853"

                        st.markdown(f"""
                        <div class='result-card'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;'>
                                <span class='badge {badge_class}'>Risque {risk_fr}</span>
                                <span style='font-size:0.75rem; color:#999;'>{model_name}</span>
                            </div>
                            <div class='result-proba' style='color:{proba_color};'>{proba:.1%}</div>
                            <div style='font-size:0.8rem; color:#999; margin-top:4px; margin-bottom:20px;'>probabilité de churn</div>
                            <hr style='border:none; border-top:1px solid #eee; margin-bottom:16px;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:16px;'>
                                <div>
                                    <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:#999; margin-bottom:4px;'>Décision</div>
                                    <div style='font-size:0.95rem; font-weight:600; color:{dec_color};'>{decision}</div>
                                </div>
                                <div>
                                    <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:#999; margin-bottom:4px;'>Revenu à risque</div>
                                    <div style='font-size:0.95rem; font-weight:600; color:#111;'>{result["revenue_at_risk"]} €/mois</div>
                                </div>
                            </div>
                            <div style='font-size:0.8rem; color:#777; line-height:1.6; background:#1a1a1a; border-radius:6px; padding:12px;'>
                                {result["interpretation"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Gauge épurée
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=proba * 100,
                            number={"suffix": "%", "font": {"color": "#f0f0f0", "size": 24, "family": "Inter"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#333", "tickfont": {"size": 10}},
                                "bar": {"color": proba_color, "thickness": 0.25},
                                "bgcolor": "#141414",
                                "borderwidth": 0,
                                "steps": [
                                    {"range": [0, 30],  "color": "rgba(52,168,83,0.08)"},
                                    {"range": [30, 60], "color": "rgba(212,154,42,0.08)"},
                                    {"range": [60, 100],"color": "rgba(224,92,92,0.08)"},
                                ],
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif"),
                            height=200,
                            margin=dict(t=20, b=10, l=30, r=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"Erreur API : {r.json().get('detail', 'inconnue')}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")
        else:
            st.markdown("""
            <div style='background:#141414; border-radius:8px; padding:48px 24px;
                        text-align:center; border: 1px dashed #1e1e1e;'>
                <div style='font-size:0.875rem; color:#444; font-weight:500;'>Renseignez le profil client</div>
                <div style='font-size:0.8rem; color:#333; margin-top:6px;'>puis cliquez sur Analyser</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Comparaison modèles":
    st.markdown("<h1>Comparaison des modèles</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.875rem; margin-bottom:24px;'>Évaluation comparative des 4 algorithmes</p>", unsafe_allow_html=True)

    metrics = load_metrics()
    df_m = pd.DataFrame([{
        "Modèle":     m["model_name"],
        "Recall":     m["test_recall"],
        "Precision":  m["test_precision"],
        "F1":         m["test_f1"],
        "ROC-AUC":    m["test_roc_auc"],
        "CV ROC-AUC": m["cv_roc_auc_mean"],
        "Temps (s)":  m["train_time_sec"],
    } for m in metrics])

    st.markdown("<div class='section-label'>Métriques</div>", unsafe_allow_html=True)
    st.dataframe(
        df_m.style.format({
            "Recall": "{:.3f}", "Precision": "{:.3f}", "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}", "CV ROC-AUC": "{:.3f}", "Temps (s)": "{:.2f}"
        }).background_gradient(
            subset=["Recall", "Precision", "F1", "ROC-AUC", "CV ROC-AUC"],
            cmap="Greys"
        ),
        use_container_width=True, hide_index=True
    )

    st.markdown("<div class='section-label'>Visualisations</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    models = [m["model_name"] for m in metrics]

    with col1:
        st.markdown("<h3>Recall & F1</h3>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Recall", x=models, y=[m["test_recall"] for m in metrics],
                             marker_color="#e05c5c",
                             text=[f"{m['test_recall']:.3f}" for m in metrics],
                             textposition="outside", textfont=dict(size=11, color="#888")))
        fig.add_trace(go.Bar(name="F1", x=models, y=[m["test_f1"] for m in metrics],
                             marker_color="#444",
                             text=[f"{m['test_f1']:.3f}" for m in metrics],
                             textposition="outside", textfont=dict(size=11, color="#888")))
        layout = plotly_base()
        layout.update(dict(
            barmode="group",
            xaxis=dict(gridcolor="#1e1e1e"),
            yaxis=dict(gridcolor="#1e1e1e", range=[0, 0.9]),
            legend=dict(font=dict(color="#888", size=11)),
            height=280,
        ))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h3>ROC-AUC — Test vs CV</h3>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Test", x=models, y=[m["test_roc_auc"] for m in metrics],
                              marker_color="#f0f0f0",
                              text=[f"{m['test_roc_auc']:.3f}" for m in metrics],
                              textposition="outside", textfont=dict(size=11, color="#888")))
        fig2.add_trace(go.Bar(name="CV", x=models, y=[m["cv_roc_auc_mean"] for m in metrics],
                              marker_color="#333",
                              text=[f"{m['cv_roc_auc_mean']:.3f}" for m in metrics],
                              textposition="outside", textfont=dict(size=11, color="#888")))
        layout2 = plotly_base()
        layout2.update(dict(
            barmode="group",
            xaxis=dict(gridcolor="#1e1e1e"),
            yaxis=dict(gridcolor="#1e1e1e", range=[0, 1.15]),
            legend=dict(font=dict(color="#888", size=11)),
            height=280,
        ))
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-label'>Matrices de confusion</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for m, col in zip(metrics, cols):
        with col:
            cm = np.array(m["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, text=cm, texttemplate="%{text}",
                colorscale=[[0, "#141414"], [1, "#f0f0f0"]],
                showscale=False,
                xgap=2, ygap=2,
            ))
            fig_cm.update_layout(
                title=dict(text=m["model_name"], font=dict(color="#888", size=11)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#141414",
                font=dict(family="Inter", color="#888"),
                xaxis=dict(tickvals=[0,1], ticktext=["Prédit 0","Prédit 1"], tickfont=dict(size=9)),
                yaxis=dict(tickvals=[0,1], ticktext=["Réel 0","Réel 1"], tickfont=dict(size=9)),
                margin=dict(t=36, b=10, l=10, r=10),
                height=200,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div class='section-label'>Recommandation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#141414; border-radius:8px; padding:16px 20px;
                font-size:0.85rem; color:#888; line-height:1.8; border-left:3px solid #f0f0f0;'>
        <span style='font-weight:600; color:#f0f0f0;'>Modèle recommandé : Random Forest</span><br>
        Meilleur compromis ROC-AUC (0.785 test / 0.983 CV) avec une précision de 26.6% sur les churners.
        La régression logistique offre un meilleur recall (64%) mais génère trop de faux positifs pour un usage CRM.
        Le Random Forest reste le meilleur choix pour prioriser les actions de rétention.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Interprétabilité":
    st.markdown("<h1>Interprétabilité SHAP</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.875rem; margin-bottom:24px;'>Pourquoi le modèle prédit-il le churn ?</p>", unsafe_allow_html=True)

    shap_data = {
        "csat_score":       0.0964,
        "payment_failures": 0.0432,
        "tenure_months":    0.0335,
        "discount_applied": 0.0240,
        "total_revenue":    0.0207,
        "gender":           0.0200,
        "monthly_logins":   0.0195,
        "survey_response":  0.0187,
        "payment_method":   0.0187,
        "referral_count":   0.0164,
    }

    interpretations = [
        ("csat_score",       "0.096", "danger", "Score satisfaction — le plus prédictif du churn"),
        ("payment_failures", "0.043", "danger", "Échecs de paiement — signal d'alerte critique"),
        ("tenure_months",    "0.034", "good",   "Ancienneté élevée — client plus stable"),
        ("discount_applied", "0.024", "neutral","Remise appliquée — risque si non suivi"),
        ("total_revenue",    "0.021", "good",   "Fort CA — client à fort enjeu"),
    ]

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("<div class='section-label'>Top 10 features — importance globale SHAP</div>", unsafe_allow_html=True)
        features = list(shap_data.keys())
        values   = list(shap_data.values())
        # couleur relative à la médiane
        median_val = np.median(values)
        colors = ["#f0f0f0" if v > median_val else "#2a2a2a" for v in values]

        fig = go.Figure(go.Bar(
            x=values[::-1], y=features[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in values[::-1]],
            textposition="outside",
            textfont=dict(color="#888", size=11),
        ))
        layout = plotly_base()
        layout.update(dict(
            xaxis=dict(gridcolor="#1e1e1e", title="Importance SHAP moyenne (|valeur|)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=380,
            margin=dict(t=10, b=10, l=16, r=60),
        ))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-label'>Interprétation métier</div>", unsafe_allow_html=True)
        color_map = {
            "danger":  ("color:#e05c5c;", "↑ risque"),
            "good":    ("color:#34a853;", "↓ risque"),
            "neutral": ("color:#d49a2a;", "neutre"),
        }
        for feat, val, level, desc in interpretations:
            style, label = color_map[level]
            st.markdown(f"""
            <div style='background:#141414; border:1px solid #1e1e1e; border-radius:6px; padding:10px 14px; margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.85rem; font-weight:500; color:#f0f0f0;'>{feat}</span>
                    <span style='font-size:0.75rem; {style} font-weight:500;'>{label} · SHAP {val}</span>
                </div>
                <div style='font-size:0.78rem; color:#555; margin-top:3px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Graphiques SHAP</div>", unsafe_allow_html=True)
    shap_figs = [
        ("12_shap_global_importance.png", "Importance globale"),
        ("13_shap_beeswarm.png",          "Beeswarm — impact directionnel"),
        ("14_shap_local_explanation.png", "Explication locale — churner"),
    ]
    tabs = st.tabs([title for _, title in shap_figs])
    for tab, (filename, title) in zip(tabs, shap_figs):
        with tab:
            img_path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"Image non trouvée : lancez `python src/explainability/shap_analysis.py` d'abord.")

    st.markdown("""
    <div style='background:#141414; border-radius:6px; padding:14px 18px; margin-top:16px;
                font-size:0.8rem; color:#555; line-height:1.8; border-left:3px solid #2a2a2a;'>
        <span style='font-weight:500; color:#888;'>Comment lire ces graphiques ?</span><br>
        Les barres claires indiquent les features dont la valeur élevée <span style='color:#e05c5c;'>augmente</span> le risque de churn.
        Les barres sombres indiquent les features qui <span style='color:#34a853;'>réduisent</span> ce risque.
        L'écart horizontal entre les points représente la variance de l'impact selon les clients.
    </div>
    """, unsafe_allow_html=True)
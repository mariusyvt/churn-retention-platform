"""
Dashboard — Churn Retention Platform
4 pages : Vue Globale | Simulateur Client | Comparaison Modèles | Interprétabilité
"""

import json
import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Retention Platform",
    page_icon="📊",
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

.main { background-color: #0d0f14; }

.metric-card {
    background: linear-gradient(135deg, #1a1d26 0%, #12151e 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    font-family: 'Syne', sans-serif;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    line-height: 1;
}
.metric-value.red { color: #f87171; }
.metric-value.yellow { color: #fbbf24; }
.metric-value.green { color: #34d399; }
.metric-value.blue { color: #60a5fa; }

.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
}
.risk-high { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }
.risk-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
.risk-low { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e5e7eb;
    border-left: 3px solid #6366f1;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800; color: #e5e7eb;'>
            📊 Churn Platform
        </div>
        <div style='font-size: 0.75rem; color: #6b7280; margin-top: 4px;'>Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    api_ok = check_api()
    status_color = "#34d399" if api_ok else "#f87171"
    status_text = "API connectée" if api_ok else "API hors ligne"
    st.markdown(f"""
    <div style='background: rgba(99,102,241,0.1); border: 1px solid #2a2d3a; border-radius: 8px;
                padding: 10px 14px; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;'>
        <span style='color: {status_color}; font-size: 0.7rem;'>●</span>
        <span style='font-size: 0.8rem; color: #9ca3af;'>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Vue Globale", "🎯 Simulateur Client", "📈 Comparaison Modèles", "🔍 Interprétabilité"],
        label_visibility="collapsed"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE GLOBALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Vue Globale":
    st.markdown("<h1 style='color:#e5e7eb; font-size:2rem; margin-bottom:4px;'>Vue Globale</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; margin-bottom:28px;'>Indicateurs clés de rétention client</p>", unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("Impossible de charger les données.")
        st.stop()

    total = len(df)
    churners = df["churn"].sum()
    churn_rate = churners / total * 100
    rev_at_risk = (df[df["churn"] == 1]["monthly_charges"].sum()) if "monthly_charges" in df.columns else 0
    avg_tenure = df["tenure_months"].mean() if "tenure_months" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Total Clients</div>
            <div class='metric-value blue'>{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Taux de Churn</div>
            <div class='metric-value red'>{churn_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Clients à Risque</div>
            <div class='metric-value yellow'>{int(churners):,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Revenu Mensuel à Risque</div>
            <div class='metric-value red'>{rev_at_risk:,.0f}€</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Distribution du Churn</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Fidèles", "Churners"],
            values=[total - churners, churners],
            hole=0.65,
            marker_colors=["#6366f1", "#f87171"],
            textfont_size=13,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            showlegend=True,
            legend=dict(font=dict(color="#9ca3af")),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            annotations=[dict(text=f"<b>{churn_rate:.1f}%</b>", x=0.5, y=0.5,
                             font_size=22, font_color="#f87171", showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Churn par Type de Contrat</div>", unsafe_allow_html=True)
        if "contract_type" in df.columns:
            contract_churn = df.groupby("contract_type")["churn"].agg(["sum", "count"]).reset_index()
            contract_churn["rate"] = contract_churn["sum"] / contract_churn["count"] * 100
            fig2 = go.Figure(go.Bar(
                x=contract_churn["contract_type"],
                y=contract_churn["rate"],
                marker_color=["#f87171", "#fbbf24", "#34d399"],
                text=[f"{v:.1f}%" for v in contract_churn["rate"]],
                textposition="outside",
                textfont=dict(color="#e5e7eb"),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#9ca3af",
                yaxis=dict(gridcolor="#1f2230", title="Taux de churn (%)"),
                xaxis=dict(gridcolor="#1f2230"),
                margin=dict(t=20, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>Distribution des Charges Mensuelles</div>", unsafe_allow_html=True)
        if "monthly_charges" in df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df[df["churn"]==0]["monthly_charges"], name="Fidèles",
                marker_color="#6366f1", opacity=0.7, nbinsx=40))
            fig3.add_trace(go.Histogram(
                x=df[df["churn"]==1]["monthly_charges"], name="Churners",
                marker_color="#f87171", opacity=0.7, nbinsx=40))
            fig3.update_layout(
                barmode="overlay",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#9ca3af",
                yaxis=dict(gridcolor="#1f2230"),
                xaxis=dict(gridcolor="#1f2230", title="Charges mensuelles (€)"),
                legend=dict(font=dict(color="#9ca3af")),
                margin=dict(t=10, b=10, l=10, r=10), height=260,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Ancienneté vs Risque de Churn</div>", unsafe_allow_html=True)
        if "tenure_months" in df.columns:
            bins = pd.cut(df["tenure_months"], bins=[0,6,12,24,36,60,120], labels=["0-6m","6-12m","12-24m","24-36m","36-60m","60m+"])
            tenure_churn = df.groupby(bins, observed=True)["churn"].mean() * 100
            fig4 = go.Figure(go.Bar(
                x=tenure_churn.index.astype(str),
                y=tenure_churn.values,
                marker_color="#6366f1",
                text=[f"{v:.1f}%" for v in tenure_churn.values],
                textposition="outside",
                textfont=dict(color="#e5e7eb"),
            ))
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#9ca3af",
                yaxis=dict(gridcolor="#1f2230", title="Taux de churn (%)"),
                xaxis=dict(gridcolor="#1f2230", title="Ancienneté"),
                margin=dict(t=10, b=10, l=10, r=10), height=260,
            )
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIMULATEUR CLIENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Simulateur Client":
    st.markdown("<h1 style='color:#e5e7eb; font-size:2rem; margin-bottom:4px;'>Simulateur Client</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; margin-bottom:28px;'>Prédisez le risque de churn d'un client en temps réel</p>", unsafe_allow_html=True)

    if not check_api():
        st.error("⚠️ L'API n'est pas accessible. Lancez `uvicorn src.api.main:app --port 8000` d'abord.")
        st.stop()

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown("<div class='section-title'>Profil Client</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Âge", 18, 80, 35)
            tenure = st.slider("Ancienneté (mois)", 0, 120, 12)
            monthly_charges = st.number_input("Charges mensuelles (€)", 0.0, 500.0, 79.99)
            total_revenue = st.number_input("Revenu total (€)", 0.0, 50000.0, float(monthly_charges * tenure))
            login_frequency = st.slider("Fréquence login (j/mois)", 0.0, 31.0, 15.0)
            monthly_logins = st.slider("Connexions/mois", 0.0, 100.0, 20.0)
            session_duration = st.slider("Durée session (min)", 0.0, 120.0, 30.0)
        with c2:
            gender = st.selectbox("Genre", ["Male", "Female", "Other"])
            contract_type = st.selectbox("Type de contrat", ["Monthly", "Quarterly", "Annual"])
            payment_method = st.selectbox("Méthode de paiement", ["Credit Card", "Bank Transfer", "PayPal", "Check"])
            discount_applied = st.slider("Remise appliquée (%)", 0.0, 50.0, 0.0)
            support_tickets = st.slider("Tickets support", 0, 20, 1)
            payment_failures = st.slider("Échecs de paiement", 0, 10, 0)
            nps_score = st.slider("NPS Score", 0.0, 10.0, 7.0)
            csat_score = st.slider("CSAT Score", 0.0, 5.0, 4.0)

        referral_count = st.slider("Nombre de parrainages", 0, 10, 0)
        model_name = st.selectbox("Modèle à utiliser", ["random_forest", "logistic_regression", "xgboost", "mlp_deep_learning"])

        predict_btn = st.button("🔍 Analyser ce client", use_container_width=True, type="primary")

    with col_result:
        st.markdown("<div class='section-title'>Résultat de l'Analyse</div>", unsafe_allow_html=True)

        if predict_btn:
            payload = {
                "age": age, "gender": gender, "contract_type": contract_type,
                "monthly_charges": monthly_charges, "total_revenue": total_revenue,
                "payment_method": payment_method, "discount_applied": discount_applied,
                "tenure_months": tenure, "login_frequency": login_frequency,
                "monthly_logins": monthly_logins, "session_duration": session_duration,
                "support_tickets": support_tickets, "payment_failures": payment_failures,
                "nps_score": nps_score, "csat_score": csat_score,
                "referral_count": referral_count, "model_name": model_name,
            }

            with st.spinner("Analyse en cours..."):
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    if r.status_code == 200:
                        result = r.json()
                        proba = result["churn_probability"]
                        risk = result["risk_level"]
                        risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}[risk]

                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1a1d26, #12151e);
                                    border: 1px solid #2a2d3a; border-radius: 12px; padding: 24px; margin-bottom: 16px;'>
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;'>
                                <span class='risk-badge {risk_class}'>Risque {risk}</span>
                                <span style='font-family:Syne,sans-serif; font-size:0.8rem; color:#6b7280;'>via {model_name}</span>
                            </div>
                            <div style='font-family:Syne,sans-serif; font-size:3rem; font-weight:800;
                                        color:{"#f87171" if risk=="High" else "#fbbf24" if risk=="Medium" else "#34d399"};
                                        text-align:center; margin: 8px 0;'>
                                {proba:.1%}
                            </div>
                            <div style='text-align:center; color:#6b7280; font-size:0.8rem; margin-bottom:16px;'>
                                probabilité de churn
                            </div>
                            <div style='background:#0d0f14; border-radius:8px; padding:12px 16px;
                                        font-size:0.85rem; color:#9ca3af; line-height:1.6;'>
                                {result["interpretation"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=proba * 100,
                            number={"suffix": "%", "font": {"color": "#e5e7eb", "size": 28}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                                "bar": {"color": "#f87171" if risk=="High" else "#fbbf24" if risk=="Medium" else "#34d399"},
                                "bgcolor": "#1a1d26",
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(52,211,153,0.15)"},
                                    {"range": [30, 60], "color": "rgba(251,191,36,0.15)"},
                                    {"range": [60, 100], "color": "rgba(248,113,113,0.15)"},
                                ],
                                "threshold": {"line": {"color": "white", "width": 2}, "value": proba*100}
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb",
                            height=220, margin=dict(t=20, b=10, l=30, r=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(f"""
                        <div style='background:#1a1d26; border:1px solid #2a2d3a; border-radius:8px;
                                    padding:12px 16px; display:flex; justify-content:space-between;'>
                            <div style='text-align:center;'>
                                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase;'>Décision</div>
                                <div style='font-weight:700; color:{"#f87171" if result["churn_prediction"]==1 else "#34d399"};'>
                                    {"CHURN" if result["churn_prediction"]==1 else "FIDÈLE"}
                                </div>
                            </div>
                            <div style='text-align:center;'>
                                <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase;'>Revenu à risque</div>
                                <div style='font-weight:700; color:#fbbf24;'>{result["revenue_at_risk"]}€/mois</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Erreur API : {r.json().get('detail', 'Inconnue')}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")
        else:
            st.markdown("""
            <div style='background:#1a1d26; border:1px dashed #2a2d3a; border-radius:12px;
                        padding:48px 24px; text-align:center; color:#4b5563;'>
                <div style='font-size:2rem; margin-bottom:12px;'>🎯</div>
                <div style='font-family:Syne,sans-serif; font-weight:600;'>Renseignez le profil client</div>
                <div style='font-size:0.85rem; margin-top:8px;'>puis cliquez sur Analyser</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Comparaison Modèles":
    st.markdown("<h1 style='color:#e5e7eb; font-size:2rem; margin-bottom:4px;'>Comparaison des Modèles</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; margin-bottom:28px;'>Évaluation comparative des 4 algorithmes entraînés</p>", unsafe_allow_html=True)

    metrics = load_metrics()
    df_m = pd.DataFrame([{
        "Modèle": m["model_name"],
        "Recall": m["test_recall"],
        "Precision": m["test_precision"],
        "F1": m["test_f1"],
        "ROC-AUC": m["test_roc_auc"],
        "CV ROC-AUC": m["cv_roc_auc_mean"],
        "Temps (s)": m["train_time_sec"],
    } for m in metrics])

    # Tableau stylisé
    st.markdown("<div class='section-title'>Tableau des Métriques</div>", unsafe_allow_html=True)

    def color_best(val, col):
        best = df_m[col].max()
        if val == best:
            return "color: #34d399; font-weight: 700;"
        return "color: #e5e7eb;"

    st.dataframe(
        df_m.style.format({
            "Recall": "{:.3f}", "Precision": "{:.3f}", "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}", "CV ROC-AUC": "{:.3f}", "Temps (s)": "{:.2f}"
        }).background_gradient(subset=["Recall","Precision","F1","ROC-AUC","CV ROC-AUC"],
                               cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Recall & F1 par Modèle</div>", unsafe_allow_html=True)
        models = [m["model_name"] for m in metrics]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Recall", x=models, y=[m["test_recall"] for m in metrics],
                            marker_color="#f87171", text=[f"{m['test_recall']:.3f}" for m in metrics],
                            textposition="outside", textfont=dict(color="#e5e7eb")))
        fig.add_trace(go.Bar(name="F1", x=models, y=[m["test_f1"] for m in metrics],
                            marker_color="#6366f1", text=[f"{m['test_f1']:.3f}" for m in metrics],
                            textposition="outside", textfont=dict(color="#e5e7eb")))
        fig.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af", yaxis=dict(gridcolor="#1f2230", range=[0, 0.85]),
            legend=dict(font=dict(color="#9ca3af")),
            margin=dict(t=20, b=10, l=10, r=10), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>ROC-AUC (Test vs CV)</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="ROC-AUC Test", x=models, y=[m["test_roc_auc"] for m in metrics],
                             marker_color="#60a5fa", text=[f"{m['test_roc_auc']:.3f}" for m in metrics],
                             textposition="outside", textfont=dict(color="#e5e7eb")))
        fig2.add_trace(go.Bar(name="ROC-AUC CV", x=models, y=[m["cv_roc_auc_mean"] for m in metrics],
                             marker_color="#34d399", text=[f"{m['cv_roc_auc_mean']:.3f}" for m in metrics],
                             textposition="outside", textfont=dict(color="#e5e7eb")))
        fig2.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af", yaxis=dict(gridcolor="#1f2230", range=[0, 1.1]),
            legend=dict(font=dict(color="#9ca3af")),
            margin=dict(t=20, b=10, l=10, r=10), height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Matrices de confusion
    st.markdown("<div class='section-title'>Matrices de Confusion</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (m, col) in enumerate(zip(metrics, cols)):
        with col:
            cm = np.array(m["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, text=cm, texttemplate="%{text}",
                colorscale=[[0, "#1a1d26"], [1, "#6366f1"]],
                showscale=False,
                xgap=2, ygap=2,
            ))
            fig_cm.update_layout(
                title=dict(text=m["model_name"].replace(" ", "<br>"), font=dict(color="#e5e7eb", size=11)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis=dict(tickvals=[0,1], ticktext=["Prédit 0","Prédit 1"], tickfont=dict(size=9)),
                yaxis=dict(tickvals=[0,1], ticktext=["Réel 0","Réel 1"], tickfont=dict(size=9)),
                margin=dict(t=40, b=10, l=10, r=10), height=200
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # Recommandation
    st.markdown("<div class='section-title'>💡 Recommandation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1d26,#12151e); border:1px solid #6366f1;
                border-radius:12px; padding:20px 24px; color:#9ca3af; line-height:1.8; font-size:0.9rem;'>
        <b style='color:#e5e7eb;'>Modèle recommandé : Random Forest</b><br>
        Meilleur compromis ROC-AUC (0.785 test / 0.983 CV) avec une précision de 26.6% sur les churners.
        La Régression Logistique offre un meilleur Recall (64%) mais au prix d'une précision très faible (19%),
        générant trop de faux positifs pour un usage CRM. Le Random Forest reste le meilleur choix pour
        prioriser les actions de rétention à fort impact.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Interprétabilité":
    st.markdown("<h1 style='color:#e5e7eb; font-size:2rem; margin-bottom:4px;'>Interprétabilité SHAP</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280; margin-bottom:28px;'>Pourquoi le modèle prédit-il le churn ?</p>", unsafe_allow_html=True)

    # Top 10 features hardcodé depuis les résultats SHAP
    shap_data = {
        "csat_score": 0.0964,
        "payment_failures": 0.0432,
        "tenure_months": 0.0335,
        "discount_applied": 0.0240,
        "total_revenue": 0.0207,
        "gender": 0.0200,
        "monthly_logins": 0.0195,
        "survey_response": 0.0187,
        "payment_method": 0.0187,
        "referral_count": 0.0164,
    }

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("<div class='section-title'>Top 10 Features — Importance Globale SHAP</div>", unsafe_allow_html=True)
        features = list(shap_data.keys())
        values = list(shap_data.values())
        median_val = np.median(values)
        colors = ["#f87171" if v > median_val else "#6366f1" for v in values]

        fig = go.Figure(go.Bar(
            x=values[::-1], y=features[::-1], orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in values[::-1]],
            textposition="outside", textfont=dict(color="#e5e7eb", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af",
            xaxis=dict(gridcolor="#1f2230", title="Importance SHAP moyenne (|valeur|)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=60), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Interprétation Métier</div>", unsafe_allow_html=True)
        interpretations = [
            ("csat_score", "0.0964", "🔴", "Score satisfaction le + prédictif du churn"),
            ("payment_failures", "0.0432", "🔴", "Échecs paiement = signal d'alerte critique"),
            ("tenure_months", "0.0335", "🟢", "Ancienneté élevée = client plus stable"),
            ("discount_applied", "0.0240", "🟡", "Remise = risque si non suivi"),
            ("total_revenue", "0.0207", "🟢", "Fort CA = client à fort enjeu"),
        ]
        for feat, val, emoji, desc in interpretations:
            st.markdown(f"""
            <div style='background:#1a1d26; border:1px solid #2a2d3a; border-radius:8px;
                        padding:10px 14px; margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-family:Syne,sans-serif; font-weight:600; color:#e5e7eb; font-size:0.85rem;'>
                        {emoji} {feat}
                    </span>
                    <span style='font-size:0.75rem; color:#6366f1; font-weight:600;'>SHAP={val}</span>
                </div>
                <div style='font-size:0.78rem; color:#6b7280; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Graphiques SHAP générés
    st.markdown("<div class='section-title'>Graphiques SHAP Générés</div>", unsafe_allow_html=True)
    shap_figs = [
        ("12_shap_global_importance.png", "Importance Globale"),
        ("13_shap_beeswarm.png", "Beeswarm — Impact Directionnel"),
        ("14_shap_local_explanation.png", "Explication Locale — Churner"),
    ]

    tabs = st.tabs([title for _, title in shap_figs])
    for tab, (filename, title) in zip(tabs, shap_figs):
        with tab:
            img_path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"Image non trouvée : {img_path}. Lance `python src/explainability/shap_analysis.py` d'abord.")

    # Légende
    st.markdown("""
    <div style='background:#1a1d26; border:1px solid #2a2d3a; border-radius:8px;
                padding:14px 20px; margin-top:16px; font-size:0.85rem; color:#9ca3af; line-height:1.8;'>
        <b style='color:#e5e7eb;'>Comment lire ces graphiques ?</b><br>
        🔴 <b style='color:#f87171;'>Rouge</b> = une valeur élevée de cette feature <b>augmente</b> le risque de churn<br>
        🔵 <b style='color:#60a5fa;'>Bleu</b> = une valeur élevée <b>réduit</b> le risque de churn<br>
        L'écart horizontal entre les points = variance de l'impact selon les clients
    </div>
    """, unsafe_allow_html=True)

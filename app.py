# =============================================================
# Employee Attrition Prediction — Streamlit Dashboard
# Project: HR Analytics | Model: XGBoost
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }

    .title-banner {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .title-banner h1 { color: white; margin: 0; font-size: 2rem; }
    .title-banner p  { color: #c5cae9; margin: 0.5rem 0 0; font-size: 1rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #3949ab;
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; color: #1a237e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 1.6rem; font-weight: 700; color: #283593; }

    .risk-high   { background:#fdecea; border:2px solid #e53935; border-radius:10px; padding:1.2rem; color:#333333; }
    .risk-medium { background:#fff8e1; border:2px solid #f9a825; border-radius:10px; padding:1.2rem; color:#333333; }
    .risk-low    { background:#e8f5e9; border:2px solid #43a047; border-radius:10px; padding:1.2rem; color:#333333; }

    .risk-high   h2 { color:#c62828; }
    .risk-medium h2 { color:#f57f17; }
    .risk-low    h2 { color:#2e7d32; }
    .risk-high   p, .risk-high   li, .risk-high   b { color:#333333; }
    .risk-medium p, .risk-medium li, .risk-medium b { color:#333333; }
    .risk-low    p, .risk-low    li, .risk-low    b { color:#333333; }

    .section-header {
        background: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 1rem 0 0.5rem;
        border-left: 4px solid #3949ab;
        font-weight: 600;
        color: #1a237e;
        font-size: 1rem;
    }

    .imp-bar  { background: #e8eaf6; border-radius: 6px; height: 10px; margin-top: 3px; }
    .imp-fill { background: linear-gradient(90deg, #3949ab, #7986cb); border-radius: 6px; height: 10px; }

    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; color: white; }

    .footer {
        text-align: center;
        color: #9e9e9e;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Saved Artifacts ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model     = joblib.load("xgb_model.pkl")
        scaler    = joblib.load("scaler.pkl")
        feat_cols = joblib.load("feature_columns.pkl")
        return model, scaler, feat_cols
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}\n\nMake sure xgb_model.pkl, scaler.pkl, and feature_columns.pkl are in the same folder as app.py.")
        st.stop()

model, scaler, feature_columns = load_artifacts()

# Top feature importance
try:
    importance_dict = dict(zip(feature_columns, model.feature_importances_))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:6]
except Exception:
    top_features = []


# ── Helper: Build input DataFrame ────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    # Fill all columns with 0 first (safe default)
    df = pd.DataFrame([{col: 0 for col in feature_columns}])

    encode_map = {
        "OverTime"  : {"No": 0, "Yes": 1},
        "Department": {"Human Resources": 0, "Research & Development": 1, "Sales": 2},
    }

    for key, val in inputs.items():
        if key in feature_columns:
            if key in encode_map:
                df[key] = encode_map[key].get(val, 0)
            else:
                df[key] = val
    return df


# ── Risk Category ─────────────────────────────────────────────
def get_risk(prob):
    if prob >= 0.60:
        return "High", "risk-high", "🔴"
    elif prob >= 0.35:
        return "Medium", "risk-medium", "🟠"
    else:
        return "Low", "risk-low", "🟢"


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="title-banner">
    <h1>🏢 Employee Attrition Prediction Dashboard</h1>
    <p>AI-powered HR Analytics · IBM Dataset · XGBoost Model</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Only Required Inputs ───────────────────────────
with st.sidebar:
    st.markdown("## 👤 Employee Profile")
    st.markdown("Fill in the details and click **Predict**.")
    st.markdown("---")

    age            = st.slider("Age", 18, 60, 30)
    department     = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    overtime       = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
    job_sat        = st.select_slider("Job Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3)
    years_company  = st.slider("Years at Company", 0, 40, 5)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 100000, 5000, step=500)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Attrition Risk")


# ── Main Layout ───────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:

    if predict_btn:
        user_inputs = {
            "Age"            : age,
            "Department"     : department,
            "OverTime"       : overtime,
            "JobSatisfaction": job_sat,
            "YearsAtCompany" : years_company,
            "MonthlyIncome"  : monthly_income,
        }

        input_df = build_input_df(user_inputs)

        try:
            prob       = float(model.predict_proba(input_df)[0][1])
            prediction = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        risk_label, risk_class, risk_icon = get_risk(prob)
        pred_label = "Yes — Will Leave" if prediction == 1 else "No — Will Stay"

        # ── Metric Cards ─────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""
        <div class="metric-card">
            <h3>Prediction</h3>
            <p>{"⚠️ Leave" if prediction==1 else "✅ Stay"}</p>
        </div>""", unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
            <h3>Attrition Probability</h3>
            <p>{prob*100:.1f}%</p>
        </div>""", unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
            <h3>Risk Category</h3>
            <p>{risk_icon} {risk_label}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Risk Box ─────────────────────────────────────────
        if risk_label == "High":
            st.markdown(f"""
            <div class="risk-high">
                <h2>🔴 HIGH ATTRITION RISK — {prob*100:.1f}%</h2>
                <p><strong>This employee has a high probability of leaving.</strong><br>
                Immediate HR intervention is recommended.</p>
            </div>""", unsafe_allow_html=True)

        elif risk_label == "Medium":
            st.markdown(f"""
            <div class="risk-medium">
                <h2>🟠 MEDIUM ATTRITION RISK — {prob*100:.1f}%</h2>
                <p><strong>This employee shows some attrition signals.</strong><br>
                Proactive monitoring and engagement is advised.</p>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2>🟢 LOW ATTRITION RISK — {prob*100:.1f}%</h2>
                <p><strong>This employee is likely to stay.</strong><br>
                Continue standard engagement practices.</p>
            </div>""", unsafe_allow_html=True)

        # ── Probability Bar ───────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Risk Probability Meter</div>", unsafe_allow_html=True)
        st.progress(min(prob, 1.0))
        st.caption(f"Attrition Probability: **{prob*100:.2f}%**")



    else:
        st.info("👈 Fill in the employee profile on the left and click **Predict Attrition Risk**.")
        st.markdown("""
        <div class="metric-card">
            <h3>How This Works</h3>
            <p style="font-size:0.95rem; color:#424242;">
            1️⃣ Enter employee details in the sidebar<br>
            2️⃣ Click the Predict button<br>
            3️⃣ View risk prediction and HR advice<br>
            4️⃣ Download the report
            </p>
        </div>""", unsafe_allow_html=True)


    

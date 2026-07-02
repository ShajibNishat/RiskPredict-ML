import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="RiskPredict-ML", layout="centered")
st.title("📈 IT Project Risk Predictor")
st.markdown("Enter your project parameters below and get an instant risk assessment.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('risk_model.pkl')

model = load_model()
risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration (days)", min_value=1, max_value=365, value=60)
        team_size = st.number_input("Team size (persons)", min_value=1, max_value=50, value=5)
        budget = st.number_input("Budget (K USD)", min_value=10, max_value=5000, value=200)
    with col2:
        past_delays = st.slider("Past delay frequency", 0.0, 1.0, 0.3)
        req_changes = st.number_input("Requirement changes (count)", min_value=0, max_value=200, value=10)
    
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    features = np.array([[duration, team_size, budget, past_delays, req_changes]])
    pred_code = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features)[0])
    risk = risk_map[pred_code]
    
    if risk == "Low":
        st.success(f"✅ Predicted Risk: **{risk}** (Confidence: {confidence:.0%})")
    elif risk == "Medium":
        st.warning(f"⚠️ Predicted Risk: **{risk}** (Confidence: {confidence:.0%})")
    else:
        st.error(f"🔴 Predicted Risk: **{risk}** (Confidence: {confidence:.0%})")
    
    st.markdown("---")
    st.caption("Note: Prediction is based on a Random Forest model trained on synthetic project data.")
    